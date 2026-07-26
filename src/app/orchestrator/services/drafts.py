"""Reading drafts off disk and deciding whether they are fit to ship.

Two callers need the same answer about a draft — ``deliver_telegram`` to refuse
it, ``export_report`` to report it — and before this module they computed it
independently. They disagreed: export resolved a draft's brief by looking it up
among the briefs for topics listed in ``topics.json``, while delivery resolved
it from the draft's own ``supporting_topic_ids``. A draft whose topic wasn't in
``topics.json`` (a backfill, or a run whose topics file was replaced) was
therefore reported "below floor" by the run report and shipped by delivery.
Delivery's reading is the correct one — the question is whether *this draft*
has evidence behind it — so that is what :func:`certify` implements, once.

Loading is separated from certification on purpose: a draft that cannot be read
at all is a different class of failure from one that reads fine but isn't
trustworthy, and only the caller knows how to report each.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from app.config import Settings
from app.orchestrator import state
from app.orchestrator.schemas import PostProposal
from app.orchestrator.services.evidence_floor import meets_evidence_floor
from app.orchestrator.services.provenance import verify_draft

logger = logging.getLogger(__name__)

GateStatus = Literal["present", "missing", "malformed"]

LoadFailure = Literal["invalid_post_id", "draft_not_found", "draft_invalid_json"]


class DraftLoadError(Exception):
    """A draft could not be read. ``reason`` is the machine-greppable code the
    calling tool surfaces in its structured error summary."""

    def __init__(self, reason: LoadFailure, message: str) -> None:
        super().__init__(message)
        self.reason: LoadFailure = reason


@dataclass(frozen=True)
class LoadedDraft:
    """One draft as it exists on disk, with its gate verdict alongside.

    ``raw`` is kept next to the validated ``proposal`` because provenance is
    signed over the on-disk dict and ``PostProposal`` drops the ``_provenance``
    block on validate — verifying the signature needs the original structure,
    not the model.
    """

    proposal: PostProposal
    raw: dict[str, Any]
    gate_verdict: dict[str, Any] | None
    gate_status: GateStatus

    @property
    def post_id(self) -> str:
        return self.proposal.post_id

    @property
    def gate_passed(self) -> bool | None:
        """Strict identity check: exactly ``True`` or ``False``.

        The verdict file is untrusted — the writer subagent is an LLM and can
        author ``drafts/<post_id>.gate.json`` with a "helpful" ``"True"`` /
        ``1`` / ``["no"]``. A truthy-but-not-bool value would sail through a
        plain ``if not passed`` check, which is exactly the "model helpfully
        overrode the gate" failure mode the gate exists to prevent. Anything
        that isn't a real bool reads as unknown.
        """
        if self.gate_verdict is None:
            return None
        passed = self.gate_verdict.get("passed")
        return passed if passed is True or passed is False else None


@dataclass(frozen=True)
class Certification:
    """Whether a draft is fit to ship. Delivery turns this into a refusal;
    export turns it into report fields."""

    provenance_ok: bool
    gate_status: GateStatus
    gate_passed: bool | None
    floor_ok: bool
    floor_reason: str = ""

    @property
    def shippable(self) -> bool:
        return self.provenance_ok and self.gate_passed is True and self.floor_ok


def _read_gate(post_id: str, data_dir: str | Path) -> tuple[dict[str, Any] | None, GateStatus]:
    path = state.gate_path(post_id, data_dir)
    if not path.exists():
        return None, "missing"
    try:
        verdict = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Malformed gate verdict at %s: %s", path, exc)
        return None, "malformed"
    return (verdict, "present") if isinstance(verdict, dict) else (None, "malformed")


def load_draft(post_id: str, data_dir: str | Path) -> LoadedDraft:
    """Load one draft with its gate verdict. Raises :class:`DraftLoadError`."""
    try:
        draft_path = state.draft_path(post_id, data_dir)
    except ValueError as exc:
        raise DraftLoadError("invalid_post_id", str(exc)) from exc

    if not draft_path.exists():
        raise DraftLoadError("draft_not_found", f"No draft at {draft_path}")

    try:
        raw = json.loads(draft_path.read_text(encoding="utf-8"))
        proposal = PostProposal.model_validate(raw)
    except Exception as exc:
        raise DraftLoadError("draft_invalid_json", str(exc)) from exc

    if not isinstance(raw, dict):
        raise DraftLoadError("draft_invalid_json", f"Non-object draft at {draft_path}")

    gate_verdict, gate_status = _read_gate(proposal.post_id, data_dir)
    return LoadedDraft(
        proposal=proposal, raw=raw, gate_verdict=gate_verdict, gate_status=gate_status
    )


def load_drafts(data_dir: str | Path) -> list[LoadedDraft]:
    """Load every draft under ``drafts/``, skipping unreadable ones with a
    warning — a bad file shouldn't sink a whole export.

    Walks the drafts subdir rather than reading from the topics file: a draft
    may exist without a topic entry (a backfill), and a topic may exist without
    a draft (research failed, or the writer was gated out). The two are
    independent.
    """
    drafts_dir = Path(data_dir) / state.DRAFTS_DIRNAME
    if not drafts_dir.exists():
        return []

    loaded: list[LoadedDraft] = []
    for draft_file in sorted(drafts_dir.glob("*.json")):
        # Skip gate verdict files — they're loaded alongside their draft.
        if draft_file.name.endswith(".gate.json"):
            continue
        try:
            loaded.append(load_draft(draft_file.name[: -len(".json")], data_dir))
        except DraftLoadError as exc:
            logger.warning("Skipping unreadable draft at %s: %s", draft_file, exc)
    return loaded


def certify(draft: LoadedDraft, settings: Settings) -> Certification:
    """Assess one draft's provenance, gate verdict and evidence floor.

    The brief is resolved from the draft's own ``supporting_topic_ids`` — the
    draft names the evidence it stands on, so that is what gets checked. A
    draft naming no topic, or whose brief is missing or unreadable, fails the
    floor closed.
    """
    floor_ok, floor_reason = False, "draft names no supporting topic"
    if draft.proposal.supporting_topic_ids:
        topic_id = draft.proposal.supporting_topic_ids[0]
        brief = state.read_brief(topic_id, settings.orchestrator_data_dir)
        if brief is None:
            floor_reason = f"no readable brief for topic {topic_id!r}"
        else:
            floor_ok, why = meets_evidence_floor(brief, settings)
            floor_reason = "" if floor_ok else f"brief {topic_id!r} below evidence floor: {why}"

    return Certification(
        provenance_ok=verify_draft(draft.raw, settings),
        gate_status=draft.gate_status,
        gate_passed=draft.gate_passed,
        floor_ok=floor_ok,
        floor_reason=floor_reason,
    )


__all__ = [
    "Certification",
    "DraftLoadError",
    "GateStatus",
    "LoadFailure",
    "LoadedDraft",
    "certify",
    "load_draft",
    "load_drafts",
]
