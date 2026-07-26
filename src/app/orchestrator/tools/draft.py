"""submit_draft — the writer subagent's draft-submission tool, and the ONLY
sanctioned way a draft lands in ``drafts/``.

Why this tool exists (2026-07-25 production trace): the writer's contract
used to be "write ``drafts/<post_id>.json`` with deepagents' built-in
``write_file``" — a tool the LLM *coordinator* also carries. The coordinator
skipped the writer-subagent entirely, wrote all five drafts itself, and
self-gated them: no linkedin-voice skill, no writer model tier, no
write-until-pass loop. This tool removes that failure mode by making draft
creation a deterministic choke point the coordinator does not have:

  1. **Validation** — the proposal is validated against ``PostProposal``
     here (not at gate time), so a malformed draft never touches disk.
  2. **Evidence floor** — the brief behind ``supporting_topic_ids[0]`` must
     clear :func:`meets_evidence_floor`; a below-floor topic is refused at
     write time, not after delivery.
  3. **Provenance** — the draft is HMAC-signed (see
     :mod:`app.orchestrator.services.provenance`); export + delivery verify
     the signature, so a ``write_file``-authored draft (no key, no
     signature) is refused downstream.

Per guiding principle #3 the tool returns a compressed summary; the draft
body stays on disk.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from app.config import Settings, get_settings
from app.orchestrator import state
from app.orchestrator.schemas import PostProposal, ResearchBrief
from app.orchestrator.services.evidence_floor import meets_evidence_floor
from app.orchestrator.services.provenance import PROVENANCE_KEY, sign_draft

logger = logging.getLogger(__name__)


class SubmitDraftArgs(BaseModel):
    """Tool input. The writer passes the post_id plus every PostProposal
    field as a nested ``proposal`` object; the tool validates, floor-checks,
    signs, and writes the draft."""

    post_id: str = Field(
        ...,
        description=(
            "The proposal's post_id. Must equal proposal.post_id; the draft "
            "is written to drafts/<post_id>.json."
        ),
    )
    proposal: PostProposal = Field(
        ...,
        description=(
            "The full PostProposal: angle, headline, body, hashtags, "
            "supporting_topic_ids (exactly one), citation_urls (only URLs "
            "from the brief's citations), confidence."
        ),
    )


def write_signed_draft(
    proposal: PostProposal, settings: Settings
) -> Path:
    """Serialize the proposal with its provenance block to
    ``drafts/<post_id>.json`` and return the path. Creates the dir if
    missing. Path-traversal-guarded by ``state.draft_path``."""
    path = state.draft_path(proposal.post_id, settings.orchestrator_data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = proposal.model_dump(mode="json")
    payload[PROVENANCE_KEY] = sign_draft(payload, settings)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    logger.info("Wrote signed draft %s to %s", proposal.post_id, path)
    return path


def _load_brief_for_floor(
    settings: Settings, topic_id: str
) -> ResearchBrief | None:
    """Load the brief behind a draft for the floor check. The verified copy
    wins; the pre-verification copy is the fallback so the refusal reason can
    name the actual status (``unverified``) instead of a bare 'missing'."""
    data_dir = settings.orchestrator_data_dir
    for path in (
        state.verified_brief_path(topic_id, data_dir),
        state.brief_path(topic_id, data_dir),
    ):
        if path.exists():
            try:
                return ResearchBrief.model_validate_json(path.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.warning("submit_draft brief load failed at %s: %s", path, exc)
                return None
    return None


async def _submit_and_write(args: dict[str, Any], settings: Settings) -> dict[str, Any]:
    """Validate + floor-check + sign + write one draft. Returns a compressed
    summary; every failure mode is structured (never raises into the agent
    loop)."""
    post_id = str(args.get("post_id") or "").strip()

    def failure(reason: str, **extra: Any) -> dict[str, Any]:
        return {"post_id": post_id, "status": "error", "reason": reason, "path": None, **extra}

    if not post_id:
        return failure("post_id is required")
    try:
        state.draft_path(post_id, settings.orchestrator_data_dir)
    except ValueError as exc:
        return failure("invalid_post_id", error=str(exc))

    raw_proposal = args.get("proposal")
    try:
        proposal = raw_proposal if isinstance(raw_proposal, PostProposal) else (
            PostProposal.model_validate(raw_proposal)
        )
    except Exception as exc:
        return failure("proposal_invalid", error=str(exc))

    if proposal.post_id != post_id:
        return failure(
            "post_id_mismatch",
            error=f"arg post_id {post_id!r} != proposal.post_id {proposal.post_id!r}",
        )

    # Evidence floor: exactly one supporting topic is required by the gate
    # anyway; check the brief behind it BEFORE the draft exists on disk.
    if len(proposal.supporting_topic_ids) != 1:
        return failure(
            "supporting_topic_ids_invalid",
            error=f"expected exactly one supporting_topic_id, got {len(proposal.supporting_topic_ids)}",
        )
    topic_id = proposal.supporting_topic_ids[0]
    brief = _load_brief_for_floor(settings, topic_id)
    if brief is None:
        return failure(
            "verification_floor",
            error=f"no readable brief for topic_id {topic_id!r}; research + verify first",
        )
    passes, why = meets_evidence_floor(brief, settings)
    if not passes:
        return failure(
            "verification_floor",
            error=(
                f"brief {topic_id!r} is below the evidence floor ({why}); "
                "do not draft this topic — report it as skipped"
            ),
        )

    path = write_signed_draft(proposal, settings)
    return {
        "post_id": post_id,
        "status": "ok",
        "reason": None,
        "path": str(path),
        "verification_status": brief.verification_status,
    }


def build_submit_draft_tool(settings: Settings | None = None) -> StructuredTool:
    """Construct the submit_draft LangChain tool. Writer-subagent-owned: the
    coordinator deliberately does NOT get it. Settings resolve lazily when not
    supplied, mirroring every sibling tool factory."""
    bound_settings = settings

    async def _async(post_id: str, proposal: dict[str, Any]) -> str:
        s = bound_settings or get_settings()
        result = await _submit_and_write({"post_id": post_id, "proposal": proposal}, s)
        return json.dumps(result, default=str)

    return StructuredTool.from_function(
        func=None,
        coroutine=_async,
        name="submit_draft",
        description=(
            "Submit one LinkedIn post draft. Validates the PostProposal, "
            "enforces the evidence floor on the brief behind "
            "supporting_topic_ids[0] (verified, or partially_verified with "
            "enough confidence + citations), signs the draft for provenance, "
            "and writes drafts/<post_id>.json. This is the ONLY way to create "
            "a draft — never write drafts/<post_id>.json with write_file. "
            "Returns a JSON summary {post_id, status, reason, path, "
            "verification_status}. status=error with reason=verification_floor "
            "means the topic must not be drafted; report it as skipped. "
            "After submitting, call quality_gate with the post_id and fix + "
            "resubmit until it passes."
        ),
        args_schema=SubmitDraftArgs,
    )


submit_draft_tool = build_submit_draft_tool()
