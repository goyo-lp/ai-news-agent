"""Per-run export bundle: read a run's artifacts, render them, write the bundle.

A run leaves its output scattered across ``orchestrator_data_dir`` (articles,
topics, briefs, drafts, gate verdicts). This module collects that into a single
dated bundle under ``<outputs_dir>/<YYYY-MM-DD>/``:

  - ``posts.md``        — the day's proposals as Markdown, one section per
                          draft, cross-linked to the briefs behind them.
  - ``run_report.json`` — counts + paths, the shape an operator or downstream
                          scraper can branch on without re-walking the data dir.
  - ``briefs.json``     — every brief as a JSON array, schema-matched to
                          ``ResearchBrief``.

Per guiding principle #3 the bundle is the *compressed run artifact*: the
report captures counts and paths, never copies of the artifacts themselves.
Every file is validated through the boundary models in
:mod:`app.orchestrator.schemas`, so a future schema drift surfaces here as a
validation failure rather than silent garbage in the export.

The export refuses to overwrite a prior same-date bundle unless explicitly told
to: the first law of run-export tools is "don't be a footgun."
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.config import Settings
from app.orchestrator import state
from app.orchestrator.schemas import ResearchBrief, TopicCandidate
from app.orchestrator.services.drafts import (
    Certification,
    LoadedDraft,
    certify,
    load_drafts,
)

logger = logging.getLogger(__name__)

POSTS_MD_FILENAME = "posts.md"
RUN_REPORT_FILENAME = "run_report.json"
BRIEFS_EXPORT_FILENAME = "briefs.json"

# Strict date-slug regex; the only accepted path-component shape. A model
# supplying a free-form date (``"../../etc/passwd"``) does not pass it.
_DATE_SLUG_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


@dataclass
class RunArtifacts:
    """Everything one run left on disk, loaded and validated."""

    topics: list[TopicCandidate] = field(default_factory=list)
    drafts: list[LoadedDraft] = field(default_factory=list)
    briefs: list[ResearchBrief] = field(default_factory=list)


def normalize_date_slug(date_arg: str | None) -> str:
    """Resolve the export leaf's YYYY-MM-DD slug: today (UTC) when None; else
    validate the caller-supplied shape strictly. Raises ``ValueError``."""
    if date_arg is None:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if not _DATE_SLUG_RE.match(date_arg):
        raise ValueError(f"Invalid date slug {date_arg!r}; must be YYYY-MM-DD exactly.")
    return date_arg


def bundle_dir_for(outputs_dir: str, date_slug: str) -> Path:
    """``<outputs_dir>/<YYYY-MM-DD>/``. The path-traversal guard lives in
    ``normalize_date_slug``; this just joins."""
    return Path(outputs_dir) / date_slug


def _read_json_or_none(path: Path) -> Any:
    """Read a JSON file, returning None if absent or malformed. The export's
    stance is "export what's there, surface absence honestly" — a missing
    articles.json is reported as a zero count, not a hard error."""
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Failed to read %s: %s", path, exc)
        return None


def _load_topics(data_dir: Path) -> list[TopicCandidate]:
    raw = _read_json_or_none(state.topics_path(data_dir))
    if not isinstance(raw, list):
        return []
    try:
        return [TopicCandidate.model_validate(item) for item in raw]
    except Exception as exc:
        logger.warning("Malformed topics at %s: %s", state.topics_path(data_dir), exc)
        return []


def _load_briefs(data_dir: Path, topics: list[TopicCandidate]) -> list[ResearchBrief]:
    """Load the brief behind each topic, verified copy preferred. Iterates the
    topics list so the bundle's brief ordering matches the run's ranked order."""
    briefs: list[ResearchBrief] = []
    seen: set[str] = set()
    for topic in topics:
        # Dedupe defensively: two topics sharing a topic_id is an upstream
        # clustering bug — surface it as one brief, don't double-count.
        if topic.topic_id in seen:
            continue
        seen.add(topic.topic_id)
        brief = state.read_brief(topic.topic_id, data_dir)
        if brief is not None:
            briefs.append(brief)
    return briefs


def load_run_artifacts(data_dir: Path) -> RunArtifacts:
    """Read one run's topics, drafts and briefs off disk."""
    topics = _load_topics(data_dir)
    return RunArtifacts(
        topics=topics,
        drafts=load_drafts(data_dir),
        briefs=_load_briefs(data_dir, topics),
    )


def draft_integrity(
    artifacts: RunArtifacts, settings: Settings
) -> dict[str, Certification]:
    """Per-draft trustworthiness, keyed by post_id.

    Reporting only — the spine, submit_draft and deliver_telegram all enforce
    these upstream; the report surfaces the rates as first-class metrics. It
    uses the same :func:`certify` delivery does, so the report cannot claim a
    draft is below floor while delivery ships it.
    """
    return {draft.post_id: certify(draft, settings) for draft in artifacts.drafts}


def format_posts_md(
    artifacts: RunArtifacts, date_slug: str, integrity: dict[str, Certification]
) -> str:
    """Render the day's posts as Markdown: one section per draft — post_id +
    headline, the body verbatim, hashtags, numbered citations, the briefs it
    draws on, and a trailer with gate / provenance / evidence state."""
    if not artifacts.drafts:
        return f"# AI News Agent posts for {date_slug}\n\n(no drafts produced)\n"

    lines: list[str] = [f"# AI News Agent posts for {date_slug}", ""]
    for draft in artifacts.drafts:
        proposal = draft.proposal
        lines += [f"## {proposal.post_id} — {proposal.headline}", "", proposal.body, ""]
        if proposal.hashtags:
            lines += [" ".join(proposal.hashtags), ""]
        if proposal.citation_urls:
            lines.append("**Citations:**")
            lines += [
                f"{idx}. {url}" for idx, url in enumerate(proposal.citation_urls, start=1)
            ]
            lines.append("")
        linked = [
            b.topic_id for b in artifacts.briefs if b.topic_id in proposal.supporting_topic_ids
        ]
        if linked:
            lines += [f"**Briefs:** {', '.join(linked)}", ""]

        gate_state = {True: "passed", False: "failed", None: "absent"}[draft.gate_passed]
        marks = integrity.get(draft.post_id)
        provenance = "signed" if marks and marks.provenance_ok else "UNSIGNED"
        floor = (
            "floor-ok"
            if marks and marks.floor_ok
            else f"below-floor ({marks.floor_reason if marks else ''})"
        )
        lines += [
            f"**Gate:** {gate_state} | **Provenance:** {provenance} | **Evidence:** {floor}",
            "",
        ]
    return "\n".join(lines).rstrip() + "\n"


def build_run_report(
    data_dir: Path,
    bundle_dir: Path,
    date_slug: str,
    artifacts: RunArtifacts,
    integrity: dict[str, Certification],
) -> dict[str, Any]:
    """Assemble the structured run report: counts + paths only, never copies of
    the artifacts themselves (principle #3)."""
    drafts = artifacts.drafts
    return {
        "date": date_slug,
        "data_dir": str(data_dir),
        "bundle_dir": str(bundle_dir),
        "files": {
            "articles": str(state.articles_path(data_dir)),
            "topics": str(state.topics_path(data_dir)),
        },
        "counts": {
            "topics": len(artifacts.topics),
            "briefs": len(artifacts.briefs),
            "drafts": len(drafts),
            "drafts_passed": sum(1 for d in drafts if d.gate_passed is True),
            "drafts_failed": sum(1 for d in drafts if d.gate_passed is False),
            "drafts_gate_absent": sum(1 for d in drafts if d.gate_verdict is None),
            # Integrity metrics: unsigned drafts were authored outside the
            # writer's submit_draft tool; below-floor means the brief is too
            # weak to ship. Both must trend to zero.
            "drafts_provenance_invalid": sum(
                1 for marks in integrity.values() if not marks.provenance_ok
            ),
            "drafts_below_floor": sum(
                1 for marks in integrity.values() if not marks.floor_ok
            ),
        },
        "topics": [
            {
                "topic_id": t.topic_id,
                "title": t.title,
                "primary_url": t.primary_url,
                "score": t.score,
            }
            for t in artifacts.topics
        ],
        "drafts": [
            {
                "post_id": d.post_id,
                "headline": d.proposal.headline,
                "supporting_topic_ids": d.proposal.supporting_topic_ids,
                "gate_passed": d.gate_passed,
                "provenance_ok": _marks(integrity, d.post_id).provenance_ok,
                "floor_ok": _marks(integrity, d.post_id).floor_ok,
                "floor_reason": _marks(integrity, d.post_id).floor_reason,
            }
            for d in drafts
        ],
    }


def _marks(integrity: dict[str, Certification], post_id: str) -> Certification:
    """An absent certification reads as untrustworthy, never as a pass."""
    return integrity.get(
        post_id,
        Certification(
            provenance_ok=False, gate_status="missing", gate_passed=None, floor_ok=False
        ),
    )


def export_run(
    settings: Settings, *, date_arg: str | None = None, overwrite: bool = False
) -> dict[str, Any]:
    """Read the data dir and write the export bundle. Returns the compressed
    summary the caller reports — never the bundle's contents."""
    try:
        date_slug = normalize_date_slug(date_arg)
    except ValueError as exc:
        return {"status": "error", "reason": "invalid_date", "error": str(exc)}

    data_dir = Path(settings.orchestrator_data_dir)
    bundle_dir = bundle_dir_for(settings.outputs_dir, date_slug)

    # Idempotency guard: refuse to blow away a prior same-date bundle unless
    # the caller asked for it explicitly. The caller can ALWAYS re-run with
    # overwrite=True.
    if bundle_dir.exists() and any(bundle_dir.iterdir()) and not overwrite:
        return {
            "status": "error",
            "reason": "overwrite_required",
            "error": (
                f"Bundle directory {bundle_dir} already exists and is non-empty; "
                "pass overwrite=True to overwrite."
            ),
            "bundle_dir": str(bundle_dir),
        }

    bundle_dir.mkdir(parents=True, exist_ok=True)

    artifacts = load_run_artifacts(data_dir)
    integrity = draft_integrity(artifacts, settings)
    run_report = build_run_report(data_dir, bundle_dir, date_slug, artifacts, integrity)

    written = {
        POSTS_MD_FILENAME: format_posts_md(artifacts, date_slug, integrity),
        RUN_REPORT_FILENAME: json.dumps(run_report, indent=2, default=str),
        BRIEFS_EXPORT_FILENAME: json.dumps(
            [b.model_dump(mode="json") for b in artifacts.briefs], indent=2, default=str
        ),
    }
    for filename, content in written.items():
        (bundle_dir / filename).write_text(content, encoding="utf-8")

    return {
        "status": "ok",
        "date": date_slug,
        "bundle_dir": str(bundle_dir),
        "files": {
            "posts_md": str(bundle_dir / POSTS_MD_FILENAME),
            "run_report": str(bundle_dir / RUN_REPORT_FILENAME),
            "briefs": str(bundle_dir / BRIEFS_EXPORT_FILENAME),
        },
        "counts": run_report["counts"],
    }


__all__ = [
    "BRIEFS_EXPORT_FILENAME",
    "POSTS_MD_FILENAME",
    "RUN_REPORT_FILENAME",
    "RunArtifacts",
    "build_run_report",
    "bundle_dir_for",
    "draft_integrity",
    "export_run",
    "format_posts_md",
    "load_run_artifacts",
    "normalize_date_slug",
]
