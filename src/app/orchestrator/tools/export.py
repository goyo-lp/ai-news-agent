"""export_report — the coordinator's per-run export tool.

Reads the artifacts a coordinator run has produced under
``orchestrator_data_dir`` (articles + topics + briefs + drafts + gate
verdicts) and writes a compact bundle under
``<outputs_dir>/<YYYY-MM-DD>/``:

  - ``posts.md``        — the day's post proposals rendered as Markdown,
                          one section per draft; cites the brief + the
                          gate verdict path for traceability.
  - ``run_report.json`` — a structured run summary (counts + paths), the
                          shape an operator / downstream analytics scraper
                          can branch on without re-walking the data dir.
  - ``briefs.json``     — every brief + its verification copy bundled into
                          one array, schema-matched to ``ResearchBrief``.

Per guiding principle #3 the export bundle is the *compressed run
artifact*. The coordinator's text reply to ``invoke`` carries the bundle
path (a compressed pointer); the coordinator's chat reply never carries
the bundle's contents. The schema-matched shape of every file in the
bundle is pinned against ``app.orchestrator.schemas``'s boundary models
(P0.2) — a future schema drift in ``PostProposal`` / ``ResearchBrief``
surfaces here as a validation failure, not as silent garbage in the
exported artifacts.

Datestamp convention: ``YYYY-MM-DD`` defaults to *today* (UTC) on tool
invocation; the caller can pass an explicit ``date`` kwarg to export a
prior run's artifacts (useful for backfill). The export is idempotent
in the sense that re-invoking on the same date overwrites the prior
bundle — OR does it? **No — by default it refuses to overwrite**, so a
live run can blow away a prior export only by passing ``overwrite=True``.
The first law of run-export tools is "don't be a footgun."

Path-traversal guarded at the date slug: only ``\\d{4}-\\d{2}-\\d{2}``
shapes are accepted (a model-supplied free-form date like
``"../../etc/passwd"`` doesn't pass the regex). Same parity-bound inline
guard pattern as the other coordinator-level tools (state.py's
centralized helper stays the deferred swap-up-onto target).

Target environment: deepagents ``create_deep_agent`` — async-only tool,
same sibling pattern as every other orchestrator tool.
"""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from app.config import Settings, get_settings
from app.orchestrator import state
from app.orchestrator.schemas import PostProposal, ResearchBrief, TopicCandidate

logger = logging.getLogger(__name__)

_ARTICLES_FILENAME = state.ARTICLES_FILENAME
_TOPICS_FILENAME = state.TOPICS_FILENAME
_BRIEFS_SUBDIR = state.BRIEFS_DIRNAME
_DRAFTS_SUBDIR = state.DRAFTS_DIRNAME

_POSTS_MD_FILENAME = "posts.md"
_RUN_REPORT_FILENAME = "run_report.json"
_BRIEFS_EXPORT_FILENAME = "briefs.json"

# Strict date-slug regex; the only accepted path-component shape. Matches
# the ADR-style "be strict about ids that go into paths" rule every other
# coordinator-level tool applies, applied to the export directory's leaf.
_DATE_SLUG_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


class ExportReportArgs(BaseModel):
    """Tool input. ``date`` is an optional ``YYYY-MM-DD`` override for the
    export's directory leaf; defaults to today (UTC). ``overwrite`` allows
    the export to blow away a prior same-date bundle; default False (the
    tool refuses with status=error reason=overwrite_required when an
    existing bundle is detected)."""

    date: str | None = Field(
        default=None,
        description=(
            "Optional YYYY-MM-DD datestamp for the export directory leaf "
            "(<outputs_dir>/<date>/). Defaults to today (UTC). Must match "
            "YYYY-MM-DD exactly; any other shape (incl. path-fragments) is "
            "rejected."
        ),
    )
    overwrite: bool = Field(
        default=False,
        description=(
            "Allow the export to overwrite a prior same-date bundle. Default "
            "False — the tool refuses with status=error reason="
            "overwrite_required when a same-date bundle directory already "
            "exists."
        ),
    )


def _normalize_date_slug(date_arg: str | None) -> str:
    """Resolve the export leaf's YYYY-MM-DD slug: today (UTC) when None;
    else validate the caller-supplied shape strictly."""
    if date_arg is None:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if not _DATE_SLUG_RE.match(date_arg):
        raise ValueError(
            f"Invalid date slug {date_arg!r}; must be YYYY-MM-DD exactly."
        )
    return date_arg


def _bundle_dir(outputs_dir: str, date_slug: str) -> Path:
    """Resolve the per-run bundle directory: ``<outputs_dir>/<YYYY-MM-DD>/``.
    The path-traversal guard is at the caller (via _normalize_date_slug's
    regex check); this just joins paths."""
    return Path(outputs_dir) / date_slug


def _read_json_or_none(path: Path) -> Any:
    """Read a JSON file, returning None if it doesn't exist or fails to parse.
    The tool's stance is "export what's there, surface absence honestly" —
    a missing articles.json is reported as ``articles_count=0`` in the run
    report, NOT a hard error. A malformed file is also None so the run
    still produces a partial export (with a warning logged)."""
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Failed to read %s: %s", path, exc)
        return None


def _load_topics(data_dir: Path) -> list[TopicCandidate]:
    """Load the topics file. Returns [] when absent or malformed —
    downstream export_paths keys off the topic list to find brief files."""
    raw = _read_json_or_none(data_dir / _TOPICS_FILENAME)
    if not isinstance(raw, list):
        return []
    try:
        return [TopicCandidate.model_validate(item) for item in raw]
    except Exception as exc:
        logger.warning("Malformed topics at %s: %s", data_dir / _TOPICS_FILENAME, exc)
        return []


def _load_drafts(data_dir: Path) -> list[tuple[PostProposal, dict[str, Any] | None]]:
    """Load every draft under ``drafts/<post_id>.json`` paired with its
    gate verdict (``drafts/<post_id>.gate.json`` if present, else None).
    Returns a list of ``(proposal, gate_verdict_or_None)`` tuples.

    Walks the drafts subdir rather than reading from a topics file — a
    draft may exist without a topic entry (e.g. a manual backfill), and a
    topic may exist without a draft (research failed, or the writer was
    gated-out). The two are independent in the export bundle.
    """
    drafts_dir = data_dir / _DRAFTS_SUBDIR
    if not drafts_dir.exists():
        return []

    pairs: list[tuple[PostProposal, dict[str, Any] | None]] = []
    for draft_file in sorted(drafts_dir.glob("*.json")):
        # Skip gate verdict files — they're loaded alongside their draft.
        if draft_file.name.endswith(".gate.json"):
            continue
        try:
            proposal = PostProposal.model_validate_json(
                draft_file.read_text(encoding="utf-8")
            )
        except Exception as exc:
            logger.warning("Malformed draft at %s: %s", draft_file, exc)
            continue
        gate_path = drafts_dir / f"{proposal.post_id}.gate.json"
        gate_verdict: dict[str, Any] | None = None
        if gate_path.exists():
            raw_gate = _read_json_or_none(gate_path)
            gate_verdict = raw_gate if isinstance(raw_gate, dict) else None
        pairs.append((proposal, gate_verdict))
    return pairs


def _load_briefs(data_dir: Path, topics: list[TopicCandidate]) -> list[ResearchBrief]:
    """Load every brief under ``briefs/<topic_id>.json`` paired with its
    verification copy (``briefs/<topic_id>.verified.json`` overrides the
    original when present — the verified file is the post-verification
    state the writer + delivery read). Iterates over the topics list so
    the export's brief ordering + topic_id coverage matches the run's
    ranked order."""
    briefs: list[ResearchBrief] = []
    if not topics:
        return briefs
    briefs_dir = data_dir / _BRIEFS_SUBDIR
    seen: set[str] = set()
    for topic in topics:
        # Avoid double-counting if two topics somehow share a topic_id
        # (clustering bug upstream — surface it via the dedup, don't blow up).
        if topic.topic_id in seen:
            continue
        seen.add(topic.topic_id)
        verified_path = briefs_dir / f"{topic.topic_id}.verified.json"
        brief_path = briefs_dir / f"{topic.topic_id}.json"
        path = verified_path if verified_path.exists() else brief_path
        if not path.exists():
            continue
        try:
            brief = ResearchBrief.model_validate_json(path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Malformed brief at %s: %s", path, exc)
            continue
        briefs.append(brief)
    return briefs


def _format_posts_md(
    drafts: list[tuple[PostProposal, dict[str, Any] | None]],
    briefs: list[ResearchBrief],
    date_slug: str,
) -> str:
    """Render the day's posts as Markdown: one section per draft, header
    is the post_id + headline, body is the post body (verbatim), followed
    by hashtags + citation URLs as a numbered list + a footer with the
    gate verdict's pass state. Briefs are cross-linked by
    supporting_topic_ids so a reader can navigate draft->brief->citations
    in the bundle."""
    if not drafts:
        return f"# AI News Agent posts for {date_slug}\n\n(no drafts produced)\n"

    lines: list[str] = [f"# AI News Agent posts for {date_slug}", ""]
    for proposal, gate_verdict in drafts:
        lines.append(f"## {proposal.post_id} — {proposal.headline}")
        lines.append("")
        lines.append(proposal.body)
        lines.append("")
        if proposal.hashtags:
            lines.append(" ".join(proposal.hashtags))
            lines.append("")
        if proposal.citation_urls:
            lines.append("**Citations:**")
            for idx, url in enumerate(proposal.citation_urls, start=1):
                lines.append(f"{idx}. {url}")
            lines.append("")
        if proposal.supporting_topic_ids:
            linked = [b for b in briefs if b.topic_id in proposal.supporting_topic_ids]
            if linked:
                lines.append(f"**Briefs:** {', '.join(b.topic_id for b in linked)}")
                lines.append("")
        gate_state = (
            "passed"
            if gate_verdict and gate_verdict.get("passed") is True
            else "failed"
            if gate_verdict and gate_verdict.get("passed") is False
            else "absent"
        )
        lines.append(f"**Gate:** {gate_state}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _build_run_report(
    data_dir: Path,
    outputs_bundle: Path,
    date_slug: str,
    topics: list[TopicCandidate],
    drafts: list[tuple[PostProposal, dict[str, Any] | None]],
    briefs: list[ResearchBrief],
) -> dict[str, Any]:
    """Assemble the structured run report dict. Captures counts + paths
    only — never copies of the artifacts themselves (principle #3)."""
    return {
        "date": date_slug,
        "data_dir": str(data_dir),
        "bundle_dir": str(outputs_bundle),
        "files": {
            "articles": str(data_dir / _ARTICLES_FILENAME),
            "topics": str(data_dir / _TOPICS_FILENAME),
        },
        "counts": {
            "topics": len(topics),
            "briefs": len(briefs),
            "drafts": len(drafts),
            "drafts_passed": sum(
                1 for _p, g in drafts if g and g.get("passed") is True
            ),
            "drafts_failed": sum(
                1 for _p, g in drafts if g and g.get("passed") is False
            ),
            "drafts_gate_absent": sum(
                1 for _p, g in drafts if g is None
            ),
        },
        "topics": [
            {
                "topic_id": t.topic_id,
                "title": t.title,
                "primary_url": t.primary_url,
                "score": t.score,
            }
            for t in topics
        ],
        "drafts": [
            {
                "post_id": p.post_id,
                "headline": p.headline,
                "supporting_topic_ids": p.supporting_topic_ids,
                # Use the same strict identity-check the deliver_telegram tool
                # (P6.2) pins — passed is exactly ``True`` / ``False`` / anything
                # else falls back to None. Catches a model-written "true"/1 string
                # leak falling through to the run_report. Loose-check would
                # silently echo the bad value here while the dependent counts
                # block (which uses ``is True``) ignores it.
                "gate_passed": (
                    bool(g.get("passed"))
                    if g is not None and g.get("passed") is True
                    else False
                    if g is not None and g.get("passed") is False
                    else None
                ),
            }
            for p, g in drafts
        ],
    }


async def _export_one(
    date_arg: str | None,
    overwrite: bool,
    settings: Settings,
) -> dict[str, Any]:
    """Read the data dir + write the export bundle. Returns the compressed
    summary the coordinator's LLM gets back — never the bundle's contents."""
    try:
        date_slug = _normalize_date_slug(date_arg)
    except ValueError as exc:
        return {
            "status": "error",
            "reason": "invalid_date",
            "error": str(exc),
        }

    data_dir = Path(settings.orchestrator_data_dir)
    bundle_dir = _bundle_dir(settings.outputs_dir, date_slug)

    # Idempotency guard: refuse to blow away a prior same-date bundle unless
    # the caller (the LLM) passed overwrite=True explicitly. The tool's
    # stance: don't be a footgun. The caller can ALWAYS re-run with overwrite.
    if bundle_dir.exists() and any(bundle_dir.iterdir()) and not overwrite:
        return {
            "status": "error",
            "reason": "overwrite_required",
            "error": (
                f"Bundle directory {bundle_dir} already exists and is non-"
                f"empty; pass overwrite=True to overwrite."
            ),
            "bundle_dir": str(bundle_dir),
        }

    bundle_dir.mkdir(parents=True, exist_ok=True)

    topics = _load_topics(data_dir)
    drafts = _load_drafts(data_dir)
    briefs = _load_briefs(data_dir, topics)

    posts_md = _format_posts_md(drafts, briefs, date_slug)
    run_report = _build_run_report(data_dir, bundle_dir, date_slug, topics, drafts, briefs)
    briefs_payload = [b.model_dump(mode="json") for b in briefs]

    posts_path = bundle_dir / _POSTS_MD_FILENAME
    run_report_path = bundle_dir / _RUN_REPORT_FILENAME
    briefs_export_path = bundle_dir / _BRIEFS_EXPORT_FILENAME

    posts_path.write_text(posts_md, encoding="utf-8")
    run_report_path.write_text(
        json.dumps(run_report, indent=2, default=str), encoding="utf-8"
    )
    briefs_export_path.write_text(
        json.dumps(briefs_payload, indent=2, default=str), encoding="utf-8"
    )

    return {
        "status": "ok",
        "date": date_slug,
        "bundle_dir": str(bundle_dir),
        "files": {
            "posts_md": str(posts_path),
            "run_report": str(run_report_path),
            "briefs": str(briefs_export_path),
        },
        "counts": run_report["counts"],
    }


def build_export_report_tool(settings: Settings | None = None) -> StructuredTool:
    """Construct the export_report LangChain tool.

    Settings resolve lazily on first call when not supplied, mirroring
    every other orchestrator tool factory."""
    bound_settings = settings

    async def _async(date: str | None = None, overwrite: bool = False) -> str:
        s = bound_settings or get_settings()
        result = await _export_one(date, overwrite, s)
        return json.dumps(result, default=str)

    return StructuredTool.from_function(
        func=None,
        coroutine=_async,
        name="export_report",
        description=(
            "Read the artifacts a coordinator run has produced under "
            "orchestrator_data_dir (articles + topics + briefs + drafts + "
            "gate verdicts) and write a compact export bundle under "
            "<outputs_dir>/<YYYY-MM-DD>/ with posts.md (rendered posts), "
            "run_report.json (structured summary with counts + paths), and "
            "briefs.json (every brief as a JSON array). Returns a JSON "
            "summary with {status, date, bundle_dir, files, counts} — never "
            "the bundle contents. Defaults to today (UTC); an optional "
            "`date` override must match YYYY-MM-DD exactly. Refuses to "
            "overwrite a prior same-date bundle unless `overwrite=True` — "
            "the first law of run-export tools is 'don't be a footgun'."
        ),
        args_schema=ExportReportArgs,
    )


# Convenience singleton — same pattern as every sibling tool.
export_report_tool = build_export_report_tool()