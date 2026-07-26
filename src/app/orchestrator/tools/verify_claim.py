"""verify_claim — the coordinator's brief-fact-checking tool.

Wraps :func:`app.orchestrator.services.brief_verifier.BriefVerifier.verify_one`
to verify a single :class:`ResearchBrief` against independently gathered web
evidence (SearXNG search + local extraction) + an OpenRouter fact-checking
model. The research subagent
(P4.1) invokes this once per brief it produces — the brief researched from
the curated news (and possibly the fetched real article) is double-checked
before it's handed to the writer subagent.

Per guiding principle #3 the brief is the *artifact* on disk — it travels in
both directions through this tool:
  * input  : the research subagent writes ``briefs/<topic_id>.json`` to the
            orchestrator data dir before calling the tool;
  * output : the verified brief is written back to
            ``briefs/<topic_id>.verified.json`` so the writer subagent and the
            quality_gate tool read the post-verification copy.

The tool returns a compressed summary: ``{topic_id, verification_status,
verification_confidence, notes_count, path}`` — never the brief body. The
status enum matches :class:`ResearchBrief.verification_status` (verified /
partially_verified / insufficient_evidence / failed).

Target environment: deepagents ``create_deep_agent`` — async-only tool, same
pattern siblings as news / technical_rank / fetch_article / web.
"""
from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from app.config import Settings, get_settings
from app.orchestrator.budgets import AttemptBudget
from app.orchestrator import state
from app.orchestrator.schemas import ResearchBrief
from app.orchestrator.services.brief_verifier import BriefVerifier

logger = logging.getLogger(__name__)

_VALID_STATUSES = {
    "unverified",
    "verified",
    "partially_verified",
    "insufficient_evidence",
    "failed",
}

class VerifyClaimArgs(BaseModel):
    """Tool input. `topic_id` identifies the brief file to load (looks up
    ``briefs/<topic_id>.json`` in the orchestrator data dir). `hours_back`
    defaults to the verifier's ~24h if not supplied."""

    topic_id: str = Field(
        ..., description="The brief's topic_id; the brief is read from briefs/<topic_id>.json."
    )
    hours_back: int | None = Field(
        default=None,
        description=(
            "Hours back to search for corroborating evidence. Omit to use the "
            "verifier's default (~24h; maps to SearXNG's day/week/month bucket)."
        ),
    )


def write_verified_brief_to_state(brief: ResearchBrief, data_dir: str) -> Path:
    """Serialize a verified brief to ``briefs/<topic_id>.verified.json`` and
    return the written path. Creates the dir if missing. Pure (no network):
    testable with a tmp directory. Mirrors the news.py / technical_rank.py
    writers — the on-disk shape round-trips through ``ResearchBrief``.

    Path-traversal on the topic_id is guarded by ``state.verified_brief_path``,
    which raises ``ValueError`` — the caller surfaces that as ``status=error``
    in the tool summary."""
    path = state.verified_brief_path(brief.topic_id, data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(brief.model_dump(mode="json"), indent=2, default=str),
        encoding="utf-8",
    )
    logger.info(
        "Wrote verified brief for %s (%s, conf=%.2f) to %s",
        brief.topic_id,
        brief.verification_status,
        brief.verification_confidence,
        path,
    )
    return path


def read_brief_from_state(data_dir: str, topic_id: str) -> ResearchBrief:
    """Load the pre-verification brief written by the research subagent into a
    :class:`ResearchBrief`. Raises ``FileNotFoundError`` deliberately — a
    missing brief is a real precondition error (the subagent skipped a write),
    not silently empty content.

    Re-validation through pydantic on load is the boundary check that catches
    a brief on disk that drifted from the boundary contract (someone hand-edited
    it, the writer produced a stale shape) before it reaches the verifier."""
    payload = json.loads(state.brief_path(topic_id, data_dir).read_text(encoding="utf-8"))
    return ResearchBrief.model_validate(payload)


def _failed(topic_id: str, reason: str) -> dict[str, Any]:
    """The tool's failure shape. Every failure mode reports ``failed`` with
    zero confidence — an unverified brief must never read as a weak pass."""
    return {
        "topic_id": topic_id,
        "verification_status": "failed",
        "verification_confidence": 0.0,
        "notes_count": 0,
        "status": "error",
        "reason": reason,
        "path": None,
    }


def _load_brief(settings: Settings, topic_id: str) -> ResearchBrief:
    """Load the brief to verify, normalizing every read failure into a
    ``ValueError`` whose message is the reason the tool reports."""
    try:
        return read_brief_from_state(settings.orchestrator_data_dir, topic_id)
    except FileNotFoundError:
        raise ValueError(f"brief not found for topic_id {topic_id!r}") from None
    except ValueError:
        # Path-traversal defense and pydantic validation drift already carry a
        # usable message.
        raise
    except Exception as exc:
        logger.warning("verify_claim brief load failed for %r: %s", topic_id, exc)
        raise ValueError(f"{type(exc).__name__}: {exc}") from exc


def _attempts_exhausted(settings: Settings, topic_id: str, budget: AttemptBudget) -> dict[str, Any]:
    """Refuse further spend, surfacing the last known verdict so the subagent
    can finish honestly instead of reporting nothing."""
    verified_path = state.verified_brief_path(topic_id, settings.orchestrator_data_dir)
    last = state.read_brief(topic_id, settings.orchestrator_data_dir)
    return {
        "topic_id": topic_id,
        "verification_status": last.verification_status if last else "unknown",
        "verification_confidence": last.verification_confidence if last else 0.0,
        "notes_count": 0,
        "status": "attempts_exhausted",
        "reason": (
            f"verification attempt budget exhausted ({budget.max_attempts} per "
            "topic); stop re-verifying and report the verdict you have"
        ),
        "path": str(verified_path) if verified_path.exists() else None,
    }


async def _verify_and_write(
    args: dict[str, Any], settings: Settings, budget: AttemptBudget
) -> dict[str, Any]:
    """Run the verifier on one brief and persist the verified copy. Returns a
    compressed summary; the brief body never rides back through the LLM."""
    topic_id = str(args.get("topic_id") or "").strip()
    if not topic_id:
        return _failed("", "topic_id is required")

    try:
        brief = _load_brief(settings, topic_id)
    except ValueError as exc:
        return _failed(topic_id, str(exc))

    # Attempt budget: refuse churn before spending. Checked after the brief
    # load so a missing brief doesn't burn an attempt.
    if budget.exhausted(topic_id):
        return _attempts_exhausted(settings, topic_id, budget)
    budget.record_attempt(topic_id)

    hours_back_raw = args.get("hours_back")
    try:
        # Wall-clock bound: one verification call may not exceed
        # verification_timeout_seconds (trace showed single calls at 80-94s).
        verified = await asyncio.wait_for(
            BriefVerifier(settings).verify_one(
                brief,
                hours_back=int(hours_back_raw) if hours_back_raw is not None else 24,
                dry_run=not (settings.openrouter_api_key or "").strip(),
            ),
            timeout=settings.verification_timeout_seconds,
        )
    except TimeoutError:
        logger.warning(
            "verify_claim timed out for %r after %ss",
            topic_id,
            settings.verification_timeout_seconds,
        )
        return _failed(
            topic_id,
            f"verification timed out after {settings.verification_timeout_seconds}s",
        )
    except Exception as exc:
        # The verifier's per-brief worker already falls back inside, so a raise
        # here is catastrophic. Record it as `failed` and never propagate into
        # the agent loop.
        logger.warning("verify_claim verifier raised for %r: %s", topic_id, exc)
        return _failed(topic_id, f"{type(exc).__name__}: {exc}")

    path = write_verified_brief_to_state(verified, settings.orchestrator_data_dir)
    status = verified.verification_status
    if status not in _VALID_STATUSES:
        # Catch a verifier that emits something the enum doesn't recognize —
        # surface it distinctly rather than silently coercing.
        logger.warning("verify_claim returned unknown status %r for %s", status, topic_id)
        status = "failed"

    return {
        "topic_id": topic_id,
        "verification_status": status,
        "verification_confidence": round(float(verified.verification_confidence), 3),
        "notes_count": len(verified.verification_notes),
        "status": "ok",
        "reason": None,
        "path": str(path),
    }


def build_verify_claim_tool(settings: Settings | None = None) -> StructuredTool:
    """Construct the verify_claim langchain tool.

    The per-topic attempt budget is created here and closed over: its lifetime
    is this tool's lifetime — one per run — so nothing has to reset it, and two
    lanes in one process can't share a budget.

    The 2026-07-25 production trace showed the research subagent calling
    verify_claim 16x across 5 topics (one call 94s) in a soften-and-reverify
    churn loop when corroboration was thin. The prompt's "re-verify ONCE"
    guidance is defense-in-depth; this budget is the mechanism."""
    bound_settings = settings
    budget = AttemptBudget(
        (bound_settings or get_settings()).verification_max_attempts
    )

    async def _async(topic_id: str, hours_back: int | None = None) -> str:
        s = bound_settings or get_settings()
        result = await _verify_and_write(
            {"topic_id": topic_id, "hours_back": hours_back}, s, budget
        )
        return json.dumps(result, default=str)

    return StructuredTool.from_function(
        func=None,
        coroutine=_async,
        name="verify_claim",
        description=(
            "Verify one research brief (briefs/<topic_id>.json) against "
            "independent web evidence + an OpenRouter fact-checking model, "
            "and write the post-verification brief to "
            "briefs/<topic_id>.verified.json. Returns a JSON summary with "
            "{topic_id, verification_status, verification_confidence, "
            "notes_count, status, reason, path} — never the brief body. Read "
            "the verified brief from `path` when `status == \"ok\"`. "
            "`verification_status` is one of "
            "verified | partially_verified | insufficient_evidence | failed. "
            "Falls back to a heuristic verdict (partially_verified) when "
            "OPENROUTER_API_KEY is unset so a subagent can exercise this "
            "without a live model key. Attempts are budgeted per topic "
            "(VERIFICATION_MAX_ATTEMPTS, default 2): once exhausted the tool "
            "returns status=\"attempts_exhausted\" with the last verdict — "
            "stop re-verifying at that point."
        ),
        args_schema=VerifyClaimArgs,
    )


verify_claim_tool = build_verify_claim_tool()