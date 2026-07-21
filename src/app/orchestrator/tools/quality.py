"""quality_gate — the coordinator's deterministic post-proposal gate.

Wraps :func:`app.orchestrator.services.quality_gate.check_proposal`. The
writer subagent (P4.2) calls this once per drafted ``PostProposal`` before
the coordinator queues it for delivery; a fail-with-reasons tells the
coordinator's retry loop to ask the writer to fix the proposal rather than
ship it. The gate is *deterministic* — no model calls, no Tavily, just word
count, single-topic, and anti-hype checks.

Per guiding principle #3 the proposal is the artifact on disk: the writer
writes ``drafts/<post_id>.json``, the gate reads it, writes the verdict to
``drafts/<post_id>.gate.json``, and returns a compressed summary. The
cleaned body is persisted with the verdict so a later retry passes the same
cleaned text back to the writer — the cleaner is deterministic, so the writer
sees what the gate saw.

Target environment: deepagents ``create_deep_agent`` — async-only tool, same
sibling pattern as news / technical_rank / fetch_article / tavily / verify_claim.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from app.config import Settings, get_settings
from app.orchestrator.schemas import PostProposal
from app.orchestrator.services.quality_gate import (
    QualityResult,
    check_proposal,
)

logger = logging.getLogger(__name__)

_DRAFTS_SUBDIR = "drafts"
_GATE_SUFFIX = ".gate.json"
_DRAFT_SUFFIX = ".json"


class QualityGateArgs(BaseModel):
    """Tool input. ``post_id`` identifies the draft on disk
    (``drafts/<post_id>.json``). The gate has no other knobs: the word-count
    window and the hype marker set are policy-internal to the
    :mod:`quality_gate` service."""

    post_id: str = Field(
        ...,
        description="The proposal's post_id; the draft is read from drafts/<post_id>.json.",
    )


def _draft_path(data_dir: str, post_id: str, *, gate: bool = False) -> Path:
    """Resolve the on-disk path for a draft or its gate verdict. ``gate=True``
    returns ``drafts/<post_id>.gate.json`` — the artifact the coordinator reads
    to decide retry-vs-deliver."""
    if "/" in post_id or post_id in {"", ".", ".."}:
        # Defense-in-depth against a model-supplied post_id escaping the
        # drafts/ subdir via path traversal. Surfaces as `status=error` in the
        # tool summary so the LLM sees a structured failure, not a raise.
        raise ValueError(f"Invalid post_id: {post_id!r}")
    suffix = _GATE_SUFFIX if gate else _DRAFT_SUFFIX
    return Path(data_dir) / _DRAFTS_SUBDIR / f"{post_id}{suffix}"


def read_proposal_from_state(data_dir: str, post_id: str) -> PostProposal:
    """Load the writer-written draft into a :class:`PostProposal`. Raises
    ``FileNotFoundError`` deliberately — a missing draft is a precondition
    error (the writer skipped a write), not silently empty content.

    Re-validation through pydantic on load is the boundary check that catches
    a draft on disk that drifted from the boundary contract before it reaches
    the gate."""
    path = _draft_path(data_dir, post_id)
    payload = json.loads(path.read_text(encoding="utf-8"))
    return PostProposal.model_validate(payload)


def write_gate_to_state(result: QualityResult, post_id: str, data_dir: str) -> Path:
    """Persist the gate verdict to ``drafts/<post_id>.gate.json`` and return
    the written path. Creates the dir if missing. Pure (no network):
    testable with a tmp directory. Mirrors the news.py / technical_rank.py /
    verify_claim.py writers — the on-disk shape round-trips through
    :class:`QualityResult` via a JSON dump + the dataclass's ``asdict``."""
    from dataclasses import asdict

    if "/" in post_id or post_id in {"", ".", ".."}:
        raise ValueError(f"Invalid post_id: {post_id!r}")
    path = _draft_path(data_dir, post_id, gate=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(asdict(result), indent=2, default=str),
        encoding="utf-8",
    )
    logger.info(
        "Wrote quality-gate verdict for %s (passed=%s, reasons=%d) to %s",
        post_id,
        result.passed,
        len(result.reasons),
        path,
    )
    return path


async def _gate_and_write(args: dict[str, Any], settings: Settings) -> dict[str, Any]:
    """Run the gate on one draft and persist the verdict. Returns a compressed
    summary; the draft body and the full reasons list stay on disk."""
    raw_post = str(args.get("post_id") or "").strip()
    if not raw_post:
        return {
            "post_id": "",
            "passed": False,
            "word_count": 0,
            "hashtags_count": 0,
            "single_topic": False,
            "has_hype": False,
            "reasons_count": 0,
            "status": "error",
            "reason": "post_id is required",
            "path": None,
        }

    failure_summary: dict[str, Any] = {
        "post_id": raw_post,
        "passed": False,
        "word_count": 0,
        "hashtags_count": 0,
        "single_topic": False,
        "has_hype": False,
        "reasons_count": 0,
        "status": "error",
        "reason": "",
        "path": None,
    }

    try:
        proposal = read_proposal_from_state(settings.orchestrator_data_dir, raw_post)
    except FileNotFoundError:
        failure_summary["reason"] = f"draft not found for post_id {raw_post!r}"
        return failure_summary
    except ValueError as exc:
        failure_summary["reason"] = str(exc)
        return failure_summary
    except Exception as exc:
        logger.warning("quality_gate draft load failed for %r: %s", raw_post, exc)
        failure_summary["reason"] = f"{type(exc).__name__}: {exc}"
        return failure_summary

    try:
        result = check_proposal(proposal)
    except Exception as exc:
        # The gate is pure, so an exception here is a service bug. Don't
        # propagate into the agent loop — surface as a structured error.
        logger.warning("quality_gate raised for %r: %s", raw_post, exc)
        failure_summary["reason"] = f"{type(exc).__name__}: {exc}"
        return failure_summary

    try:
        path = write_gate_to_state(result, raw_post, settings.orchestrator_data_dir)
    except ValueError as exc:
        failure_summary["reason"] = str(exc)
        return failure_summary

    gate_status = "passed" if result.passed else "failed"
    return {
        "post_id": raw_post,
        "passed": result.passed,
        "word_count": result.word_count,
        "hashtags_count": result.hashtags_count,
        "single_topic": result.single_topic,
        "has_hype": result.has_hype,
        "reasons_count": len(result.reasons),
        "status": gate_status,
        "reason": None,
        "path": str(path),
    }


def build_quality_gate_tool(settings: Settings | None = None) -> StructuredTool:
    """Construct the quality_gate langchain tool."""
    bound_settings = settings

    async def _async(post_id: str) -> str:
        s = bound_settings or get_settings()
        result = await _gate_and_write({"post_id": post_id}, s)
        return json.dumps(result, default=str)

    return StructuredTool.from_function(
        func=None,
        coroutine=_async,
        name="quality_gate",
        description=(
            "Run the deterministic post-proposal quality gate against the draft "
            "at drafts/<post_id>.json: word count 105–182 (after cleaning), "
            "exactly one supporting_topic_id, no hype markers, at least 3 "
            "usable hashtags. Writes the verdict to drafts/<post_id>.gate.json. "
            "Returns a JSON summary with {post_id, passed, word_count, "
            "hashtags_count, single_topic, has_hype, reasons_count, status, "
            "reason, path} — never the draft body or the full reasons list. "
            "Read the verdict from `path` (the cleaned_body and reasons live "
            "there). `status` is `passed` (all checks ok), `failed` (one or "
            "more checks failed; `reasons_count` > 0; the verdict artifact "
            "lists each), or `error` (load/gate raised; no verdict written)."
        ),
        args_schema=QualityGateArgs,
    )


quality_gate_tool = build_quality_gate_tool()