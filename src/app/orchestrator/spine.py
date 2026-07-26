"""Deterministic propose spine — the default ``propose`` path.

Replaces the LLM-planner coordinator for the standard path:

    fetch_curated_ai_news → technical_rank → editor veto → select (cap +
    domain diversity) → research fan-out (per-topic timeout) → evidence
    floor → writer fan-out (per-topic timeout) → report

LLM judgment stays exactly where it earns its tokens: one batched editorial
veto at selection, research synthesis inside the research subagent, and the
writing itself inside the writer subagent. Everything that was a *prompt
contract* in the coordinator path — the topic cap, per-topic timeouts, the
evidence floor, writer-only drafting, no-inline-research — is code here.

Motivation (2026-07-25 production trace, 2,445 runs / 8m52s / 230 LLM calls):
the coordinator mechanically followed its 7-step pipeline — then skipped the
writer-subagent entirely, authored all five drafts itself via ``write_file``,
and self-gated them. A planner that adds no judgment over a script while
needing its guardrails babysat in prose is not an agent; it's an expensive,
flaky state machine. This module is the cheap, reliable state machine, and the
only propose path.

The run's outcome shape lives in :mod:`app.orchestrator.spine_result`.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from datetime import datetime, timezone
from typing import Any, Protocol

from langsmith import traceable

from app.config import Settings
from app.orchestrator import state
from app.orchestrator.schemas import TopicCandidate
from app.orchestrator.services.drafts import DraftLoadError, load_draft
from app.orchestrator.services.editor_veto import veto_irrelevant_topics
from app.orchestrator.services.evidence_floor import meets_evidence_floor
from app.orchestrator.spine_result import (
    SelectedTopic,
    SkippedTopic,
    SpineResult,
    SpineStatus,
    TaskOutcome,
    WrittenPost,
)
from app.orchestrator.subagents.research import build_research_agent
from app.orchestrator.subagents.writer import build_writer_agent
from app.orchestrator.tools.news import build_fetch_curated_ai_news_tool
from app.orchestrator.tools.technical_rank import build_technical_rank_tool

logger = logging.getLogger(__name__)


class _AgentLike(Protocol):
    """The slice of a compiled deep agent the spine drives. Structural, so
    tests can inject a scripted stub without LangGraph plumbing."""

    async def ainvoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any: ...


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested directly)
# ---------------------------------------------------------------------------


def select_topics(topics: list[TopicCandidate], cap: int) -> list[TopicCandidate]:
    """Diversity-aware greedy selection, deterministic.

    First pass: take topics in score order, one per ``primary_domain``. If
    the cap isn't reached, second pass fills remaining slots in score order
    regardless of domain. This is the rule the coordinator prompt described
    in prose ("weigh score against source diversity, no fixed quota") — as
    code it can't be talked out of. Input order is the rank order (topics
    arrive score-sorted from technical_rank); we sort defensively anyway.
    """
    if cap <= 0:
        return []
    ordered = sorted(topics, key=lambda t: t.score, reverse=True)
    picked: list[TopicCandidate] = []
    seen_domains: set[str] = set()
    for topic in ordered:
        if len(picked) >= cap:
            break
        if topic.primary_domain not in seen_domains:
            picked.append(topic)
            seen_domains.add(topic.primary_domain)
    if len(picked) < cap:
        for topic in ordered:
            if len(picked) >= cap:
                break
            if topic not in picked:
                picked.append(topic)
    return sorted(picked, key=lambda t: t.score, reverse=True)


_SLUG_STRIP_RE = re.compile(r"[^a-z0-9]+")


def post_id_for(title: str, date_slug: str, taken: set[str]) -> str:
    """Deterministic post_id: ``post-<yyyymmdd>-<title-slug>``. The legacy
    path let a model invent ids (``ccae-001``, ``ruff-001`` — ungreppable
    and non-unique across days). ``taken`` carries the run's already-issued
    ids so two similar titles get suffixed ``-2``/``-3``."""
    base = _SLUG_STRIP_RE.sub("-", title.lower()).strip("-")[:48].strip("-")
    base = base or "topic"
    candidate = f"post-{date_slug.replace('-', '')}-{base}"
    suffix = 2
    unique = candidate
    while unique in taken:
        unique = f"{candidate}-{suffix}"
        suffix += 1
    taken.add(unique)
    return unique


def _read_topics(settings: Settings) -> list[TopicCandidate]:
    path = state.topics_path(settings.orchestrator_data_dir)
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return [TopicCandidate.model_validate(item) for item in payload]
    except Exception as exc:
        logger.warning("spine: could not read topics file %s: %s", path, exc)
        return []


def _final_message_text(result: Any) -> str:
    """Pull the agent's final AI message text out of an ainvoke result —
    the same shape SubAgentMiddleware returned as the task() result."""
    messages = (result or {}).get("messages") if isinstance(result, dict) else None
    if not messages:
        return ""
    last = messages[-1]
    content = getattr(last, "content", "")
    if isinstance(content, list):
        return " ".join(
            str(part.get("text", "")) for part in content if isinstance(part, dict)
        )
    return str(content)


# ---------------------------------------------------------------------------
# Stage drivers
# ---------------------------------------------------------------------------


async def _run_agent_task(
    agent: _AgentLike,
    *,
    description: str,
    topic_id: str,
    timeout_seconds: int,
) -> TaskOutcome:
    """Invoke one subagent for one topic with a hard wall-clock bound — the
    per-topic timeout the research.py docstring always promised and the
    LLM-coordinator path never implemented. A timeout/error kills ONE
    topic's task, never the run."""
    try:
        result = await asyncio.wait_for(
            agent.ainvoke({"messages": [{"role": "user", "content": description}]}),
            timeout=timeout_seconds,
        )
        return TaskOutcome(
            topic_id=topic_id, status="ok", detail=_final_message_text(result)[:500]
        )
    except TimeoutError:
        logger.warning(
            "spine: task for topic %s timed out after %ss", topic_id, timeout_seconds
        )
        return TaskOutcome(
            topic_id=topic_id,
            status="timeout",
            detail=f"timed out after {timeout_seconds}s",
        )
    except Exception as exc:
        logger.warning("spine: task for topic %s failed: %s", topic_id, exc)
        return TaskOutcome(
            topic_id=topic_id, status="error", detail=f"{type(exc).__name__}: {exc}"
        )


async def _fan_out(
    agent: _AgentLike,
    tasks: list[tuple[str, str]],
    *,
    timeout_seconds: int,
) -> list[TaskOutcome]:
    """Run one subagent task per ``(topic_id, description)`` concurrently, each
    hard-bounded. Results come back in input order, so callers can zip them
    against whatever they derived the tasks from."""
    return list(
        await asyncio.gather(
            *(
                _run_agent_task(
                    agent,
                    description=description,
                    topic_id=topic_id,
                    timeout_seconds=timeout_seconds,
                )
                for topic_id, description in tasks
            )
        )
    )


def _research_description(topic: TopicCandidate) -> str:
    """The task description a research subagent boots with. Same contract as
    the delegation rules: topic_id + one-line summary + primary_url, plus the
    cluster's supporting_urls so corroboration-first actually has the URLs
    (the legacy path withheld them, which is part of why researchers leaned
    on the dead SearXNG)."""
    supporting = "\n".join(f"  - {u}" for u in topic.supporting_urls[:5]) or "  (none)"
    return (
        f"Research topic {topic.topic_id}: {topic.title}\n"
        f"summary_hint: {topic.summary_hint}\n"
        f"primary_url: {topic.primary_url}\n"
        f"supporting_urls:\n{supporting}"
    )


def _writer_description(topic_id: str, post_id: str) -> str:
    return (
        f"Write the LinkedIn post for topic_id={topic_id} as post_id={post_id}. "
        f"The verified brief is at briefs/{topic_id}.verified.json under your "
        "data directory."
    )


# ---------------------------------------------------------------------------
# The spine
# ---------------------------------------------------------------------------


def _stop(result: SpineResult, status: SpineStatus, *, error: str | None = None) -> SpineResult:
    """Terminate the run with an explicit status. Every exit from the pipeline
    goes through here — including the successful one — so the status a caller
    reads is always one a stage deliberately set."""
    result.status = status
    result.error = error
    return result


@traceable(run_type="chain", name="propose-spine", tags=["orchestrator", "spine"])
async def run_propose_spine(
    settings: Settings,
    *,
    run_id: str,
    limit: int | None = None,
    research_agent: _AgentLike | None = None,
    writer_agent: _AgentLike | None = None,
) -> SpineResult:
    """Run the deterministic propose pipeline for one run.

    ``research_agent`` / ``writer_agent`` are the test seam: compiled deep
    agents are built from ``settings`` when not supplied; tests inject
    scripted stubs implementing ``ainvoke``. Returns the run summary (also
    logged); the artifacts themselves land on disk under the orchestrator data
    dir, so export + delivery read them from there.
    """
    # Status is set only by _stop(), at every exit including the successful
    # one — so a stage added later cannot fall through and report "ok" on a run
    # that produced nothing.
    result = SpineResult(run_id=run_id, status="no_articles")

    # 1. Fetch + 2. Rank — deterministic tools driven from code (no planner
    # tokens spent sequencing them).
    fetch_tool = build_fetch_curated_ai_news_tool(settings)
    fetch_summary = json.loads(await fetch_tool.ainvoke({"limit": limit}))
    result.fetched = int(fetch_summary.get("count", 0))
    if result.fetched == 0:
        return _stop(result, "no_articles")

    rank_tool = build_technical_rank_tool(settings)
    rank_summary = json.loads(await rank_tool.ainvoke({}))
    if rank_summary.get("status") == "error":
        return _stop(result, "rank_failed", error=rank_summary.get("reason"))

    topics = _read_topics(settings)
    result.topics = len(topics)
    if not topics:
        return _stop(result, "no_topics")

    # 3. Editorial veto over the whole pool (one batched LLM call, fail-open),
    # then deterministic diversity-aware selection under the cap — the cap is
    # enforced HERE, in code, not in a prompt.
    llm_unavailable = not (settings.openrouter_api_key or "").strip()
    pool, vetoes = await veto_irrelevant_topics(
        topics, settings, dry_run=llm_unavailable
    )
    result.vetoed = vetoes

    selected = select_topics(pool, max(1, settings.max_topics_per_run))
    result.selected = [
        SelectedTopic(
            topic_id=t.topic_id, title=t.title, domain=t.primary_domain, score=t.score
        )
        for t in selected
    ]
    if not selected:
        return _stop(result, "no_selection")

    # 4. Research fan-out — concurrent, each task hard-bounded by
    # research_task_timeout_seconds. This is the code-side enforcement the
    # research subagent's docstring always referenced.
    researcher = research_agent or build_research_agent(settings)
    result.research = await _fan_out(
        researcher,
        [(t.topic_id, _research_description(t)) for t in selected],
        timeout_seconds=settings.research_task_timeout_seconds,
    )

    # 5. Evidence floor — only floored topics may be drafted. This is the
    # deterministic version of the coordinator's "skip insufficient_evidence"
    # prose rule, now with teeth: confidence + citation breadth thresholds.
    writable: list[TopicCandidate] = []
    for topic in selected:
        brief = state.read_brief(topic.topic_id, settings.orchestrator_data_dir)
        if brief is None:
            result.skipped_floor.append(
                SkippedTopic(topic_id=topic.topic_id, reason="no brief on disk")
            )
            continue
        passes, why = meets_evidence_floor(brief, settings)
        if passes:
            writable.append(topic)
        else:
            result.skipped_floor.append(SkippedTopic(topic_id=topic.topic_id, reason=why))
    if not writable:
        return _stop(result, "nothing_above_floor")

    # 6. Writer fan-out — concurrent, hard-bounded, deterministic post_ids.
    writer = writer_agent or build_writer_agent(settings)
    date_slug = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    taken: set[str] = set()
    assignments = [(topic, post_id_for(topic.title, date_slug, taken)) for topic in writable]
    write_results = await _fan_out(
        writer,
        [(topic.topic_id, _writer_description(topic.topic_id, post_id)) for topic, post_id in assignments],
        timeout_seconds=settings.writer_task_timeout_seconds,
    )

    # 7. Report each draft's gate verdict from disk. The delivery layer
    # re-certifies everything anyway; this is the run summary's honesty check,
    # and it reads the verdict through the same loader delivery uses so the two
    # can't disagree about what "gated" means.
    for (topic, post_id), outcome in zip(assignments, write_results, strict=True):
        try:
            gate_passed = load_draft(post_id, settings.orchestrator_data_dir).gate_passed
        except DraftLoadError as exc:
            logger.warning("spine: no readable draft for %s: %s", post_id, exc)
            gate_passed = None
        result.written.append(
            WrittenPost(
                topic_id=topic.topic_id,
                post_id=post_id,
                writer_status=outcome.status,
                gate_passed=gate_passed is True,
            )
        )
    logger.info(
        "spine complete: %d fetched, %d topics, %d selected, %d vetoed, %d floored-out, %d/%d drafts gated",
        result.fetched,
        result.topics,
        len(result.selected),
        len(result.vetoed),
        len(result.skipped_floor),
        result.drafts_passed,
        len(result.written),
    )
    return _stop(result, "ok")


__all__ = [
    "SpineResult",
    "post_id_for",
    "run_propose_spine",
    "select_topics",
]
