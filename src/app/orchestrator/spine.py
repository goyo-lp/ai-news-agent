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
flaky state machine. This module is the cheap, reliable state machine. The
legacy LLM-coordinator path remains available via ``propose
--legacy-coordinator`` for comparison.
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
from app.orchestrator.schemas import ResearchBrief, TopicCandidate
from app.orchestrator.services.editor_veto import veto_irrelevant_topics
from app.orchestrator.services.evidence_floor import meets_evidence_floor
from app.orchestrator.subagents.research import build_research_agent
from app.orchestrator.subagents.writer import build_writer_agent
from app.orchestrator.tools import verify_claim as verify_claim_tool_mod
from app.orchestrator.tools import web as web_tool_mod
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


def _read_verified_brief(settings: Settings, topic_id: str) -> ResearchBrief | None:
    try:
        path = state.verified_brief_path(topic_id, settings.orchestrator_data_dir)
    except ValueError:
        return None
    if not path.exists():
        return None
    try:
        return ResearchBrief.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("spine: unreadable verified brief for %s: %s", topic_id, exc)
        return None


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
) -> dict[str, Any]:
    """Invoke one subagent for one topic with a hard wall-clock bound — the
    per-topic timeout the research.py docstring always promised and the
    LLM-coordinator path never implemented. A timeout/error kills ONE
    topic's task, never the run."""
    try:
        result = await asyncio.wait_for(
            agent.ainvoke({"messages": [{"role": "user", "content": description}]}),
            timeout=timeout_seconds,
        )
        return {
            "topic_id": topic_id,
            "status": "ok",
            "summary": _final_message_text(result)[:500],
        }
    except TimeoutError:
        logger.warning(
            "spine: task for topic %s timed out after %ss", topic_id, timeout_seconds
        )
        return {
            "topic_id": topic_id,
            "status": "timeout",
            "error": f"timed out after {timeout_seconds}s",
        }
    except Exception as exc:
        logger.warning("spine: task for topic %s failed: %s", topic_id, exc)
        return {
            "topic_id": topic_id,
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
        }


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


@traceable(run_type="chain", name="propose-spine", tags=["orchestrator", "spine"])
async def run_propose_spine(
    settings: Settings,
    *,
    run_id: str,
    limit: int | None = None,
    research_agent: _AgentLike | None = None,
    writer_agent: _AgentLike | None = None,
) -> dict[str, Any]:
    """Run the deterministic propose pipeline for one run.

    ``research_agent`` / ``writer_agent`` are the test seam: compiled deep
    agents are built from ``settings`` when not supplied; tests inject
    scripted stubs implementing ``ainvoke``. Returns a structured run
    summary (also logged); the artifacts themselves land on disk under the
    orchestrator data dir, exactly as the LLM-coordinator path produced
    them, so export + delivery are unchanged.
    """
    summary: dict[str, Any] = {
        "run_id": run_id,
        "fetched": 0,
        "topics": 0,
        "vetoed": [],
        "selected": [],
        "research": [],
        "skipped_floor": [],
        "written": [],
        "drafts_passed": 0,
    }

    # One-process-one-run hygiene: reset the tool-layer churn guards so a
    # prior invocation in this process (tests, `both` re-entry) can't leak
    # budget into this run.
    verify_claim_tool_mod._reset_attempt_counters()
    web_tool_mod._reset_search_circuit()

    # 1. Fetch + 2. Rank — the same deterministic tools the coordinator
    # called, driven from code (no planner tokens spent sequencing them).
    fetch_tool = build_fetch_curated_ai_news_tool(settings)
    fetch_result = json.loads(await fetch_tool.ainvoke({"limit": limit}))
    summary["fetched"] = int(fetch_result.get("count", 0))
    if summary["fetched"] == 0:
        summary["status"] = "no_articles"
        return summary

    rank_tool = build_technical_rank_tool(settings)
    rank_result = json.loads(await rank_tool.ainvoke({}))
    if rank_result.get("status") == "error":
        summary["status"] = "rank_failed"
        summary["error"] = rank_result.get("reason")
        return summary

    topics = _read_topics(settings)
    summary["topics"] = len(topics)
    if not topics:
        summary["status"] = "no_topics"
        return summary

    # 3. Editorial veto over the whole pool (one batched LLM call, fail-open),
    # then deterministic diversity-aware selection under the cap — the cap is
    # enforced HERE, in code, not in a prompt.
    dry_run = not (settings.openrouter_api_key or "").strip()
    pool, vetoes = await veto_irrelevant_topics(topics, settings, dry_run=dry_run)
    summary["vetoed"] = vetoes

    cap = max(1, settings.max_topics_per_run)
    selected = select_topics(pool, cap)
    summary["selected"] = [
        {"topic_id": t.topic_id, "title": t.title, "domain": t.primary_domain, "score": t.score}
        for t in selected
    ]
    if not selected:
        summary["status"] = "no_selection"
        return summary

    # 4. Research fan-out — concurrent, each task hard-bounded by
    # research_task_timeout_seconds. This is the code-side enforcement the
    # research subagent's docstring always referenced.
    researcher = research_agent or build_research_agent(settings)
    research_results = await asyncio.gather(
        *(
            _run_agent_task(
                researcher,
                description=_research_description(topic),
                topic_id=topic.topic_id,
                timeout_seconds=settings.research_task_timeout_seconds,
            )
            for topic in selected
        )
    )
    summary["research"] = [
        {"topic_id": r["topic_id"], "status": r["status"]} for r in research_results
    ]

    # 5. Evidence floor — only floored topics may be drafted. This is the
    # deterministic version of the coordinator's "skip insufficient_evidence"
    # prose rule, now with teeth: confidence + citation breadth thresholds.
    writable: list[TopicCandidate] = []
    for topic in selected:
        brief = _read_verified_brief(settings, topic.topic_id)
        if brief is None:
            summary["skipped_floor"].append(
                {"topic_id": topic.topic_id, "reason": "no verified brief"}
            )
            continue
        passes, why = meets_evidence_floor(brief, settings)
        if passes:
            writable.append(topic)
        else:
            summary["skipped_floor"].append(
                {"topic_id": topic.topic_id, "reason": why}
            )
    if not writable:
        summary["status"] = "nothing_above_floor"
        return summary

    # 6. Writer fan-out — concurrent, hard-bounded, deterministic post_ids.
    writer = writer_agent or build_writer_agent(settings)
    date_slug = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    taken: set[str] = set()
    assignments = [(topic, post_id_for(topic.title, date_slug, taken)) for topic in writable]
    write_results = await asyncio.gather(
        *(
            _run_agent_task(
                writer,
                description=_writer_description(topic.topic_id, post_id),
                topic_id=topic.topic_id,
                timeout_seconds=settings.writer_task_timeout_seconds,
            )
            for topic, post_id in assignments
        )
    )

    # 7. Verify each draft's gate verdict from disk (the delivery layer
    # re-checks everything anyway; this is the run summary's honesty check).
    for (topic, post_id), result in zip(assignments, write_results):
        gate_passed = False
        try:
            gate_file = state.gate_path(post_id, settings.orchestrator_data_dir)
            if gate_file.exists():
                verdict = json.loads(gate_file.read_text(encoding="utf-8"))
                gate_passed = verdict.get("passed") is True
        except Exception as exc:
            logger.warning("spine: gate read failed for %s: %s", post_id, exc)
        summary["written"].append(
            {
                "topic_id": topic.topic_id,
                "post_id": post_id,
                "writer_status": result["status"],
                "gate_passed": gate_passed,
            }
        )
    summary["drafts_passed"] = sum(1 for w in summary["written"] if w["gate_passed"])
    summary["status"] = "ok"
    logger.info(
        "spine complete: %d fetched, %d topics, %d selected, %d vetoed, %d floored-out, %d/%d drafts gated",
        summary["fetched"],
        summary["topics"],
        len(summary["selected"]),
        len(summary["vetoed"]),
        len(summary["skipped_floor"]),
        summary["drafts_passed"],
        len(summary["written"]),
    )
    return summary


__all__ = ["post_id_for", "run_propose_spine", "select_topics"]
