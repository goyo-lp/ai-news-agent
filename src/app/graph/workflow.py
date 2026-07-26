from __future__ import annotations

from datetime import timedelta
from typing import Any, cast

import httpx
from langgraph.func import RetryPolicy
from langgraph.graph import END, StateGraph

from app.graph.state import AgentState
from app.nodes.deliver import deliver_node
from app.nodes.enrich import enrich_node
from app.nodes.ingest import ingest_node
from app.nodes.rank import rank_node
from app.nodes.summarize import summarize_node


def _retry_policy() -> RetryPolicy:
    """Retry a node on transient network / HTTP errors with exponential backoff."""
    return RetryPolicy(
        initial_interval=1.0,
        backoff_factor=2.0,
        max_interval=15.0,
        max_attempts=3,
        jitter=True,
        retry_on=(httpx.HTTPError, httpx.TimeoutException, ConnectionError),
    )


def _node_error_handler(state: AgentState) -> AgentState:
    """Pinned node-level error handler: log, surface via state['errors'], continue.

    LangGraph invokes this when a node exhausts its retry policy or raises an
    unrecoverable error. Per-article failures are already handled inside each
    node's services layer, so this only fires on catastrophic node failures
    (e.g. DNS down, async client construction failure, parser bug).
    """
    next_state: AgentState = cast(AgentState, dict(state))
    existing: list[str] = list(next_state.get("errors", []))
    existing.append("Node failed after retries; see prior log warnings for details.")
    next_state["errors"] = existing
    return next_state


def route_after_rank(state: AgentState) -> str:
    """Skip summarize/deliver when nothing survived the rank filters."""
    return "summarize" if state.get("articles_selected") else END


def _add_core_nodes(graph: StateGraph, retry: RetryPolicy) -> None:
    """Register the shared curation prefix: ingest -> enrich -> rank ->
    summarize. Used by build_workflow (which continues on to deliver) and
    build_curation_workflow (which stops here), so the two pipelines can't
    drift apart."""
    graph.add_node(
        "ingest",
        ingest_node,
        retry_policy=retry,
        timeout=timedelta(seconds=60),
        error_handler=_node_error_handler,
    )
    graph.add_node(
        "enrich",
        enrich_node,
        retry_policy=retry,
        timeout=timedelta(seconds=120),
        error_handler=_node_error_handler,
    )
    # 90s, not 45s: rank now bounds its own LLM re-rank
    # (llm_rerank_timeout_seconds) and degrades to the deterministic ranking on
    # timeout, so this is a backstop against a genuinely hung node rather than
    # the control on LLM latency. At 45s it *was* that control, and a slow
    # relevance call silently zeroed out the whole digest.
    graph.add_node(
        "rank",
        rank_node,
        retry_policy=retry,
        timeout=timedelta(seconds=90),
        error_handler=_node_error_handler,
    )
    graph.add_node(
        "summarize",
        summarize_node,
        retry_policy=retry,
        timeout=timedelta(seconds=120),
        error_handler=_node_error_handler,
    )

    graph.set_entry_point("ingest")
    graph.add_edge("ingest", "enrich")
    graph.add_edge("enrich", "rank")
    graph.add_conditional_edges("rank", route_after_rank, {"summarize": "summarize", END: END})


def build_workflow() -> Any:
    """Assemble and compile the 5-node pipeline: ingest -> enrich -> rank ->
    (summarize -> deliver | END). Each node shares the same retry policy and a
    pinned error handler; rank has a conditional early exit (route_after_rank)
    when nothing survives filtering."""
    graph = StateGraph(AgentState)
    retry = _retry_policy()

    _add_core_nodes(graph, retry)
    graph.add_node(
        "deliver",
        deliver_node,
        retry_policy=retry,
        timeout=timedelta(seconds=90),
        error_handler=_node_error_handler,
    )
    graph.add_edge("summarize", "deliver")
    graph.add_edge("deliver", END)

    return graph.compile()


def build_curation_workflow() -> Any:
    """Assemble and compile the curation-only pipeline: ingest -> enrich -> rank
    -> summarize, stopping before deliver. Used by run_curation() to produce
    structured articles for the orchestrator without sending anything; the
    full pipeline (build_workflow, deliver included) is untouched."""
    graph = StateGraph(AgentState)
    retry = _retry_policy()

    _add_core_nodes(graph, retry)
    graph.add_edge("summarize", END)

    return graph.compile()
