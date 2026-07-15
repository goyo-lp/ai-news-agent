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


def build_workflow() -> Any:
    """Assemble and compile the 5-node pipeline: ingest -> enrich -> rank ->
    (summarize -> deliver | END). Each node shares the same retry policy and a
    pinned error handler; rank has a conditional early exit (route_after_rank)
    when nothing survives filtering."""
    graph = StateGraph(AgentState)

    retry = _retry_policy()

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
    graph.add_node(
        "rank",
        rank_node,
        retry_policy=retry,
        timeout=timedelta(seconds=45),
        error_handler=_node_error_handler,
    )
    graph.add_node(
        "summarize",
        summarize_node,
        retry_policy=retry,
        timeout=timedelta(seconds=120),
        error_handler=_node_error_handler,
    )
    graph.add_node(
        "deliver",
        deliver_node,
        retry_policy=retry,
        timeout=timedelta(seconds=90),
        error_handler=_node_error_handler,
    )

    graph.set_entry_point("ingest")
    graph.add_edge("ingest", "enrich")
    graph.add_edge("enrich", "rank")
    graph.add_conditional_edges("rank", route_after_rank, {"summarize": "summarize", END: END})
    graph.add_edge("summarize", "deliver")
    graph.add_edge("deliver", END)

    return graph.compile()
