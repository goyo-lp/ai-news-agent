from __future__ import annotations

from typing import Any, TypedDict, cast


class AgentState(TypedDict, total=False):
    """Shared pipeline state for one workflow.ainvoke() run. All fields are
    per-run except delivery_history, which rank_node loads from disk and
    deliver_node reuses (rather than re-reading the file) when recording sends."""
    run_id: str
    started_at: str
    dry_run: bool
    limit: int
    fetch_defaults: dict[str, Any]
    sources: list[dict[str, Any]]
    articles_raw: list[dict[str, Any]]
    articles_enriched: list[dict[str, Any]]
    articles_selected: list[dict[str, Any]]
    delivery_history: list[dict[str, Any]]
    delivery_results: list[dict[str, Any]]
    errors: list[str]


def copy_state(state: AgentState) -> AgentState:
    """Return a shallow copy of state for node mutation."""
    return cast(AgentState, dict(state))


def merge_errors(state: AgentState, new_errors: list[str]) -> None:
    """Append non-fatal errors to state["errors"] without mutating the previous list."""
    if not new_errors:
        return
    state["errors"] = [*state.get("errors", []), *new_errors]
