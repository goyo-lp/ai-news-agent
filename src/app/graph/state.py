from __future__ import annotations

from typing import Any, TypedDict


class AgentState(TypedDict, total=False):
    run_id: str
    started_at: str
    dry_run: bool
    limit: int
    fetch_defaults: dict[str, Any]
    sources: list[dict[str, Any]]
    articles_raw: list[dict[str, Any]]
    articles_enriched: list[dict[str, Any]]
    articles_ranked: list[dict[str, Any]]
    articles_top20: list[dict[str, Any]]
    delivery_results: list[dict[str, Any]]
    errors: list[str]
