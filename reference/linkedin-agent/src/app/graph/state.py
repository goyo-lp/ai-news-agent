from __future__ import annotations

from typing import Any, TypedDict


class AgentState(TypedDict, total=False):
    run_id: str
    started_at: str
    run_date: str
    dry_run: bool
    hours_back: int
    max_topics: int

    style_samples_dir_override: str

    discovered_items: list[dict[str, Any]]
    normalized_items: list[dict[str, Any]]
    ranked_topics: list[dict[str, Any]]
    deep_research_briefs: list[dict[str, Any]]
    research_briefs: list[dict[str, Any]]
    adaptive_briefs: list[dict[str, Any]]
    verified_briefs: list[dict[str, Any]]
    style_profile: dict[str, Any]
    generated_posts: list[dict[str, Any]]
    linkedin_posts: list[dict[str, Any]]
    delivery_results: list[dict[str, Any]]

    quality_checks: list[str]
    adaptive_errors: list[str]
    verify_errors: list[str]
    style_profile_errors: list[str]
    artifact_export_dir: str
    export_dir: str
    report: dict[str, Any]
    errors: list[str]
