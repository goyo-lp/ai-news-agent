from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import cast

from app.graph.state import AgentState
from app.schemas import (
    RunReport,
    parse_delivery_results,
    parse_discovered,
    parse_linkedin_posts,
    parse_ranked_topics,
    parse_research_briefs,
)
from app.services.output_writer import OutputWriter
from app.services.tracing import traceable
from app.config import get_settings

logger = logging.getLogger(__name__)


@traceable(name="export_outputs_node")
async def export_outputs_node(state: AgentState) -> AgentState:
    settings = get_settings()

    run_id = str(state.get("run_id", "unknown"))
    run_date = str(state.get("run_date", datetime.now().date().isoformat()))
    started_at = str(state.get("started_at", datetime.now(timezone.utc).isoformat()))

    discovered_count = len(state.get("discovered_items", []))
    normalized_count = len(state.get("normalized_items", []))
    topics_count = len(state.get("ranked_topics", []))

    seed_items = parse_discovered(state.get("discovered_items"))
    ranked_topics = parse_ranked_topics(state.get("ranked_topics"))
    adaptive_briefs = parse_research_briefs(state.get("adaptive_briefs"))
    briefs = parse_research_briefs(state.get("research_briefs"))
    posts = parse_linkedin_posts(state.get("linkedin_posts"))
    delivery_results = parse_delivery_results(state.get("delivery_results"))
    deliveries_sent = len([item for item in delivery_results if item.status == "sent"])
    deliveries_failed = len([item for item in delivery_results if item.status == "error"])

    report = RunReport(
        run_id=run_id,
        run_date=run_date,
        started_at=started_at,
        completed_at=datetime.now(timezone.utc).isoformat(),
        dry_run=bool(state.get("dry_run", False)),
        discovered_count=discovered_count,
        normalized_count=normalized_count,
        topics_count=topics_count,
        briefs_count=len(briefs),
        posts_count=len(posts),
        deliveries_attempted=len(delivery_results),
        deliveries_sent=deliveries_sent,
        deliveries_failed=deliveries_failed,
        quality_checks=list(state.get("quality_checks", [])),
        errors=list(state.get("errors", [])),
    )

    writer = OutputWriter(settings)
    output_dir = writer.write_outputs(
        run_date=run_date,
        seed_items=seed_items,
        ranked_topics=ranked_topics,
        adaptive_briefs=adaptive_briefs or briefs,
        briefs=briefs,
        posts=posts,
        report=report,
    )

    next_state = cast(AgentState, dict(state))
    next_state["export_dir"] = str(output_dir)
    next_state["report"] = report.model_dump(mode="json")

    logger.info("Export complete: %s", output_dir)
    return next_state
