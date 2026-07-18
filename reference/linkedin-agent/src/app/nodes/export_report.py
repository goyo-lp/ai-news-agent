from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

from app.config import get_settings
from app.graph.state import AgentState
from app.schemas import (
    RunReport,
    parse_delivery_results,
    parse_linkedin_posts,
    parse_research_briefs,
)
from app.services.output_writer import OutputWriter
from app.services.tracing import traceable

logger = logging.getLogger(__name__)


@traceable(name="export_report_node")
async def export_report_node(state: AgentState) -> AgentState:
    settings = get_settings()

    run_id = str(state.get("run_id", "unknown"))
    run_date = str(state.get("run_date", datetime.now().date().isoformat()))
    started_at = str(state.get("started_at", datetime.now(timezone.utc).isoformat()))

    discovered_count = len(state.get("discovered_items", []))
    normalized_count = len(state.get("normalized_items", []))
    topics_count = len(state.get("ranked_topics", []))
    briefs = parse_research_briefs(state.get("research_briefs"))
    posts = parse_linkedin_posts(state.get("linkedin_posts"))
    delivery_results = parse_delivery_results(state.get("delivery_results"))
    deliveries_sent = len([item for item in delivery_results if item.status == "sent"])
    deliveries_failed = len([item for item in delivery_results if item.status == "error"])
    merged_errors = _merge_errors(
        base=list(state.get("errors", [])),
        extras=list(state.get("style_profile_errors", [])),
    )

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
        errors=merged_errors,
    )

    raw_export_dir = str(state.get("artifact_export_dir") or state.get("export_dir") or "").strip()
    output_dir = Path(raw_export_dir).expanduser() if raw_export_dir else None
    writer = OutputWriter(settings)
    report_path = writer.write_run_report(
        report=report,
        run_date=run_date,
        output_dir=output_dir,
    )

    logger.info("Run report export complete: %s", report_path)
    return {
        "export_dir": str(report_path.parent),
        "report": report.model_dump(mode="json"),
    }


def _merge_errors(base: list[str], extras: list[str]) -> list[str]:
    merged: list[str] = []
    for error in [*base, *extras]:
        normalized = " ".join(str(error).split()).strip()
        if not normalized or normalized in merged:
            continue
        merged.append(normalized)
    return merged
