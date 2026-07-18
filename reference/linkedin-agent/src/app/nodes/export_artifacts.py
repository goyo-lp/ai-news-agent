from __future__ import annotations

import logging
from datetime import datetime

from app.config import get_settings
from app.graph.state import AgentState
from app.schemas import parse_discovered, parse_linkedin_posts, parse_ranked_topics, parse_research_briefs
from app.services.output_writer import OutputWriter
from app.services.tracing import traceable

logger = logging.getLogger(__name__)


@traceable(name="export_artifacts_node")
async def export_artifacts_node(state: AgentState) -> AgentState:
    settings = get_settings()
    run_date = str(state.get("run_date", datetime.now().date().isoformat()))

    seed_items = parse_discovered(state.get("discovered_items"))
    ranked_topics = parse_ranked_topics(state.get("ranked_topics"))
    adaptive_briefs = parse_research_briefs(state.get("adaptive_briefs"))
    briefs = parse_research_briefs(state.get("research_briefs"))
    posts = parse_linkedin_posts(state.get("linkedin_posts"))

    writer = OutputWriter(settings)
    output_dir = writer.write_artifacts(
        run_date=run_date,
        seed_items=seed_items,
        ranked_topics=ranked_topics,
        adaptive_briefs=adaptive_briefs or briefs,
        briefs=briefs,
        posts=posts,
    )

    logger.info("Artifact export complete: %s", output_dir)
    return {"artifact_export_dir": str(output_dir)}
