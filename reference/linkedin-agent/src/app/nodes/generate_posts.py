from __future__ import annotations

import logging

from app.config import get_settings
from app.graph.state import AgentState
from app.schemas import parse_research_briefs, parse_style_profile, serialize_models
from app.services.post_generator import PostGenerator
from app.services.tracing import traceable

logger = logging.getLogger(__name__)


@traceable(name="generate_posts_node")
async def generate_posts_node(state: AgentState) -> AgentState:
    settings = get_settings()
    dry_run = bool(state.get("dry_run", False))

    briefs = parse_research_briefs(state.get("research_briefs"))
    style_profile = parse_style_profile(state.get("style_profile"))

    generator = PostGenerator(settings)
    posts = await generator.generate_posts(briefs=briefs, style_profile=style_profile, dry_run=dry_run)

    logger.info("Post generation complete: %s posts", len(posts))
    return {"generated_posts": serialize_models(posts)}
