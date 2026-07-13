from __future__ import annotations

import logging

from app.config import get_settings
from app.graph.state import AgentState
from app.schemas.article import parse_articles, serialize_articles
from app.services.openrouter_client import OpenRouterClient
from app.services.tracing import traceable

logger = logging.getLogger(__name__)


@traceable(name="summarize_node")
async def summarize_node(state: AgentState) -> AgentState:
    settings = get_settings()
    dry_run = bool(state.get("dry_run", False))

    selected_articles = parse_articles(state.get("articles_selected"))
    client = OpenRouterClient(settings)
    summarized = await client.summarize_articles(selected_articles, dry_run=dry_run)

    next_state: AgentState = dict(state)
    next_state["articles_selected"] = serialize_articles(summarized)

    logger.info("Summarization complete: %s items", len(summarized))
    return next_state
