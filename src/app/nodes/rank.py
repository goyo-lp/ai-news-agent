from __future__ import annotations

import logging

from app.config import get_settings
from app.graph.state import AgentState
from app.schemas.article import parse_articles, serialize_articles
from app.services.scoring import rank_articles
from app.services.tracing import traceable

logger = logging.getLogger(__name__)


@traceable(name="rank_node")
async def rank_node(state: AgentState) -> AgentState:
    settings = get_settings()

    enriched_articles = parse_articles(state.get("articles_enriched"))
    limit = int(state.get("limit", settings.max_articles_per_run))
    limit = max(1, min(limit, settings.max_articles_per_run))

    ranked = rank_articles(enriched_articles, limit=limit)

    next_state: AgentState = dict(state)
    next_state["articles_ranked"] = serialize_articles(ranked)
    next_state["articles_top20"] = serialize_articles(ranked)

    logger.info("Ranking complete: selected %s items", len(ranked))
    return next_state
