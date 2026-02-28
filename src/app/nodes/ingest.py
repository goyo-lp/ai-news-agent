from __future__ import annotations

import logging

from app.config import get_settings
from app.graph.state import AgentState
from app.schemas.article import serialize_articles
from app.services.rss_client import RSSClient, dedupe_articles
from app.services.tracing import traceable

logger = logging.getLogger(__name__)


@traceable(name="ingest_node")
async def ingest_node(state: AgentState) -> AgentState:
    settings = get_settings()
    rss_client = RSSClient(settings)

    fetch_defaults, sources = rss_client.load_sources()
    articles, errors = await rss_client.fetch_all(sources)
    deduped = dedupe_articles(articles)

    next_state: AgentState = dict(state)
    next_state["fetch_defaults"] = fetch_defaults.model_dump(mode="json")
    next_state["sources"] = [source.model_dump(mode="json") for source in sources]
    next_state["articles_raw"] = serialize_articles(deduped)

    existing_errors = list(next_state.get("errors", []))
    existing_errors.extend(errors)
    next_state["errors"] = existing_errors

    logger.info("Ingestion complete: %s raw items", len(deduped))
    return next_state
