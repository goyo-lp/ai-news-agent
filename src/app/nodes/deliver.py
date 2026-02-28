from __future__ import annotations

import logging

from app.config import get_settings
from app.graph.state import AgentState
from app.schemas.article import parse_articles
from app.services.telegram_client import TelegramClient
from app.services.tracing import traceable

logger = logging.getLogger(__name__)


@traceable(name="deliver_node")
async def deliver_node(state: AgentState) -> AgentState:
    settings = get_settings()
    dry_run = bool(state.get("dry_run", False))

    articles = parse_articles(state.get("articles_top20"))
    telegram_client = TelegramClient(settings)
    results = await telegram_client.send_articles(articles, dry_run=dry_run)

    next_state: AgentState = dict(state)
    next_state["delivery_results"] = results

    failures = [item for item in results if item.get("status") == "error"]
    logger.info("Delivery complete: %s sent, %s failed", len(results) - len(failures), len(failures))

    if failures:
        existing_errors = list(next_state.get("errors", []))
        existing_errors.extend(
            f"Delivery failure ({item.get('article_id')}): {item.get('error', 'unknown')}"
            for item in failures
        )
        next_state["errors"] = existing_errors

    return next_state
