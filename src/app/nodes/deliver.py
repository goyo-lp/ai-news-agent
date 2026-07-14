from __future__ import annotations

import logging

from app.config import get_settings
from app.graph.state import AgentState, copy_state, merge_errors
from app.schemas.article import parse_articles
from app.services.history import record_deliveries
from app.services.telegram_client import TelegramClient
from app.services.tracing import traceable

logger = logging.getLogger(__name__)


@traceable(name="deliver_node")
async def deliver_node(state: AgentState) -> AgentState:
    settings = get_settings()
    dry_run = bool(state.get("dry_run", False))

    articles = parse_articles(state.get("articles_selected"))
    telegram_client = TelegramClient(settings)
    results = await telegram_client.send_articles(articles, dry_run=dry_run)

    if not dry_run:
        record_deliveries(
            settings.history_file,
            articles,
            results,
            settings.history_retention_days,
        )

    next_state = copy_state(state)
    next_state["delivery_results"] = results

    failures = [item for item in results if item.get("status") == "error"]
    logger.info("Delivery complete: %s sent, %s failed", len(results) - len(failures), len(failures))

    merge_errors(
        next_state,
        [
            f"Delivery failure ({item.get('article_id')}): {item.get('error', 'unknown')}"
            for item in failures
        ],
    )

    return next_state
