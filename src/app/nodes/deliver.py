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
    # The news digest path routes to the "news" bot explicitly (P6.1
    # made this kwarg first-class; previously the news path was the
    # only path and consumed the legacy top-level credentials directly).
    # Pinning the name here means a future multi-bot misconfiguration
    # surfaces as a real "news profile not configured" error instead of
    # silently routing the digest to the LinkedIn chat (or vice versa).
    results = await telegram_client.send_articles(articles, dry_run=dry_run, bot="news")

    if not dry_run:
        record_deliveries(
            settings.history_file,
            articles,
            results,
            state.get("delivery_history", []),
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
