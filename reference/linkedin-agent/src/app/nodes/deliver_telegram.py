from __future__ import annotations

import logging

from app.config import get_settings
from app.graph.state import AgentState
from app.schemas import parse_linkedin_posts, serialize_models
from app.services.telegram_client import TelegramClient
from app.services.tracing import traceable

logger = logging.getLogger(__name__)


@traceable(name="deliver_telegram_node")
async def deliver_telegram_node(state: AgentState) -> AgentState:
    settings = get_settings()
    dry_run = bool(state.get("dry_run", False))

    posts = parse_linkedin_posts(state.get("linkedin_posts"))
    client = TelegramClient(settings)
    delivery_results = await client.deliver_posts(posts=posts, dry_run=dry_run)

    sent_count = len([result for result in delivery_results if result.status == "sent"])
    failed_count = len([result for result in delivery_results if result.status == "error"])
    dry_count = len([result for result in delivery_results if result.status == "dry_run"])

    logger.info(
        "Telegram delivery complete: total=%s sent=%s failed=%s dry_run=%s",
        len(delivery_results),
        sent_count,
        failed_count,
        dry_count,
    )
    return {"delivery_results": serialize_models(delivery_results)}
