from __future__ import annotations

import logging
from typing import cast

from app.config import get_settings
from app.graph.state import AgentState
from app.schemas import serialize_models
from app.services.rss_seed_client import RSSSeedClient
from app.services.scoring import select_seed_items
from app.services.source_policy import SourcePolicy
from app.services.tracing import traceable

logger = logging.getLogger(__name__)


@traceable(name="discover_news_node")
async def discover_news_node(state: AgentState) -> AgentState:
    settings = get_settings()
    dry_run = bool(state.get("dry_run", False))

    client = RSSSeedClient(settings)
    daily_items, errors = await client.fetch_daily_seed_items(dry_run=dry_run)

    policy = SourcePolicy.from_file(settings.trusted_sources_path)
    seeded_items = select_seed_items(
        daily_items,
        limit=max(5, settings.seed_articles_per_run),
        policy=policy,
    )

    next_state = cast(AgentState, dict(state))
    next_state["discovered_items"] = serialize_models(seeded_items)

    existing_errors = list(next_state.get("errors", []))
    existing_errors.extend(errors)
    next_state["errors"] = existing_errors

    logger.info(
        "Discovery complete (RSS seed): raw_daily=%s selected_seed=%s",
        len(daily_items),
        len(seeded_items),
    )
    return next_state
