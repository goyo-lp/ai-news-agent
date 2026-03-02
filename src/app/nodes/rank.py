from __future__ import annotations

import logging
from datetime import datetime, timezone

from app.config import get_settings
from app.graph.state import AgentState
from app.schemas.article import Article, parse_articles, serialize_articles
from app.services.scoring import rank_articles
from app.services.tracing import traceable

logger = logging.getLogger(__name__)


def filter_articles_published_today(
    articles: list[Article],
    now: datetime | None = None,
) -> list[Article]:
    reference_now = now if now is not None else datetime.now().astimezone()
    if reference_now.tzinfo is None:
        reference_now = reference_now.replace(tzinfo=timezone.utc)
    local_tz = reference_now.tzinfo
    today = reference_now.date()

    filtered: list[Article] = []
    for article in articles:
        published_at = article.published_at
        if published_at is None:
            continue
        if published_at.tzinfo is None:
            published_at = published_at.replace(tzinfo=timezone.utc)
        if published_at.astimezone(local_tz).date() == today:
            filtered.append(article)

    return filtered


@traceable(name="rank_node")
async def rank_node(state: AgentState) -> AgentState:
    settings = get_settings()

    enriched_articles = parse_articles(state.get("articles_enriched"))
    limit = int(state.get("limit", settings.max_articles_per_run))
    limit = max(1, min(limit, settings.max_articles_per_run))

    local_now = datetime.now().astimezone()
    todays_articles = filter_articles_published_today(enriched_articles, now=local_now)
    filtered_count = len(enriched_articles) - len(todays_articles)
    if filtered_count:
        logger.info(
            "Date filter kept %s/%s items for %s",
            len(todays_articles),
            len(enriched_articles),
            local_now.date().isoformat(),
        )

    ranked = rank_articles(todays_articles, limit=limit)

    next_state: AgentState = dict(state)
    next_state["articles_ranked"] = serialize_articles(ranked)
    next_state["articles_top20"] = serialize_articles(ranked)

    logger.info("Ranking complete: selected %s items", len(ranked))
    return next_state
