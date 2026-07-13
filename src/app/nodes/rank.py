from __future__ import annotations

import logging
from datetime import datetime, timezone

from app.config import get_settings
from app.graph.state import AgentState
from app.schemas.article import Article, parse_articles, serialize_articles
from app.services.history import delivered_titles, filter_previously_delivered, load_history
from app.services.openrouter_client import OpenRouterClient
from app.services.scoring import rank_articles
from app.services.tracing import traceable

logger = logging.getLogger(__name__)

# Deterministic ranking narrows to this pool before the single batched LLM
# relevance call re-orders it; the final list is then cut to the run limit.
_LLM_RERANK_POOL = 40
_LLM_BLEND_WEIGHT = 0.3


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
    dry_run = bool(state.get("dry_run", False))

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

    history = load_history(settings.history_file, settings.history_retention_days)
    fresh_articles = filter_previously_delivered(todays_articles, history)
    dropped_count = len(todays_articles) - len(fresh_articles)
    if dropped_count:
        logger.info("History filter dropped %s previously delivered items", dropped_count)

    candidates = rank_articles(
        fresh_articles,
        limit=max(limit, _LLM_RERANK_POOL),
        recent_titles=delivered_titles(history),
        max_per_source=settings.max_articles_per_source,
    )

    client = OpenRouterClient(settings)
    llm_scores = await client.score_articles_relevance(candidates, dry_run=dry_run)
    if llm_scores:
        for article in candidates:
            llm_score = llm_scores.get(article.id)
            if llm_score is not None:
                article.score = round(
                    (1 - _LLM_BLEND_WEIGHT) * (article.score or 0.0)
                    + _LLM_BLEND_WEIGHT * llm_score,
                    5,
                )
        candidates.sort(
            key=lambda a: (
                a.score or 0.0,
                a.published_at or datetime.min.replace(tzinfo=timezone.utc),
            ),
            reverse=True,
        )
        logger.info("LLM re-rank blended scores for %s items", len(llm_scores))

    selected = candidates[:limit]

    next_state: AgentState = dict(state)
    next_state["articles_selected"] = serialize_articles(selected)

    logger.info("Ranking complete: selected %s items", len(selected))
    return next_state
