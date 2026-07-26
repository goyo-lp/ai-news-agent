from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

from app.config import Settings, get_settings
from app.graph.state import AgentState, copy_state, merge_errors
from app.schemas.article import Article, parse_articles, serialize_articles
from app.services.history import delivered_titles, filter_previously_delivered, load_history
from app.services.openrouter_client import OpenRouterClient
from app.services.scoring import article_sort_key, rank_articles
from app.services.tracing import traceable
from app.utils import resolve_reference_now

logger = logging.getLogger(__name__)

# Deterministic ranking narrows to this pool before the single batched LLM
# relevance call re-orders it; the final list is then cut to the run limit.
_LLM_RERANK_POOL = 40
_LLM_BLEND_WEIGHT = 0.3


def filter_articles_published_today(
    articles: list[Article],
    now: datetime | None = None,
) -> list[Article]:
    """Keep only articles whose published_at falls on today's date in `now`'s
    timezone (defaults to local time). Articles with no published_at are dropped
    rather than assumed current."""
    reference_now = resolve_reference_now(now)
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


async def _blend_scores(
    candidates: list[Article],
    settings: Settings,
    *,
    dry_run: bool,
) -> tuple[dict[str, float], str | None]:
    """Score relevance with one batched LLM call, bounded by
    ``llm_rerank_timeout_seconds``.

    The blend is an *enrichment*: an empty result means "keep the deterministic
    ranking", which is already this call's documented contract on failure. It
    was not bounded, so on 2026-07-26 a slow-but-successful call ran past the
    rank node's graph timeout, killed the node, and discarded 40 perfectly good
    deterministically-ranked articles — the digest reported `selected=0` and
    sent nothing. A timeout here degrades instead: worse ordering, still a
    digest.
    """
    try:
        return await asyncio.wait_for(
            OpenRouterClient(settings).score_articles_relevance(
                candidates, dry_run=dry_run
            ),
            timeout=settings.llm_rerank_timeout_seconds,
        )
    except TimeoutError:
        logger.warning(
            "LLM re-rank exceeded %ss; keeping the deterministic ranking",
            settings.llm_rerank_timeout_seconds,
        )
        return {}, (
            f"LLM relevance scoring timed out after "
            f"{settings.llm_rerank_timeout_seconds}s; deterministic ranking kept"
        )


@traceable(name="rank_node")
async def rank_node(state: AgentState) -> AgentState:
    """Filter to today's not-previously-delivered articles, cluster+score them
    deterministically, optionally blend in one batched LLM relevance call over
    the top pool, then cut to the run limit. Also stashes the loaded delivery
    history in state so deliver_node doesn't have to re-read it from disk."""
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
    fresh_articles = filter_previously_delivered(todays_articles, history, now=local_now)
    dropped_count = len(todays_articles) - len(fresh_articles)
    if dropped_count:
        logger.info("History filter dropped %s previously delivered items", dropped_count)

    candidates = rank_articles(
        fresh_articles,
        limit=max(limit, _LLM_RERANK_POOL),
        recent_titles=delivered_titles(history),
        max_per_source=settings.max_articles_per_source,
    )

    llm_scores, llm_error = await _blend_scores(candidates, settings, dry_run=dry_run)
    if llm_scores:
        for article in candidates:
            llm_score = llm_scores.get(article.id)
            if llm_score is not None:
                article.score = round(
                    (1 - _LLM_BLEND_WEIGHT) * (article.score or 0.0)
                    + _LLM_BLEND_WEIGHT * llm_score,
                    5,
                )
        candidates.sort(key=article_sort_key, reverse=True)
        logger.info("LLM re-rank blended scores for %s items", len(llm_scores))

    selected = candidates[:limit]

    next_state = copy_state(state)
    next_state["articles_selected"] = serialize_articles(selected)
    next_state["delivery_history"] = history
    if llm_error:
        merge_errors(next_state, [llm_error])

    logger.info("Ranking complete: selected %s items", len(selected))
    return next_state
