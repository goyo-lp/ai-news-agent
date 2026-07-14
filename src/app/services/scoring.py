from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher

from app.schemas.article import Article
from app.services.scoring_keywords import (
    DEAL_KEYWORDS,
    ENTERPRISE_ADOPTION_KEYWORDS,
    FRONTIER_KEYWORDS,
    HIGH_RELEVANCE_PHRASES,
    LOW_PRIORITY_KEYWORDS,
    LOW_PRIORITY_PHRASES,
    PRODUCT_LAUNCH_KEYWORDS,
    STARTUP_FUNDING_KEYWORDS,
    STOPWORDS,
    TECH_DEVELOPMENT_KEYWORDS,
)

_SOURCE_WEIGHTS = {
    "openai blog": 1.0,
    "google deepmind blog": 1.0,
    "anthropic blog": 1.0,
    "meta ai blog": 1.0,
    "google ai blog": 0.95,
    "mit technology review": 0.92,
    "techcrunch (ai)": 0.9,
    "simon willison": 0.9,
    "the verge (ai)": 0.88,
    "wired (ai)": 0.88,
    "ars technica (ai)": 0.88,
    "venturebeat (ai)": 0.87,
    "hacker news (top)": 0.85,
    "the guardian (ai)": 0.85,
    "github trending": 0.8,
    "zdnet (ai)": 0.8,
    # High-volume, low-editorial feeds: demote so they season the digest
    # instead of flooding it.
    "reddit r/machinelearning": 0.62,
    "reddit r/artificial": 0.6,
    "arxiv.org (cs.ai)": 0.5,
}

_WORD_RE = re.compile(r"[a-z0-9]+")

_DATETIME_MIN_UTC = datetime.min.replace(tzinfo=timezone.utc)


@dataclass
class StoryCluster:
    id: str
    members: list[Article]


def _normalize_text(value: str) -> str:
    return " ".join(_WORD_RE.findall(value.lower()))


def _tokenize(value: str) -> set[str]:
    return {token for token in _WORD_RE.findall(value.lower()) if token not in STOPWORDS}


def _source_weight(source_name: str) -> float:
    return _SOURCE_WEIGHTS.get(source_name.lower().strip(), 0.7)


def _boost_from_hits(hits: int, base: float, extra: float, max_extra_hits: int = 2) -> float:
    if hits <= 0:
        return 0.0
    return base + (min(hits - 1, max_extra_hits) * extra)


def _count_phrase_hits(text: str, phrases: set[str]) -> int:
    return sum(1 for phrase in phrases if phrase in text)


def _relevance_score(article: Article) -> float:
    content = f"{article.effective_title} {article.effective_summary_source}".strip()
    normalized_content = _normalize_text(content)
    if not normalized_content:
        return 0.2

    tokens = _tokenize(normalized_content)
    score = 0.15

    product_hits = len(tokens & PRODUCT_LAUNCH_KEYWORDS)
    tech_hits = len(tokens & TECH_DEVELOPMENT_KEYWORDS)
    startup_hits = len(tokens & STARTUP_FUNDING_KEYWORDS)
    enterprise_hits = len(tokens & ENTERPRISE_ADOPTION_KEYWORDS)
    deal_hits = len(tokens & DEAL_KEYWORDS)
    frontier_hits = len(tokens & FRONTIER_KEYWORDS)
    total_priority_hits = (
        product_hits + tech_hits + startup_hits + enterprise_hits + deal_hits + frontier_hits
    )

    # Hand-tuned boosts: product launches rank highest, frontier-company and
    # funding news next; extra same-category hits add a little, capped by
    # _boost_from_hits so one keyword-stuffed title can't dominate.
    score += _boost_from_hits(product_hits, base=0.28, extra=0.03)
    score += _boost_from_hits(tech_hits, base=0.22, extra=0.03)
    score += _boost_from_hits(startup_hits, base=0.24, extra=0.03)
    score += _boost_from_hits(enterprise_hits, base=0.22, extra=0.03)
    score += _boost_from_hits(deal_hits, base=0.20, extra=0.03)
    score += _boost_from_hits(frontier_hits, base=0.24, extra=0.04)

    high_phrase_hits = _count_phrase_hits(normalized_content, HIGH_RELEVANCE_PHRASES)
    low_phrase_hits = _count_phrase_hits(normalized_content, LOW_PRIORITY_PHRASES)
    low_keyword_hits = len(tokens & LOW_PRIORITY_KEYWORDS)

    score += min(high_phrase_hits, 2) * 0.07

    if low_phrase_hits or low_keyword_hits:
        penalty = 0.08 if total_priority_hits > 0 else 0.22
        penalty += min(low_phrase_hits + low_keyword_hits, 3) * 0.03
        score -= penalty

    return max(0.0, min(score, 1.0))


def _recency_score(published_at: datetime | None) -> float:
    if published_at is None:
        return 0.3

    now = datetime.now(timezone.utc)
    hours_old = max((now - published_at).total_seconds() / 3600.0, 0.0)
    if hours_old <= 6:
        return 1.0
    if hours_old <= 24:
        return 0.8
    if hours_old <= 48:
        return 0.6
    if hours_old <= 96:
        return 0.4
    return 0.2


def _title_similarity(left: str, right: str) -> float:
    left_norm = _normalize_text(left)
    right_norm = _normalize_text(right)
    if not left_norm or not right_norm:
        return 0.0

    left_tokens = _tokenize(left_norm)
    right_tokens = _tokenize(right_norm)
    union = left_tokens | right_tokens
    if not union:
        return 0.0

    jaccard = len(left_tokens & right_tokens) / len(union)
    sequence = SequenceMatcher(None, left_norm, right_norm).ratio()

    return (0.65 * jaccard) + (0.35 * sequence)


def _novelty_score(article: Article, recent_titles: list[str] | None) -> float:
    """1.0 = unseen story; approaches 0.0 as the title matches recently delivered ones."""
    if not recent_titles:
        return 1.0
    # O(recent_titles); acceptable at current history scale (~14 days).
    max_similarity = max(
        _title_similarity(article.effective_title, title) for title in recent_titles
    )
    return max(0.0, 1.0 - max_similarity)


def _is_time_aligned(left: datetime | None, right: datetime | None, max_hours: int = 120) -> bool:
    if left is None or right is None:
        return True
    delta = abs((left - right).total_seconds()) / 3600.0
    return delta <= max_hours


def _same_story(left: Article, right: Article) -> bool:
    left_tokens = _tokenize(left.effective_title)
    right_tokens = _tokenize(right.effective_title)
    overlap_count = len(left_tokens & right_tokens)
    if overlap_count < 2:
        return False

    min_token_count = max(min(len(left_tokens), len(right_tokens)), 1)
    overlap_ratio = overlap_count / min_token_count
    title_similarity = _title_similarity(left.effective_title, right.effective_title)
    if title_similarity >= 0.78 and overlap_ratio >= 0.5:
        return True

    if (
        title_similarity >= 0.62
        and overlap_ratio >= 0.7
        and _is_time_aligned(left.published_at, right.published_at)
    ):
        return True

    return False


def article_sort_key(article: Article) -> tuple[float, datetime]:
    """Ranking order: score first, publish time as tiebreaker (use with reverse=True)."""
    return (article.score or 0.0, article.published_at or _DATETIME_MIN_UTC)


def cluster_articles(articles: list[Article]) -> list[StoryCluster]:
    ordered = sorted(
        articles,
        key=lambda item: item.published_at or _DATETIME_MIN_UTC,
        reverse=True,
    )

    clusters: list[StoryCluster] = []
    token_to_cluster_indices: dict[str, list[int]] = {}
    for article in ordered:
        article_tokens = _tokenize(article.effective_title)
        candidate_indices: set[int] = set()
        for token in article_tokens:
            candidate_indices.update(token_to_cluster_indices.get(token, []))

        matched_cluster: StoryCluster | None = None
        for idx in sorted(candidate_indices):
            representative = clusters[idx].members[0]
            if _same_story(article, representative):
                matched_cluster = clusters[idx]
                break

        if matched_cluster is None:
            new_index = len(clusters)
            clusters.append(StoryCluster(id=article.id, members=[article]))
            for token in article_tokens:
                token_to_cluster_indices.setdefault(token, []).append(new_index)
        else:
            matched_cluster.members.append(article)

    return clusters


def score_article(
    article: Article,
    cluster_size: int = 1,
    recent_titles: list[str] | None = None,
) -> float:
    relevance = _relevance_score(article)
    recency = _recency_score(article.published_at)
    source_weight = _source_weight(article.source_name)
    duplication_signal = min(article.duplicate_count / 5.0, 1.0)
    cluster_signal = min(max(cluster_size, 1) / 5.0, 1.0)
    novelty = _novelty_score(article, recent_titles)

    # Hand-tuned blend: keyword relevance dominates, freshness second; source
    # reputation and cross-feed corroboration adjust the middle; novelty is a
    # small tiebreaker against stories resembling recent deliveries.
    score = (
        0.38 * relevance
        + 0.24 * recency
        + 0.14 * source_weight
        + 0.10 * duplication_signal
        + 0.09 * cluster_signal
        + 0.05 * novelty
    )
    return round(score, 5)


def _select_with_source_cap(
    representatives: list[Article],
    limit: int,
    max_per_source: int,
) -> tuple[list[Article], list[Article]]:
    """Pass 1: strict per-source cap.

    Picks the highest-scoring representatives up to `max_per_source` per source,
    in score-descending order. Returns (selected, passed_over) where passed_over
    holds the articles skipped because their source hit the cap.
    """
    selected: list[Article] = []
    passed_over: list[Article] = []
    per_source: dict[str, int] = {}
    for article in representatives:
        if len(selected) >= limit:
            break
        if per_source.get(article.source_name, 0) >= max_per_source:
            passed_over.append(article)
            continue
        selected.append(article)
        per_source[article.source_name] = per_source.get(article.source_name, 0) + 1
    return selected, passed_over


def _backfill_past_cap(
    passed_over: list[Article],
    count: int,
    backfill_penalty: float,
) -> list[Article]:
    """Pass 2: fill remaining slots from passed-over articles, with a score penalty.

    Each extra same-source article beyond the cap penalizes by `backfill_penalty`
    (the (cap+1)th article loses 1*penalty, the (cap+2)th loses 2*penalty, ...).
    This keeps single-source floods from dominating yet lets a quiet day reach
    the configured limit instead of returning a shortened digest.
    """
    extras_per_source: dict[str, int] = {}
    pool: list[tuple[float, datetime, Article]] = []
    for article in passed_over:
        backfill_count = extras_per_source.get(article.source_name, 0)
        penalized_score = (article.score or 0.0) - backfill_penalty * (backfill_count + 1)
        pool.append((penalized_score, article.published_at or _DATETIME_MIN_UTC, article))
        extras_per_source[article.source_name] = backfill_count + 1

    pool.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [article for _, _, article in pool[:count]]


def rank_articles(
    articles: list[Article],
    limit: int,
    recent_titles: list[str] | None = None,
    max_per_source: int | None = None,
    backfill_penalty: float = 0.05,
) -> list[Article]:
    candidates = [article.model_copy(deep=True) for article in articles]
    story_clusters = cluster_articles(candidates)

    representatives: list[Article] = []
    for cluster in story_clusters:
        cluster_size = len(cluster.members)
        for member in cluster.members:
            member.cluster_id = cluster.id
            member.cluster_size = cluster_size
            member.score = score_article(
                member,
                cluster_size=cluster_size,
                recent_titles=recent_titles,
            )

        representatives.append(max(cluster.members, key=article_sort_key))

    representatives.sort(key=article_sort_key, reverse=True)

    if max_per_source is None:
        return representatives[:limit]

    selected, passed_over = _select_with_source_cap(representatives, limit, max_per_source)
    if len(selected) < limit and passed_over:
        selected.extend(_backfill_past_cap(passed_over, limit - len(selected), backfill_penalty))
    return selected
