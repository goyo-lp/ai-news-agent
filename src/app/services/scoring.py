from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher

from app.schemas.article import Article

_SOURCE_WEIGHTS = {
    "openai blog": 1.0,
    "google deepmind blog": 1.0,
    "anthropic blog": 1.0,
    "meta ai blog": 1.0,
    "mit technology review": 0.92,
    "techcrunch (ai)": 0.9,
    "the verge (ai)": 0.88,
    "wired (ai)": 0.88,
    "venturebeat (ai)": 0.87,
    "the guardian (ai)": 0.85,
    "zdnet (ai)": 0.8,
}

_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "into",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "to",
    "story",
    "stories",
    "news",
    "update",
    "updates",
    "with",
}

_WORD_RE = re.compile(r"[a-z0-9]+")


@dataclass
class StoryCluster:
    id: str
    members: list[Article]


def _normalize_text(value: str) -> str:
    return " ".join(_WORD_RE.findall(value.lower()))


def _tokenize(value: str) -> set[str]:
    return {token for token in _WORD_RE.findall(value.lower()) if token not in _STOPWORDS}


def _source_weight(source_name: str) -> float:
    return _SOURCE_WEIGHTS.get(source_name.lower().strip(), 0.7)


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


def _novelty_score(article: Article) -> float:
    title_tokens = set(article.effective_title.lower().split())
    if not title_tokens:
        return 0.0
    return min(len(title_tokens) / 20.0, 1.0)


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


def cluster_articles(articles: list[Article]) -> list[StoryCluster]:
    ordered = sorted(
        articles,
        key=lambda item: item.published_at or datetime.min.replace(tzinfo=timezone.utc),
        reverse=True,
    )

    clusters: list[StoryCluster] = []
    for article in ordered:
        matched_cluster: StoryCluster | None = None
        for cluster in clusters:
            representative = cluster.members[0]
            if _same_story(article, representative):
                matched_cluster = cluster
                break

        if matched_cluster is None:
            clusters.append(StoryCluster(id=article.id, members=[article]))
        else:
            matched_cluster.members.append(article)

    return clusters


def score_article(article: Article, cluster_size: int = 1) -> float:
    recency = _recency_score(article.published_at)
    source_weight = _source_weight(article.source_name)
    duplication_signal = min(article.duplicate_count / 5.0, 1.0)
    cluster_signal = min(max(cluster_size, 1) / 5.0, 1.0)
    novelty = _novelty_score(article)

    score = (
        0.40 * recency
        + 0.27 * source_weight
        + 0.14 * duplication_signal
        + 0.14 * cluster_signal
        + 0.05 * novelty
    )
    return round(score, 5)


def rank_articles(articles: list[Article], limit: int) -> list[Article]:
    candidates = [article.model_copy(deep=True) for article in articles]
    story_clusters = cluster_articles(candidates)

    representatives: list[Article] = []
    for cluster in story_clusters:
        cluster_size = len(cluster.members)
        for member in cluster.members:
            member.cluster_id = cluster.id
            member.cluster_size = cluster_size
            member.score = score_article(member, cluster_size=cluster_size)

        representative = max(
            cluster.members,
            key=lambda item: (
                item.score or 0.0,
                item.published_at or datetime.min.replace(tzinfo=timezone.utc),
            ),
        )
        representatives.append(representative)

    representatives.sort(
        key=lambda a: (
            a.score or 0.0,
            a.published_at or datetime.min.replace(tzinfo=timezone.utc),
        ),
        reverse=True,
    )
    return representatives[:limit]
