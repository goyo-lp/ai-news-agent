"""Group articles into distinct news events without model judgment."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime

import langsmith as ls
import structlog

from ai_news_agent.observability import redact
from ai_news_agent.schemas import Article, ArticleText, StoryCluster
from ai_news_agent.storage import SQLiteStore

_LOGGER = structlog.get_logger(__name__)

TITLE_MERGE_JACCARD = 0.5
TITLE_BOOST_JACCARD = 0.35
ENTITY_BOOST_JACCARD = 0.5
AMBIGUOUS_JACCARD = 0.3
DATE_WINDOW_HOURS = 48.0

_TOKEN = re.compile(r"[a-z0-9]+")
_ENTITY = re.compile(r"\b[A-Z][A-Za-z0-9&]*(?:\s+[A-Z][A-Za-z0-9&]*){0,3}")


@dataclass(slots=True)
class ClusterResult:
    cluster: StoryCluster
    recycled: bool
    collapsed_duplicates: int = 0


@dataclass(slots=True)
class ClusteringSummary:
    attempted: int = 0
    clusters: int = 0
    articles_clustered: int = 0
    duplicates_collapsed: int = 0
    recycled: int = 0
    results: tuple[ClusterResult, ...] = ()
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    finished_at: datetime = field(default_factory=lambda: datetime.now(UTC))


def title_tokens(title: str) -> frozenset[str]:
    """Tokenize a headline for overlap comparison."""

    return frozenset(_TOKEN.findall(title.lower()))


def title_entities(title: str) -> frozenset[str]:
    """Extract capitalized phrases as a cheap entity signal."""

    return frozenset(match.group(0) for match in _ENTITY.finditer(title))


def jaccard(first: frozenset[str], second: frozenset[str]) -> float:
    """Return token-set Jaccard similarity, 0.0 when either side is empty."""

    if not first or not second:
        return 0.0
    return len(first & second) / len(first | second)


def hours_apart(first: datetime, second: datetime) -> float:
    """Return the absolute publication gap in hours."""

    return abs((first - second).total_seconds()) / 3600.0


def stable_cluster_id(article_ids: tuple[str, ...]) -> str:
    """Return a stable cluster ID derived from sorted member IDs."""

    digest = hashlib.sha256(",".join(sorted(article_ids)).encode("utf-8")).hexdigest()
    return f"cluster-{digest[:12]}"


def _dedupe_urls(articles: tuple[Article, ...]) -> tuple[tuple[Article, ...], int]:
    """Collapse exact canonical-URL duplicates, keeping earliest first seen."""

    kept: dict[str, Article] = {}
    collapsed = 0
    for article in sorted(articles, key=lambda item: (item.first_seen_at, item.id)):
        key = str(article.canonical_url).strip().lower()
        if key in kept:
            collapsed += 1
            continue
        kept[key] = article
    return tuple(kept.values()), collapsed


def _content_hashes(
    articles: tuple[Article, ...], store: SQLiteStore | None
) -> dict[str, str]:
    """Map article IDs to full-text content hashes when retrievable."""

    if store is None:
        return {}
    hashes: dict[str, str] = {}
    for article in articles:
        text = store.get(ArticleText, f"{article.id}-text")
        if text is not None:
            hashes[article.id] = text.content_hash
    return hashes


def _should_merge(
    first: Article,
    second: Article,
    hashes: dict[str, str],
) -> tuple[bool, str]:
    """Decide whether two articles describe one event, with a reason.

    Ambiguous near-misses stay separate; the returned reason records the
    scores so a future model-judged pass can revisit exactly these pairs.
    """

    first_hash = hashes.get(first.id)
    if first_hash is not None and first_hash == hashes.get(second.id):
        return True, "identical article content"
    tokens_first = title_tokens(first.title)
    tokens_second = title_tokens(second.title)
    similarity = jaccard(tokens_first, tokens_second)
    entities = jaccard(title_entities(first.title), title_entities(second.title))
    gap = hours_apart(first.published_at, second.published_at)
    if gap > DATE_WINDOW_HOURS:
        return False, f"published {gap:.0f}h apart"
    if similarity >= TITLE_MERGE_JACCARD:
        shared = len(tokens_first & tokens_second)
        total = len(tokens_first | tokens_second)
        return True, f"shared {shared}/{total} title tokens, {gap:.0f}h apart"
    if similarity >= TITLE_BOOST_JACCARD and entities >= ENTITY_BOOST_JACCARD:
        return True, f"shared entities plus title overlap {similarity:.2f}"
    if similarity >= AMBIGUOUS_JACCARD:
        return False, (
            "ambiguous near-miss kept separate "
            f"(title overlap {similarity:.2f}, entity overlap {entities:.2f}); "
            "needs model judgment to merge"
        )
    return False, f"title overlap {similarity:.2f} too low"


def _group_events(
    articles: tuple[Article, ...], hashes: dict[str, str]
) -> tuple[list[list[Article]], list[tuple[str, str, str]], dict[tuple[str, str], str]]:
    """Cluster articles with union-find over pairwise merge decisions.

    Returns groups, merge decisions as (article, article, reason) triples,
    and ambiguous near-miss notes keyed by article-ID pair.
    """

    parents: dict[str, str] = {article.id: article.id for article in articles}
    merges: list[tuple[str, str, str]] = []
    near_misses: dict[tuple[str, str], str] = {}

    def _find(item: str) -> str:
        while parents[item] != item:
            parents[item] = parents[parents[item]]
            item = parents[item]
        return item

    ordered = sorted(articles, key=lambda item: item.id)
    for index, first in enumerate(ordered):
        for second in ordered[index + 1 :]:
            merge, reason = _should_merge(first, second, hashes)
            if merge:
                parents[_find(first.id)] = _find(second.id)
                merges.append((first.id, second.id, reason))
            elif "ambiguous near-miss" in reason:
                near_misses[(first.id, second.id)] = reason
    groups: dict[str, list[Article]] = {}
    for article in articles:
        groups.setdefault(_find(article.id), []).append(article)
    return list(groups.values()), merges, near_misses


def _representative(
    members: list[Article], hashes: dict[str, str]
) -> tuple[Article, str]:
    """Pick the most evidenced article, preferring the earliest publication."""

    def _score(article: Article) -> tuple[int, int]:
        return (1 if article.id in hashes else 0, 1 if article.authors else 0)

    best = max(
        sorted(members, key=lambda item: (item.published_at, item.id)),
        key=_score,
    )
    signals: list[str] = []
    if best.id in hashes:
        signals.append("has full text")
    if best.authors:
        signals.append("has byline")
    detail = (
        f"representative {best.id} ({', '.join(signals)})"
        if signals
        else (f"representative {best.id} (earliest publication)")
    )
    return best, detail


@ls.traceable(
    name="cluster-articles",
    run_type="chain",
    process_inputs=redact,
    process_outputs=redact,
)
def cluster_articles(
    articles: tuple[Article, ...],
    *,
    now: datetime | None = None,
    store: SQLiteStore | None = None,
    published_article_ids: frozenset[str] = frozenset(),
) -> ClusteringSummary:
    """Dedupe, group, and persist articles as distinct story clusters.

    Pairwise matching is deterministic; ambiguous near-misses are kept
    separate with their scores recorded in the rationale, which is the seam
    where a future model-judged pass can revisit them. Clusters whose every
    member was already published are flagged recycled, while new articles on
    an old topic form fresh clusters (substantive follow-ups qualify).
    """

    started = now or datetime.now(UTC)
    unique, collapsed = _dedupe_urls(articles)
    hashes = _content_hashes(unique, store)
    groups, merges, near_misses = _group_events(unique, hashes)

    results: list[ClusterResult] = []
    for members in sorted(
        groups, key=lambda group: sorted(item.id for item in group)[0]
    ):
        ordered_ids = tuple(sorted(item.id for item in members))
        id_set = set(ordered_ids)
        best, detail = _representative(members, hashes)
        recycled = bool(published_article_ids) and all(
            item.id in published_article_ids for item in members
        )
        parts = [detail]
        if len(members) > 1:
            reason = next(
                reason
                for first_id, second_id, reason in merges
                if first_id in id_set and second_id in id_set
            )
            parts.append(f"grouped: {reason}")
        else:
            parts.append("singleton: no merge partner found")
        for (first_id, second_id), note in sorted(near_misses.items()):
            if first_id in id_set or second_id in id_set:
                parts.append(f"{first_id} vs {second_id}: {note}")
        if recycled:
            parts.append(f"recycled: all {len(members)} articles already published")
        cluster = StoryCluster(
            id=stable_cluster_id(ordered_ids),
            article_ids=ordered_ids,
            representative_article_id=best.id,
            rationale="; ".join(parts),
            created_at=started,
        )
        if store is not None:
            store.save(cluster)
        results.append(ClusterResult(cluster=cluster, recycled=recycled))

    summary = ClusteringSummary(
        attempted=len(articles),
        clusters=len(results),
        articles_clustered=sum(len(item.cluster.article_ids) for item in results),
        duplicates_collapsed=collapsed,
        recycled=sum(1 for item in results if item.recycled),
        results=tuple(results),
        started_at=started,
        finished_at=datetime.now(UTC),
    )
    _LOGGER.info(
        "clustering_complete",
        attempted=summary.attempted,
        clusters=summary.clusters,
        duplicates_collapsed=summary.duplicates_collapsed,
        recycled=summary.recycled,
    )
    return summary


__all__ = [
    "AMBIGUOUS_JACCARD",
    "DATE_WINDOW_HOURS",
    "TITLE_BOOST_JACCARD",
    "TITLE_MERGE_JACCARD",
    "ClusterResult",
    "ClusteringSummary",
    "cluster_articles",
    "hours_apart",
    "jaccard",
    "stable_cluster_id",
    "title_entities",
    "title_tokens",
]
