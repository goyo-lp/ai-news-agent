"""Choose the top stories and reserves from scored clusters."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime

import langsmith as ls
import structlog

from ai_news_agent.observability import redact
from ai_news_agent.schemas import Article, Score, Source, StoryCluster
from ai_news_agent.storage import SQLiteStore

_LOGGER = structlog.get_logger(__name__)

MAX_DIGEST = 10
DEFAULT_MAX_RESERVE = 5
QUALITY_FLOOR = 65.0
MIN_RELEVANCE = 3
MIN_EVIDENCE = 3
DEFAULT_MAX_PER_FAMILY = 2
DEFAULT_MAX_PER_TOPIC = 4
DIVERSITY_SWAP_LIMIT = 15.0

_TOPIC_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("model-release", ("model", "weight", "checkpoint", "open weight", "release")),
    ("policy", ("policy", "regulation", "law", "court", "executive order")),
    ("security", ("security", "vulnerability", "breach", "jailbreak", "misuse")),
    ("research", ("paper", "benchmark", "study", "evaluation", "preprint")),
    ("market", ("funding", "acquisition", "valuation", "layoff", "ipo")),
)


def topic_bucket(title: str) -> str:
    """Bucket a headline into a coarse topic for soft diversity."""

    lowered = title.lower()
    for bucket, keywords in _TOPIC_KEYWORDS:
        if any(keyword in lowered for keyword in keywords):
            return bucket
    return "general"


def publisher_family(article: Article, store: SQLiteStore) -> str:
    """Return the publisher family, falling back to the source ID."""

    source = store.get(Source, article.source_id)
    return source.publisher_family if source is not None else article.source_id


def eligibility_reason(score: Score) -> str | None:
    """Return None when a score clears the quality floor, else the reason."""

    if not score.evidence_ids:
        return "missing evidence references"
    if score.relevance < MIN_RELEVANCE:
        return f"relevance {score.relevance} below minimum {MIN_RELEVANCE}"
    if score.evidence < MIN_EVIDENCE:
        return f"evidence {score.evidence} below minimum {MIN_EVIDENCE}"
    if score.weighted_total < QUALITY_FLOOR:
        return f"total {score.weighted_total:.1f} below floor {QUALITY_FLOOR:.0f}"
    return None


@dataclass(slots=True)
class SelectionDecision:
    cluster_id: str
    selected: bool
    reason: str
    weighted_total: float = 0.0
    rank: int | None = None
    reserve_rank: int | None = None
    tradeoff_note: str | None = None


@dataclass(slots=True)
class SelectionSummary:
    attempted: int = 0
    eligible: int = 0
    selected: int = 0
    reserve: int = 0
    shortfall: int = 0
    selected_ids: tuple[str, ...] = ()
    reserve_ids: tuple[str, ...] = ()
    decisions: tuple[SelectionDecision, ...] = ()
    diversity_notes: tuple[str, ...] = ()
    shortfall_reason: str | None = None
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    finished_at: datetime = field(default_factory=lambda: datetime.now(UTC))


def _eligible_candidates(
    clusters: tuple[StoryCluster, ...],
    scores_by_cluster: dict[str, Score],
    articles_by_id: dict[str, Article],
) -> tuple[list[tuple[StoryCluster, Score]], list[SelectionDecision]]:
    candidates: list[tuple[StoryCluster, Score]] = []
    decisions: list[SelectionDecision] = []
    seen_urls: set[str] = set()
    for cluster in sorted(clusters, key=lambda item: item.id):
        score = scores_by_cluster.get(cluster.id)
        if score is None:
            decisions.append(
                SelectionDecision(
                    cluster_id=cluster.id,
                    selected=False,
                    reason="missing editorial score",
                )
            )
            continue
        article = articles_by_id.get(cluster.representative_article_id)
        if article is None:
            decisions.append(
                SelectionDecision(
                    cluster_id=cluster.id,
                    selected=False,
                    reason="representative article missing from store",
                    weighted_total=score.weighted_total,
                )
            )
            continue
        canonical = str(article.canonical_url).strip().lower()
        if canonical in seen_urls:
            decisions.append(
                SelectionDecision(
                    cluster_id=cluster.id,
                    selected=False,
                    reason="duplicate event already represented",
                    weighted_total=score.weighted_total,
                )
            )
            continue
        blocked = eligibility_reason(score)
        if blocked is not None:
            decisions.append(
                SelectionDecision(
                    cluster_id=cluster.id,
                    selected=False,
                    reason=blocked,
                    weighted_total=score.weighted_total,
                )
            )
            continue
        seen_urls.add(canonical)
        candidates.append((cluster, score))
    candidates.sort(key=lambda item: (-item[1].weighted_total, item[0].id))
    return candidates, decisions


def _apply_diversity(
    candidates: list[tuple[StoryCluster, Score]],
    articles_by_id: dict[str, Article],
    store: SQLiteStore,
    *,
    max_selected: int,
    max_per_family: int,
    max_per_topic: int,
) -> tuple[list[tuple[StoryCluster, Score]], tuple[str, ...]]:
    selected: list[tuple[StoryCluster, Score]] = []
    deferred: list[tuple[StoryCluster, Score]] = []
    family_counts: Counter[str] = Counter()
    topic_counts: Counter[str] = Counter()
    for cluster, score in candidates:
        if len(selected) >= max_selected:
            deferred.append((cluster, score))
            continue
        article = articles_by_id[cluster.representative_article_id]
        family = publisher_family(article, store)
        topic = topic_bucket(article.title)
        if (
            family_counts[family] >= max_per_family
            or topic_counts[topic] >= max_per_topic
        ):
            deferred.append((cluster, score))
            continue
        selected.append((cluster, score))
        family_counts[family] += 1
        topic_counts[topic] += 1
    notes: list[str] = []
    for cluster, score in list(deferred):
        if len(selected) >= max_selected:
            break
        selected.append((cluster, score))
        article = articles_by_id[cluster.representative_article_id]
        family_counts[publisher_family(article, store)] += 1
        topic_counts[topic_bucket(article.title)] += 1
    pure_top = [cluster.id for cluster, _ in candidates[:max_selected]]
    actual_ids = {cluster.id for cluster, _ in selected[:max_selected]}
    for cluster, score in candidates:
        if cluster.id in pure_top and cluster.id not in actual_ids:
            article = articles_by_id[cluster.representative_article_id]
            family = publisher_family(article, store)
            topic = topic_bucket(article.title)
            for other, other_score in selected:
                if other.id not in pure_top:
                    notes.append(
                        f"diversity: selected {other.id} "
                        f"({other_score.weighted_total:.1f}) over {cluster.id} "
                        f"({score.weighted_total:.1f}) to limit "
                        f"{family}/{topic}"
                    )
                    break
            break
    selected.sort(key=lambda item: (-item[1].weighted_total, item[0].id))
    # One soft swap: prefer a deferred non-violator over the lowest violator
    # when the gap is small, keeping consequential outliers on top.
    if len(selected) == max_selected and deferred:
        families: Counter[str] = Counter()
        topics: Counter[str] = Counter()
        for cluster, _ in selected:
            article = articles_by_id[cluster.representative_article_id]
            families[publisher_family(article, store)] += 1
            topics[topic_bucket(article.title)] += 1
        for index in range(len(selected) - 1, -1, -1):
            cluster, score = selected[index]
            article = articles_by_id[cluster.representative_article_id]
            family = publisher_family(article, store)
            topic = topic_bucket(article.title)
            if families[family] <= max_per_family and topics[topic] <= max_per_topic:
                continue
            for deferred_index, (other, other_score) in enumerate(deferred):
                if other in [item[0] for item in selected]:
                    continue
                other_article = articles_by_id[other.representative_article_id]
                other_family = publisher_family(other_article, store)
                other_topic = topic_bucket(other_article.title)
                if (
                    families[other_family] >= max_per_family
                    or topics[other_topic] >= max_per_topic
                ):
                    continue
                gap = score.weighted_total - other_score.weighted_total
                if 0 < gap <= DIVERSITY_SWAP_LIMIT:
                    selected[index] = (other, other_score)
                    notes.append(
                        f"diversity: selected {other.id} "
                        f"({other_score.weighted_total:.1f}) over {cluster.id} "
                        f"({score.weighted_total:.1f}) to limit "
                        f"{family}/{topic}"
                    )
                    deferred[deferred_index] = (cluster, score)
                    break
            break
    selected.sort(key=lambda item: (-item[1].weighted_total, item[0].id))
    return selected, tuple(notes)


@ls.traceable(
    name="select-digest",
    run_type="chain",
    process_inputs=redact,
    process_outputs=redact,
)
def select_digest(
    clusters: tuple[StoryCluster, ...],
    scores_by_cluster: dict[str, Score],
    articles_by_id: dict[str, Article],
    store: SQLiteStore,
    *,
    now: datetime | None = None,
    max_selected: int = MAX_DIGEST,
    max_reserve: int = DEFAULT_MAX_RESERVE,
    max_per_family: int = DEFAULT_MAX_PER_FAMILY,
    max_per_topic: int = DEFAULT_MAX_PER_TOPIC,
) -> SelectionSummary:
    """Select up to ten qualifying stories plus a ranked reserve list.

    Eligibility enforces distinct events and the editorial quality floor in
    code. Soft publisher/topic preferences reorder but never force padding:
    quiet days publish fewer than ten with a recorded shortfall reason.
    """

    started = now or datetime.now(UTC)
    candidates, rejected = _eligible_candidates(
        clusters, scores_by_cluster, articles_by_id
    )
    selected, diversity_notes = _apply_diversity(
        candidates,
        articles_by_id,
        store,
        max_selected=max_selected,
        max_per_family=max_per_family,
        max_per_topic=max_per_topic,
    )
    selected_ids = tuple(cluster.id for cluster, _ in selected[:max_selected])
    remaining = [item for item in candidates if item[0].id not in set(selected_ids)]
    reserve_ids = tuple(item[0].id for item in remaining[:max_reserve])

    decisions: list[SelectionDecision] = list(rejected)
    rank_by_id = {
        cluster_id: index + 1 for index, cluster_id in enumerate(selected_ids)
    }
    reserve_by_id = {
        cluster_id: index + 1 for index, cluster_id in enumerate(reserve_ids)
    }
    note_by_id: dict[str, str] = {}
    for note in diversity_notes:
        # Notes name both clusters as "<selected> ... <replaced>".
        parts = note.split()
        if len(parts) >= 6:
            note_by_id.setdefault(parts[2], note)
            note_by_id.setdefault(parts[5], note)
    for cluster, score in candidates:
        if cluster.id in rank_by_id:
            decisions.append(
                SelectionDecision(
                    cluster_id=cluster.id,
                    selected=True,
                    reason="clears quality floor",
                    weighted_total=score.weighted_total,
                    rank=rank_by_id[cluster.id],
                    tradeoff_note=note_by_id.get(cluster.id),
                )
            )
        elif cluster.id in reserve_by_id:
            decisions.append(
                SelectionDecision(
                    cluster_id=cluster.id,
                    selected=False,
                    reason="reserve candidate",
                    weighted_total=score.weighted_total,
                    reserve_rank=reserve_by_id[cluster.id],
                    tradeoff_note=note_by_id.get(cluster.id),
                )
            )
        elif cluster.id not in {item.cluster_id for item in rejected}:
            decisions.append(
                SelectionDecision(
                    cluster_id=cluster.id,
                    selected=False,
                    reason="outside top selection and reserve",
                    weighted_total=score.weighted_total,
                )
            )
    shortfall = max(0, max_selected - len(selected_ids))
    shortfall_reason = (
        f"selected {len(selected_ids)} of {max_selected} slots; "
        f"{len(candidates)} eligible of {len(clusters)} attempted"
        if shortfall
        else None
    )
    summary = SelectionSummary(
        attempted=len(clusters),
        eligible=len(candidates),
        selected=len(selected_ids),
        reserve=len(reserve_ids),
        shortfall=shortfall,
        selected_ids=selected_ids,
        reserve_ids=reserve_ids,
        decisions=tuple(sorted(decisions, key=lambda item: item.cluster_id)),
        diversity_notes=diversity_notes,
        shortfall_reason=shortfall_reason,
        started_at=started,
        finished_at=datetime.now(UTC),
    )
    _LOGGER.info(
        "digest_selection_complete",
        attempted=summary.attempted,
        eligible=summary.eligible,
        selected=summary.selected,
        reserve=summary.reserve,
        shortfall=summary.shortfall,
    )
    return summary


def render_draft(
    summary: SelectionSummary,
    clusters_by_id: dict[str, StoryCluster],
    articles_by_id: dict[str, Article],
    scores_by_cluster: dict[str, Score],
    store: SQLiteStore,
) -> str:
    """Render a reviewable markdown draft from a selection summary."""

    lines = ["# Digest draft (unverified)", ""]
    if summary.shortfall_reason:
        lines.append(f"> Shortfall: {summary.shortfall_reason}")
        lines.append("")
    for rank, cluster_id in enumerate(summary.selected_ids, start=1):
        cluster = clusters_by_id.get(cluster_id)
        score = scores_by_cluster.get(cluster_id)
        article = (
            articles_by_id.get(cluster.representative_article_id)
            if cluster is not None
            else None
        )
        if cluster is None or score is None or article is None:
            lines.append(f"## {rank}. {cluster_id} (missing context)")
            continue
        source = store.get(Source, article.source_id)
        name = source.name if source is not None else article.source_id
        lines.append(f"## {rank}. [{article.title}]({article.url})")
        lines.append(f"Source: {name} · Published: {article.published_at.isoformat()}")
        lines.append(
            f"Score: {score.weighted_total:.1f} "
            f"(R{score.relevance} I{score.impact} N{score.novelty} "
            f"E{score.evidence} T{score.timeliness})"
        )
        lines.append(f"Rationale: {score.rationale}")
        lines.append(f"Evidence: {', '.join(score.evidence_ids)}")
        lines.append("")
    if summary.reserve_ids:
        lines.append("## Reserve")
        lines.append("")
        for rank, cluster_id in enumerate(summary.reserve_ids, start=1):
            score = scores_by_cluster.get(cluster_id)
            total = f"{score.weighted_total:.1f}" if score is not None else "n/a"
            lines.append(f"{rank}. {cluster_id} ({total})")
        lines.append("")
    if summary.diversity_notes:
        lines.append("## Diversity notes")
        lines.append("")
        lines.extend(f"- {note}" for note in summary.diversity_notes)
        lines.append("")
    return "\n".join(lines).strip() + "\n"


__all__ = [
    "DEFAULT_MAX_PER_FAMILY",
    "DEFAULT_MAX_PER_TOPIC",
    "DEFAULT_MAX_RESERVE",
    "DIVERSITY_SWAP_LIMIT",
    "MAX_DIGEST",
    "MIN_EVIDENCE",
    "MIN_RELEVANCE",
    "QUALITY_FLOOR",
    "SelectionDecision",
    "SelectionSummary",
    "eligibility_reason",
    "publisher_family",
    "render_draft",
    "select_digest",
    "topic_bucket",
]
