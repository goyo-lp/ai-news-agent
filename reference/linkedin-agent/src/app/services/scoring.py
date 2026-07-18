from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher

from app.schemas import DiscoveredItem, RankedTopic
from app.services.source_policy import SourcePolicy

_WORD_RE = re.compile(r"[a-z0-9]+")
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
    "with",
    "this",
    "new",
}

_TECH_KEYWORDS = {
    "model",
    "models",
    "agent",
    "agents",
    "launch",
    "release",
    "released",
    "benchmark",
    "reasoning",
    "training",
    "inference",
    "multimodal",
    "architecture",
    "chip",
    "chips",
    "accelerator",
    "open",
    "source",
    "api",
    "sdk",
    "framework",
}

_BUSINESS_KEYWORDS = {
    "startup",
    "funding",
    "raised",
    "series",
    "valuation",
    "partnership",
    "acquisition",
    "enterprise",
    "deploy",
    "deployment",
    "customer",
    "adoption",
    "contract",
}

_LOW_SIGNAL_KEYWORDS = {
    "roundup",
    "webinar",
    "event",
    "conference",
    "podcast",
    "opinion",
    "tutorial",
    "guide",
}


@dataclass
class StoryCluster:
    id: str
    members: list[DiscoveredItem]


def _normalize_text(value: str) -> str:
    return " ".join(_WORD_RE.findall(value.lower()))


def _tokenize(value: str) -> set[str]:
    return {token for token in _WORD_RE.findall(value.lower()) if token not in _STOPWORDS}


def _published_or_min(value: datetime | None) -> datetime:
    if value is None:
        return datetime.min.replace(tzinfo=timezone.utc)
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


def _recency_score(published_at: datetime | None) -> float:
    if published_at is None:
        return 0.3

    published = published_at.astimezone(timezone.utc) if published_at.tzinfo else published_at.replace(tzinfo=timezone.utc)
    now = datetime.now(timezone.utc)
    hours_old = max((now - published).total_seconds() / 3600.0, 0.0)

    if hours_old <= 6:
        return 1.0
    if hours_old <= 24:
        return 0.85
    if hours_old <= 48:
        return 0.7
    if hours_old <= 96:
        return 0.45
    return 0.2


def _relevance_score(item: DiscoveredItem) -> float:
    text = f"{item.title} {item.snippet or ''} {item.raw_content or ''}".lower()
    tokens = _tokenize(text)
    if not tokens:
        return 0.2

    tech_hits = len(tokens & _TECH_KEYWORDS)
    business_hits = len(tokens & _BUSINESS_KEYWORDS)
    low_signal_hits = len(tokens & _LOW_SIGNAL_KEYWORDS)

    score = 0.2
    if tech_hits > 0:
        score += min(0.45, 0.2 + 0.05 * tech_hits)
    if business_hits > 0:
        score += min(0.3, 0.12 + 0.05 * business_hits)

    if low_signal_hits > 0 and tech_hits == 0 and business_hits == 0:
        score -= min(0.35, 0.1 * low_signal_hits)

    return max(0.0, min(score, 1.0))


def _novelty_score(item: DiscoveredItem) -> float:
    tokens = _tokenize(item.title)
    if not tokens:
        return 0.0
    return min(1.0, len(tokens) / 14.0)


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
    delta = abs((_published_or_min(left) - _published_or_min(right)).total_seconds()) / 3600.0
    return delta <= max_hours


def _same_story(left: DiscoveredItem, right: DiscoveredItem) -> bool:
    left_tokens = _tokenize(left.title)
    right_tokens = _tokenize(right.title)
    overlap_count = len(left_tokens & right_tokens)
    if overlap_count < 2:
        return False

    min_token_count = max(min(len(left_tokens), len(right_tokens)), 1)
    overlap_ratio = overlap_count / min_token_count
    title_similarity = _title_similarity(left.title, right.title)

    if title_similarity >= 0.78 and overlap_ratio >= 0.45:
        return True
    if title_similarity >= 0.62 and overlap_ratio >= 0.70 and _is_time_aligned(left.published_at, right.published_at):
        return True
    return False


def cluster_items(items: list[DiscoveredItem]) -> list[StoryCluster]:
    ordered = sorted(items, key=lambda item: _published_or_min(item.published_at), reverse=True)

    clusters: list[StoryCluster] = []
    for item in ordered:
        matched_cluster: StoryCluster | None = None
        for cluster in clusters:
            representative = cluster.members[0]
            if _same_story(item, representative):
                matched_cluster = cluster
                break

        if matched_cluster is None:
            clusters.append(StoryCluster(id=item.id, members=[item]))
        else:
            matched_cluster.members.append(item)

    return clusters


def _score_item(item: DiscoveredItem, cluster_size: int, policy: SourcePolicy) -> float:
    relevance = _relevance_score(item)
    recency = _recency_score(item.published_at)
    source_weight = policy.weight_for(item.domain)
    corroboration = min(max(cluster_size, 1) / 4.0, 1.0)
    novelty = _novelty_score(item)

    score = (
        0.42 * relevance
        + 0.23 * recency
        + 0.20 * source_weight
        + 0.10 * corroboration
        + 0.05 * novelty
    )
    return round(score, 5)


def _score_item_with_overrides(
    item: DiscoveredItem,
    cluster_size: int,
    policy: SourcePolicy,
    technical_override: float | None = None,
    implementation_override: float | None = None,
    hype_override: float | None = None,
) -> float:
    technical_depth = technical_override if technical_override is not None else _relevance_score(item)
    implementation_specificity = implementation_override if implementation_override is not None else 0.35
    hype_score = hype_override if hype_override is not None else 0.0

    recency = _recency_score(item.published_at)
    source_weight = policy.weight_for(item.domain)
    corroboration = min(max(cluster_size, 1) / 4.0, 1.0)
    novelty = _novelty_score(item)
    business_only_penalty = _business_only_penalty(item, technical_depth, implementation_specificity)

    score = (
        0.38 * technical_depth
        + 0.20 * implementation_specificity
        + 0.18 * recency
        + 0.12 * source_weight
        + 0.07 * corroboration
        + 0.05 * novelty
        - 0.18 * hype_score
        - business_only_penalty
    )
    return round(max(0.0, min(score, 1.0)), 5)


def _business_only_penalty(
    item: DiscoveredItem,
    technical_depth: float,
    implementation_specificity: float,
) -> float:
    text = f"{item.title} {item.snippet or ''} {item.raw_content or ''}".lower()
    tokens = _tokenize(text)
    has_business = len(tokens & _BUSINESS_KEYWORDS) > 0
    if not has_business:
        return 0.0

    # Keep funding/deal stories only when paired with technical implementation detail.
    if technical_depth >= 0.55 or implementation_specificity >= 0.45:
        return 0.0
    return 0.16


def _build_rationale(item: DiscoveredItem, cluster_size: int, policy: SourcePolicy) -> str:
    tier = policy.tier_for(item.domain)
    source_label = "tier-1" if tier == 1 else "tier-2" if tier == 2 else "untrusted"
    signal = "high-signal"
    if _relevance_score(item) < 0.45:
        signal = "mixed-signal"

    corroboration = "multi-source" if cluster_size > 1 else "single-source"
    return f"{signal}, {corroboration}, {source_label}"


def rank_topics(
    items: list[DiscoveredItem],
    limit: int,
    policy: SourcePolicy,
    technical_overrides: dict[str, float] | None = None,
    implementation_overrides: dict[str, float] | None = None,
    hype_overrides: dict[str, float] | None = None,
) -> list[RankedTopic]:
    if not items or limit <= 0:
        return []

    candidates = [item.model_copy(deep=True) for item in items]
    clusters = cluster_items(candidates)

    topics: list[RankedTopic] = []
    for cluster in clusters:
        cluster_size = len(cluster.members)
        cluster_domains = {member.domain for member in cluster.members}

        if not policy.cluster_allowed(cluster_domains):
            continue

        for member in cluster.members:
            member.cluster_id = cluster.id
            member.cluster_size = cluster_size
            member.score = _score_item_with_overrides(
                member,
                cluster_size=cluster_size,
                policy=policy,
                technical_override=(technical_overrides or {}).get(member.id),
                implementation_override=(implementation_overrides or {}).get(member.id),
                hype_override=(hype_overrides or {}).get(member.id),
            )
            member.source_tier = policy.tier_for(member.domain)
            member.source_weight = policy.weight_for(member.domain)

        representative = max(
            cluster.members,
            key=lambda item: (
                item.score or 0.0,
                _published_or_min(item.published_at),
            ),
        )

        summary_hint = representative.snippet or representative.raw_content or representative.title
        summary_hint = " ".join(summary_hint.split())[:260]

        supporting_urls = []
        for member in cluster.members:
            if member.url not in supporting_urls:
                supporting_urls.append(member.url)
            if len(supporting_urls) >= 5:
                break

        topics.append(
            RankedTopic(
                topic_id=cluster.id,
                title=representative.title,
                summary_hint=summary_hint,
                primary_url=representative.url,
                primary_domain=representative.domain,
                published_at=representative.published_at,
                score=float(representative.score or 0.0),
                cluster_size=cluster_size,
                rationale=_build_rationale(representative, cluster_size=cluster_size, policy=policy),
                supporting_urls=supporting_urls,
            )
        )

    topics.sort(key=lambda topic: (topic.score, _published_or_min(topic.published_at)), reverse=True)
    return topics[:limit]


def select_seed_items(
    items: list[DiscoveredItem],
    limit: int,
    policy: SourcePolicy,
) -> list[DiscoveredItem]:
    if not items or limit <= 0:
        return []

    def seed_score(item: DiscoveredItem) -> float:
        relevance = _relevance_score(item)
        recency = _recency_score(item.published_at)
        source_weight = policy.weight_for(item.domain)
        duplicate_signal = min(max(item.duplicate_count, 1) / 4.0, 1.0)
        return (
            0.38 * relevance
            + 0.32 * recency
            + 0.20 * source_weight
            + 0.10 * duplicate_signal
        )

    selected = [item.model_copy(deep=True) for item in items]
    selected.sort(
        key=lambda item: (
            seed_score(item),
            _published_or_min(item.published_at),
        ),
        reverse=True,
    )
    return selected[:limit]
