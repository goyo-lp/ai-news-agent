"""Deterministic ranking ported from the reference LinkedIn agent (Phase 2,
P2.1). Re-clusters curated articles, scores each for technical implementation
value, and demotes hype- and business-only stories — the explicit "demote hype
without technical detail" requirement.

Why a port and not a wrap of the News Agent's own ``src/app/services/scoring``:
the News pipeline's ranker blends generic relevance + an LLM rerank pass but has
no ``technical_depth`` / ``implementation_specificity`` / ``hype_score`` axes
and no ``business_only_penalty``. The reference ranker does exactly that, on a
``DiscoveredItem`` surface that maps cleanly onto ``CuratedArticle``. Porting it
keeps the Stage A / Stage B split (Decision J) and the hype demotion intact.

Inputs are orchestrator-internal ``DiscoveredItem`` adapters, *not* the
boundary ``CuratedArticle``: the boundary model is the on-disk contract, the
adapter is the in-memory shape the ranker consumes. Keeping the adapter
local to the orchestrator stops the boundary from leaking cluster-internal
fields (``query``, ``source_tier``) into the coordinator's stable surface.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
from urllib.parse import urlparse

from pydantic import BaseModel

from app.orchestrator.schemas import CuratedArticle, TopicCandidate

# ---------------------------------------------------------------------------
# Tokenization + keyword sets (verbatim port of reference scoring.py).
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in",
    "into", "is", "it", "its", "of", "on", "or", "that", "the", "their",
    "to", "with", "this", "new",
}

_TECH_KEYWORDS = {
    "model", "models", "agent", "agents", "launch", "release", "released",
    "benchmark", "reasoning", "training", "inference", "multimodal",
    "architecture", "chip", "chips", "accelerator", "open", "source", "api",
    "sdk", "framework",
}

_BUSINESS_KEYWORDS = {
    "startup", "funding", "raised", "series", "valuation", "partnership",
    "acquisition", "enterprise", "deploy", "deployment", "customer",
    "adoption", "contract",
}

_LOW_SIGNAL_KEYWORDS = {
    "roundup", "webinar", "event", "conference", "podcast", "opinion",
    "tutorial", "guide",
}


# ---------------------------------------------------------------------------
# DiscoveredItem — orchestrator-internal adapter for ranking. NOT a boundary
# contract; lives here, with the ranker, because nothing outside the orchestrator
# depends on its shape.
# ---------------------------------------------------------------------------

class DiscoveredItem(BaseModel):
    """Adapter consumed by the ranker. Projected from ``CuratedArticle`` via
    :func:`curated_articles_to_items`; projected back to ``TopicCandidate`` by
    :func:`rank_topics`."""

    id: str
    query: str = ""
    title: str
    url: str
    domain: str
    published_at: datetime | None = None
    snippet: str | None = None
    raw_content: str | None = None

    duplicate_count: int = 1
    source_weight: float = 0.65
    source_tier: int = 3

    score: float | None = None
    cluster_id: str | None = None
    cluster_size: int = 1

    model_config = {"extra": "ignore"}


# ---------------------------------------------------------------------------
# SourcePolicy — permissive default. The News Agent's RSS sources are already
# curated (Decision E: the news pipeline is the sole producer), so the LinkedIn
# agent's tiered-allowlist trust gating over scraped-discovery results is not
# needed here. We port the shape (not the closed-default) so an operator can drop
# a trusted-sources.yaml later; out of the box every curated domain reads as
# tier-1 and every cluster is allowed.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SourcePolicy:
    tier1_domains: frozenset[str]
    tier2_domains: frozenset[str]
    # When True, `cluster_allowed` and `tier_for` short-circuit to permissive
    # values regardless of the tier sets. This is the default-out-of-box mode;
    # a loaded trusted-sources.yaml flips it off by populating real tiers.
    trust_all: bool = True

    @classmethod
    def permissive(cls) -> SourcePolicy:
        return cls(tier1_domains=frozenset(), tier2_domains=frozenset(), trust_all=True)

    def tier_for(self, domain: str) -> int:
        if self.trust_all:
            return 1
        normalized = _normalize_domain(domain)
        if normalized in self.tier1_domains:
            return 1
        if normalized in self.tier2_domains:
            return 2
        return 3

    def weight_for(self, domain: str) -> float:
        tier = self.tier_for(domain)
        if tier == 1:
            return 1.0
        if tier == 2:
            return 0.85
        return 0.6

    def cluster_allowed(self, domains: set[str]) -> bool:
        # trust-all mode trusts the upstream curation: every cluster that made
        # it to articles.json came from a configured, included source.
        if self.trust_all:
            return bool(domains)
        normalized = {_normalize_domain(d) for d in domains}
        if not normalized:
            return False
        trusted = {
            d for d in normalized if d in self.tier1_domains or d in self.tier2_domains
        }
        return bool(trusted)


def _normalize_domain(value: str) -> str:
    return value.lower().replace("www.", "").strip()


# ---------------------------------------------------------------------------
# Clustering + scoring. Verbatim port of reference scoring.py with the
# `select_seed_items` and non-override `_score_item` paths dropped (not needed
# for technical ranking).
# ---------------------------------------------------------------------------

@dataclass
class StoryCluster:
    id: str
    members: list[DiscoveredItem]


def _normalize_text(value: str) -> str:
    return " ".join(_WORD_RE.findall(value.lower()))


def _tokenize(value: str) -> set[str]:
    return {t for t in _WORD_RE.findall(value.lower()) if t not in _STOPWORDS}


def _published_or_min(value: datetime | None) -> datetime:
    if value is None:
        return datetime.min.replace(tzinfo=timezone.utc)
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


def _recency_score(published_at: datetime | None) -> float:
    if published_at is None:
        return 0.3
    published = (
        published_at.astimezone(timezone.utc)
        if published_at.tzinfo
        else published_at.replace(tzinfo=timezone.utc)
    )
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
    if (
        title_similarity >= 0.62
        and overlap_ratio >= 0.70
        and _is_time_aligned(left.published_at, right.published_at)
    ):
        return True
    return False


def cluster_items(items: list[DiscoveredItem]) -> list[StoryCluster]:
    ordered = sorted(items, key=lambda i: _published_or_min(i.published_at), reverse=True)
    clusters: list[StoryCluster] = []
    for item in ordered:
        matched: StoryCluster | None = None
        for cluster in clusters:
            if _same_story(item, cluster.members[0]):
                matched = cluster
                break
        if matched is None:
            clusters.append(StoryCluster(id=item.id, members=[item]))
        else:
            matched.members.append(item)
    return clusters


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
    """Demote funding/deal stories unless paired with concrete technical detail.
    The explicit 'demote hype without technical detail' lever (plan P2.1)."""
    text = f"{item.title} {item.snippet or ''} {item.raw_content or ''}".lower()
    tokens = _tokenize(text)
    has_business = len(tokens & _BUSINESS_KEYWORDS) > 0
    if not has_business:
        return 0.0
    if technical_depth >= 0.55 or implementation_specificity >= 0.45:
        return 0.0
    return 0.16


def _build_rationale(item: DiscoveredItem, cluster_size: int, policy: SourcePolicy) -> str:
    tier = policy.tier_for(item.domain)
    source_label = "tier-1" if tier == 1 else "tier-2" if tier == 2 else "untrusted"
    signal = "high-signal" if _relevance_score(item) >= 0.45 else "mixed-signal"
    corroboration = "multi-source" if cluster_size > 1 else "single-source"
    return f"{signal}, {corroboration}, {source_label}"


def rank_topics(
    items: list[DiscoveredItem],
    limit: int,
    policy: SourcePolicy,
    technical_overrides: dict[str, float] | None = None,
    implementation_overrides: dict[str, float] | None = None,
    hype_overrides: dict[str, float] | None = None,
) -> list[TopicCandidate]:
    """Re-cluster ``items`` and return the top-``limit`` topics as boundary
    ``TopicCandidate`` contracts.

    Determinism caveat: scoring is deterministic *for a fixed wall-clock*.
    ``_recency_score`` buckets by ``datetime.now(timezone.utc)``, so the same
    input can yield different scores across a 6h/24h/48h/96h bucket boundary.
    P2.1's parity tests pin ``published_at`` to ``now`` (recency = 1.0) so the
    ordering it asserts is stable; a future byte-for-byte snapshot harness
    should inject the clock rather than rely on wall-time."""
    if not items or limit <= 0:
        return []

    candidates = [item.model_copy(deep=True) for item in items]
    clusters = cluster_items(candidates)

    topics: list[TopicCandidate] = []
    for cluster in clusters:
        cluster_size = len(cluster.members)
        cluster_domains = {m.domain for m in cluster.members}
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
            key=lambda i: (i.score or 0.0, _published_or_min(i.published_at)),
        )
        summary_hint = representative.snippet or representative.raw_content or representative.title
        summary_hint = " ".join(summary_hint.split())[:260]

        supporting_urls: list[str] = []
        for m in cluster.members:
            if m.url not in supporting_urls:
                supporting_urls.append(m.url)
            if len(supporting_urls) >= 5:
                break

        topics.append(
            TopicCandidate(
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

    topics.sort(key=lambda t: (t.score, _published_or_min(t.published_at)), reverse=True)
    return topics[:limit]


# ---------------------------------------------------------------------------
# CuratedArticle -> DiscoveredItem projection.
# ---------------------------------------------------------------------------

def domain_from_url(url: str) -> str:
    """Extract a normalized registrable-ish host from a URL. Lowercased,
    ``www.`` stripped. Empty for malformed input so a bad URL never crashes the
    ranker (it scores with the default untrusted weight)."""
    host = (urlparse(url).hostname or "").lower()
    return host.removeprefix("www.")


def curated_articles_to_items(articles: list[CuratedArticle], policy: SourcePolicy) -> list[DiscoveredItem]:
    """Project the on-disk boundary ``CuratedArticle`` list into the in-memory
    shape the ranker consumes. The mapping is the seam that lets the boundary
    contract in :mod:`app.orchestrator.schemas` stay stable while the ranker
    evolves its internal fields freely.

    ``summary_context`` (Article.effective_summary_source — OG description or
    RSS description) maps to ``snippet``; ``summary`` (the model-generated
    summary) maps to ``raw_content``. The ranker treats ``raw_content`` as the
    richer of the two when computing relevance/technical depth, which matches
    the reference's intent."""
    items: list[DiscoveredItem] = []
    for article in articles:
        domain = domain_from_url(article.url)
        items.append(
            DiscoveredItem(
                id=article.id,
                query="",
                title=article.title,
                url=article.url,
                domain=domain,
                published_at=article.published_at,
                snippet=article.summary_context or None,
                raw_content=article.summary or None,
                duplicate_count=article.duplicate_count,
                # CuratedArticle.cluster_id/cluster_size are intentionally NOT
                # projected here: rank_topics re-clusters from scratch (it
                # buckets on `_same_story` titles, ignoring the news pipeline's
                # own cluster_id) and overwrites both fields on every member.
                # Projecting them would be dead data suggesting the upstream
                # clustering is preserved when it isn't.
                source_weight=policy.weight_for(domain),
                source_tier=policy.tier_for(domain),
            )
        )
    return items


__all__ = [
    "DiscoveredItem",
    "SourcePolicy",
    "cluster_items",
    "rank_topics",
    "curated_articles_to_items",
    "domain_from_url",
]