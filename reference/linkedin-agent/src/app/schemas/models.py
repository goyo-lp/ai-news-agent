from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Sequence

from pydantic import BaseModel, Field


class DiscoveredItem(BaseModel):
    id: str
    query: str
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


class RankedTopic(BaseModel):
    topic_id: str
    title: str
    summary_hint: str
    primary_url: str
    primary_domain: str
    published_at: datetime | None = None
    score: float
    cluster_size: int
    rationale: str
    supporting_urls: list[str] = Field(default_factory=list)


class Citation(BaseModel):
    title: str
    url: str
    domain: str
    published_at: datetime | None = None


class ResearchBrief(BaseModel):
    topic_id: str
    headline: str
    summary: str
    technical_significance: str
    business_impact: str
    why_now: str
    key_points: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    verification_status: str = "unverified"
    verification_confidence: float = 0.0
    verification_notes: list[str] = Field(default_factory=list)


class StyleProfile(BaseModel):
    sample_count: int = 0
    sentence_count: int = 0
    avg_sentence_words: float = 0.0
    common_openers: list[str] = Field(default_factory=list)
    vocabulary: list[str] = Field(default_factory=list)
    tone_traits: list[str] = Field(default_factory=list)
    cta_patterns: list[str] = Field(default_factory=list)


class LinkedInPost(BaseModel):
    post_id: str
    angle: str
    headline: str
    body: str
    hashtags: list[str] = Field(default_factory=list)
    supporting_topic_ids: list[str] = Field(default_factory=list)
    citation_urls: list[str] = Field(default_factory=list)
    confidence: float = 0.0


class DeliveryResult(BaseModel):
    post_id: str
    status: str
    message_id: int | None = None
    error: str | None = None


class RunReport(BaseModel):
    run_id: str
    run_date: str
    started_at: str
    completed_at: str
    dry_run: bool
    discovered_count: int
    normalized_count: int
    topics_count: int
    briefs_count: int
    posts_count: int
    deliveries_attempted: int = 0
    deliveries_sent: int = 0
    deliveries_failed: int = 0
    quality_checks: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


def serialize_models(items: Sequence[BaseModel]) -> list[dict[str, Any]]:
    return [item.model_dump(mode="json") for item in items]


def parse_discovered(payload: list[dict[str, Any]] | None) -> list[DiscoveredItem]:
    if not payload:
        return []
    return [DiscoveredItem.model_validate(item) for item in payload]


def parse_ranked_topics(payload: list[dict[str, Any]] | None) -> list[RankedTopic]:
    if not payload:
        return []
    return [RankedTopic.model_validate(item) for item in payload]


def parse_research_briefs(payload: list[dict[str, Any]] | None) -> list[ResearchBrief]:
    if not payload:
        return []
    return [ResearchBrief.model_validate(item) for item in payload]


def parse_style_profile(payload: dict[str, Any] | None) -> StyleProfile:
    if not payload:
        return StyleProfile()
    return StyleProfile.model_validate(payload)


def parse_linkedin_posts(payload: list[dict[str, Any]] | None) -> list[LinkedInPost]:
    if not payload:
        return []
    return [LinkedInPost.model_validate(item) for item in payload]


def parse_delivery_results(payload: list[dict[str, Any]] | None) -> list[DeliveryResult]:
    if not payload:
        return []
    return [DeliveryResult.model_validate(item) for item in payload]


def model_to_json(value: BaseModel) -> str:
    return json.dumps(value.model_dump(mode="json"), indent=2)
