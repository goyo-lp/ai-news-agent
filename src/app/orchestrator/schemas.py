"""Shared data contracts for the orchestrator boundary — the single source of
truth for what crosses between the coordinator and its tools/subagents.

Field parity:
  CuratedArticle  <- app.schemas.article.Article (this repo's news pipeline)
  TopicCandidate  <- RankedTopic in reference/linkedin-agent/src/app/schemas/models.py
  ResearchBrief   <- ResearchBrief in the same reference file (identical fields)
  PostProposal    <- LinkedInPost in the same reference file
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class CuratedArticle(BaseModel):
    """Output of the news curation pipeline (ingest -> enrich -> rank ->
    summarize), handed across the orchestrator boundary.

    Field parity with Article: id, source_name, url, published_at, image_url,
    score, duplicate_count, cluster_id, cluster_size carry the same name/type.
    title and description carry Article.effective_title /
    effective_summary_source instead of the raw RSS/OG variants — those are
    pipeline-internal and aren't exposed across the boundary. summary is
    Article.summary and may be None if summarization was skipped or failed,
    in which case description remains as fallback text.
    """

    id: str
    source_name: str
    title: str
    url: str
    published_at: datetime | None = None
    description: str | None = None
    image_url: str | None = None
    summary: str | None = None
    score: float | None = None
    duplicate_count: int = 1
    cluster_id: str | None = None
    cluster_size: int = 1


class TopicCandidate(BaseModel):
    """A ranked, deduplicated candidate topic proposed for research.

    Field parity with RankedTopic: identical fields/types.
    """

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
    """A source citation backing a ResearchBrief. Field parity with the
    reference Citation model: identical fields/types."""

    title: str
    url: str
    domain: str
    published_at: datetime | None = None


class ResearchBrief(BaseModel):
    """Verified research output for one topic candidate, ready for post
    drafting.

    Field parity with ResearchBrief: identical fields/types, including the
    nested Citation model.
    """

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


class PostProposal(BaseModel):
    """A drafted post proposal awaiting the quality gate and delivery.

    Field parity with LinkedInPost: identical fields/types; renamed to
    PostProposal to reflect its pre-delivery status at the coordinator
    boundary.
    """

    post_id: str
    angle: str
    headline: str
    body: str
    hashtags: list[str] = Field(default_factory=list)
    supporting_topic_ids: list[str] = Field(default_factory=list)
    citation_urls: list[str] = Field(default_factory=list)
    confidence: float = 0.0
