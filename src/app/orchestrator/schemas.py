"""Shared data contracts for the orchestrator boundary — the single source of
truth for what crosses between the coordinator and its tools/subagents.

Schema split:
  app.schemas          pipeline-internal models (logger/parser-facing, evolve
                       freely as nodes change). Article lives there.
  app.orchestrator.schemas
                       boundary contracts (coordinator <-> tools/subagents).
                       Tighter and more stable than the internal models; this
                       is the surface external actors depend on. Do not reach
                       across the boundary into app.schemas from a tool.

Field parity is enforced constructively (see CuratedArticle.from_article), not
just documented:
  CuratedArticle  <- app.schemas.article.Article (this repo's news pipeline)
  TopicCandidate  <- RankedTopic in reference/linkedin-agent/src/app/schemas/models.py
  ResearchBrief   <- ResearchBrief in the same reference file (identical fields)
  PostProposal    <- LinkedInPost in the same reference file
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from app.schemas.article import Article


class CuratedArticle(BaseModel):
    """Output of the news curation pipeline (ingest -> enrich -> rank ->
    summarize), handed across the orchestrator boundary.

    This is a *projection* of Article, not a copy: it exposes the resolved
    title/summary-context (Article.effective_title / effective_summary_source)
    and hides the pipeline-internal raw variants (source_rss, og_title,
    og_description, rss_image_url). summary is Article.summary and may be
    None if summarization was skipped or failed, in which case summary_context
    still carries fallback text for the researcher.
    """

    id: str
    source_name: str
    title: str
    url: str
    published_at: datetime | None = None
    # Holds Article.effective_summary_source (og_description or RSS description),
    # not a raw RSS field — named to make the projection honest.
    summary_context: str = ""
    image_url: str | None = None
    summary: str | None = None
    score: float | None = None
    duplicate_count: int = 1
    cluster_id: str | None = None
    cluster_size: int = 1

    @classmethod
    def from_article(cls, a: Article) -> CuratedArticle:
        """Construct the boundary projection from a pipeline Article. This
        method is the parity contract: if Article gains/renames a projected
        field, this is the single place that breaks, not a scattering of
        callers. Tests pin it (see tests/test_schemas.py)."""
        return cls(
            id=a.id,
            source_name=a.source_name,
            title=a.effective_title,
            url=a.url,
            published_at=a.published_at,
            summary_context=a.effective_summary_source,
            image_url=a.image_url,
            summary=a.summary,
            score=a.score,
            duplicate_count=a.duplicate_count,
            cluster_id=a.cluster_id,
            cluster_size=a.cluster_size,
        )


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
    verification_status: Literal["unverified", "verified", "failed"] = "unverified"
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
