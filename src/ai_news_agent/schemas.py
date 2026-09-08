"""Validated records shared by collection, editorial, and delivery stages."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Annotated, Self

from pydantic import (
    AnyHttpUrl,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    computed_field,
    field_validator,
    model_validator,
)

RecordId = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
ScoreValue = Annotated[int, Field(ge=0, le=5)]


class StrictModel(BaseModel):
    """Base model that rejects unknown fields and accidental mutation."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class TimestampedModel(StrictModel):
    """Base model that requires timezone-aware timestamps."""

    @field_validator("*", mode="after")
    @classmethod
    def require_aware_datetimes(cls, value: object) -> object:
        if isinstance(value, datetime) and value.utcoffset() is None:
            raise ValueError("timestamps must include a timezone")
        return value


class SourceTier(StrEnum):
    CORE = "core"
    BREADTH = "breadth"


class SourceCategory(StrEnum):
    LABS_PLATFORMS = "labs-platforms"
    INDEPENDENT_REPORTING = "independent-reporting"
    RESEARCH_ACADEMIA = "research-academia"
    BUILDERS_ANALYSIS = "builders-analysis"
    POLICY_PUBLIC_INTEREST = "policy-public-interest"


class RouteType(StrEnum):
    RSS = "rss"
    ATOM = "atom"
    OFFICIAL_PAGE_ADAPTER = "official_page_adapter"


class Source(TimestampedModel):
    id: RecordId
    name: Text
    publisher_family: RecordId
    category: SourceCategory
    tier: SourceTier
    homepage_url: AnyHttpUrl
    route_type: RouteType
    route_url: AnyHttpUrl
    feed_scope: Text
    editorial_notes: tuple[Text, ...] = ()
    enabled: bool = False
    disabled_reason: Text | None = "pending-validation"
    last_checked_at: datetime | None = None
    last_success_at: datetime | None = None

    @model_validator(mode="after")
    def require_disabled_reason(self) -> Self:
        if not self.enabled and self.disabled_reason is None:
            raise ValueError("disabled sources require disabled_reason")
        if self.enabled and self.disabled_reason is not None:
            raise ValueError("enabled sources cannot have disabled_reason")
        return self


class Article(TimestampedModel):
    id: RecordId
    source_id: RecordId
    url: AnyHttpUrl
    canonical_url: AnyHttpUrl
    title: Text
    authors: tuple[Text, ...] = ()
    published_at: datetime
    updated_at: datetime | None = None
    first_seen_at: datetime
    raw_feed_ref: RecordId | None = None


class ArticleText(TimestampedModel):
    """Source-authored text, deliberately separate from generated summaries."""

    id: RecordId
    article_id: RecordId
    text: Text
    content_hash: Annotated[str, StringConstraints(pattern=r"^[a-f0-9]{64}$")]
    fetched_at: datetime


class StoryCluster(TimestampedModel):
    id: RecordId
    article_ids: Annotated[tuple[RecordId, ...], Field(min_length=1)]
    representative_article_id: RecordId
    rationale: Text
    created_at: datetime

    @model_validator(mode="after")
    def representative_belongs_to_cluster(self) -> Self:
        if self.representative_article_id not in self.article_ids:
            raise ValueError("representative article must belong to the cluster")
        if len(set(self.article_ids)) != len(self.article_ids):
            raise ValueError("article_ids must be unique")
        return self


class Score(StrictModel):
    id: RecordId
    story_cluster_id: RecordId
    relevance: ScoreValue
    impact: ScoreValue
    novelty: ScoreValue
    evidence: ScoreValue
    timeliness: ScoreValue
    evidence_ids: tuple[RecordId, ...] = ()
    rationale: Text

    @computed_field
    @property
    def weighted_total(self) -> float:
        """Compute the accepted editorial score on a 100-point scale."""

        weighted = (
            self.relevance * 30
            + self.impact * 25
            + self.novelty * 20
            + self.evidence * 15
            + self.timeliness * 10
        )
        return weighted / 5


class EvidenceKind(StrEnum):
    FEED_ENTRY = "feed_entry"
    ARTICLE_TEXT = "article_text"
    PRIMARY_DOCUMENT = "primary_document"
    SUPPORTING_REPORT = "supporting_report"


class Evidence(TimestampedModel):
    id: RecordId
    article_id: RecordId
    kind: EvidenceKind
    source_url: AnyHttpUrl
    locator: Text
    excerpt: Text
    content_hash: Annotated[str, StringConstraints(pattern=r"^[a-f0-9]{64}$")]
    captured_at: datetime


class SummaryStatus(StrEnum):
    DRAFT = "draft"
    VERIFIED = "verified"
    REJECTED = "rejected"


class Summary(StrictModel):
    id: RecordId
    story_cluster_id: RecordId
    representative_article_id: RecordId
    sentences: tuple[Text, Text, Text]
    evidence_by_sentence: tuple[
        tuple[RecordId, ...], tuple[RecordId, ...], tuple[RecordId, ...]
    ]
    status: SummaryStatus = SummaryStatus.DRAFT


class VerificationVerdict(StrEnum):
    PASS = "pass"
    REVISE = "revise"
    REPLACE = "replace"
    REJECT = "reject"


class FindingSeverity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class VerificationFinding(StrictModel):
    severity: FindingSeverity
    message: Text
    claim: Text | None = None
    evidence_ids: tuple[RecordId, ...] = ()


class VerificationResult(TimestampedModel):
    id: RecordId
    summary_id: RecordId
    verdict: VerificationVerdict
    findings: tuple[VerificationFinding, ...] = ()
    checked_at: datetime
    attempt: Annotated[int, Field(ge=1)] = 1


class RunStatus(StrEnum):
    RUNNING = "running"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILED = "failed"


class CoverageStats(StrictModel):
    attempted: Annotated[int, Field(ge=0)] = 0
    successful: Annotated[int, Field(ge=0)] = 0
    unchanged: Annotated[int, Field(ge=0)] = 0
    failed: Annotated[int, Field(ge=0)] = 0
    unavailable: Annotated[int, Field(ge=0)] = 0

    @model_validator(mode="after")
    def outcomes_do_not_exceed_attempts(self) -> Self:
        outcomes = self.successful + self.unchanged + self.failed + self.unavailable
        if outcomes > self.attempted:
            raise ValueError("source outcomes cannot exceed attempted count")
        return self


class Run(TimestampedModel):
    id: RecordId
    status: RunStatus
    started_at: datetime
    finished_at: datetime | None = None
    coverage: CoverageStats = CoverageStats()
    policy_version: Text
    code_version: Text
    termination_reason: Text | None = None

    @model_validator(mode="after")
    def finished_state_is_consistent(self) -> Self:
        if self.status is RunStatus.RUNNING and self.finished_at is not None:
            raise ValueError("a running run cannot have finished_at")
        if self.status is not RunStatus.RUNNING and self.finished_at is None:
            raise ValueError("a finished run requires finished_at")
        return self


StoredRecord = (
    Source
    | Article
    | ArticleText
    | StoryCluster
    | Score
    | Evidence
    | Summary
    | VerificationResult
    | Run
)
