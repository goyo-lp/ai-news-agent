from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.orchestrator.schemas import (
    Citation,
    CuratedArticle,
    PostProposal,
    ResearchBrief,
    TopicCandidate,
)
from app.schemas.article import Article


def _sample_article(**overrides: object) -> Article:
    base = dict(
        id="a1",
        source_name="Example Feed",
        source_rss="https://example.com/feed.xml",
        title="Raw RSS Title",
        url="https://example.com/a1",
        description="RSS description fallback.",
    )
    base.update(overrides)
    return Article(**base)  # type: ignore[arg-type]


def test_from_article_projects_effective_title_and_summary_context() -> None:
    """The boundary projection must surface resolved fields, not raw RSS/OG."""
    a = _sample_article(
        og_title="Polished OG Title",
        og_description="OG description for the researcher.",
    )
    curated = CuratedArticle.from_article(a)

    assert curated.title == "Polished OG Title"
    assert curated.summary_context == "OG description for the researcher."
    assert curated.id == "a1"
    assert curated.source_name == "Example Feed"
    assert curated.url == "https://example.com/a1"


def test_from_article_falls_back_to_rss_when_og_missing() -> None:
    a = _sample_article()  # no og_title / og_description
    curated = CuratedArticle.from_article(a)

    assert curated.title == "Raw RSS Title"
    assert curated.summary_context == "RSS description fallback."


def test_from_article_hides_pipeline_internal_fields() -> None:
    """The whole point of the projection: source_rss, og_*, rss_image_url
    must not leak across the boundary."""
    a = _sample_article(
        og_title="OG",
        og_description="OG desc",
        rss_image_url="https://example.com/img.png",
    )
    CuratedArticle.from_article(a)
    fields = set(CuratedArticle.model_fields.keys())

    assert "source_rss" not in fields
    assert "og_title" not in fields
    assert "og_description" not in fields
    assert "rss_image_url" not in fields
    assert "source_url" not in fields


def test_from_article_carries_cluster_and_score() -> None:
    a = _sample_article(cluster_id="c1", cluster_size=4, score=0.91, duplicate_count=3)
    curated = CuratedArticle.from_article(a)

    assert curated.cluster_id == "c1"
    assert curated.cluster_size == 4
    assert curated.score == 0.91
    assert curated.duplicate_count == 3


def test_curated_article_defaults() -> None:
    curated = CuratedArticle.from_article(_sample_article())
    assert curated.cluster_size == 1
    assert curated.duplicate_count == 1
    assert curated.summary is None
    assert curated.image_url is None


def test_research_brief_verification_status_rejects_bogus() -> None:
    with pytest.raises(ValidationError):
        ResearchBrief(
            topic_id="t1",
            headline="h",
            summary="s",
            technical_significance="ts",
            business_impact="bi",
            why_now="wn",
            verification_status="bogus",  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "status",
    ["unverified", "verified", "partially_verified", "insufficient_evidence", "failed"],
)
def test_research_brief_verification_status_accepts_full_lifecycle(status: str) -> None:
    """P2.4 widens the verification_status enum to the BriefVerifier's actual
    outputs (verified / partially_verified / insufficient_evidence) plus the
    catastrophic-tool-failure sentinel (failed). All five must validate so the
    verify_claim tool can write each one without a model validator tripping."""
    brief = ResearchBrief(
        topic_id="t1",
        headline="h",
        summary="s",
        technical_significance="ts",
        business_impact="bi",
        why_now="wn",
        verification_status=status,  # type: ignore[arg-type]
        verification_confidence=0.5,
    )
    assert brief.verification_status == status


def test_research_brief_round_trip() -> None:
    brief = ResearchBrief(
        topic_id="t1",
        headline="A Big Announcement",
        summary="What happened and why it matters.",
        technical_significance="New architecture improves throughput.",
        business_impact="Cuts inference cost for adopters.",
        why_now="Ships this week.",
        key_points=["Point one", "Point two"],
        risks=["Unverified benchmark claims"],
        citations=[
            Citation(title="Source A", url="https://example.com/a1", domain="example.com")
        ],
        verification_status="verified",
        verification_confidence=0.9,
        verification_notes=["Confirmed via two independent sources."],
    )
    restored = ResearchBrief.model_validate_json(brief.model_dump_json())
    assert restored == brief


def test_topic_candidate_round_trip() -> None:
    topic = TopicCandidate(
        topic_id="t1",
        title="A Big Announcement",
        summary_hint="One-line hint for the researcher.",
        primary_url="https://example.com/a1",
        primary_domain="example.com",
        score=0.75,
        cluster_size=2,
        rationale="Strong technical signal across sources.",
        supporting_urls=["https://example.com/a1", "https://another.com/a1"],
    )
    restored = TopicCandidate.model_validate_json(topic.model_dump_json())
    assert restored == topic


def test_post_proposal_round_trip() -> None:
    post = PostProposal(
        post_id="p1",
        angle="technical-deep-dive",
        headline="A Big Announcement",
        body="Full post body text.",
        hashtags=["#AI", "#MachineLearning"],
        supporting_topic_ids=["t1"],
        citation_urls=["https://example.com/a1"],
        confidence=0.8,
    )
    restored = PostProposal.model_validate_json(post.model_dump_json())
    assert restored == post