from __future__ import annotations

from app.orchestrator.schemas import (
    Citation,
    CuratedArticle,
    PostProposal,
    ResearchBrief,
    TopicCandidate,
)


def test_curated_article_json_round_trip() -> None:
    article = CuratedArticle(
        id="a1",
        source_name="Example Feed",
        title="A Big Announcement",
        url="https://example.com/a1",
        description="Longer fallback text.",
        image_url="https://example.com/a1.jpg",
        summary="Short AI summary.",
        score=0.87,
        cluster_id="c1",
        cluster_size=3,
    )
    restored = CuratedArticle.model_validate_json(article.model_dump_json())
    assert restored == article


def test_topic_candidate_json_round_trip() -> None:
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


def test_research_brief_json_round_trip() -> None:
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


def test_post_proposal_json_round_trip() -> None:
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
