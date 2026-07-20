from datetime import datetime, timezone

import pytest

from app.config import Settings
from app.schemas import Citation, ResearchBrief, StyleProfile
from app.services.post_generator import PostGenerator


def _brief(topic_id: str, headline: str, url: str) -> ResearchBrief:
    return ResearchBrief(
        topic_id=topic_id,
        headline=headline,
        summary=f"{headline} summary",
        technical_significance="Technical significance",
        business_impact="Business impact",
        why_now="Why now",
        key_points=["Key point"],
        risks=["Risk"],
        citations=[
            Citation(
                title=headline,
                url=url,
                domain="example.com",
                published_at=datetime.now(timezone.utc),
            )
        ],
    )


@pytest.mark.asyncio
async def test_post_generator_fallback_returns_five_posts() -> None:
    settings = Settings(_env_file=None, openrouter_api_key=None)
    generator = PostGenerator(settings)

    briefs = [
        _brief("t1", "Model launch", "https://example.com/model"),
        _brief("t2", "Enterprise deployment", "https://example.com/enterprise"),
        _brief("t3", "AI infrastructure", "https://example.com/infra"),
        _brief("t4", "Agentic orchestration framework", "https://example.com/orchestration"),
        _brief("t5", "Evaluation tooling update", "https://example.com/evals"),
    ]
    style = StyleProfile(
        sample_count=1,
        sentence_count=2,
        avg_sentence_words=12.0,
        common_openers=["one thing"],
        vocabulary=["ai", "systems"],
        tone_traits=["direct"],
        cta_patterns=["What do you think?"],
    )

    posts = await generator.generate_posts(briefs=briefs, style_profile=style, dry_run=True)

    assert len(posts) == 5
    assert all(post.angle == "technical_reflection" for post in posts)
    assert all(len(post.supporting_topic_ids) == 1 for post in posts)
    assert all(post.citation_urls for post in posts)
