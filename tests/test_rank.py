from datetime import datetime, timedelta, timezone

from app.schemas.article import Article
from app.services.scoring import rank_articles


def _article(
    article_id: str,
    source: str,
    hours_old: int,
    duplicate_count: int = 1,
    title: str | None = None,
    description: str | None = None,
) -> Article:
    return Article(
        id=article_id,
        source_name=source,
        source_rss="https://example.com/feed",
        title=title or f"{source} title {article_id}",
        url=f"https://example.com/{article_id}",
        published_at=datetime.now(timezone.utc) - timedelta(hours=hours_old),
        description=description,
        duplicate_count=duplicate_count,
    )


def test_rank_articles_limits_output() -> None:
    articles = [
        _article(str(idx), "OpenAI Blog", idx, title=f"Unique story {idx}")
        for idx in range(30)
    ]
    ranked = rank_articles(articles, limit=20)
    assert len(ranked) == 20


def test_rank_articles_orders_descending_score() -> None:
    fresh = _article("fresh", "OpenAI Blog", 1, duplicate_count=3)
    old = _article("old", "Unknown", 120, duplicate_count=1)

    ranked = rank_articles([old, fresh], limit=20)
    assert ranked[0].id == "fresh"
    assert (ranked[0].score or 0.0) >= (ranked[1].score or 0.0)


def test_rank_articles_clusters_same_story_across_sources() -> None:
    techcrunch = _article(
        "tc1",
        "TechCrunch (AI)",
        2,
        title="OpenAI launches new multimodal model for developers",
    )
    verge = _article(
        "vg1",
        "The Verge (AI)",
        3,
        title="OpenAI launches a new multimodal model for developers",
    )
    unrelated = _article(
        "other1",
        "MIT Technology Review",
        1,
        title="NVIDIA unveils next generation AI accelerator chips",
    )

    ranked = rank_articles([techcrunch, verge, unrelated], limit=20)
    assert len(ranked) == 2
    assert any(item.id == "other1" for item in ranked)
    clustered = [item for item in ranked if item.id != "other1"][0]
    assert clustered.cluster_size == 2


def test_rank_articles_prioritizes_high_relevance_over_event_roundups() -> None:
    high_relevance = _article(
        "high",
        "Unknown",
        8,
        title="Startup raises Series B to launch enterprise AI agent platform",
    )
    event_roundup = _article(
        "low",
        "OpenAI Blog",
        2,
        title="Top AI events and webinar roundup for this week",
    )

    ranked = rank_articles([event_roundup, high_relevance], limit=20)
    assert ranked[0].id == "high"


def test_rank_articles_uses_description_for_relevance_signals() -> None:
    startup_deal = _article(
        "deal",
        "Unknown",
        4,
        title="Daily AI update",
        description="A startup raised seed funding and signed an enterprise partnership deal.",
    )
    generic_event = _article(
        "event",
        "Unknown",
        1,
        title="Daily AI update",
        description="Conference event recap and webinar schedule for this month.",
    )

    ranked = rank_articles([generic_event, startup_deal], limit=20)
    assert ranked[0].id == "deal"
