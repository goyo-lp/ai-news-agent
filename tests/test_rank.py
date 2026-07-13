from datetime import datetime, timedelta, timezone

from app.nodes.rank import filter_articles_published_today
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


def _dated_article(article_id: str, published_at: datetime | None) -> Article:
    return Article(
        id=article_id,
        source_name="Test Source",
        source_rss="https://example.com/feed",
        title=f"Title {article_id}",
        url=f"https://example.com/{article_id}",
        published_at=published_at,
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


def test_rank_articles_caps_items_per_source() -> None:
    distinct_titles = [
        "Medical agents for clinical decision benchmarks",
        "Warehouse robotics planning with vision transformers",
        "Speech synthesis evaluation across low resource languages",
        "Graph neural networks for protein folding dynamics",
        "Reinforcement learning in autonomous driving simulators",
        "Quantum circuit optimization via evolutionary search",
        "Federated training privacy guarantees under drift",
        "Tabular foundation embeddings for credit scoring",
        "Video diffusion architectures with temporal attention",
        "Multilingual retrieval augmentation for legal corpora",
    ]
    flood = [
        _article(f"arxiv{idx}", "arXiv.org (cs.AI)", 1, title=title)
        for idx, title in enumerate(distinct_titles)
    ]
    other = _article("news1", "TechCrunch (AI)", 2, title="Anthropic launches new Claude model")

    ranked = rank_articles(flood + [other], limit=20, max_per_source=3)

    per_source: dict[str, int] = {}
    for item in ranked:
        per_source[item.source_name] = per_source.get(item.source_name, 0) + 1
    assert per_source["arXiv.org (cs.AI)"] == 3
    assert per_source["TechCrunch (AI)"] == 1
    assert len(ranked) == 4


def test_rank_articles_boosts_frontier_company_mentions() -> None:
    frontier = _article(
        "frontier",
        "Unknown",
        2,
        title="Anthropic ships Claude update with new coding features",
    )
    generic = _article(
        "generic",
        "Unknown",
        2,
        title="Researchers ship update with new coding features",
    )

    ranked = rank_articles([generic, frontier], limit=20)
    assert ranked[0].id == "frontier"


def test_rank_articles_penalizes_recently_delivered_titles() -> None:
    repeat = _article(
        "repeat",
        "Unknown",
        2,
        title="OpenAI launches new multimodal model for developers",
    )
    fresh = _article(
        "fresh",
        "Unknown",
        2,
        title="NVIDIA unveils next generation AI accelerator chips",
    )
    recent_titles = ["OpenAI launches new multimodal model for developers"]

    ranked = rank_articles([repeat, fresh], limit=20, recent_titles=recent_titles)
    baseline = rank_articles([repeat, fresh], limit=20)

    repeat_with_history = next(item for item in ranked if item.id == "repeat")
    repeat_baseline = next(item for item in baseline if item.id == "repeat")
    assert (repeat_with_history.score or 0.0) < (repeat_baseline.score or 0.0)


def test_filter_articles_published_today_keeps_same_day() -> None:
    now = datetime(2026, 3, 2, 12, 0, tzinfo=timezone.utc)
    today = _dated_article("today", datetime(2026, 3, 2, 1, 0, tzinfo=timezone.utc))
    old = _dated_article("old", datetime(2026, 3, 1, 23, 59, tzinfo=timezone.utc))

    filtered = filter_articles_published_today([today, old], now=now)
    assert [item.id for item in filtered] == ["today"]


def test_filter_articles_published_today_respects_reference_timezone() -> None:
    eastern = timezone(timedelta(hours=-5))
    now = datetime(2026, 3, 2, 0, 30, tzinfo=eastern)
    old_local_day = _dated_article("old-local", datetime(2026, 3, 2, 3, 30, tzinfo=timezone.utc))
    today_local = _dated_article("today-local", datetime(2026, 3, 2, 6, 0, tzinfo=timezone.utc))

    filtered = filter_articles_published_today([old_local_day, today_local], now=now)
    assert [item.id for item in filtered] == ["today-local"]


def test_filter_articles_published_today_excludes_missing_dates() -> None:
    now = datetime(2026, 3, 2, 12, 0, tzinfo=timezone.utc)
    undated = _dated_article("undated", None)

    filtered = filter_articles_published_today([undated], now=now)
    assert filtered == []
