from datetime import datetime, timedelta, timezone

from app.schemas import DiscoveredItem
from app.services.scoring import rank_topics
from app.services.source_policy import SourcePolicy


def _item(
    item_id: str,
    title: str,
    domain: str,
    hours_old: int,
    snippet: str | None = None,
) -> DiscoveredItem:
    return DiscoveredItem(
        id=item_id,
        query="ai development",
        title=title,
        url=f"https://{domain}/{item_id}",
        domain=domain,
        published_at=datetime.now(timezone.utc) - timedelta(hours=hours_old),
        snippet=snippet,
    )


def test_rank_topics_prioritizes_high_signal() -> None:
    policy = SourcePolicy(
        tier1_domains={"openai.com"},
        tier2_domains={"techcrunch.com"},
    )

    high_signal = _item(
        "high",
        "OpenAI launches a new reasoning model for enterprise agents",
        "openai.com",
        hours_old=8,
    )
    low_signal = _item(
        "low",
        "Weekly AI webinar roundup and event recap",
        "techcrunch.com",
        hours_old=2,
    )

    ranked = rank_topics([low_signal, high_signal], limit=5, policy=policy)
    assert ranked[0].topic_id == "high"


def test_rank_topics_clusters_same_story() -> None:
    policy = SourcePolicy(
        tier1_domains={"openai.com"},
        tier2_domains={"theverge.com"},
    )

    left = _item(
        "a1",
        "OpenAI releases multimodal reasoning model for developers",
        "openai.com",
        hours_old=3,
    )
    right = _item(
        "a2",
        "OpenAI releases a multimodal reasoning model for developers",
        "theverge.com",
        hours_old=4,
    )

    ranked = rank_topics([left, right], limit=5, policy=policy)
    assert len(ranked) == 1
    assert ranked[0].cluster_size == 2
