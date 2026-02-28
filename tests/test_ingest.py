from datetime import datetime, timezone

from app.schemas.article import Article
from app.services.rss_client import dedupe_articles, normalize_url


def test_normalize_url_removes_tracking_params() -> None:
    url = "https://example.com/story?utm_source=x&id=123&fbclid=abc"
    assert normalize_url(url) == "https://example.com/story?id=123"


def test_dedupe_articles_keeps_newest_and_counts_duplicates() -> None:
    older = Article(
        id="a1",
        source_name="Test",
        source_rss="https://example.com/feed",
        title="Hello",
        url="https://example.com/story?utm_source=x",
        published_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    newer = Article(
        id="a2",
        source_name="Test",
        source_rss="https://example.com/feed",
        title="Hello Updated",
        url="https://example.com/story",
        published_at=datetime(2026, 1, 2, tzinfo=timezone.utc),
    )

    deduped = dedupe_articles([older, newer])
    assert len(deduped) == 1
    assert deduped[0].id == "a2"
    assert deduped[0].duplicate_count == 2
