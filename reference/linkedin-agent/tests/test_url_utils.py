from datetime import datetime, timezone

from app.schemas import DiscoveredItem
from app.services.url_utils import dedupe_items, normalize_url


def test_normalize_url_removes_tracking_params() -> None:
    url = "https://example.com/story?utm_source=x&id=123&fbclid=abc"
    assert normalize_url(url) == "https://example.com/story?id=123"


def test_dedupe_items_keeps_newest_and_counts_duplicates() -> None:
    older = DiscoveredItem(
        id="a1",
        query="q",
        title="Old",
        url="https://example.com/story?utm_source=x",
        domain="example.com",
        published_at=datetime(2026, 3, 1, tzinfo=timezone.utc),
    )
    newer = DiscoveredItem(
        id="a2",
        query="q",
        title="New",
        url="https://example.com/story",
        domain="example.com",
        published_at=datetime(2026, 3, 2, tzinfo=timezone.utc),
    )

    deduped = dedupe_items([older, newer])
    assert len(deduped) == 1
    assert deduped[0].id == "a2"
    assert deduped[0].duplicate_count == 2
