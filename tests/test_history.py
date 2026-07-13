import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from app.schemas.article import Article
from app.services.history import (
    delivered_titles,
    filter_previously_delivered,
    load_history,
    record_deliveries,
)


def _article(article_id: str, url: str, title: str) -> Article:
    return Article(
        id=article_id,
        source_name="Test Source",
        source_rss="https://example.com/feed",
        title=title,
        url=url,
    )


def test_record_and_load_round_trip(tmp_path: Path) -> None:
    history_file = tmp_path / "history.json"
    article = _article("a1", "https://example.com/a1", "OpenAI launches new model")
    results = [{"article_id": "a1", "status": "sent"}]

    record_deliveries(history_file, [article], results, retention_days=14)
    history = load_history(history_file, retention_days=14)

    assert len(history) == 1
    assert history[0]["url"] == "https://example.com/a1"
    assert delivered_titles(history) == ["OpenAI launches new model"]


def test_record_skips_failed_deliveries(tmp_path: Path) -> None:
    history_file = tmp_path / "history.json"
    article = _article("a1", "https://example.com/a1", "Title")
    results = [{"article_id": "a1", "status": "error"}]

    record_deliveries(history_file, [article], results, retention_days=14)

    assert not history_file.exists()


def test_load_history_prunes_expired_entries(tmp_path: Path) -> None:
    history_file = tmp_path / "history.json"
    now = datetime(2026, 7, 13, 12, 0, tzinfo=timezone.utc)
    entries = [
        {"url": "https://example.com/old", "title": "Old", "delivered_at": (now - timedelta(days=20)).isoformat()},
        {"url": "https://example.com/new", "title": "New", "delivered_at": (now - timedelta(days=2)).isoformat()},
    ]
    history_file.write_text(json.dumps(entries), encoding="utf-8")

    history = load_history(history_file, retention_days=14, now=now)

    assert [entry["url"] for entry in history] == ["https://example.com/new"]


def test_load_history_missing_or_corrupt_file(tmp_path: Path) -> None:
    assert load_history(tmp_path / "missing.json", retention_days=14) == []

    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text("not json", encoding="utf-8")
    assert load_history(corrupt, retention_days=14) == []


def test_filter_previously_delivered_drops_known_urls(tmp_path: Path) -> None:
    delivered = _article("a1", "https://example.com/a1", "Seen story")
    fresh = _article("a2", "https://example.com/a2", "New story")
    now = datetime(2026, 7, 13, 12, 0, tzinfo=timezone.utc)
    history = [
        {
            "url": "https://example.com/a1",
            "title": "Seen story",
            "delivered_at": (now - timedelta(days=1)).isoformat(),
        }
    ]

    filtered = filter_previously_delivered([delivered, fresh], history, now=now)

    assert [article.id for article in filtered] == ["a2"]


def test_filter_previously_delivered_keeps_same_day_deliveries(tmp_path: Path) -> None:
    """Same-day runs shouldn't starve the digest: only prior-day deliveries drop."""
    delivered_today = _article("a1", "https://example.com/a1", "Seen earlier today")
    fresh = _article("a2", "https://example.com/a2", "New story")
    now = datetime(2026, 7, 13, 18, 0, tzinfo=timezone.utc)
    history = [
        {
            "url": "https://example.com/a1",
            "title": "Seen earlier today",
            "delivered_at": datetime(2026, 7, 13, 8, 0, tzinfo=timezone.utc).isoformat(),
        }
    ]

    filtered = filter_previously_delivered([delivered_today, fresh], history, now=now)

    # Both pass: same-day deliveries are exempt.
    assert [article.id for article in filtered] == ["a1", "a2"]


def test_filter_previously_delivered_empty_history_keeps_all(tmp_path: Path) -> None:
    a1 = _article("a1", "https://example.com/a1", "Story A")
    a2 = _article("a2", "https://example.com/a2", "Story B")
    now = datetime(2026, 7, 13, 12, 0, tzinfo=timezone.utc)

    assert filter_previously_delivered([a1, a2], [], now=now) == [a1, a2]
