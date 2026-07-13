from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from app.schemas.article import Article

logger = logging.getLogger(__name__)


def load_history(
    path: str | Path,
    retention_days: int,
    now: datetime | None = None,
) -> list[dict[str, Any]]:
    """Return delivery records newer than the retention window."""
    file_path = Path(path)
    if not file_path.exists():
        return []

    try:
        entries = json.loads(file_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Could not read delivery history %s: %s", file_path, exc)
        return []
    if not isinstance(entries, list):
        logger.warning("Delivery history %s has unexpected format; ignoring", file_path)
        return []

    reference_now = now if now is not None else datetime.now(timezone.utc)
    cutoff = reference_now - timedelta(days=retention_days)

    kept: list[dict[str, Any]] = []
    for entry in entries:
        try:
            delivered_at = datetime.fromisoformat(entry["delivered_at"])
        except (KeyError, TypeError, ValueError):
            continue
        if delivered_at.tzinfo is None:
            delivered_at = delivered_at.replace(tzinfo=timezone.utc)
        if delivered_at >= cutoff:
            kept.append(entry)
    return kept


def delivered_urls(history: list[dict[str, Any]]) -> set[str]:
    return {entry["url"] for entry in history if entry.get("url")}


def delivered_titles(history: list[dict[str, Any]]) -> list[str]:
    return [entry["title"] for entry in history if entry.get("title")]


def filter_previously_delivered(
    articles: list[Article],
    history: list[dict[str, Any]],
    *,
    now: datetime | None = None,
) -> list[Article]:
    """Drop articles whose URL was delivered on a prior day.

    Articles delivered earlier *today* are kept so same-day testing / re-runs
    don't starve the digest. Only deliveries from previous days count as
    "previously delivered". Timezone for "today" follows `now` (defaults to
    local now, matching the rank node's today filter).
    """
    if not history:
        return articles

    reference_now = now if now is not None else datetime.now().astimezone()
    if reference_now.tzinfo is None:
        reference_now = reference_now.replace(tzinfo=timezone.utc)
    today = reference_now.astimezone().date()

    seen: set[str] = set()
    for entry in history:
        url = entry.get("url")
        if not url:
            continue
        try:
            delivered_at = datetime.fromisoformat(entry["delivered_at"])
        except (KeyError, TypeError, ValueError):
            continue
        if delivered_at.tzinfo is None:
            delivered_at = delivered_at.replace(tzinfo=timezone.utc)
        if delivered_at.astimezone(reference_now.tzinfo).date() < today:
            seen.add(url)
    return [article for article in articles if article.url not in seen]


def record_deliveries(
    path: str | Path,
    articles: list[Article],
    results: list[dict[str, Any]],
    retention_days: int,
    now: datetime | None = None,
) -> None:
    """Append successfully sent articles to the history file, pruning expired entries."""
    sent_ids = {item.get("article_id") for item in results if item.get("status") == "sent"}
    if not sent_ids:
        return

    reference_now = now if now is not None else datetime.now(timezone.utc)
    history = load_history(path, retention_days, now=reference_now)
    known_urls = delivered_urls(history)

    for article in articles:
        if article.id in sent_ids and article.url not in known_urls:
            history.append(
                {
                    "url": article.url,
                    "title": article.effective_title,
                    "delivered_at": reference_now.isoformat(),
                }
            )
            known_urls.add(article.url)

    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
