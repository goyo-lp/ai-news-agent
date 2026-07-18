from __future__ import annotations

from datetime import datetime, timezone
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from app.schemas import DiscoveredItem

_TRACKING_PARAMS = {
    "utm_source",
    "utm_medium",
    "utm_campaign",
    "utm_term",
    "utm_content",
    "utm_id",
    "gclid",
    "fbclid",
    "mc_cid",
    "mc_eid",
}


def normalize_url(url: str) -> str:
    parsed = urlparse(url.strip())
    cleaned_query = [
        (key, value)
        for key, value in parse_qsl(parsed.query, keep_blank_values=True)
        if key not in _TRACKING_PARAMS
    ]
    normalized = parsed._replace(fragment="", query=urlencode(cleaned_query, doseq=True))
    return urlunparse(normalized)


def domain_from_url(url: str) -> str:
    parsed = urlparse(url)
    return parsed.netloc.lower().replace("www.", "").strip()


def _published_or_min(value: datetime | None) -> datetime:
    if value is None:
        return datetime.min.replace(tzinfo=timezone.utc)
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


def dedupe_items(items: list[DiscoveredItem]) -> list[DiscoveredItem]:
    deduped: dict[str, DiscoveredItem] = {}
    for item in items:
        normalized_url = normalize_url(item.url)
        updated = item.model_copy(deep=True)
        updated.url = normalized_url
        updated.domain = domain_from_url(normalized_url)

        existing = deduped.get(normalized_url)
        if existing is None:
            deduped[normalized_url] = updated
            continue

        if _published_or_min(updated.published_at) > _published_or_min(existing.published_at):
            updated.duplicate_count = existing.duplicate_count + 1
            deduped[normalized_url] = updated
        else:
            existing.duplicate_count += 1

    return list(deduped.values())
