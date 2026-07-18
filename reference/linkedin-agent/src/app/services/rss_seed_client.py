from __future__ import annotations

import asyncio
import hashlib
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any

import feedparser  # type: ignore[import-untyped]
import httpx
import yaml  # type: ignore[import-untyped]

from app.config import Settings
from app.schemas import DiscoveredItem
from app.services.url_utils import dedupe_items, domain_from_url, normalize_url

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RSSSource:
    name: str
    url: str
    rss: str


class RSSSeedClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def load_sources(self) -> list[RSSSource]:
        path = self.settings.sources_path
        if not path.exists():
            raise FileNotFoundError(f"Sources file not found: {path}")

        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        raw_sources = payload.get("sources") or []

        sources: list[RSSSource] = []
        for item in raw_sources:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "").strip()
            rss = str(item.get("rss") or "").strip()
            url = str(item.get("url") or rss).strip()
            if not name or not rss:
                continue
            sources.append(RSSSource(name=name, url=url, rss=rss))

        return sources

    async def fetch_daily_seed_items(self, dry_run: bool = False) -> tuple[list[DiscoveredItem], list[str]]:
        if dry_run:
            return self._mock_seed_items(limit=max(5, self.settings.seed_articles_per_run)), []

        sources = self.load_sources()
        if not sources:
            return [], ["No RSS sources configured"]

        timeout = httpx.Timeout(self.settings.request_timeout_seconds)
        semaphore = asyncio.Semaphore(max(1, self.settings.rss_http_concurrency))

        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            async def worker(source: RSSSource) -> tuple[list[DiscoveredItem], str | None]:
                try:
                    async with semaphore:
                        items = await self._fetch_source(client, source)
                    logger.info("Fetched %s items from %s", len(items), source.name)
                    return items, None
                except Exception as exc:
                    error = f"RSS fetch failed ({source.name}): {exc}"
                    logger.warning(error)
                    return [], error

            results = await asyncio.gather(*(worker(source) for source in sources))

        collected: list[DiscoveredItem] = []
        errors: list[str] = []
        for items, maybe_error in results:
            collected.extend(items)
            if maybe_error:
                errors.append(maybe_error)

        deduped = dedupe_items(collected)
        daily = filter_items_published_today(deduped)
        return daily, errors

    async def _fetch_source(self, client: httpx.AsyncClient, source: RSSSource) -> list[DiscoveredItem]:
        headers = {"User-Agent": "AILinkedImpostingAgent/0.1"}
        response = await client.get(source.rss, headers=headers)
        response.raise_for_status()

        parsed = feedparser.parse(response.text)
        entries = parsed.entries[: self.settings.max_feed_items_per_source]

        items: list[DiscoveredItem] = []
        for raw_entry in entries:
            entry = dict(raw_entry)
            raw_url = str(entry.get("link") or "").strip()
            title = str(entry.get("title") or "").strip()
            if not raw_url or not title:
                continue

            normalized_url = normalize_url(raw_url)
            published_at = _parse_entry_datetime(entry)
            snippet = str(entry.get("summary") or entry.get("description") or "").strip() or None
            item_id = _build_item_id(source.name, normalized_url, title)

            items.append(
                DiscoveredItem(
                    id=item_id,
                    query="rss_seed",
                    title=title,
                    url=normalized_url,
                    domain=domain_from_url(normalized_url),
                    published_at=published_at,
                    snippet=snippet,
                    raw_content=snippet,
                )
            )

        return items

    def _mock_seed_items(self, limit: int) -> list[DiscoveredItem]:
        now = datetime.now(timezone.utc)
        templates = [
            (
                "Open-source agent framework introduces A2A protocol for enterprise workflow orchestration",
                "https://techcrunch.com/a2a-framework",
                "The release details protocol-level changes, context transfer semantics, and deployment tradeoffs.",
            ),
            (
                "Research team benchmarks hybrid retrieval + memory routing for long-horizon agents",
                "https://www.technologyreview.com/hybrid-retrieval-memory",
                "The paper compares latency, cost, and failure modes across routing strategies.",
            ),
            (
                "Model serving stack adds dynamic context window management for multi-step reasoning",
                "https://openai.com/news/context-window-mgmt",
                "Engineers describe token budgeting, cache invalidation, and eval outcomes.",
            ),
            (
                "Enterprise tooling update improves agent observability with trace-level replay",
                "https://deepmind.google/discover/blog/agent-observability/",
                "The tooling shows step replay, guardrail events, and regression triage workflow.",
            ),
            (
                "New eval harness reports reliability gaps in autonomous task decomposition",
                "https://huggingface.co/blog/eval-harness",
                "Coverage includes benchmark design, scoring methodology, and practical mitigations.",
            ),
        ]

        items: list[DiscoveredItem] = []
        for idx in range(limit):
            title, url, snippet = templates[idx % len(templates)]
            item_id = _build_item_id("Mock Source", f"{url}?v={idx}", f"{title} {idx}")
            items.append(
                DiscoveredItem(
                    id=item_id,
                    query="rss_seed_mock",
                    title=f"{title} ({idx + 1})",
                    url=normalize_url(f"{url}?id={idx}"),
                    domain=domain_from_url(url),
                    published_at=now,
                    snippet=snippet,
                    raw_content=snippet,
                )
            )
        return items


def filter_items_published_today(
    items: list[DiscoveredItem],
    now: datetime | None = None,
) -> list[DiscoveredItem]:
    reference_now = now if now is not None else datetime.now().astimezone()
    if reference_now.tzinfo is None:
        reference_now = reference_now.replace(tzinfo=timezone.utc)

    local_tz = reference_now.tzinfo
    today = reference_now.date()

    filtered: list[DiscoveredItem] = []
    for item in items:
        published_at = item.published_at
        if published_at is None:
            continue
        if published_at.tzinfo is None:
            published_at = published_at.replace(tzinfo=timezone.utc)
        if published_at.astimezone(local_tz).date() == today:
            filtered.append(item)

    return filtered


def _build_item_id(source_name: str, url: str, title: str) -> str:
    payload = f"{source_name}|{url}|{title.lower().strip()}".encode("utf-8", errors="ignore")
    return hashlib.sha256(payload).hexdigest()[:24]


def _parse_entry_datetime(entry: dict[str, Any]) -> datetime | None:
    parsed_struct = entry.get("published_parsed") or entry.get("updated_parsed")
    if parsed_struct is not None:
        try:
            return datetime(
                parsed_struct.tm_year,
                parsed_struct.tm_mon,
                parsed_struct.tm_mday,
                parsed_struct.tm_hour,
                parsed_struct.tm_min,
                parsed_struct.tm_sec,
                tzinfo=timezone.utc,
            )
        except Exception:
            pass

    date_text = entry.get("published") or entry.get("updated")
    if not date_text:
        return None

    try:
        parsed = parsedate_to_datetime(str(date_text))
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed
    except Exception:
        return None
