from __future__ import annotations

import asyncio
import hashlib
import logging
from datetime import datetime, timezone
from typing import Any, cast

import httpx
from dateutil import parser as date_parser  # type: ignore[import-untyped]

from app.config import Settings
from app.schemas import DiscoveredItem
from app.services.api_usage_tracker import record_tavily_extract, record_tavily_search
from app.services.url_utils import domain_from_url, normalize_url

logger = logging.getLogger(__name__)


class TavilyClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def search_many(
        self,
        queries: list[str],
        hours_back: int,
        max_results_per_query: int,
        dry_run: bool,
    ) -> tuple[list[DiscoveredItem], list[str]]:
        semaphore = asyncio.Semaphore(max(1, self.settings.tavily_http_concurrency))
        timeout = httpx.Timeout(self.settings.request_timeout_seconds)

        async with httpx.AsyncClient(timeout=timeout) as client:
            async def worker(query: str) -> tuple[list[DiscoveredItem], str | None]:
                try:
                    async with semaphore:
                        items = await self.search_news(
                            client=client,
                            query=query,
                            hours_back=hours_back,
                            max_results=max_results_per_query,
                            dry_run=dry_run,
                        )
                    return items, None
                except Exception as exc:
                    error = f"Tavily search failed for '{query}': {exc}"
                    logger.warning(error)
                    return [], error

            gathered = await asyncio.gather(*(worker(query) for query in queries))

        all_items: list[DiscoveredItem] = []
        errors: list[str] = []
        for items, maybe_error in gathered:
            all_items.extend(items)
            if maybe_error:
                errors.append(maybe_error)
        return all_items, errors

    async def search_news(
        self,
        client: httpx.AsyncClient,
        query: str,
        hours_back: int,
        max_results: int,
        dry_run: bool,
    ) -> list[DiscoveredItem]:
        if dry_run:
            return self._mock_search(query, max_results=max_results)

        if not (self.settings.tavily_api_key or "").strip():
            raise RuntimeError("Missing TAVILY_API_KEY")

        payload: dict[str, Any] = {
            "api_key": self.settings.tavily_api_key,
            "query": query,
            "topic": self.settings.tavily_topic,
            "search_depth": self.settings.tavily_search_depth,
            "max_results": max_results,
            "include_answer": False,
            "include_images": False,
            "include_raw_content": True,
            "time_range": self.settings.tavily_time_range,
            "days": max(1, min(7, int((hours_back + 23) / 24))),
        }

        response = await client.post(f"{self.settings.tavily_base_url}/search", json=payload)
        response.raise_for_status()
        data = response.json()

        results = data.get("results") or []
        record_tavily_search(results_count=len(results) if isinstance(results, list) else 0)
        items: list[DiscoveredItem] = []
        for raw in results:
            parsed = self._result_to_item(raw, query=query)
            if parsed is not None:
                items.append(parsed)

        return items

    async def extract_contents(
        self,
        client: httpx.AsyncClient,
        urls: list[str],
        dry_run: bool,
    ) -> dict[str, str]:
        unique_urls = []
        seen: set[str] = set()
        for url in urls:
            normalized = normalize_url(url)
            if normalized not in seen:
                seen.add(normalized)
                unique_urls.append(normalized)

        if not unique_urls:
            return {}

        if dry_run:
            return {url: "Dry-run mock content." for url in unique_urls}

        if not (self.settings.tavily_api_key or "").strip():
            return {}

        payload: dict[str, Any] = {
            "api_key": self.settings.tavily_api_key,
            "urls": unique_urls,
            "include_images": False,
        }

        try:
            record_tavily_extract(url_count=len(unique_urls))
            response = await client.post(f"{self.settings.tavily_base_url}/extract", json=payload)
            response.raise_for_status()
            data = response.json()
        except Exception as exc:
            logger.warning("Tavily extract failed: %s", exc)
            return {}

        extracted_payload = data.get("results") or data.get("data") or []
        extracted: dict[str, str] = {}
        if isinstance(extracted_payload, list):
            for item in extracted_payload:
                if not isinstance(item, dict):
                    continue
                url = normalize_url(str(item.get("url") or "").strip())
                if not url:
                    continue
                content = str(
                    item.get("raw_content")
                    or item.get("content")
                    or item.get("text")
                    or ""
                ).strip()
                if content:
                    extracted[url] = content
        return extracted

    def _result_to_item(self, raw: Any, query: str) -> DiscoveredItem | None:
        if not isinstance(raw, dict):
            return None

        title = str(raw.get("title") or "").strip()
        url = str(raw.get("url") or "").strip()
        if not title or not url:
            return None

        normalized_url = normalize_url(url)
        domain = domain_from_url(normalized_url)
        item_id = hashlib.sha256(f"{query}|{normalized_url}".encode("utf-8")).hexdigest()[:24]

        published_at = _parse_datetime(
            raw.get("published_date")
            or raw.get("published_at")
            or raw.get("date")
            or raw.get("published")
        )

        snippet = str(raw.get("content") or raw.get("snippet") or "").strip() or None
        raw_content = str(raw.get("raw_content") or "").strip() or None

        return DiscoveredItem(
            id=item_id,
            query=query,
            title=title,
            url=normalized_url,
            domain=domain,
            published_at=published_at,
            snippet=snippet,
            raw_content=raw_content,
        )

    def _mock_search(self, query: str, max_results: int) -> list[DiscoveredItem]:
        now = datetime.now(timezone.utc)
        templates = [
            (
                "OpenAI releases a new reasoning model with better tool use",
                "https://openai.com/news/new-reasoning-model",
                "The model improves planning, multi-step execution, and tool reliability.",
            ),
            (
                "Google DeepMind announces enterprise Gemini deployment updates",
                "https://deepmind.google/discover/blog/gemini-enterprise-updates/",
                "Deployment highlights include evaluation guardrails and lower latency serving.",
            ),
            (
                "NVIDIA unveils next-gen AI inference chips for production workloads",
                "https://blogs.nvidia.com/blog/ai-inference-chip-launch/",
                "The release targets higher throughput and better cost efficiency.",
            ),
        ]

        items: list[DiscoveredItem] = []
        for idx, (title, url, snippet) in enumerate(templates[:max_results]):
            item_id = hashlib.sha256(f"{query}|{url}".encode("utf-8")).hexdigest()[:24]
            items.append(
                DiscoveredItem(
                    id=item_id,
                    query=query,
                    title=title,
                    url=normalize_url(url),
                    domain=domain_from_url(url),
                    published_at=now,
                    snippet=snippet,
                    raw_content=f"{snippet} Query context: {query}",
                )
            )
        return items


def _parse_datetime(value: Any) -> datetime | None:
    if value is None:
        return None

    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value

    try:
        parsed = cast(datetime, date_parser.parse(str(value)))
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed
    except Exception:
        return None
