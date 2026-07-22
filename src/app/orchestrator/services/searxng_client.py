"""SearXNG search client — the keyless, self-hosted replacement for Tavily
search.

SearXNG is a self-hosted metasearch engine (run it via the bundled
``docker-compose.searxng.yml``); it aggregates results from public engines and
exposes a JSON API at ``GET {base}/search?format=json``. No API key, no billing
relationship — the operator points ``SEARXNG_BASE_URL`` at their instance.

Surface used by the orchestrator's research subagent (P4.1) and the brief
verifier (P2.4): ``search_news`` — a news search returning ``DiscoveredItem``
results. Article-text extraction is not here — it's the keyless local
:mod:`app.orchestrator.services.web_extract` (SSRF-guarded fetch + trafilatura).

Dry-run: when ``SEARXNG_BASE_URL`` is unset the client returns deterministic
mock results (mirroring the old "no key -> mock" behavior), so a subagent can
iterate with no service running. A configured base URL that fails at request
time surfaces as :class:`SearxngSearchError`.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
from datetime import datetime, timezone
from typing import Any

import httpx

from app.config import Settings
from app.orchestrator.services.ranking import DiscoveredItem, domain_from_url
from app.orchestrator.usage import record_searxng_search
from app.services.rss_client import normalize_url

logger = logging.getLogger(__name__)


class SearxngSearchError(Exception):
    """Raised when a configured SearXNG search request fails (HTTP error,
    transport error, JSON decode error). An unconfigured base URL is NOT an
    error — it falls back to dry-run mock results."""


def _time_range_for(hours_back: int) -> str:
    """Map an hours-back window onto SearXNG's coarse ``time_range`` buckets
    (day/week/month/year). SearXNG has no finer granularity, so a same-day
    verification window becomes ``day``."""
    if hours_back <= 24:
        return "day"
    if hours_back <= 24 * 7:
        return "week"
    if hours_back <= 24 * 31:
        return "month"
    return "year"


class SearxngClient:
    """Wraps the SearXNG ``/search`` JSON endpoint. Construct with ``Settings``;
    reads the ``searxng_*`` knobs at call time so an operator starting the
    instance after process start is picked up."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def search_many(
        self,
        queries: list[str],
        hours_back: int,
        max_results_per_query: int,
        dry_run: bool,
    ) -> tuple[list[DiscoveredItem], list[str]]:
        """Run several queries concurrently (bounded by
        ``searxng_http_concurrency``). Returns ``(all_items, per_query_errors)``;
        a per-query failure degrades to an empty result for that query rather
        than aborting the batch."""
        semaphore = asyncio.Semaphore(max(1, self.settings.searxng_http_concurrency))
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
                    error = f"SearXNG search failed for {query!r}: {exc}"
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
        base = (self.settings.searxng_base_url or "").strip().rstrip("/")
        # No instance configured -> deterministic mock. The empty base URL is the
        # single "search unavailable" signal, so both callers (the web_search
        # tool and the verifier) agree without needing to derive dry-run from
        # unrelated knobs — a missing instance never silently drops evidence.
        if dry_run or not base:
            return self._mock_search(query, max_results=max_results)

        params = {
            "q": query,
            "format": "json",
            "categories": self.settings.searxng_categories,
            "language": self.settings.searxng_language,
            "time_range": _time_range_for(hours_back),
        }

        try:
            response = await client.get(f"{base}/search", params=params)
            response.raise_for_status()
            data = response.json()
        except Exception as exc:
            raise SearxngSearchError(f"SearXNG search request failed: {exc}") from exc

        record_searxng_search()  # observability only — self-hosted, no cost

        results = data.get("results") or []
        items: list[DiscoveredItem] = []
        if isinstance(results, list):
            for raw in results[: max(0, max_results)]:
                parsed = self._result_to_item(raw, query=query)
                if parsed is not None:
                    items.append(parsed)
        return items

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

        published_at = _parse_datetime(raw.get("publishedDate") or raw.get("published_date"))
        snippet = str(raw.get("content") or "").strip() or None

        return DiscoveredItem(
            id=item_id,
            query=query,
            title=title,
            url=normalized_url,
            domain=domain,
            published_at=published_at,
            snippet=snippet,
            # SearXNG returns a snippet, not full article text — extraction is a
            # separate step (web_extract). No raw_content here.
            raw_content=None,
        )

    def _mock_search(self, query: str, max_results: int) -> list[DiscoveredItem]:
        """Deterministic dry-run fixtures so a subagent exercising search without
        a running SearXNG instance produces stable artifacts."""
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
        for title, url, snippet in templates[: max(0, min(max_results, len(templates)))]:
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
    """Parse a SearXNG-supplied datetime string to a tz-aware datetime using
    stdlib only. Returns None for anything ``datetime.fromisoformat`` can't
    handle — downstream consumers treat None as 'unknown recency' rather than
    failing. Naive datetimes are pinned to UTC."""
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value
    try:
        text = str(value).strip()
        if not text:
            return None
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


__all__ = ["SearxngClient", "SearxngSearchError"]
