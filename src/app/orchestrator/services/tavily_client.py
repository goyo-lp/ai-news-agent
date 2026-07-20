"""Tavily client ported from the reference LinkedIn agent
(``reference/linkedin-agent/src/app/services/tavily_client.py``).

Tavily has two surfaces the orchestrator's research subagent uses (P4.1):
  * ``search_news`` — news-mode search returning ``DiscoveredItem`` results.
  * ``extract_contents`` — extract endpoint: ``{url: text}`` for given URLs.

Intentional divergences from the reference (both documented here so the port's
gaps don't drift silently):
  1. ``api_usage_tracker.record_tavily_search`` / ``record_tavily_extract`` are
     NOT ported here: usage/cost tracking lands in Phase 7 (PR P7.2). A
     ``# TODO(P7.2)`` marks each seam so the integrator knows exactly where to
     wire the counters.
  2. Date parsing uses stdlib ``datetime.fromisoformat`` only — the reference
     pulled in ``python-dateutil`` for tolerant parsing. Tavily publishes
     ISO-ish datetime strings; non-ISO fallbacks (legacy RSS date formats) drop
     to ``None`` rather than dragging in a new runtime dependency for a per-URL
     cosmetic field that downstream consumers already handle as ``None``.
  3. Reuses the orchestrator-internal ``DiscoveredItem`` from
     :mod:`app.orchestrator.services.ranking` (already defined there for the
     technical_rank path) and the existing ``domain_from_url`` helper from the
     same module. The reference defined its own copies in ``services.url_utils``
     — porting those would produce two source-of-truth for the same helper in
     one repo. ``normalize_url`` comes from the News Agent's own
     ``app.services.rss_client`` (single source of truth for URL
     canonicalization across the host).
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
from app.services.rss_client import normalize_url

logger = logging.getLogger(__name__)


class TavilySearchError(Exception):
    """Raised when the Tavily search call fails in non-dry-run mode."""


class TavilyExtractError(Exception):
    """Raised when the Tavily extract endpoint call fails (HTTP error /
    transport error / JSON decode error) in non-dry-run mode. Distinguished
    at the tool layer from `success_count=0` (which means the endpoint answered
    with no extractable content for any URL) so a research subagent can tell
    'endpoint down' from 'endpoint answered, nothing usable'."""


class TavilyClient:
    """Wraps the Tavily ``/search`` and ``/extract`` HTTP endpoints. Construct
    with ``Settings``; the client picks up the env-driven ``tavily_*`` knobs at
    call time rather than construction so an operator setting the key after
    process start is picked up."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def search_many(
        self,
        queries: list[str],
        hours_back: int,
        max_results_per_query: int,
        dry_run: bool,
    ) -> tuple[list[DiscoveredItem], list[str]]:
        """Run several queries concurrently (bounded by ``tavily_http_concurrency``).
        Returns ``(all_items, per_query_errors)`` — same shape as the reference;
        a per-query failure degrades to an empty result for that query rather
        than aborting the batch."""
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
                    error = f"Tavily search failed for {query!r}: {exc}"
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
            raise TavilySearchError("Missing TAVILY_API_KEY")

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

        response = await client.post(
            f"{self.settings.tavily_base_url}/search", json=payload
        )
        response.raise_for_status()
        data = response.json()

        results = data.get("results") or []
        # TODO(P7.2): wire api_usage_tracker.record_tavily_search here once
        # Phase 7 introduces usage tracking.
        items: list[DiscoveredItem] = []
        if isinstance(results, list):
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
        """Extract per-URL cleaned text via the Tavily extract endpoint. De-dups
        URLs by normalized form first. Returns ``{normalized_url: text}``.

        Empty dict (no exception) on: empty input (short-circuit, no network)
        and missing TAVILY_API_KEY in non-dry-run (no-op rather than raised —
        the research subagent can keep working with other evidence). Tavily dry
        run returns mock content per URL. Endpoint failure raises
        :class:`TavilyExtractError` so the tool can surface it as `status=
        "error"`, distinct from 'endpoint answered but no content'."""
        unique_urls: list[str] = []
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
            # TODO(P7.2): wire api_usage_tracker.record_tavily_extract here.
            response = await client.post(
                f"{self.settings.tavily_base_url}/extract", json=payload
            )
            response.raise_for_status()
            data = response.json()
        except Exception as exc:
            # Re-raise so the tool layer can distinguish 'endpoint down' from
            # 'endpoint answered, no extractable content'. The reference
            # returned {} silently — that was fine for an internal call graph,
            # but a tool-fed subagent needs the distinction (verifier retry path
            # checks the summary's `status` field).
            logger.warning("Tavily extract failed: %s", exc)
            raise TavilyExtractError(str(exc)) from exc

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
                    item.get("raw_content") or item.get("content") or item.get("text") or ""
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
        """Deterministic dry-run fixtures so a subagent exercising the tool
        produces stable artifacts (deterministic-looking ids from the
        sha256(query|url) key, fixed published_at=now in the run). The three
        templates mirror the reference's so behavior parity is checkable."""
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
    """Parse a Tavily-supplied datetime-ish string to a tz-aware datetime using
    stdlib only. Returns None for anything ``datetime.fromisoformat`` can't
    handle — downstream consumers treat None as 'unknown recency' rather than
    failing, so a non-ISO date costs one weaker recency score, not a crash.

    Naive datetimes are pinned to UTC (Tavily results generally carry tz, but
    pinning is the safe default for the odd legacy format that parses naive)."""
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


__all__ = ["TavilyClient", "TavilySearchError", "TavilyExtractError"]