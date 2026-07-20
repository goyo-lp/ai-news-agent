"""tavily_search / tavily_extract — the coordinator's research-evidence tools.

Wraps :class:`app.orchestrator.services.tavily_client.TavilyClient` as two
langchain ``StructuredTool`` instances. Per Decision E, Tavily stays *only* as
per-topic research evidence (the research subagent verifies a brief by
running a focused query, optionally extracting the URLs it finds); the
LinkedIn agent's own Tavily/RSS discovery path is dropped.

Both tools follow the news / technical_rank / fetch_article pattern:
async-only ``StructuredTool``, factory with lazily-resolved settings
(tests inject directly), structured data persists to the orchestrator data
dir under ``tavily/``, the return value is a compressed JSON summary
(guiding principle #3) — never the result list or extracted text.

The mock/dry-run path in TavilyClient means a subagent exercising these
tools without ``TAVILY_API_KEY`` still gets stable training data — the
research subagent safety net (P4.1) can iterate without an API key during
dev.
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import httpx
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from app.config import Settings, get_settings
from app.orchestrator.services.ranking import DiscoveredItem
from app.orchestrator.services.tavily_client import (
    TavilyClient,
    TavilyExtractError,
    TavilySearchError,
)
from app.services.rss_client import normalize_url

logger = logging.getLogger(__name__)

_TAVILY_SUBDIR = "tavily"
_SEARCH_SUBDIR = "search"
_EXTRACT_SUBDIR = "extracted"


def _slug(value: str, length: int = 12) -> str:
    """Stable filesystem slug for a Tavily artifact (query or batch key).
    The hash alone keeps the filename filesystem-safe even for hostile input
    (a query scraped with path separators or unicode); the prefix-length cap
    keeps directory listings readable."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:length]


# ---------------------------------------------------------------------------
# tavily_search
# ---------------------------------------------------------------------------


class TavilySearchArgs(BaseModel):
    """Tool input. `query` is required (no best-guess default); `hours_back`
    and `max_results` are optional and clamp to sensible Tavily ranges inside
    the client, mirroring the reference defaults of 24h / 8 results."""

    query: str = Field(..., description="Natural-language research query.")
    hours_back: int | None = Field(
        default=None,
        description=(
            "How far back to search, in hours. Maps to Tavily's `days` "
            "param, clamped to [1, 7]. Omit to default to ~24h."
        ),
    )
    max_results: int | None = Field(
        default=None,
        description="Max Tavily results to return for this query. Omit to use the client default."
    )


async def _run_search(args: dict[str, Any], settings: Settings) -> dict[str, Any]:
    query = str(args.get("query") or "").strip()
    if not query:
        return {"query": "", "status": "error", "reason": "query is required", "path": None}

    hours_back_raw = args.get("hours_back")
    hours_back = int(hours_back_raw) if hours_back_raw is not None else 24
    max_results_raw = args.get("max_results")
    max_results = int(max_results_raw) if max_results_raw is not None else 8
    dry_run = not (settings.tavily_api_key or "").strip()

    client = TavilyClient(settings)
    timeout = httpx.Timeout(settings.request_timeout_seconds)
    empty_summary = {
        "query": query,
        "result_count": 0,
        "status": "ok",
        "reason": None,
        "path": None,
    }
    try:
        async with httpx.AsyncClient(timeout=timeout) as http:
            try:
                items = await client.search_news(
                    client=http,
                    query=query,
                    hours_back=hours_back,
                    max_results=max_results,
                    dry_run=dry_run,
                )
            except TavilySearchError as exc:
                logger.warning("tavily_search failed for %r: %s", query, exc)
                return {**empty_summary, "status": "error", "reason": str(exc)}
            except Exception as exc:
                logger.warning("tavily_search unexpected for %r: %s", query, exc)
                return {**empty_summary, "status": "error", "reason": f"{type(exc).__name__}: {exc}"}
    except Exception as exc:  # httpx.AsyncClient construction failure
        logger.warning("tavily_search http client failed: %s", exc)
        return {**empty_summary, "status": "error", "reason": f"{type(exc).__name__}: {exc}"}

    path = write_search_to_state(items, query, settings.orchestrator_data_dir)
    return {
        "query": query,
        "result_count": len(items),
        "status": "ok",
        "reason": None,
        "path": str(path),
    }


def write_search_to_state(items: list[DiscoveredItem], query: str, data_dir: str) -> Path:
    """Serialize a Tavily search's results to ``tavily/search/<query-slug>.json``
    and return the written path. Creates the dir if missing. Pure (no network):
    testable with a tmp directory. Mirrors the news.py / technical_rank.py
    writers — the on-disk shape round-trips through ``DiscoveredItem``."""
    root = Path(data_dir) / _TAVILY_SUBDIR / _SEARCH_SUBDIR
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{_slug(query)}.json"
    path.write_text(
        json.dumps([i.model_dump(mode="json") for i in items], indent=2, default=str),
        encoding="utf-8",
    )
    logger.info("Wrote %d Tavily search results for %r to %s", len(items), query, path)
    return path


def build_tavily_search_tool(settings: Settings | None = None) -> StructuredTool:
    """Construct the tavily_search langchain tool."""
    bound_settings = settings

    async def _async(query: str, hours_back: int | None = None, max_results: int | None = None) -> str:
        s = bound_settings or get_settings()
        result = await _run_search(
            {"query": query, "hours_back": hours_back, "max_results": max_results}, s
        )
        return json.dumps(result, default=str)

    return StructuredTool.from_function(
        func=None,
        coroutine=_async,
        name="tavily_search",
        description=(
            "Run a Tavily news search for one query and persist the normalized "
            "results to tavily/search/<query-slug>.json. Returns a JSON summary "
            "with {query, result_count, status, reason, path} — never the results "
            "themselves. Read them from `path` when `status == \"ok\"`. Falls "
            "back to a mocked result set when TAVILY_API_KEY is unset so a "
            "subagent can exercise this without a live key."
        ),
        args_schema=TavilySearchArgs,
    )


# ---------------------------------------------------------------------------
# tavily_extract
# ---------------------------------------------------------------------------


class TavilyExtractArgs(BaseModel):
    """Tool input. `urls` is required and non-empty; an empty list short-circuits
    to an empty-result summary rather than calling the extract endpoint."""

    urls: list[str] = Field(
        ...,
        description="List of absolute http(s) URLs to extract cleaned text for via Tavily's extract endpoint.",
    )


async def _run_extract(args: dict[str, Any], settings: Settings) -> dict[str, Any]:
    raw_urls = args.get("urls") or []
    if not isinstance(raw_urls, list) or not raw_urls:
        return {
            "url_count": 0,
            "success_count": 0,
            "status": "error",
            "reason": "urls is required and must be non-empty",
            "path": None,
        }

    # Normalize + dedupe at the tool layer so url_count and success_count
    # share one definition of 'URL' (the deduped set). extract_contents also
    # dedupes internally as defense-in-depth; the duplicate work is free.
    seen: set[str] = set()
    urls: list[str] = []
    for u in raw_urls:
        normalized = normalize_url(str(u or ""))
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        urls.append(normalized)
    if not urls:
        return {
            "url_count": 0,
            "success_count": 0,
            "status": "error",
            "reason": "no valid URLs after normalization",
            "path": None,
        }

    dry_run = not (settings.tavily_api_key or "").strip()
    client = TavilyClient(settings)
    timeout = httpx.Timeout(settings.request_timeout_seconds)

    try:
        async with httpx.AsyncClient(timeout=timeout) as http:
            extracted = await client.extract_contents(
                client=http, urls=urls, dry_run=dry_run
            )
    except TavilyExtractError as exc:
        # Endpoint failed outright (HTTP/transport/JSON) — distinct from
        # 'endpoint answered, no content' (success_count=0, status=ok below).
        logger.warning("tavily_extract endpoint failed: %s", exc)
        return {
            "url_count": len(urls),
            "success_count": 0,
            "status": "error",
            "reason": str(exc),
            "path": None,
        }
    except Exception as exc:
        logger.warning("tavily_extract unexpected: %s", exc)
        return {
            "url_count": len(urls),
            "success_count": 0,
            "status": "error",
            "reason": f"{type(exc).__name__}: {exc}",
            "path": None,
        }

    path = write_extract_to_state(extracted, urls, settings.orchestrator_data_dir)
    return {
        "url_count": len(urls),
        "success_count": len(extracted),
        "status": "ok",
        "reason": None,
        "path": str(path),
    }


def write_extract_to_state(extracted: dict[str, str], urls: list[str], data_dir: str) -> Path:
    """Serialize a Tavily extract batch to ``tavily/extracted/<batch-slug>.json``
    as ``{urls: [...], results: {url: text}}`` and return the written path.
    The batch slug is keyed on the full input set so the same batch re-runs
    overwrite (idempotent); different batches land in different files because
    the slug is the sha256 of the joined URL list."""
    root = Path(data_dir) / _TAVILY_SUBDIR / _EXTRACT_SUBDIR
    root.mkdir(parents=True, exist_ok=True)
    slug = _slug("\n".join(urls))
    path = root / f"{slug}.json"
    payload = {"urls": urls, "results": extracted}
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    logger.info("Wrote Tavily extract batch (%d/%d urls) to %s", len(extracted), len(urls), path)
    return path


def build_tavily_extract_tool(settings: Settings | None = None) -> StructuredTool:
    """Construct the tavily_extract langchain tool."""
    bound_settings = settings

    async def _async(urls: list[str]) -> str:
        s = bound_settings or get_settings()
        result = await _run_extract({"urls": urls}, s)
        return json.dumps(result, default=str)

    return StructuredTool.from_function(
        func=None,
        coroutine=_async,
        name="tavily_extract",
        description=(
            "Extract cleaned per-URL text via Tavily's extract endpoint and "
            "persist the batch to tavily/extracted/<batch-slug>.json as "
            "{urls: [...], results: {url: text}}. Returns a JSON summary with "
            "{url_count, success_count, status, reason, path} — never the "
            "extracted text. Read it from `path` when `status == \"ok\"`. "
            "status values: ok (endpoint answered; success_count may be 0 when "
            "no URL had extractable content), error (endpoint down / "
            "HTTP failure / unexpected exception — `reason` is set, no artifact). "
            "Dry-run (no TAVILY_API_KEY) returns mock text per URL."
        ),
        args_schema=TavilyExtractArgs,
    )


# Convenience singletons for `create_deep_agent(tools=[...])`.
tavily_search_tool = build_tavily_search_tool()
tavily_extract_tool = build_tavily_extract_tool()