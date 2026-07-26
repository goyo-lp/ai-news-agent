"""web_search / web_extract — the coordinator's research-evidence tools.

`web_search` wraps :class:`app.orchestrator.services.searxng_client.SearxngClient`
(self-hosted SearXNG — keyless, no billing). `web_extract` wraps the keyless
local :func:`app.orchestrator.services.web_extract.extract_url_texts`
(SSRF-guarded fetch + trafilatura). Neither depends on a paid third-party
service. Per Decision E, these stay *only* as per-topic research evidence (the
research subagent verifies a brief by running a focused query, then extracting
the URLs it finds); the LinkedIn agent's own discovery path is dropped.

Both tools follow the news / technical_rank / fetch_article pattern:
async-only ``StructuredTool``, factory with lazily-resolved settings
(tests inject directly), structured data persists to the orchestrator data
dir under ``web/``, the return value is a compressed JSON summary (guiding
principle #3) — never the result list or extracted text. `web_extract` does
real local extraction with no dry-run mock, matching fetch_article; `web_search`
returns deterministic mock results when ``SEARXNG_BASE_URL`` is unset so a
subagent can iterate with no instance running.
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
from app.orchestrator.budgets import SearchCircuit
from app.orchestrator.services.ranking import DiscoveredItem
from app.orchestrator.services.searxng_client import (
    SearxngClient,
    SearxngSearchError,
)
from app.orchestrator.services.web_extract import extract_url_texts
from app.services.rss_client import normalize_url

logger = logging.getLogger(__name__)

# All research-evidence artifacts live under a single provider-neutral `web/`
# dir: search results in web/search/, extracted text in web/extracted/.
_WEB_SUBDIR = "web"
_SEARCH_SUBDIR = "search"
_EXTRACT_SUBDIR = "extracted"

def _slug(value: str, length: int = 12) -> str:
    """Stable filesystem slug for a research artifact (query or batch key).
    The hash alone keeps the filename filesystem-safe even for hostile input
    (a query scraped with path separators or unicode); the prefix-length cap
    keeps directory listings readable."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:length]


# ---------------------------------------------------------------------------
# web_search
# ---------------------------------------------------------------------------


class WebSearchArgs(BaseModel):
    """Tool input. `query` is required (no best-guess default); `hours_back`
    and `max_results` are optional, defaulting to 24h / 8 results."""

    query: str = Field(..., description="Natural-language research query.")
    hours_back: int | None = Field(
        default=None,
        description=(
            "How far back to search, in hours. Maps onto SearXNG's coarse "
            "time_range bucket (day/week/month/year). Omit to default to ~24h."
        ),
    )
    max_results: int | None = Field(
        default=None,
        description="Max results to keep for this query. Omit to use the client default (8)."
    )


async def _run_search(
    args: dict[str, Any], settings: Settings, circuit: SearchCircuit
) -> dict[str, Any]:
    query = str(args.get("query") or "").strip()
    if not query:
        return {"query": "", "status": "error", "reason": "query is required", "path": None}

    hours_back_raw = args.get("hours_back")
    hours_back = int(hours_back_raw) if hours_back_raw is not None else 24
    max_results_raw = args.get("max_results")
    max_results = int(max_results_raw) if max_results_raw is not None else 8
    # No configured SearXNG instance -> dry-run mock (mirrors the old "no key"
    # behavior), so the tool works before an instance is stood up.
    dry_run = not (settings.searxng_base_url or "").strip()

    # Circuit breaker: only meaningful against a real instance (the dry-run
    # mock always returns results, so it can never trip this).
    if not dry_run and circuit.is_open:
        return {
            "query": query,
            "result_count": 0,
            "status": "circuit_open",
            "reason": (
                f"SearXNG returned empty/failed results "
                f"{circuit.consecutive_empty} times in a row — the search "
                "backend is not producing evidence this run. STOP searching; "
                "corroborate via fetch_article on the primary_url and "
                "web_extract on the topic's supporting_urls instead."
            ),
            "path": None,
        }

    client = SearxngClient(settings)
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
            except SearxngSearchError as exc:
                logger.warning("web_search failed for %r: %s", query, exc)
                circuit.record(found_results=False)
                return {**empty_summary, "status": "error", "reason": str(exc)}
            except Exception as exc:
                logger.warning("web_search unexpected for %r: %s", query, exc)
                circuit.record(found_results=False)
                return {**empty_summary, "status": "error", "reason": f"{type(exc).__name__}: {exc}"}
    except Exception as exc:  # httpx.AsyncClient construction failure
        logger.warning("web_search http client failed: %s", exc)
        circuit.record(found_results=False)
        return {**empty_summary, "status": "error", "reason": f"{type(exc).__name__}: {exc}"}

    circuit.record(found_results=bool(items))

    path = write_search_to_state(items, query, settings.orchestrator_data_dir)
    result = {
        "query": query,
        "result_count": len(items),
        "status": "ok",
        "reason": None,
        "path": str(path),
    }
    if not items and not dry_run:
        # Tell the researcher how close the circuit is to opening so it can
        # budget its own pivot instead of discovering it the hard way.
        result["reason"] = (
            f"0 results ({circuit.consecutive_empty}/{circuit.threshold} before "
            "search circuit opens); prefer web_extract on supporting_urls"
        )
    return result


def write_search_to_state(items: list[DiscoveredItem], query: str, data_dir: str) -> Path:
    """Serialize a search's results to ``web/search/<query-slug>.json`` and
    return the written path. Creates the dir if missing. Pure (no network):
    testable with a tmp directory. Mirrors the news.py / technical_rank.py
    writers — the on-disk shape round-trips through ``DiscoveredItem``."""
    root = Path(data_dir) / _WEB_SUBDIR / _SEARCH_SUBDIR
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{_slug(query)}.json"
    path.write_text(
        json.dumps([i.model_dump(mode="json") for i in items], indent=2, default=str),
        encoding="utf-8",
    )
    logger.info("Wrote %d search results for %r to %s", len(items), query, path)
    return path


def build_web_search_tool(settings: Settings | None = None) -> StructuredTool:
    """Construct the web_search langchain tool (SearXNG-backed).

    The circuit breaker is created here and closed over, so its lifetime is the
    lifetime of this tool — one per run, no cross-run leakage, nothing to
    reset."""
    bound_settings = settings
    circuit = SearchCircuit(
        (bound_settings or get_settings()).searxng_empty_circuit_breaker
    )

    async def _async(query: str, hours_back: int | None = None, max_results: int | None = None) -> str:
        s = bound_settings or get_settings()
        result = await _run_search(
            {"query": query, "hours_back": hours_back, "max_results": max_results}, s, circuit
        )
        return json.dumps(result, default=str)

    return StructuredTool.from_function(
        func=None,
        coroutine=_async,
        name="web_search",
        description=(
            "Run a web news search for one query via the self-hosted SearXNG "
            "instance (keyless, no third-party service) and persist the "
            "normalized results to web/search/<query-slug>.json. Returns a JSON "
            "summary with {query, result_count, status, reason, path} — never "
            "the results themselves. Read them from `path` when `status == "
            "\"ok\"`. Returns a mocked result set when SEARXNG_BASE_URL is unset "
            "so a subagent can exercise this with no instance running. When "
            "the configured instance returns empty results repeatedly the tool "
            "returns status=\"circuit_open\": STOP searching and corroborate "
            "via fetch_article + web_extract on supporting_urls instead."
        ),
        args_schema=WebSearchArgs,
    )


# ---------------------------------------------------------------------------
# web_extract
# ---------------------------------------------------------------------------


class WebExtractArgs(BaseModel):
    """Tool input. `urls` is required and non-empty; an empty list short-circuits
    to an empty-result summary rather than fetching anything."""

    urls: list[str] = Field(
        ...,
        description="List of absolute http(s) URLs to extract cleaned article text for.",
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
    # share one definition of 'URL' (the deduped set). extract_url_texts also
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

    # Real local extraction (SSRF-guarded fetch + trafilatura); no dry-run mock,
    # matching the fetch_article tool. Per-URL failures are skipped inside the
    # service, so the normal path is always ok with success_count reflecting how
    # many URLs yielded text; only an unexpected service-level exception errors.
    try:
        extracted = await extract_url_texts(urls, settings)
    except Exception as exc:
        logger.warning("web_extract unexpected: %s", exc)
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
    """Serialize an extract batch to ``web/extracted/<batch-slug>.json`` as
    ``{urls: [...], results: {url: text}}`` and return the written path. The
    batch slug is keyed on the full input set so the same batch re-runs overwrite
    (idempotent); different batches land in different files because the slug is
    the sha256 of the joined URL list."""
    root = Path(data_dir) / _WEB_SUBDIR / _EXTRACT_SUBDIR
    root.mkdir(parents=True, exist_ok=True)
    slug = _slug("\n".join(urls))
    path = root / f"{slug}.json"
    payload = {"urls": urls, "results": extracted}
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    logger.info("Wrote web extract batch (%d/%d urls) to %s", len(extracted), len(urls), path)
    return path


def build_web_extract_tool(settings: Settings | None = None) -> StructuredTool:
    """Construct the web_extract langchain tool."""
    bound_settings = settings

    async def _async(urls: list[str]) -> str:
        s = bound_settings or get_settings()
        result = await _run_extract({"urls": urls}, s)
        return json.dumps(result, default=str)

    return StructuredTool.from_function(
        func=None,
        coroutine=_async,
        name="web_extract",
        description=(
            "Extract cleaned per-URL article text locally (SSRF-guarded fetch + "
            "readability extraction — no API key, no third-party service) and "
            "persist the batch to web/extracted/<batch-slug>.json as "
            "{urls: [...], results: {url: text}}. Returns a JSON summary with "
            "{url_count, success_count, status, reason, path} — never the "
            "extracted text. Read it from `path` when `status == \"ok\"`. "
            "status is ok (success_count is how many URLs yielded text; may be 0 "
            "when none did) or error (unexpected failure — `reason` set, no "
            "artifact). A single unreachable/blocked URL is skipped, not an error."
        ),
        args_schema=WebExtractArgs,
    )


# Convenience singletons for `create_deep_agent(tools=[...])`.
web_search_tool = build_web_search_tool()
web_extract_tool = build_web_extract_tool()
