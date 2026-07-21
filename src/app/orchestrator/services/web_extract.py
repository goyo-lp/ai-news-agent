"""Local article-text extraction — the keyless, no-service replacement for
Tavily's ``/extract`` endpoint.

Each URL is fetched through the News Agent's own SSRF-guarded, streaming,
size-capped HTTP path (``get_capped`` + ``reject_private_network_requests``)
and run through :mod:`trafilatura` for readability extraction. No API, no key,
no third-party dependency — the only cost is the network fetch we were already
going to pay.

Why SSRF matters here: the URLs come from research briefs / search results,
which can be attacker-influenced (a prompt-injected article could plant a
citation pointing at ``169.254.169.254`` or ``localhost``). So this reuses the
same guard ``fetch_article`` does, on every request including redirect hops.

Contract: returns ``{normalized_url: text}`` for the URLs that yielded content.
A per-URL failure (blocked host, oversize, transport error, no extractable
text) is skipped, not raised — extraction of one source should never sink the
batch. An empty input, or an all-failed batch, simply returns ``{}``.
"""
from __future__ import annotations

import asyncio
import logging

import httpx
import trafilatura

from app.config import Settings
from app.services.http_utils import get_capped, reject_private_network_requests
from app.services.rss_client import normalize_url

logger = logging.getLogger(__name__)

# Mirror fetch_article's budgets: same per-request download cap and the same
# persisted-text slice, so a URL extracted here and one fetched via
# fetch_article carry comparable payloads.
_MAX_DOWNLOAD_BYTES = 2_000_000
_MAX_TEXT_CHARS = 20_000


def _dedupe_normalized(urls: list[str]) -> list[str]:
    unique: list[str] = []
    seen: set[str] = set()
    for url in urls:
        normalized = normalize_url(str(url or ""))
        if normalized and normalized not in seen:
            seen.add(normalized)
            unique.append(normalized)
    return unique


async def _extract_one(client: httpx.AsyncClient, url: str, headers: dict[str, str]) -> str | None:
    """Fetch + extract one URL. Returns cleaned text (≤ _MAX_TEXT_CHARS) or
    None on any failure. Never raises — the caller skips a None."""
    try:
        response = await get_capped(client, url, headers, _MAX_DOWNLOAD_BYTES)
    except Exception as exc:
        # Blocked host / oversize / transport error — skip this source.
        logger.info("web_extract: fetch failed for %s: %s", url, exc)
        return None

    if "text/html" not in response.headers.get("content-type", ""):
        return None

    try:
        extracted = trafilatura.extract(response.text or "")
    except Exception as exc:
        logger.info("web_extract: trafilatura failed for %s: %s", url, exc)
        return None

    if not extracted:
        return None
    return " ".join(extracted.split())[:_MAX_TEXT_CHARS]


async def extract_url_texts(
    urls: list[str],
    settings: Settings,
    *,
    dry_run: bool = False,
) -> dict[str, str]:
    """Extract cleaned article text for each URL, locally. See module docstring
    for the contract. ``dry_run`` returns deterministic mock text without any
    network access (the verifier's dry-run path uses it); the extract *tool*
    does real extraction, matching the fetch_article precedent."""
    unique_urls = _dedupe_normalized(urls)
    if not unique_urls:
        return {}

    if dry_run:
        return {url: "Dry-run mock content." for url in unique_urls}

    headers = {"User-Agent": settings.user_agent}
    timeout = httpx.Timeout(settings.request_timeout_seconds)
    semaphore = asyncio.Semaphore(max(1, settings.http_concurrency))

    async with httpx.AsyncClient(
        timeout=timeout,
        follow_redirects=True,
        event_hooks={"request": [reject_private_network_requests]},
    ) as client:
        async def worker(url: str) -> tuple[str, str | None]:
            async with semaphore:
                return url, await _extract_one(client, url, headers)

        gathered = await asyncio.gather(*(worker(url) for url in unique_urls))

    return {url: text for url, text in gathered if text}


__all__ = ["extract_url_texts"]
