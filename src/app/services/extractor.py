from __future__ import annotations

import asyncio
import logging
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup

from app.config import Settings
from app.schemas.article import Article, FetchRules
from app.services.http_utils import (
    ResponseTooLargeError,
    get_capped,
    reject_private_network_requests,
)
from app.services.rss_client import normalize_url

logger = logging.getLogger(__name__)

_MAX_DOWNLOAD_BYTES = 2_000_000


def extract_open_graph_fields(html: str) -> tuple[str | None, str | None, str | None]:
    """Extract og:*/twitter:*/description meta tags in a single DOM pass.

    Builds property/name lookup tables once instead of calling soup.find() per
    candidate key (each of which walks the tree from scratch); property matches
    still take precedence over name matches for the same key, matching the prior
    per-key search order.
    """
    soup = BeautifulSoup(html, "lxml")

    # Record the first tag's content per key (even if empty) — mirrors the old
    # find(property=key) or find(name=key) short-circuit: a property tag's mere
    # presence for a key blocks falling back to a name tag for that same key,
    # even if the property tag's content turns out to be empty.
    by_property: dict[str, str] = {}
    by_name: dict[str, str] = {}
    for tag in soup.find_all("meta"):
        content = str(tag.get("content") or "").strip()
        property_key = tag.get("property")
        if property_key and property_key not in by_property:
            by_property[str(property_key)] = content
        name_key = tag.get("name")
        if name_key and name_key not in by_name:
            by_name[str(name_key)] = content

    def first_present(*keys: str) -> str | None:
        for key in keys:
            if key in by_property:
                value = by_property[key]
            elif key in by_name:
                value = by_name[key]
            else:
                continue
            if value:
                return value
        return None

    og_title = first_present("og:title", "twitter:title")
    og_description = first_present("og:description", "description", "twitter:description")
    og_image = first_present("og:image", "twitter:image", "twitter:image:src")
    return og_title, og_description, og_image


def is_domain_blocked(url: str, blocked_domains: list[str]) -> bool:
    """Match a host against an operator-configured blocklist (exact or subdomain)."""
    host = (urlparse(url).hostname or "").lower()
    for blocked in blocked_domains:
        target = blocked.lower().strip()
        if host == target or host.endswith(f".{target}"):
            return True
    return False


class OpenGraphExtractor:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def enrich_articles(
        self,
        articles: list[Article],
        source_rules: dict[str, FetchRules],
    ) -> tuple[list[Article], list[str]]:
        """Enrich all articles concurrently (bounded by http_concurrency); returns
        (enriched articles — always same length/order as input, one copy per
        article — and non-fatal per-article error descriptions)."""
        timeout = httpx.Timeout(self.settings.request_timeout_seconds)
        semaphore = asyncio.Semaphore(self.settings.http_concurrency)

        async with httpx.AsyncClient(
            timeout=timeout,
            follow_redirects=True,
            event_hooks={"request": [reject_private_network_requests]},
        ) as client:
            async def worker(article: Article) -> tuple[Article, str | None]:
                async with semaphore:
                    return await self.enrich_article(client, article, source_rules.get(article.source_name, FetchRules()))

            results = await asyncio.gather(*(worker(article) for article in articles))

        enriched: list[Article] = []
        errors: list[str] = []
        for article, maybe_error in results:
            enriched.append(article)
            if maybe_error:
                errors.append(maybe_error)

        return enriched, errors

    async def enrich_article(
        self,
        client: httpx.AsyncClient,
        article: Article,
        rules: FetchRules,
    ) -> tuple[Article, str | None]:
        enriched = article.model_copy(deep=True)
        enriched.url = normalize_url(enriched.url)
        error: str | None = None

        if not is_domain_blocked(enriched.url, rules.blocked_domains):
            error = await self._fetch_open_graph(client, enriched, rules)

        if not enriched.image_url and rules.image_fallback_rss_enclosure and enriched.rss_image_url:
            enriched.image_url = enriched.rss_image_url

        return enriched, error

    async def _fetch_open_graph(
        self,
        client: httpx.AsyncClient,
        enriched: Article,
        rules: FetchRules,
    ) -> str | None:
        """Fetch the article page and copy Open Graph fields onto it, in place."""
        headers: dict[str, str] = {}
        if rules.requires_user_agent:
            headers["User-Agent"] = self.settings.user_agent

        try:
            response = await get_capped(client, enriched.url, headers, _MAX_DOWNLOAD_BYTES)
        except ResponseTooLargeError:
            logger.warning(
                "Enrichment skipped (%s): response too large (>%s bytes)",
                enriched.source_name,
                _MAX_DOWNLOAD_BYTES,
            )
            return f"Enrichment skipped ({enriched.source_name}): response too large"
        except Exception as exc:
            error = f"Enrichment failed ({enriched.source_name}): {exc}"
            logger.warning(error)
            return error

        enriched.url = normalize_url(str(response.url))

        content_type = response.headers.get("content-type", "")
        if "text/html" in content_type:
            og_title, og_description, og_image = extract_open_graph_fields(response.text)
            if og_title:
                enriched.og_title = og_title
            if og_description:
                enriched.og_description = og_description
            if og_image:
                enriched.image_url = og_image

        return None
