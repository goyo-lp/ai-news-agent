from __future__ import annotations

import asyncio
import logging
from typing import Any
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup

from app.config import Settings
from app.schemas.article import Article, FetchRules
from app.services.rss_client import normalize_url

logger = logging.getLogger(__name__)


def extract_open_graph_fields(html: str) -> tuple[str | None, str | None, str | None]:
    soup = BeautifulSoup(html, "lxml")

    def meta_value(*keys: str) -> str | None:
        for key in keys:
            tag = soup.find("meta", attrs={"property": key}) or soup.find("meta", attrs={"name": key})
            if tag and tag.get("content"):
                return str(tag["content"]).strip()
        return None

    og_title = meta_value("og:title", "twitter:title")
    og_description = meta_value("og:description", "description", "twitter:description")
    og_image = meta_value("og:image", "twitter:image", "twitter:image:src")
    return og_title, og_description, og_image


def is_domain_blocked(url: str, blocked_domains: list[str]) -> bool:
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
        timeout = httpx.Timeout(self.settings.request_timeout_seconds)
        semaphore = asyncio.Semaphore(self.settings.http_concurrency)

        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            async def worker(article: Article) -> tuple[Article, str | None]:
                async with semaphore:
                    return await self._enrich_one(client, article, source_rules.get(article.source_name, FetchRules()))

            results = await asyncio.gather(*(worker(article) for article in articles))

        enriched: list[Article] = []
        errors: list[str] = []
        for article, maybe_error in results:
            enriched.append(article)
            if maybe_error:
                errors.append(maybe_error)

        return enriched, errors

    async def _enrich_one(
        self,
        client: httpx.AsyncClient,
        article: Article,
        rules: FetchRules,
    ) -> tuple[Article, str | None]:
        enriched = article.model_copy(deep=True)
        normalized_url = normalize_url(enriched.url)
        enriched.url = normalized_url

        if is_domain_blocked(normalized_url, rules.blocked_domains):
            if rules.image_fallback_rss_enclosure and enriched.rss_image_url:
                enriched.image_url = enriched.rss_image_url
            return enriched, None

        headers: dict[str, str] = {}
        if rules.requires_user_agent:
            headers["User-Agent"] = self.settings.user_agent

        try:
            response = await client.get(normalized_url, headers=headers)
            response.raise_for_status()
        except Exception as exc:
            if rules.image_fallback_rss_enclosure and enriched.rss_image_url:
                enriched.image_url = enriched.rss_image_url
            return enriched, f"Enrichment failed ({enriched.source_name}): {exc}"

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

        if not enriched.image_url and rules.image_fallback_rss_enclosure and enriched.rss_image_url:
            enriched.image_url = enriched.rss_image_url

        return enriched, None
