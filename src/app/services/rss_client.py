from __future__ import annotations

import asyncio
import hashlib
import logging
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import feedparser
import httpx
import yaml

from app.config import Settings
from app.schemas.article import Article, FetchRules, SourceConfig, SourcesFile

logger = logging.getLogger(__name__)

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
    cleaned_query = [(k, v) for k, v in parse_qsl(parsed.query, keep_blank_values=True) if k not in _TRACKING_PARAMS]
    normalized = parsed._replace(fragment="", query=urlencode(cleaned_query, doseq=True))
    return urlunparse(normalized)


def parse_entry_datetime(entry: dict[str, Any]) -> datetime | None:
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
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except Exception:
        return None


def extract_entry_image(entry: dict[str, Any]) -> str | None:
    media_content = entry.get("media_content") or []
    if isinstance(media_content, list):
        for item in media_content:
            if isinstance(item, dict) and item.get("url"):
                return str(item["url"])

    media_thumbnail = entry.get("media_thumbnail") or []
    if isinstance(media_thumbnail, list):
        for item in media_thumbnail:
            if isinstance(item, dict) and item.get("url"):
                return str(item["url"])

    links = entry.get("links") or []
    if isinstance(links, list):
        for link in links:
            if not isinstance(link, dict):
                continue
            link_type = str(link.get("type") or "")
            if link_type.startswith("image/") and link.get("href"):
                return str(link["href"])

    image = entry.get("image")
    if isinstance(image, dict) and image.get("href"):
        return str(image["href"])

    return None


def build_article_id(source_name: str, url: str, title: str) -> str:
    payload = f"{source_name}|{url}|{title.lower().strip()}".encode("utf-8", errors="ignore")
    return hashlib.sha256(payload).hexdigest()[:24]


def dedupe_articles(articles: list[Article]) -> list[Article]:
    deduped: dict[str, Article] = {}
    for article in articles:
        key = normalize_url(article.url)
        existing = deduped.get(key)
        if existing is None:
            deduped[key] = article
            continue

        existing_published = existing.published_at or datetime.min.replace(tzinfo=timezone.utc)
        incoming_published = article.published_at or datetime.min.replace(tzinfo=timezone.utc)
        if incoming_published > existing_published:
            article.duplicate_count = existing.duplicate_count + 1
            deduped[key] = article
        else:
            existing.duplicate_count += 1

    return list(deduped.values())


class RSSClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def load_sources(self) -> tuple[FetchRules, list[SourceConfig]]:
        with open(self.settings.sources_file, "r", encoding="utf-8") as source_file:
            data = yaml.safe_load(source_file) or {}
        parsed = SourcesFile.model_validate(data)
        return parsed.fetch_defaults, parsed.sources

    async def fetch_source(self, client: httpx.AsyncClient, source: SourceConfig) -> list[Article]:
        headers = {"User-Agent": self.settings.user_agent}
        response = await client.get(source.rss, headers=headers)
        response.raise_for_status()

        parsed = feedparser.parse(response.text)
        entries = parsed.entries[: self.settings.max_feed_items_per_source]

        articles: list[Article] = []
        for raw_entry in entries:
            entry = dict(raw_entry)
            raw_url = str(entry.get("link") or "").strip()
            if not raw_url:
                continue

            url = normalize_url(raw_url)
            title = str(entry.get("title") or "Untitled Article").strip()
            description = str(entry.get("summary") or entry.get("description") or "").strip() or None
            published_at = parse_entry_datetime(entry)
            rss_image_url = extract_entry_image(entry)

            article = Article(
                id=build_article_id(source.name, url, title),
                source_name=source.name,
                source_rss=source.rss,
                source_url=source.url,
                title=title,
                url=url,
                published_at=published_at,
                description=description,
                rss_image_url=rss_image_url,
            )
            articles.append(article)

        return articles

    async def fetch_all(self, sources: list[SourceConfig]) -> tuple[list[Article], list[str]]:
        timeout = httpx.Timeout(self.settings.request_timeout_seconds)
        semaphore = asyncio.Semaphore(self.settings.http_concurrency)

        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            async def worker(source: SourceConfig) -> tuple[list[Article], str | None]:
                try:
                    async with semaphore:
                        source_articles = await self.fetch_source(client, source)
                    logger.info("Fetched %s items from %s", len(source_articles), source.name)
                    return source_articles, None
                except Exception as exc:
                    error = f"Source fetch failed ({source.name}): {exc}"
                    logger.warning(error)
                    return [], error

            results = await asyncio.gather(*(worker(source) for source in sources))

        all_articles: list[Article] = []
        errors: list[str] = []
        for source_articles, maybe_error in results:
            all_articles.extend(source_articles)
            if maybe_error:
                errors.append(maybe_error)

        return all_articles, errors
