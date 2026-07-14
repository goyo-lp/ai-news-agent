from __future__ import annotations

import httpx

from app.config import Settings
from app.schemas.article import SourceConfig
from app.services.rss_client import RSSClient


RSS_XML = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0" xmlns:media="http://search.yahoo.com/mrss/">
  <channel>
    <title>Test Feed</title>
    <link>https://example.com</link>
    <description>Test feed</description>
    <item>
      <title>Story One</title>
      <link>https://example.com/one</link>
      <description>First story description.</description>
    </item>
    <item>
      <title>Story Two</title>
      <link></link>
      <description>Empty link item, should be skipped.</description>
    </item>
    <item>
      <title>Story Three</title>
      <link>https://example.com/three</link>
      <description>Has a media thumbnail.</description>
      <media:thumbnail url="https://cdn.example.com/three.png" />
    </item>
  </channel>
</rss>
"""


def _settings() -> Settings:
    return Settings(_env_file=None, max_feed_items_per_source=50, user_agent="test-ua")


def _ok_handler(request: httpx.Request) -> httpx.Response:
    return httpx.Response(
        200,
        text=RSS_XML,
        headers={"content-type": "application/rss+xml; charset=utf-8"},
    )


def _err_handler(request: httpx.Request) -> httpx.Response:
    return httpx.Response(500, text="internal server error")


async def test_fetch_source_parses_rss_items() -> None:
    settings = _settings()
    client = RSSClient(settings)
    source = SourceConfig(name="Test", url="https://example.com", rss="https://example.com/feed")

    async with httpx.AsyncClient(transport=httpx.MockTransport(_ok_handler), follow_redirects=True) as http:
        articles = await client.fetch_source(http, source)

    assert [article.title for article in articles] == ["Story One", "Story Three"]
    assert articles[0].url == "https://example.com/one"
    assert articles[1].rss_image_url == "https://cdn.example.com/three.png"


async def test_fetch_source_skips_empty_link_items() -> None:
    settings = _settings()
    client = RSSClient(settings)
    source = SourceConfig(name="Test", url="https://example.com", rss="https://example.com/feed")

    async with httpx.AsyncClient(transport=httpx.MockTransport(_ok_handler), follow_redirects=True) as http:
        articles = await client.fetch_source(http, source)

    assert all(article.url for article in articles)
    assert len(articles) == 2


async def test_fetch_all_reports_error_on_500(monkeypatch) -> None:
    settings = _settings()
    client = RSSClient(settings)
    sources = [SourceConfig(name="Test", url="https://example.com", rss="https://example.com/feed")]

    transport = httpx.MockTransport(_err_handler)
    real_async_client = httpx.AsyncClient

    def patched_async_client(*args: object, **kwargs: object) -> httpx.AsyncClient:
        kwargs.setdefault("transport", transport)
        return real_async_client(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(httpx, "AsyncClient", patched_async_client)

    articles, errors = await client.fetch_all(sources)
    assert articles == []
    assert len(errors) == 1
    assert "Test" in errors[0]
