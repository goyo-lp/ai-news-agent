from __future__ import annotations

import httpx
import pytest

from app.config import Settings
from app.orchestrator.services.ranking import DiscoveredItem
from app.orchestrator.services import tavily_client as svc_mod
from app.orchestrator.services.tavily_client import TavilyClient, TavilySearchError, _parse_datetime


def _settings(api_key: str | None = None) -> Settings:
    return Settings(
        _env_file=None,
        tavily_api_key=api_key,
        tavily_base_url="https://api.tavily.example",
        request_timeout_seconds=10,
    )


def _client_with(handler) -> httpx.AsyncClient:
    """A real httpx.AsyncClient with a MockTransport — search_news takes an
    injected client argument, so we just pass this in rather than patching
    global httpx construction."""
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _patch_search_many_httpx(monkeypatch: pytest.MonkeyPatch, handler) -> None:
    """search_many constructs its own httpx.AsyncClient internally, so the test
    has to patch the AsyncClient symbol on the tavily_client module."""
    real = httpx.AsyncClient

    def _factory(*args: object, **kwargs: object) -> httpx.AsyncClient:
        kwargs["transport"] = httpx.MockTransport(handler)
        return real(**kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(svc_mod.httpx, "AsyncClient", _factory)


def test_empty_settings_yields_tavily_defaults() -> None:
    s = Settings(_env_file=None)
    assert s.tavily_api_key is None
    assert s.tavily_topic == "news"


async def test_search_news_dry_run_returns_mock_results_without_network() -> None:
    """No key, dry_run=True -> deterministic mock fixtures; zero network IO."""
    settings = _settings(api_key=None)
    client = TavilyClient(settings)

    async with httpx.AsyncClient() as http:
        items = await client.search_news(http, "test query", hours_back=24, max_results=2, dry_run=True)

    assert len(items) == 2
    assert all(isinstance(i, DiscoveredItem) for i in items)


async def test_search_news_dry_run_is_deterministic_across_calls() -> None:
    settings = _settings(api_key=None)
    client = TavilyClient(settings)
    async with httpx.AsyncClient() as http:
        first = await client.search_news(http, "q", hours_back=24, max_results=3, dry_run=True)
        second = await client.search_news(http, "q", hours_back=24, max_results=3, dry_run=True)
    assert [i.id for i in first] == [i.id for i in second]


async def test_search_news_without_key_raises_in_non_dry_run() -> None:
    settings = _settings(api_key=None)
    client = TavilyClient(settings)
    async with httpx.AsyncClient() as http:
        with pytest.raises(TavilySearchError):
            await client.search_news(http, "q", hours_back=24, max_results=2, dry_run=False)


async def test_search_news_parses_real_response_to_discovered_items() -> None:
    response = {
        "results": [
            {
                "title": "OpenAI launches new model",
                "url": "https://openai.com/news/x",
                "content": "Snippet text.",
                "raw_content": "Longer body text.",
                "published_date": "2025-01-15T12:30:45Z",
            },
            {"title": "", "url": "https://skip.example/empty-title"},
            "not-a-dict",
        ]
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=response)

    settings = _settings(api_key="sk-test")
    client = TavilyClient(settings)

    async with _client_with(handler) as http:
        items = await client.search_news(http, "q", hours_back=24, max_results=2, dry_run=False)

    assert len(items) == 1
    item = items[0]
    assert item.title == "OpenAI launches new model"
    assert item.url == "https://openai.com/news/x"
    assert item.domain == "openai.com"
    assert item.snippet == "Snippet text."
    assert item.raw_content == "Longer body text."
    assert item.published_at is not None
    assert item.published_at.year == 2025


async def test_search_news_raises_on_http_status_failure() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "boom"})

    settings = _settings(api_key="sk-test")
    client = TavilyClient(settings)

    with pytest.raises(httpx.HTTPStatusError):
        async with _client_with(handler) as http:
            await client.search_news(http, "q", hours_back=24, max_results=2, dry_run=False)


# Note: article-text extraction is no longer a Tavily surface — it moved to the
# keyless local app.orchestrator.services.web_extract (tested in
# tests/test_web_extract.py).


# --- _parse_datetime --------------------------------------------------------


def test_parse_datetime_iso_with_z() -> None:
    parsed = _parse_datetime("2025-01-15T12:30:45Z")
    assert parsed is not None
    assert parsed.year == 2025
    assert parsed.tzinfo is not None


def test_parse_datetime_naive_iso_pinned_to_utc() -> None:
    parsed = _parse_datetime("2025-01-15T12:30:45")
    assert parsed is not None
    assert parsed.tzinfo is not None


def test_parse_datetime_non_iso_returns_none() -> None:
    assert _parse_datetime("Jan 15 2025") is None
    assert _parse_datetime(None) is None
    assert _parse_datetime("") is None


# --- search_many batch ------------------------------------------------------


async def test_search_many_collects_per_query_results_and_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """search_many runs N queries concurrently and aggregates (items, errors);
    a per-query failure degrades to [] + error rather than aborting the batch."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "boom"})

    _patch_search_many_httpx(monkeypatch, handler)
    settings = _settings(api_key="sk-test")
    client = TavilyClient(settings)

    items, errors = await client.search_many(
        queries=["q1", "q2"],
        hours_back=24,
        max_results_per_query=3,
        dry_run=False,
    )
    assert items == []
    assert len(errors) == 2
    assert all("q1" in e or "q2" in e for e in errors)