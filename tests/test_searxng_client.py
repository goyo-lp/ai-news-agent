from __future__ import annotations

import httpx
import pytest

from app.config import Settings
from app.orchestrator.services import searxng_client as svc_mod
from app.orchestrator.services.ranking import DiscoveredItem
from app.orchestrator.services.searxng_client import (
    SearxngClient,
    SearxngSearchError,
    _parse_datetime,
    _time_range_for,
)


def _settings(base_url: str = "") -> Settings:
    return Settings(_env_file=None, searxng_base_url=base_url, request_timeout_seconds=10)


def _client_with(handler) -> httpx.AsyncClient:
    """A real httpx.AsyncClient with a MockTransport — search_news takes an
    injected client argument, so we just pass this in rather than patching
    global httpx construction."""
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _patch_search_many_httpx(monkeypatch: pytest.MonkeyPatch, handler) -> None:
    """search_many constructs its own httpx.AsyncClient internally, so the test
    has to patch the AsyncClient symbol on the searxng_client module."""
    real = httpx.AsyncClient

    def _factory(*args: object, **kwargs: object) -> httpx.AsyncClient:
        kwargs["transport"] = httpx.MockTransport(handler)
        return real(**kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(svc_mod.httpx, "AsyncClient", _factory)


def test_empty_settings_yields_searxng_defaults() -> None:
    s = Settings(_env_file=None)
    assert s.searxng_base_url == ""
    assert s.searxng_categories == "news"
    assert s.searxng_language == "en"


def test_time_range_buckets() -> None:
    assert _time_range_for(6) == "day"
    assert _time_range_for(24) == "day"
    assert _time_range_for(48) == "week"
    assert _time_range_for(24 * 20) == "month"
    assert _time_range_for(24 * 400) == "year"


async def test_search_news_dry_run_returns_mock_results_without_network() -> None:
    """No base URL, dry_run=True -> deterministic mock fixtures; zero network."""
    client = SearxngClient(_settings())
    async with httpx.AsyncClient() as http:
        items = await client.search_news(http, "test query", hours_back=24, max_results=2, dry_run=True)

    assert len(items) == 2
    assert all(isinstance(i, DiscoveredItem) for i in items)


async def test_search_news_dry_run_is_deterministic_across_calls() -> None:
    client = SearxngClient(_settings())
    async with httpx.AsyncClient() as http:
        first = await client.search_news(http, "q", hours_back=24, max_results=3, dry_run=True)
        second = await client.search_news(http, "q", hours_back=24, max_results=3, dry_run=True)
    assert [i.id for i in first] == [i.id for i in second]


async def test_search_news_without_base_url_returns_mock_even_when_not_dry_run() -> None:
    """An unconfigured instance is not an error — it falls back to mock results,
    so the verifier (which derives dry_run from the model key, not the SearXNG
    URL) never silently loses corroboration when SearXNG isn't stood up yet."""
    client = SearxngClient(_settings(base_url=""))
    async with httpx.AsyncClient() as http:
        items = await client.search_news(http, "q", hours_back=24, max_results=2, dry_run=False)
    assert len(items) == 2  # mock fixtures, no network


async def test_search_news_parses_searxng_response_to_discovered_items() -> None:
    response = {
        "results": [
            {
                "title": "OpenAI launches new model",
                "url": "https://openai.com/news/x",
                "content": "Snippet text.",
                "publishedDate": "2025-01-15T12:30:45Z",
                "engine": "google news",
            },
            {"title": "", "url": "https://skip.example/empty-title"},
            "not-a-dict",
        ]
    }

    captured: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["format"] = request.url.params.get("format", "")
        captured["time_range"] = request.url.params.get("time_range", "")
        return httpx.Response(200, json=response)

    client = SearxngClient(_settings(base_url="https://searxng.example"))
    async with _client_with(handler) as http:
        items = await client.search_news(http, "q", hours_back=24, max_results=2, dry_run=False)

    # Hit the SearXNG JSON search endpoint with the day bucket.
    assert captured["path"] == "/search"
    assert captured["format"] == "json"
    assert captured["time_range"] == "day"

    assert len(items) == 1  # empty-title + non-dict dropped
    item = items[0]
    assert item.title == "OpenAI launches new model"
    assert item.url == "https://openai.com/news/x"
    assert item.domain == "openai.com"
    assert item.snippet == "Snippet text."
    # SearXNG returns a snippet, not full text -> raw_content is None.
    assert item.raw_content is None
    assert item.published_at is not None
    assert item.published_at.year == 2025


async def test_search_news_trailing_slash_base_url_is_normalized() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url).startswith("https://searxng.example/search")
        return httpx.Response(200, json={"results": []})

    client = SearxngClient(_settings(base_url="https://searxng.example/"))
    async with _client_with(handler) as http:
        await client.search_news(http, "q", hours_back=24, max_results=2, dry_run=False)


async def test_search_news_wraps_http_failure_in_searxng_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "boom"})

    client = SearxngClient(_settings(base_url="https://searxng.example"))
    with pytest.raises(SearxngSearchError):
        async with _client_with(handler) as http:
            await client.search_news(http, "q", hours_back=24, max_results=2, dry_run=False)


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
    client = SearxngClient(_settings(base_url="https://searxng.example"))

    items, errors = await client.search_many(
        queries=["q1", "q2"],
        hours_back=24,
        max_results_per_query=3,
        dry_run=False,
    )
    assert items == []
    assert len(errors) == 2
    assert all("q1" in e or "q2" in e for e in errors)
