from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest

import ai_news_agent.ingestion as ingestion
from ai_news_agent.ingestion import (
    FeedContent,
    FeedFetchError,
    FeedParseError,
    SourceOutcome,
    canonicalize_url,
    ensure_public_http_url,
    entry_to_records,
    fetch_with_retries,
    ingest_one_source,
    ingest_sources,
    parse_feed,
    parse_feed_datetime,
    stable_article_id,
    strip_html,
)
from ai_news_agent.schemas import Article, Source
from ai_news_agent.sources import load_source_configs
from ai_news_agent.storage import SQLiteStore

FIXTURES = Path(__file__).parent / "fixtures" / "feeds"
CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "sources.yaml"
NOW = datetime(2026, 9, 8, 12, 0, tzinfo=UTC)


def read_fixture(name: str) -> bytes:
    return (FIXTURES / name).read_bytes()


def rss_config(source_id: str = "source-01"):
    configs = {config.id: config for config in load_source_configs(CONFIG_PATH)}
    return configs[source_id]


def make_store(tmp_path: Path) -> SQLiteStore:
    store = SQLiteStore(tmp_path / "ingest.db")
    store.migrate()
    return store


def fake_fetcher(body: bytes, **kwargs):
    def _fetch(
        url,
        *,
        timeout=15.0,
        max_bytes=15_000_000,
        max_retries=2,
        etag=None,
        last_modified=None,
        sleep=None,
    ):
        assert url.startswith("http")
        return FeedContent(url=url, status_code=200, body=body, **kwargs)

    return _fetch


# URL validation and normalization


def test_public_url_guard_rejects_private_and_credentials() -> None:
    with pytest.raises(ValueError, match="only http"):
        ensure_public_http_url("ftp://example.com/feed.xml")
    with pytest.raises(ValueError, match="credentials"):
        ensure_public_http_url("https://user:pass@example.com/feed")
    for bad in (
        "http://localhost/feed",
        "http://127.0.0.1/feed",
        "http://10.0.0.5/feed",
        "http://192.168.1.10/feed",
        "http://169.254.169.254/latest",
        "http://example.local/feed",
        "http://[::1]/feed",
    ):
        with pytest.raises(ValueError, match=r"private|non-routable"):
            ensure_public_http_url(bad)
    assert ensure_public_http_url("https://example.com/feed.xml")


def test_canonicalize_strips_trackers_and_fragments() -> None:
    canonical = canonicalize_url(
        "https://Example.COM/ai/a?utm_source=feed&fbclid=x&keep=1#frag",
        "https://example.com/",
    )

    assert canonical == "https://example.com/ai/a?keep=1"
    relative = canonicalize_url("/ai/b/", "https://example.com/feed.xml")
    assert relative == "https://example.com/ai/b"
    with pytest.raises(ValueError, match="only http"):
        canonicalize_url("file:///etc/passwd", "https://example.com/")


def test_stable_article_id_is_deterministic() -> None:
    first = stable_article_id("source-01", "https://example.com/A")
    second = stable_article_id("source-01", "https://example.com/a ")
    other = stable_article_id("source-02", "https://example.com/a")

    assert first == second
    assert first != other


# Date and feed parsing


def test_parse_feed_datetime_handles_formats() -> None:
    assert parse_feed_datetime("Sun, 07 Sep 2026 11:00:00 GMT") == datetime(
        2026, 9, 7, 11, 0, tzinfo=UTC
    )
    assert parse_feed_datetime("2026-09-07T10:00:00Z") == datetime(
        2026, 9, 7, 10, 0, tzinfo=UTC
    )
    assert parse_feed_datetime(None) is None
    assert parse_feed_datetime("   ") is None
    assert parse_feed_datetime("not-a-date") is None
    naive = parse_feed_datetime("2026-09-07T10:00:00")
    assert naive is not None and naive.tzinfo is not None


def test_parse_valid_rss_normalizes_entries() -> None:
    entries = parse_feed(read_fixture("valid_rss.xml"), "https://example.com/ai/")

    assert len(entries) == 2
    assert entries[0].url == "https://example.com/ai/model-a"
    assert entries[0].authors == ("Example Author",)
    assert entries[0].published_at is not None
    assert "open weights" in entries[0].excerpt
    assert entries[0].locator == "feed-entry:feed-entry-model-a"


def test_parse_valid_atom_strips_tracking_ids() -> None:
    entries = parse_feed(read_fixture("valid_atom.xml"), "https://example.com/ai/")

    assert len(entries) == 2
    assert entries[0].url == "https://example.com/ai/agent-toolkit"
    assert entries[0].authors == ("Toolkit Author",)
    assert entries[0].updated_at is not None


def test_parse_malformed_xml_raises_without_guessing() -> None:
    with pytest.raises(FeedParseError, match="malformed"):
        parse_feed(read_fixture("malformed.xml"), "https://example.com/")


def test_parse_unsupported_root_raises() -> None:
    with pytest.raises(FeedParseError, match="unsupported"):
        parse_feed(b"<html><body>not a feed</body></html>", "https://example.com/")


def test_parse_missing_dates_falls_back_and_quarantines() -> None:
    entries = parse_feed(read_fixture("missing_dates.xml"), "https://example.com/ai/")

    assert len(entries) == 1
    assert entries[0].url == "https://example.com/ai/undated"
    assert entries[0].published_at is None


def test_strip_html_bounds_length() -> None:
    assert strip_html("<p>Hello <b>world</b></p>") == "Hello world"
    assert len(strip_html("<p>" + ("x " * 5000) + "</p>", limit=100)) == 100


def test_entry_to_records_falls_back_to_first_seen() -> None:
    entries = parse_feed(read_fixture("missing_dates.xml"), "https://example.com/ai/")
    article, evidence = entry_to_records("source-01", entries[0], now=NOW)

    assert article.published_at == NOW
    assert article.first_seen_at == NOW
    assert article.raw_feed_ref == "feed-entry:feed-entry-undated"
    assert evidence.kind.value == "feed_entry"
    assert evidence.article_id == article.id


# Single-source ingestion


def test_ingest_success_persists_articles_and_evidence(tmp_path: Path) -> None:
    with make_store(tmp_path) as store:
        result = ingest_one_source(
            rss_config(),
            store,
            now=NOW,
            fetcher=fake_fetcher(read_fixture("valid_rss.xml")),
        )

        assert result.outcome == SourceOutcome.SUCCESS
        assert result.new_articles == 2
        assert store.get(Article, result.articles[0].id) is not None
        source = store.get(Source, "source-01")
        assert source is not None and source.last_success_at == NOW


def test_ingest_is_idempotent_across_identical_runs(tmp_path: Path) -> None:
    with make_store(tmp_path) as store:
        fetcher = fake_fetcher(read_fixture("valid_rss.xml"))
        first = ingest_one_source(rss_config(), store, now=NOW, fetcher=fetcher)
        second = ingest_one_source(rss_config(), store, now=NOW, fetcher=fetcher)

        assert first.outcome == SourceOutcome.SUCCESS
        assert second.outcome == SourceOutcome.UNCHANGED
        assert second.new_articles == 0
        count = store.connection.execute(
            "SELECT COUNT(*) FROM records WHERE record_type = 'article'"
        ).fetchone()[0]
        assert count == 2


def test_ingest_dedupes_repeated_entries_within_one_feed(tmp_path: Path) -> None:
    with make_store(tmp_path) as store:
        result = ingest_one_source(
            rss_config(),
            store,
            now=NOW,
            fetcher=fake_fetcher(read_fixture("repeated_entries.xml")),
        )

        assert result.outcome == SourceOutcome.SUCCESS
        assert result.new_articles == 1


def test_ingest_marks_304_not_modified_as_unchanged(tmp_path: Path) -> None:
    def _fetch(url, **kwargs):
        return FeedContent(url=url, status_code=304, not_modified=True)

    with make_store(tmp_path) as store:
        result = ingest_one_source(rss_config(), store, now=NOW, fetcher=_fetch)

        assert result.outcome == SourceOutcome.UNCHANGED


def test_ingest_marks_malformed_xml_as_failed(tmp_path: Path) -> None:
    with make_store(tmp_path) as store:
        result = ingest_one_source(
            rss_config(),
            store,
            now=NOW,
            fetcher=fake_fetcher(read_fixture("malformed.xml")),
        )

        assert result.outcome == SourceOutcome.FAILED
        assert "malformed" in (result.error or "")


def test_ingest_marks_timeouts_as_failed(tmp_path: Path) -> None:
    def _fetch(url, **kwargs):
        raise FeedFetchError("feed timeout after 15.0s: " + url)

    with make_store(tmp_path) as store:
        result = ingest_one_source(rss_config(), store, now=NOW, fetcher=_fetch)

        assert result.outcome == SourceOutcome.FAILED
        assert "timeout" in (result.error or "")


def test_ingest_marks_adapter_sources_unavailable(tmp_path: Path) -> None:
    configs = {config.id: config for config in load_source_configs(CONFIG_PATH)}
    with make_store(tmp_path) as store:
        result = ingest_one_source(configs["source-02"], store, now=NOW)

        assert result.outcome == SourceOutcome.UNAVAILABLE


def test_ingest_rejects_private_feed_url(tmp_path: Path) -> None:
    config = rss_config()
    payload = config.model_dump(mode="json")
    payload["route"]["url"] = "http://127.0.0.1/feed.xml"
    from ai_news_agent.sources import SourceConfig

    private = SourceConfig.model_validate(payload)
    with make_store(tmp_path) as store:
        result = ingest_one_source(
            private, store, now=NOW, fetcher=fake_fetcher(b"<rss/>")
        )

        assert result.outcome == SourceOutcome.FAILED


# Multi-source ingestion and coverage


def test_ingest_sources_records_mixed_coverage(tmp_path: Path) -> None:
    configs = load_source_configs(CONFIG_PATH)
    subset = tuple(
        c for c in configs if c.id in ("source-01", "source-02", "source-35")
    )

    def _fetch(url, **kwargs):
        if "openai" in url:
            return FeedContent(
                url=url, status_code=200, body=read_fixture("valid_rss.xml")
            )
        if "simonwillison" in url:
            raise FeedFetchError("feed timeout after 15.0s: " + url)
        raise AssertionError(f"unexpected url {url}")

    with make_store(tmp_path) as store:
        summary = ingest_sources(
            subset,
            store,
            digest_run_id="run-mixed",
            now=NOW,
            max_workers=1,
            fetcher=_fetch,
            sleep=lambda _: None,
        )

        assert summary.coverage.attempted == 3
        assert summary.coverage.successful == 1
        assert summary.coverage.failed == 1
        assert summary.coverage.unavailable == 1
        assert summary.new_article_count == 2


def test_ingest_sources_runs_concurrently_and_persists_run(tmp_path: Path) -> None:
    configs = load_source_configs(CONFIG_PATH)
    subset = tuple(c for c in configs if c.id in ("source-01", "source-35"))

    def _fetch(url, **kwargs):
        if "openai" in url:
            return FeedContent(
                url=url, status_code=200, body=read_fixture("valid_rss.xml")
            )
        return FeedContent(
            url=url, status_code=200, body=read_fixture("valid_atom.xml")
        )

    with make_store(tmp_path) as store:
        summary = ingest_sources(
            subset,
            store,
            digest_run_id="run-concurrent",
            now=NOW,
            max_workers=2,
            fetcher=_fetch,
            sleep=lambda _: None,
        )

        assert summary.coverage.successful == 2
        assert summary.new_article_count == 4
        run = store.connection.execute(
            "SELECT payload_json FROM records WHERE record_type = 'run'"
        ).fetchone()
        assert run is not None


# Retry behavior


def test_fetch_with_retries_succeeds_after_transient_failure(monkeypatch) -> None:
    calls = {"count": 0}

    def _flaky(url, *, timeout, max_bytes, etag, last_modified):
        calls["count"] += 1
        if calls["count"] == 1:
            raise FeedFetchError("feed timeout after 15.0s: " + url)
        return FeedContent(url=url, status_code=200, body=b"<rss/>")

    monkeypatch.setattr(ingestion, "fetch_feed", _flaky)
    content = fetch_with_retries("https://example.com/feed", sleep=lambda _: None)

    assert content.status_code == 200
    assert calls["count"] == 2


def test_fetch_with_retries_fails_fast_on_404(monkeypatch) -> None:
    calls = {"count": 0}

    def _missing(url, *, timeout, max_bytes, etag, last_modified):
        calls["count"] += 1
        raise FeedFetchError("feed HTTP 404: " + url)

    monkeypatch.setattr(ingestion, "fetch_feed", _missing)
    with pytest.raises(FeedFetchError, match="404"):
        fetch_with_retries("https://example.com/feed", sleep=lambda _: None)

    assert calls["count"] == 1


def test_fetch_with_retries_gives_up_after_budget(monkeypatch) -> None:
    def _down(url, *, timeout, max_bytes, etag, last_modified):
        raise FeedFetchError("feed HTTP 503: " + url)

    monkeypatch.setattr(ingestion, "fetch_feed", _down)
    with pytest.raises(FeedFetchError, match="503"):
        fetch_with_retries(
            "https://example.com/feed", max_retries=1, sleep=lambda _: None
        )


# HTTP layer with mocked transport


class _FakeStreamResponse:
    def __init__(
        self,
        url="https://example.com/feed.xml",
        status=200,
        body=b"<rss/>",
        headers=None,
    ):
        self._url = url
        self.status_code = status
        self._body = body
        self.headers = headers or {}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    @property
    def url(self):
        return self._url

    def raise_for_status(self):
        if self.status_code >= 400:
            request = httpx.Request("GET", self._url)
            response = httpx.Response(self.status_code, request=request)
            raise httpx.HTTPStatusError("error", request=request, response=response)

    def iter_bytes(self, chunk_size=65536):
        yield self._body


def test_fetch_feed_success_and_conditional_headers(monkeypatch) -> None:
    seen = {}

    def _stream(method, url, *, headers, timeout, follow_redirects):
        seen.update(headers)
        return _FakeStreamResponse(
            body=read_fixture("valid_rss.xml"),
            headers={"ETag": '"abc"', "Last-Modified": "Sun, 07 Sep 2026 11:00:00 GMT"},
        )

    monkeypatch.setattr(httpx, "stream", _stream)
    content = ingestion.fetch_feed(
        "https://example.com/feed.xml", etag='"abc"', last_modified="then"
    )

    assert content.status_code == 200
    assert content.etag == '"abc"'
    assert seen["If-None-Match"] == '"abc"'
    assert b"Model A" in content.body


def test_fetch_feed_handles_304(monkeypatch) -> None:
    monkeypatch.setattr(
        httpx, "stream", lambda *a, **k: _FakeStreamResponse(status=304)
    )
    content = ingestion.fetch_feed("https://example.com/feed.xml")

    assert content.not_modified is True


def test_fetch_feed_enforces_size_bound(monkeypatch) -> None:
    def _stream(*a, **k):
        return _FakeStreamResponse(body=b"x" * 100)

    monkeypatch.setattr(httpx, "stream", _stream)
    with pytest.raises(FeedFetchError, match="exceeds"):
        ingestion.fetch_feed("https://example.com/feed.xml", max_bytes=10)


def test_fetch_feed_maps_transport_errors(monkeypatch) -> None:
    def _timeout(*a, **k):
        raise httpx.ConnectTimeout("slow")

    monkeypatch.setattr(httpx, "stream", _timeout)
    with pytest.raises(FeedFetchError, match="timeout"):
        ingestion.fetch_feed("https://example.com/feed.xml", timeout=0.1)

    def _status(*a, **k):
        return _FakeStreamResponse(status=404)

    monkeypatch.setattr(httpx, "stream", _status)
    with pytest.raises(FeedFetchError, match="HTTP 404"):
        ingestion.fetch_feed("https://example.com/feed.xml")

    def _broken(*a, **k):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(httpx, "stream", _broken)
    with pytest.raises(FeedFetchError, match="transport error"):
        ingestion.fetch_feed("https://example.com/feed.xml")
