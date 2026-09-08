from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest

import ai_news_agent.extraction as extraction
from ai_news_agent.extraction import (
    ArticleBlockedError,
    ArticleFetchError,
    FetchedPage,
    HostRateLimiter,
    InsufficientEvidenceError,
    article_has_text,
    extract_article,
    fetch_article_html,
    fetch_with_retries,
    retrieve_article,
    retrieve_articles,
)
from ai_news_agent.schemas import Article, ArticleText, Evidence
from ai_news_agent.storage import SQLiteStore

FIXTURES = Path(__file__).parent / "fixtures" / "articles"
NOW = datetime(2026, 9, 8, 12, 0, tzinfo=UTC)


def read_fixture(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def make_article(
    article_id: str = "article-01", url: str = "https://example.com/ai/b"
) -> Article:
    return Article(
        id=article_id,
        source_id="source-01",
        url=url,
        canonical_url=url,
        title="Fixture story",
        published_at=NOW,
        first_seen_at=NOW,
    )


def make_store(tmp_path: Path) -> SQLiteStore:
    store = SQLiteStore(tmp_path / "articles.db")
    store.migrate()
    return store


def fake_fetcher(body: str, url: str | None = None):
    def _fetch(url_in, *, timeout=15.0, max_bytes=5_000_000, max_retries=2, sleep=None):
        return FetchedPage(url=url or url_in, body=body.encode("utf-8"))

    return _fetch


# Extraction from HTML


def test_extract_valid_article_strips_boilerplate() -> None:
    article = extract_article(read_fixture("valid.html"), "https://example.com/ai/b")

    assert article.title == "Fixture Lab releases Model B with open weights"
    assert article.authors == ("Example Author",)
    assert article.published_at == datetime(2026, 9, 7, 11, 0, tzinfo=UTC)
    assert article.updated_at == datetime(2026, 9, 7, 12, 30, tzinfo=UTC)
    assert "reference inference server" in article.text
    assert not article.truncated
    for boilerplate in (
        "navigation link",
        "sidebar",
        "tracking script",
        "newsletter signup",
    ):
        assert boilerplate not in article.text


def test_extract_prefers_heading_when_no_meta_title() -> None:
    html = read_fixture("thin.html").replace("<title>Brief note</title>", "")

    with pytest.raises(InsufficientEvidenceError, match="thin extract"):
        extract_article(html, "https://example.com/ai/brief")


def test_extract_detects_paywall_markers() -> None:
    html = read_fixture("paywalled.html")
    assert "isAccessibleForFree" in html

    with pytest.raises(InsufficientEvidenceError, match="paywall"):
        extract_article(html, "https://example.com/ai/big")

    stripped = html.replace(
        '<script type="application/ld+json">{"@type":"NewsArticle",'
        '"isAccessibleForFree":"False"}</script>',
        "",
    )
    with pytest.raises(InsufficientEvidenceError, match="paywall detected"):
        extract_article(stripped, "https://example.com/ai/big")


def test_extract_detects_access_blocks() -> None:
    with pytest.raises(InsufficientEvidenceError, match="access block"):
        extract_article(read_fixture("challenge.html"), "https://example.com/ai/x")


def test_extract_rejects_thin_and_untitled_pages() -> None:
    with pytest.raises(InsufficientEvidenceError, match="thin extract"):
        extract_article(read_fixture("thin.html"), "https://example.com/ai/brief")
    with pytest.raises(InsufficientEvidenceError, match="no title"):
        extract_article(
            "<html><body><p>Some text here.</p></body></html>", "https://e.com/"
        )


def test_extract_truncates_long_text() -> None:
    article = extract_article(
        read_fixture("valid.html"), "https://example.com/ai/b", max_chars=100
    )

    assert article.truncated is True
    assert len(article.text) == 100


def test_extract_falls_back_to_time_element_for_dates() -> None:
    html = read_fixture("valid.html").replace(
        '<meta property="article:published_time" content="2026-09-07T11:00:00Z">',
        '<time datetime="2026-09-07T11:00:00Z">September 7, 2026</time>',
    )
    article = extract_article(html, "https://example.com/ai/b")

    assert article.published_at == datetime(2026, 9, 7, 11, 0, tzinfo=UTC)


# HTTP fetching


class _FakeStreamResponse:
    def __init__(
        self, url="https://example.com/ai/b", status=200, body=b"<html/>", headers=None
    ):
        self._url = url
        self.status_code = status
        self._body = body
        self.headers = headers or {"content-type": "text/html; charset=utf-8"}

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


def _html_stream(monkeypatch, **kwargs) -> None:
    body = read_fixture("valid.html").encode("utf-8")
    monkeypatch.setattr(
        httpx, "stream", lambda *a, **k: _FakeStreamResponse(body=body, **kwargs)
    )


def test_fetch_article_returns_html_body(monkeypatch) -> None:
    _html_stream(monkeypatch)

    page = fetch_article_html("https://example.com/ai/b")

    assert page.url == "https://example.com/ai/b"
    assert b"Model B" in page.body


def test_fetch_article_rejects_unsafe_destination() -> None:
    with pytest.raises(ValueError, match="private"):
        fetch_article_html("http://127.0.0.1/ai/b")


def test_fetch_article_maps_block_and_error_status(monkeypatch) -> None:
    monkeypatch.setattr(
        httpx, "stream", lambda *a, **k: _FakeStreamResponse(status=403)
    )
    with pytest.raises(ArticleBlockedError, match="access blocked"):
        fetch_article_html("https://example.com/ai/b")

    monkeypatch.setattr(
        httpx, "stream", lambda *a, **k: _FakeStreamResponse(status=404)
    )
    with pytest.raises(ArticleFetchError, match="HTTP 404"):
        fetch_article_html("https://example.com/ai/b")


def test_fetch_article_rejects_non_html(monkeypatch) -> None:
    monkeypatch.setattr(
        httpx,
        "stream",
        lambda *a, **k: _FakeStreamResponse(
            body=b"%PDF-1.4", headers={"content-type": "application/pdf"}
        ),
    )
    with pytest.raises(ArticleFetchError, match="unsupported content type"):
        fetch_article_html("https://example.com/ai/b")


def test_fetch_article_enforces_size_bound(monkeypatch) -> None:
    monkeypatch.setattr(
        httpx, "stream", lambda *a, **k: _FakeStreamResponse(body=b"x" * 100)
    )
    with pytest.raises(ArticleFetchError, match="exceeds"):
        fetch_article_html("https://example.com/ai/b", max_bytes=10)


def test_fetch_article_validates_redirect_hops(monkeypatch) -> None:
    hops = [
        _FakeStreamResponse(status=302, headers={"location": "/ai/c"}),
        _FakeStreamResponse(status=302, headers={"location": "http://127.0.0.1/ai/c"}),
    ]
    monkeypatch.setattr(httpx, "stream", lambda *a, **k: hops.pop(0))
    with pytest.raises(ValueError, match="private"):
        fetch_article_html("https://example.com/ai/b")


def test_fetch_article_follows_safe_redirect(monkeypatch) -> None:
    body = read_fixture("valid.html").encode("utf-8")
    hops = [
        _FakeStreamResponse(status=301, headers={"location": "/ai/final"}),
        _FakeStreamResponse(url="https://example.com/ai/final", body=body),
    ]
    monkeypatch.setattr(httpx, "stream", lambda *a, **k: hops.pop(0))

    page = fetch_article_html("https://example.com/ai/b")

    assert page.url == "https://example.com/ai/final"
    assert b"Model B" in page.body


def test_fetch_article_rejects_redirect_loops(monkeypatch) -> None:
    monkeypatch.setattr(
        httpx,
        "stream",
        lambda *a, **k: _FakeStreamResponse(status=302, headers={"location": "/ai/b"}),
    )
    with pytest.raises(ArticleFetchError, match="too many redirects"):
        fetch_article_html("https://example.com/ai/b", max_redirects=2)


def test_fetch_article_rejects_empty_redirect(monkeypatch) -> None:
    monkeypatch.setattr(
        httpx, "stream", lambda *a, **k: _FakeStreamResponse(status=302, headers={})
    )
    with pytest.raises(ArticleFetchError, match="without location"):
        fetch_article_html("https://example.com/ai/b")


def test_fetch_article_maps_transport_errors(monkeypatch) -> None:
    monkeypatch.setattr(
        httpx,
        "stream",
        lambda *a, **k: (_ for _ in ()).throw(httpx.ConnectTimeout("slow")),
    )
    with pytest.raises(ArticleFetchError, match="timeout"):
        fetch_article_html("https://example.com/ai/b", timeout=0.1)

    monkeypatch.setattr(
        httpx, "stream", lambda *a, **k: (_ for _ in ()).throw(httpx.ConnectError("x"))
    )
    with pytest.raises(ArticleFetchError, match="transport error"):
        fetch_article_html("https://example.com/ai/b")


def test_fetch_with_retries_succeeds_after_transient_failure(monkeypatch) -> None:
    calls = {"count": 0}

    def _flaky(url, *, timeout, max_bytes):
        calls["count"] += 1
        if calls["count"] == 1:
            raise ArticleFetchError("article HTTP 503: " + url)
        return FetchedPage(url=url, body=b"<html/>")

    monkeypatch.setattr(extraction, "fetch_article_html", _flaky)
    page = fetch_with_retries("https://example.com/ai/b", sleep=lambda _: None)

    assert page.body == b"<html/>"
    assert calls["count"] == 2


def test_fetch_with_retries_never_retries_blocks(monkeypatch) -> None:
    calls = {"count": 0}

    def _blocked(url, *, timeout, max_bytes):
        calls["count"] += 1
        raise ArticleBlockedError("access blocked (HTTP 403): " + url)

    monkeypatch.setattr(extraction, "fetch_article_html", _blocked)
    with pytest.raises(ArticleBlockedError, match="access blocked"):
        fetch_with_retries("https://example.com/ai/b", sleep=lambda _: None)

    assert calls["count"] == 1


def test_fetch_with_retries_fails_fast_on_404(monkeypatch) -> None:
    calls = {"count": 0}

    def _missing(url, *, timeout, max_bytes):
        calls["count"] += 1
        raise ArticleFetchError("article HTTP 404: " + url)

    monkeypatch.setattr(extraction, "fetch_article_html", _missing)
    with pytest.raises(ArticleFetchError, match="404"):
        fetch_with_retries("https://example.com/ai/b", sleep=lambda _: None)

    assert calls["count"] == 1


def test_fetch_with_retries_fails_fast_on_size_limit(monkeypatch) -> None:
    calls = {"count": 0}

    def _huge(url, *, timeout, max_bytes):
        calls["count"] += 1
        raise ArticleFetchError("article exceeds 10 byte limit: " + url)

    monkeypatch.setattr(extraction, "fetch_article_html", _huge)
    with pytest.raises(ArticleFetchError, match="exceeds"):
        fetch_with_retries("https://example.com/ai/b", sleep=lambda _: None)

    assert calls["count"] == 1


# Rate limiting


def test_rate_limiter_spaces_same_host_requests() -> None:
    clock = {"now": 100.0}
    slept: list[float] = []
    limiter = HostRateLimiter(1.0, clock=lambda: clock["now"], sleep=slept.append)

    limiter.wait("https://example.com/a")
    limiter.wait("https://other.com/a")
    assert slept == []
    limiter.wait("https://example.com/b")
    assert slept == [1.0]


# Retrieval and caching


def test_retrieve_article_persists_text_and_evidence(tmp_path: Path) -> None:
    with make_store(tmp_path) as store:
        result = retrieve_article(
            make_article(),
            store,
            now=NOW,
            fetcher=fake_fetcher(read_fixture("valid.html")),
        )

        assert result.status == "retrieved"
        assert result.cached is False
        text = store.get(ArticleText, "article-01-text")
        assert text is not None
        assert text.article_id == "article-01"
        assert text.fetched_at == NOW
        assert "reference inference server" in text.text
        evidence = store.get(Evidence, "article-01-article-text")
        assert evidence is not None
        assert evidence.kind.value == "article_text"
        assert evidence.locator == "article-body"
        assert article_has_text(store, "article-01") is True


def test_retrieve_article_is_idempotent_for_identical_content(tmp_path: Path) -> None:
    fetcher = fake_fetcher(read_fixture("valid.html"))
    with make_store(tmp_path) as store:
        first = retrieve_article(make_article(), store, now=NOW, fetcher=fetcher)
        later = NOW.replace(hour=13)
        second = retrieve_article(make_article(), store, now=later, fetcher=fetcher)

        assert first.status == "retrieved"
        assert second.status == "retrieved"
        assert second.cached is True
        assert store.get(ArticleText, "article-01-text").fetched_at == NOW


def test_retrieve_article_updates_changed_content(tmp_path: Path) -> None:
    with make_store(tmp_path) as store:
        retrieve_article(
            make_article(),
            store,
            now=NOW,
            fetcher=fake_fetcher(read_fixture("valid.html")),
        )
        changed = read_fixture("valid.html").replace(
            "reference inference server", "brand new inference cluster"
        )
        result = retrieve_article(
            make_article(), store, now=NOW, fetcher=fake_fetcher(changed)
        )

        assert result.cached is False
        assert (
            "brand new inference cluster"
            in store.get(ArticleText, "article-01-text").text
        )


def test_retrieve_article_marks_blocked_pages_insufficient(tmp_path: Path) -> None:
    with make_store(tmp_path) as store:
        for name in ("paywalled.html", "challenge.html", "thin.html"):
            article = make_article(f"article-{name}")
            result = retrieve_article(
                article, store, now=NOW, fetcher=fake_fetcher(read_fixture(name))
            )

            assert result.status == "insufficient", name
            assert result.reason
            assert store.get(ArticleText, f"article-{name}-text") is None
            assert article_has_text(store, f"article-{name}") is False


def test_retrieve_article_marks_transport_failures_failed(tmp_path: Path) -> None:
    def _failing(url, **kwargs):
        raise ArticleFetchError("article timeout after 15.0s: " + url)

    with make_store(tmp_path) as store:
        result = retrieve_article(make_article(), store, now=NOW, fetcher=_failing)

        assert result.status == "failed"
        assert "timeout" in result.reason
        assert article_has_text(store, "article-01") is False


def test_retrieve_article_rejects_private_urls(tmp_path: Path) -> None:
    article = make_article(url="http://127.0.0.1/ai/b")
    with make_store(tmp_path) as store:
        result = retrieve_article(
            article, store, now=NOW, fetcher=fake_fetcher("<html/>")
        )

        assert result.status == "failed"


def test_retrieve_articles_reports_mixed_outcomes(tmp_path: Path) -> None:
    articles = (
        make_article("article-ok", "https://example.com/ai/ok"),
        make_article("article-pay", "https://example.com/ai/pay"),
        make_article("article-down", "https://down.example/ai/down"),
    )

    def _fetch(url, **kwargs):
        if "down.example" in url:
            raise ArticleFetchError("article timeout: " + url)
        if "/pay" in url:
            return FetchedPage(url=url, body=read_fixture("paywalled.html").encode())
        return FetchedPage(url=url, body=read_fixture("valid.html").encode())

    with make_store(tmp_path) as store:
        summary = retrieve_articles(
            articles, store, now=NOW, min_interval=0, fetcher=_fetch
        )

        assert summary.attempted == 3
        assert summary.retrieved == 1
        assert summary.insufficient == 1
        assert summary.failed == 1
