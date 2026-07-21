"""Local article-text extraction (SSRF-guarded fetch + trafilatura) — the
keyless replacement for Tavily's /extract endpoint."""
from __future__ import annotations

import httpx
import pytest

from app.config import Settings
from app.orchestrator.services import web_extract as we
from app.orchestrator.services.web_extract import extract_url_texts


def _settings() -> Settings:
    return Settings(_env_file=None, user_agent="test-agent", request_timeout_seconds=5, http_concurrency=4)


def _patch_transport(monkeypatch: pytest.MonkeyPatch, handler) -> None:
    """Inject a MockTransport into the client web_extract builds, while keeping
    its real kwargs — crucially the SSRF event hook and follow_redirects — so
    the guard still fires against a blocked host."""
    real = httpx.AsyncClient

    def factory(*args: object, **kwargs: object) -> httpx.AsyncClient:
        kwargs["transport"] = httpx.MockTransport(handler)
        return real(**kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(we.httpx, "AsyncClient", factory)


def _html(body: str) -> httpx.Response:
    return httpx.Response(200, text=body, headers={"content-type": "text/html; charset=utf-8"})


async def test_empty_and_whitespace_urls_short_circuit() -> None:
    assert await extract_url_texts([], _settings()) == {}
    assert await extract_url_texts(["   ", ""], _settings()) == {}


async def test_dry_run_returns_mock_without_network(monkeypatch: pytest.MonkeyPatch) -> None:
    # No transport patch: dry-run must not build a client or touch the network.
    out = await extract_url_texts(
        ["https://a.example/x", "https://a.example/x", "https://b.example/y"],
        _settings(),
        dry_run=True,
    )
    # Deduped to two, mock text each.
    assert set(out) == {"https://a.example/x", "https://b.example/y"}
    assert all(v == "Dry-run mock content." for v in out.values())


async def test_extracts_text_via_trafilatura(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return _html("<html><body><article><p>hi</p></article></body></html>")

    _patch_transport(monkeypatch, handler)
    monkeypatch.setattr(we.trafilatura, "extract", lambda _html: "Clean extracted body.")

    out = await extract_url_texts(["https://a.example/x"], _settings())
    assert out == {"https://a.example/x": "Clean extracted body."}


async def test_non_html_content_type_is_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"%PDF-1.4", headers={"content-type": "application/pdf"})

    _patch_transport(monkeypatch, handler)
    # trafilatura must not even be consulted for non-HTML.
    monkeypatch.setattr(we.trafilatura, "extract", lambda _html: "should not be used")

    assert await extract_url_texts(["https://a.example/doc.pdf"], _settings()) == {}


async def test_empty_extraction_is_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return _html("<html><body></body></html>")

    _patch_transport(monkeypatch, handler)
    monkeypatch.setattr(we.trafilatura, "extract", lambda _html: None)  # nothing extractable

    assert await extract_url_texts(["https://a.example/x"], _settings()) == {}


async def test_fetch_failure_is_skipped_not_raised(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if "good" in str(request.url):
            return _html("<article><p>hi</p></article>")
        return httpx.Response(500, text="boom")

    _patch_transport(monkeypatch, handler)
    monkeypatch.setattr(we.trafilatura, "extract", lambda _html: "Good body.")

    out = await extract_url_texts(
        ["https://a.example/good", "https://a.example/bad"], _settings()
    )
    # The 500 is skipped; the batch still returns the one that worked.
    assert out == {"https://a.example/good": "Good body."}


async def test_ssrf_guarded_blocked_host_is_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return _html("<article><p>hi</p></article>")

    _patch_transport(monkeypatch, handler)
    monkeypatch.setattr(we.trafilatura, "extract", lambda _html: "Body.")

    out = await extract_url_texts(
        ["https://a.example/ok", "http://169.254.169.254/latest/meta-data"],
        _settings(),
    )
    # The cloud-metadata address is rejected by the SSRF guard and skipped; the
    # public URL still resolves.
    assert list(out) == ["https://a.example/ok"]


async def test_real_trafilatura_extracts_article_body(monkeypatch: pytest.MonkeyPatch) -> None:
    """End-to-end with the real trafilatura (no extract stub) AND a genuinely
    streaming response body — so the get_capped read path (which would otherwise
    leave the body unreadable) is exercised for real, not masked by preset
    response content."""
    article = (
        "<html><head><title>New Model Ships</title></head><body>"
        "<nav>menu</nav>"
        "<article>"
        "<h1>A new open-weights model was released today</h1>"
        "<p>The research lab published an open-weights language model that it says "
        "improves multi-step reasoning and tool use across long documents.</p>"
        "<p>According to the accompanying technical report, the model was trained on "
        "a mixture of public code and web text, and reaches competitive scores on "
        "several public reasoning benchmarks while running at lower cost.</p>"
        "<p>Independent engineers noted that the release includes the weights, a "
        "permissive license, and reference inference code, which lowers the barrier "
        "for enterprises evaluating it in production workflows.</p>"
        "</article>"
        "<footer>copyright</footer>"
        "</body></html>"
    )

    async def stream_body() -> object:
        # chunk it so the response is genuinely streamed (no preset _content).
        for i in range(0, len(article), 64):
            yield article[i : i + 64].encode("utf-8")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=stream_body(), headers={"content-type": "text/html"})

    _patch_transport(monkeypatch, handler)

    out = await extract_url_texts(["https://news.example/model"], _settings())
    text = out.get("https://news.example/model", "")
    assert "open-weights language model" in text
    # Chrome (nav/footer) is stripped by the readability extraction.
    assert "menu" not in text and "copyright" not in text
