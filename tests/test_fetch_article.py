from __future__ import annotations

import httpx
import pytest

from app.config import Settings
from app.orchestrator.services.fetch_article import (
    ArticleNotHtmlError,
    BlockedArticleError,
    FetchedArticle,
    FetchArticleError,
    OversizedArticleError,
    _clean_text,
    fetch_article,
    slug_for_url,
)
from app.orchestrator.services import fetch_article as svc_mod


def _settings() -> Settings:
    return Settings(_env_file=None, request_timeout_seconds=10, user_agent="test-agent")


def _patch_httpx(monkeypatch: pytest.MonkeyPatch, handler) -> None:
    """Replace ``httpx.AsyncClient`` on the fetch_article service module so
    every construction inside it routes a single MockTransport handler —
    leaving the real get_capped streaming loop and event hooks intact.

    Captures the real AsyncClient *before* patching so the factory delegates to
    it instead of recursing into the patched one."""
    real_async_client = httpx.AsyncClient

    def _factory(*args: object, **kwargs: object) -> httpx.AsyncClient:
        kwargs["transport"] = httpx.MockTransport(handler)
        return real_async_client(**kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(svc_mod.httpx, "AsyncClient", _factory)


def test_slug_for_url_is_stable_and_filesystem_safe() -> None:
    slug_a = slug_for_url("https://example.com/a")
    slug_b = slug_for_url("https://example.com/b")
    assert slug_a.endswith(".json")
    assert slug_a.startswith("example.com-")
    assert slug_a != slug_b
    assert slug_for_url("https://example.com/a") == slug_a


def test_slug_for_url_handles_no_host_and_ip_literals() -> None:
    """No-host URLs degrade to ``nohost-<hash>.json``; IP literals (incl.
    IPv6) collapse colons to dashes so the filename stays filesystem-safe while
    the 12-char sha256 keeps same-IP URLs distinct."""
    no_host = slug_for_url("not-a-url")
    assert no_host.startswith("nohost-")
    assert no_host.endswith(".json")

    ipv4 = slug_for_url("http://192.0.2.5/x")
    assert ipv4.startswith("192.0.2.5-")
    assert ipv4.endswith(".json")

    ipv6_a = slug_for_url("http://[2001:db8::1]/a")
    ipv6_b = slug_for_url("http://[2001:db8::1]/b")
    assert ipv6_a.endswith(".json")
    assert "/" not in ipv6_a and ":" not in ipv6_a  # filesystem-safe
    assert ipv6_a != ipv6_b  # same host, different path -> different slug


def test_clean_text_strips_script_style_and_nav_noise() -> None:
    html = (
        "<html><head><script>var evil=1</script><style>a{}</style></head>"
        "<body><nav>Home About</nav><header>Logo</header><footer>Copyright</footer>"
        "<aside>Related stories</aside><form>Subscribe</form>"
        "<iframe src='x'>fallback</iframe><svg>icon</svg>"
        "<template>x</template><noscript>ns</noscript>"
        "<article><p>Hello world. Real content.</p></article></body></html>"
    )
    text = _clean_text(html)
    assert "evil" not in text
    assert "Home About" not in text
    assert "Logo" not in text
    assert "Copyright" not in text
    assert "Related stories" not in text
    assert "Subscribe" not in text
    assert "fallback" not in text
    assert "icon" not in text
    assert "Hello world. Real content." in text


def test_clean_text_truncates_to_cap() -> None:
    assert len(_clean_text("<p>" + ("x " * 50_000) + "</p>")) <= 20_000


async def test_fetch_article_extracts_open_graph_and_body(monkeypatch: pytest.MonkeyPatch) -> None:
    html = (
        "<html><head>"
        '<meta property="og:title" content="OpenAI launches new model" />'
        '<meta property="og:description" content="A short description." />'
        '<meta property="og:image" content="https://img.example.com/cover.png" />'
        "</head><body><article><p>Body paragraph with real detail.</p>"
        "<script>ignored()</script></article></body></html>"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/html; charset=utf-8"},
            content=html.encode("utf-8"),
        )

    _patch_httpx(monkeypatch, handler)

    result = await fetch_article("https://example.com/post", _settings())

    assert isinstance(result, FetchedArticle)
    assert result.title == "OpenAI launches new model"
    assert result.description == "A short description."
    assert result.image_url == "https://img.example.com/cover.png"
    assert "Body paragraph with real detail." in result.text
    assert "ignored" not in result.text
    assert result.final_url  # populated from the (mocked) final response URL


async def test_fetch_article_blocks_private_ip() -> None:
    """SSRF guard: a literal loopback host is refused before any request leaves
    the process — the event hook fires on the literal host before DNS."""
    with pytest.raises(BlockedArticleError):
        await fetch_article("http://127.0.0.1/admin", _settings())


async def test_fetch_article_blocks_redirect_to_metadata_host(monkeypatch: pytest.MonkeyPatch) -> None:
    """SSRF guard runs per-hop: a public URL that 302s to the cloud metadata
    service is refused at the redirect, not after fetching it."""

    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == "http://public.example.com/redirect":
            return httpx.Response(302, headers={"location": "http://169.254.169.4/latest/meta-data/"})
        return httpx.Response(200, text="nope")  # unreachable

    _patch_httpx(monkeypatch, handler)
    with pytest.raises(BlockedArticleError):
        await fetch_article("http://public.example.com/redirect", _settings())


async def test_fetch_article_raises_oversized_when_body_exceeds_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(svc_mod, "_MAX_DOWNLOAD_BYTES", 64)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"x" * 4096)

    _patch_httpx(monkeypatch, handler)
    with pytest.raises(OversizedArticleError):
        await fetch_article("http://example.com/huge", _settings())


async def test_fetch_article_rejects_non_html_content_type(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "application/pdf"},
            content=b"%PDF-1.4 junk",
        )

    _patch_httpx(monkeypatch, handler)
    with pytest.raises(ArticleNotHtmlError):
        await fetch_article("http://example.com/doc.pdf", _settings())


async def test_fetch_article_translates_http_transport_error_to_fetch_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("DNS down")

    _patch_httpx(monkeypatch, handler)
    with pytest.raises(FetchArticleError):
        await fetch_article("http://example.com/boom", _settings())