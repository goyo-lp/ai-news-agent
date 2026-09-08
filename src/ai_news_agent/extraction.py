"""Cached article retrieval with boilerplate stripping and block detection."""

from __future__ import annotations

import hashlib
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from html.parser import HTMLParser
from urllib.parse import urljoin, urlparse

import httpx
import langsmith as ls
import structlog

from ai_news_agent.ingestion import ensure_public_http_url, parse_feed_datetime
from ai_news_agent.observability import redact
from ai_news_agent.schemas import Article, ArticleText, Evidence, EvidenceKind
from ai_news_agent.storage import SQLiteStore

_LOGGER = structlog.get_logger(__name__)

DEFAULT_TIMEOUT_SECONDS = 15.0
DEFAULT_MAX_BYTES = 5_000_000
DEFAULT_MAX_RETRIES = 2
DEFAULT_MAX_REDIRECTS = 5
DEFAULT_MIN_INTERVAL_SECONDS = 1.0
DEFAULT_MIN_TEXT_CHARS = 400
TEXT_CHAR_LIMIT = 20_000
EVIDENCE_EXCERPT_LIMIT = 1000

Fetcher = Callable[..., "FetchedPage"]
Sleeper = Callable[[float], None]
Clock = Callable[[], float]

_PAYWALL_MARKERS = (
    "subscribe to continue reading",
    "sign in to continue",
    "log in to continue reading",
    "this content is for members",
    "already a subscriber",
    "start your free trial to continue",
)
_BLOCK_MARKERS = (
    "verify you are human",
    "enable javascript",
    "captcha",
    "access denied",
    "request blocked",
    "are you a robot",
    "just a moment",
)
_ACCESSIBLE_FOR_FREE = re.compile(
    r'"isAccessibleForFree"\s*:\s*(false|"false")', re.IGNORECASE
)
_SKIP_TAGS = frozenset(
    {"script", "style", "nav", "header", "footer", "aside", "form", "noscript"}
)
_TEXT_TAGS = frozenset({"p", "h1", "h2", "h3", "li", "blockquote"})


class ArticleFetchError(RuntimeError):
    """Raised when an article page cannot be retrieved within bounds."""


class ArticleBlockedError(ArticleFetchError):
    """Raised when the server refuses access (401/403/451)."""


class InsufficientEvidenceError(RuntimeError):
    """Raised when a retrieved page cannot support a full-article summary."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(slots=True)
class FetchedPage:
    url: str
    body: bytes
    content_type: str = "text/html"


@dataclass(slots=True)
class ExtractedArticle:
    final_url: str
    title: str
    authors: tuple[str, ...]
    published_at: datetime | None
    updated_at: datetime | None
    text: str
    truncated: bool


@dataclass(slots=True)
class ArticleResult:
    article_id: str
    status: str
    text_id: str | None = None
    evidence_id: str | None = None
    reason: str | None = None
    cached: bool = False


@dataclass(slots=True)
class RetrievalSummary:
    attempted: int = 0
    retrieved: int = 0
    cached: int = 0
    insufficient: int = 0
    failed: int = 0
    results: tuple[ArticleResult, ...] = ()
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    finished_at: datetime = field(default_factory=lambda: datetime.now(UTC))


class _ArticleParser(HTMLParser):
    """Collect title, authors, dates, and body text while skipping chrome."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._skip_depth = 0
        self._in_title = False
        self._in_heading = False
        self._in_text = False
        self._in_article = False
        self._title_tag = ""
        self._heading = ""
        self._current: list[str] = []
        self.blocks: list[str] = []
        self.meta: dict[str, str] = {}
        self.times: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag in _SKIP_TAGS:
            self._skip_depth += 1
            return
        if self._skip_depth:
            return
        if tag == "article":
            self._in_article = True
        if tag == "title":
            self._in_title = True
        if tag in ("h1", "h2", "h3") and not self._heading:
            self._in_heading = True
        if tag in _TEXT_TAGS:
            self._in_text = True
            self._current = []
        if tag == "meta":
            entry = {key.lower(): (value or "") for key, value in attrs}
            name = (
                (
                    entry.get("property")
                    or entry.get("name")
                    or entry.get("itemprop")
                    or ""
                )
                .strip()
                .lower()
            )
            if name and entry.get("content", "").strip():
                self.meta.setdefault(name, entry["content"].strip())
        if tag == "time":
            entry = dict(attrs)
            if entry.get("datetime", "").strip():
                self.times.append(entry["datetime"].strip())

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in _SKIP_TAGS and self._skip_depth:
            self._skip_depth -= 1
            return
        if self._skip_depth:
            return
        if tag == "article":
            self._in_article = False
        if tag == "title":
            self._in_title = False
        if tag in ("h1", "h2", "h3") and self._in_heading:
            self._in_heading = False
        if tag in _TEXT_TAGS and self._in_text:
            self._in_text = False
            if self._current and tag != "h1":
                text = re.sub(r"\s+", " ", "".join(self._current)).strip()
                if text:
                    self.blocks.append(text)
            self._current = []

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        if self._in_title:
            self._title_tag += data
        elif self._in_heading and not self._heading:
            self._heading += data
        elif self._in_text:
            self._current.append(data)


def _parse_html(html: str) -> _ArticleParser:
    parser = _ArticleParser()
    parser.feed(html)
    parser.close()
    return parser


def extract_article(
    html: str,
    base_url: str,
    *,
    min_chars: int = DEFAULT_MIN_TEXT_CHARS,
    max_chars: int = TEXT_CHAR_LIMIT,
) -> ExtractedArticle:
    """Extract readable text and metadata from raw HTML.

    Raises InsufficientEvidenceError when the page shows paywall, access-block,
    or challenge markers, or when the remaining body text is too thin to
    support a full-article summary. Retrieved text is untrusted data: it is
    returned as inert strings and never interpreted as instructions.
    """

    lowered = html.lower()
    if _ACCESSIBLE_FOR_FREE.search(html):
        raise InsufficientEvidenceError("paywall: page marked not freely accessible")
    for marker in _BLOCK_MARKERS:
        if marker in lowered:
            raise InsufficientEvidenceError(f"access block detected: {marker}")
    for marker in _PAYWALL_MARKERS:
        if marker in lowered:
            raise InsufficientEvidenceError(f"paywall detected: {marker}")

    parser = _parse_html(html)
    title = (
        parser.meta.get("og:title")
        or parser.meta.get("twitter:title")
        or parser._title_tag.strip()
        or parser._heading.strip()
    )
    if not title:
        raise InsufficientEvidenceError("no title found")
    authors = _split_authors(
        parser.meta.get("author")
        or parser.meta.get("article:author")
        or parser.meta.get("twitter:creator")
        or ""
    )
    published_at = parse_feed_datetime(
        parser.meta.get("article:published_time")
        or parser.meta.get("datepublished")
        or next(iter(parser.times), None)
    )
    updated_at = parse_feed_datetime(
        parser.meta.get("article:modified_time")
        or parser.meta.get("datemodified")
        or None
    )
    text = " ".join(parser.blocks)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) < min_chars:
        raise InsufficientEvidenceError(
            f"thin extract: {len(text)} chars below {min_chars} minimum"
        )
    truncated = len(text) > max_chars
    return ExtractedArticle(
        final_url=base_url,
        title=title,
        authors=authors,
        published_at=published_at,
        updated_at=updated_at,
        text=text[:max_chars],
        truncated=truncated,
    )


def _split_authors(value: str) -> tuple[str, ...]:
    authors = [
        part.strip().lstrip("@")
        for part in re.split(r"[,;]| and ", value)
        if part.strip()
    ]
    return tuple(dict.fromkeys(authors))


def fetch_article_html(
    url: str,
    *,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    max_bytes: int = DEFAULT_MAX_BYTES,
    max_redirects: int = DEFAULT_MAX_REDIRECTS,
) -> FetchedPage:
    """Fetch one article page, validating every redirect hop.

    Each redirect target is resolved against the previous URL and rejected
    when it points at a private, local, non-HTTP, or credential-bearing
    destination.
    """

    ensure_public_http_url(url)
    current = url
    try:
        for _ in range(max_redirects + 1):
            with httpx.stream(
                "GET",
                current,
                headers={"User-Agent": "ai-news-agent/0.1 (+article-retrieval)"},
                timeout=timeout,
                follow_redirects=False,
            ) as response:
                if response.status_code in (301, 302, 303, 307, 308):
                    location = response.headers.get("location", "").strip()
                    if not location:
                        raise ArticleFetchError(f"redirect without location: {url}")
                    current = ensure_public_http_url(urljoin(current, location))
                    continue
                if response.status_code in (401, 403, 451):
                    raise ArticleBlockedError(
                        f"access blocked (HTTP {response.status_code}): {url}"
                    )
                response.raise_for_status()
                content_type = response.headers.get("content-type", "")
                if "html" not in content_type.lower():
                    raise ArticleFetchError(
                        f"unsupported content type {content_type!r}: {url}"
                    )
                chunks: list[bytes] = []
                total = 0
                for chunk in response.iter_bytes(chunk_size=65536):
                    total += len(chunk)
                    if total > max_bytes:
                        raise ArticleFetchError(
                            f"article exceeds {max_bytes} byte limit: {url}"
                        )
                    chunks.append(chunk)
                return FetchedPage(
                    url=current, body=b"".join(chunks), content_type=content_type
                )
    except httpx.TimeoutException as exc:
        raise ArticleFetchError(f"article timeout after {timeout}s: {url}") from exc
    except httpx.HTTPStatusError as exc:
        raise ArticleFetchError(
            f"article HTTP {exc.response.status_code}: {url}"
        ) from exc
    except httpx.HTTPError as exc:
        raise ArticleFetchError(f"article transport error: {url}: {exc}") from exc
    raise ArticleFetchError(f"too many redirects: {url}")


def _retryable(message: str) -> bool:
    if "exceeds" in message or "unsupported content type" in message:
        return False
    if "HTTP 4" in message and "HTTP 429" not in message:
        return False
    return (
        "timeout" in message
        or "transport error" in message
        or "HTTP 429" in message
        or "HTTP 500" in message
        or "HTTP 502" in message
        or "HTTP 503" in message
        or "HTTP 504" in message
    )


def fetch_with_retries(
    url: str,
    *,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    max_bytes: int = DEFAULT_MAX_BYTES,
    max_retries: int = DEFAULT_MAX_RETRIES,
    sleep: Sleeper = time.sleep,
) -> FetchedPage:
    """Retry timeouts, 429s, and 5xx with backoff; fail fast otherwise."""

    last_error: ArticleFetchError | None = None
    for attempt in range(max_retries + 1):
        try:
            return fetch_article_html(url, timeout=timeout, max_bytes=max_bytes)
        except ArticleBlockedError:
            raise
        except ArticleFetchError as exc:
            last_error = exc
            if not _retryable(str(exc)) or attempt >= max_retries:
                raise
            sleep(float(2**attempt))
            _LOGGER.info(
                "article_fetch_retry", url=url, attempt=attempt + 1, error=str(exc)
            )
    raise last_error or ArticleFetchError(f"article fetch failed: {url}")


class HostRateLimiter:
    """Enforce a minimum interval between requests to the same host."""

    def __init__(
        self,
        min_interval: float = DEFAULT_MIN_INTERVAL_SECONDS,
        *,
        clock: Clock = time.monotonic,
        sleep: Sleeper = time.sleep,
    ) -> None:
        self.min_interval = min_interval
        self._clock = clock
        self._sleep = sleep
        self._last_seen: dict[str, float] = {}

    def wait(self, url: str) -> None:
        """Sleep until the host interval has elapsed since the last call."""

        host = (urlparse(url).hostname or "").lower()
        now = self._clock()
        previous = self._last_seen.get(host)
        if previous is not None:
            delay = self.min_interval - (now - previous)
            if delay > 0:
                self._sleep(delay)
                now = self._clock()
        self._last_seen[host] = now


def article_has_text(store: SQLiteStore, article_id: str) -> bool:
    """Return whether an article has retrieved full text for summarization."""

    return store.get(ArticleText, f"{article_id}-text") is not None


@ls.traceable(
    name="retrieve-article",
    run_type="chain",
    process_inputs=redact,
    process_outputs=redact,
)
def retrieve_article(
    article: Article,
    store: SQLiteStore,
    *,
    now: datetime,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    max_bytes: int = DEFAULT_MAX_BYTES,
    max_retries: int = DEFAULT_MAX_RETRIES,
    min_chars: int = DEFAULT_MIN_TEXT_CHARS,
    rate_limiter: HostRateLimiter | None = None,
    fetcher: Fetcher | None = None,
    sleep: Sleeper = time.sleep,
) -> ArticleResult:
    """Fetch, extract, and cache the full text of one article.

    Blocked, paywalled, and thin pages yield an ``insufficient`` result and
    persist nothing, so they can never qualify for full-article summaries.
    Identical re-fetches keep the original ``fetched_at`` timestamp.
    """

    text_id = f"{article.id}-text"
    evidence_id = f"{article.id}-article-text"
    try:
        ensure_public_http_url(str(article.url))
        if rate_limiter is not None:
            rate_limiter.wait(str(article.url))
        if fetcher is not None:
            page = fetcher(
                str(article.url),
                timeout=timeout,
                max_bytes=max_bytes,
                max_retries=max_retries,
                sleep=sleep,
            )
        else:
            page = fetch_with_retries(
                str(article.url),
                timeout=timeout,
                max_bytes=max_bytes,
                max_retries=max_retries,
                sleep=sleep,
            )
        ensure_public_http_url(page.url)
        html = page.body.decode("utf-8", errors="replace")
        extracted = extract_article(html, page.url, min_chars=min_chars)
    except (ArticleBlockedError, InsufficientEvidenceError) as exc:
        _LOGGER.info("article_insufficient", article_id=article.id, reason=str(exc))
        return ArticleResult(
            article_id=article.id, status="insufficient", reason=str(exc)
        )
    except (ArticleFetchError, ValueError, UnicodeError) as exc:
        _LOGGER.warning("article_fetch_failed", article_id=article.id, error=str(exc))
        return ArticleResult(article_id=article.id, status="failed", reason=str(exc))

    content_hash = hashlib.sha256(extracted.text.encode("utf-8")).hexdigest()
    existing = store.get(ArticleText, text_id)
    if existing is not None and existing.content_hash == content_hash:
        return ArticleResult(
            article_id=article.id,
            status="retrieved",
            text_id=text_id,
            evidence_id=evidence_id,
            cached=True,
        )
    locator = "article-body:truncated" if extracted.truncated else "article-body"
    store.save(
        ArticleText(
            id=text_id,
            article_id=article.id,
            text=extracted.text,
            content_hash=content_hash,
            fetched_at=now,
        )
    )
    store.save(
        Evidence(
            id=evidence_id,
            article_id=article.id,
            kind=EvidenceKind.ARTICLE_TEXT,
            source_url=extracted.final_url,
            locator=locator,
            excerpt=extracted.text[:EVIDENCE_EXCERPT_LIMIT],
            content_hash=content_hash,
            captured_at=now,
        )
    )
    _LOGGER.info(
        "article_retrieved",
        article_id=article.id,
        text_chars=len(extracted.text),
        content_hash=content_hash[:12],
    )
    return ArticleResult(
        article_id=article.id,
        status="retrieved",
        text_id=text_id,
        evidence_id=evidence_id,
    )


@ls.traceable(
    name="retrieve-articles",
    run_type="chain",
    process_inputs=redact,
    process_outputs=redact,
)
def retrieve_articles(
    articles: tuple[Article, ...],
    store: SQLiteStore,
    *,
    now: datetime | None = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    max_bytes: int = DEFAULT_MAX_BYTES,
    max_retries: int = DEFAULT_MAX_RETRIES,
    min_chars: int = DEFAULT_MIN_TEXT_CHARS,
    min_interval: float = DEFAULT_MIN_INTERVAL_SECONDS,
    fetcher: Fetcher | None = None,
    sleep: Sleeper = time.sleep,
) -> RetrievalSummary:
    """Retrieve full text for each article sequentially with host politeness."""

    started = now or datetime.now(UTC)
    limiter = HostRateLimiter(min_interval, sleep=sleep)
    results = tuple(
        retrieve_article(
            article,
            store,
            now=started,
            timeout=timeout,
            max_bytes=max_bytes,
            max_retries=max_retries,
            min_chars=min_chars,
            rate_limiter=limiter,
            fetcher=fetcher,
            sleep=sleep,
        )
        for article in articles
    )
    summary = RetrievalSummary(
        attempted=len(results),
        retrieved=sum(1 for item in results if item.status == "retrieved"),
        cached=sum(1 for item in results if item.cached),
        insufficient=sum(1 for item in results if item.status == "insufficient"),
        failed=sum(1 for item in results if item.status == "failed"),
        results=results,
        started_at=started,
        finished_at=datetime.now(UTC),
    )
    _LOGGER.info(
        "retrieval_complete",
        attempted=summary.attempted,
        retrieved=summary.retrieved,
        insufficient=summary.insufficient,
        failed=summary.failed,
    )
    return summary


__all__ = [
    "DEFAULT_MAX_BYTES",
    "DEFAULT_TIMEOUT_SECONDS",
    "ArticleBlockedError",
    "ArticleFetchError",
    "ArticleResult",
    "ExtractedArticle",
    "FetchedPage",
    "HostRateLimiter",
    "InsufficientEvidenceError",
    "RetrievalSummary",
    "article_has_text",
    "extract_article",
    "fetch_article_html",
    "fetch_with_retries",
    "retrieve_article",
    "retrieve_articles",
]
