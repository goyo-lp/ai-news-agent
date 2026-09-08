"""Bounded, idempotent RSS/Atom collection with explicit health records."""

from __future__ import annotations

import hashlib
import ipaddress
import re
import sqlite3
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from enum import StrEnum
from urllib.parse import parse_qsl, urlencode, urljoin, urlparse, urlunparse
from xml.etree import ElementTree as ET

import httpx
import langsmith as ls
import structlog

from ai_news_agent.adapters import adapter_for_source
from ai_news_agent.observability import redact
from ai_news_agent.schemas import (
    Article,
    CoverageStats,
    Evidence,
    EvidenceKind,
    Run,
    RunStatus,
    Source,
)
from ai_news_agent.sources import SourceConfig
from ai_news_agent.storage import SQLiteStore

_LOGGER = structlog.get_logger(__name__)

DEFAULT_TIMEOUT_SECONDS = 15.0
DEFAULT_MAX_WORKERS = 8
DEFAULT_MAX_BYTES = 15_000_000
DEFAULT_MAX_RETRIES = 2
DEFAULT_MAX_ENTRIES = 500
_EVIDENCE_EXCERPT_LIMIT = 2000
_TRACKING_PARAMS = {"fbclid", "gclid", "mc_cid", "mc_eid"}
_ATOM_NS = "{http://www.w3.org/2005/Atom}"
_HTML_TAG = re.compile(r"<[^>]+>")

Fetcher = Callable[..., "FeedContent"]
Sleeper = Callable[[float], None]


class FeedFetchError(RuntimeError):
    """Raised when a feed cannot be retrieved within bounded retries."""


class FeedParseError(RuntimeError):
    """Raised when feed bytes cannot be parsed without inventing content."""


class SourceOutcome(StrEnum):
    SUCCESS = "successful"
    UNCHANGED = "unchanged"
    FAILED = "failed"
    UNAVAILABLE = "unavailable"


@dataclass(slots=True)
class FeedContent:
    url: str
    status_code: int
    body: bytes = b""
    etag: str | None = None
    last_modified: str | None = None
    not_modified: bool = False


@dataclass(slots=True)
class ParsedEntry:
    url: str
    title: str
    published_at: datetime | None = None
    updated_at: datetime | None = None
    authors: tuple[str, ...] = ()
    excerpt: str = ""
    locator: str = ""


@dataclass(slots=True)
class SourceIngestResult:
    source_id: str
    outcome: SourceOutcome
    articles: tuple[Article, ...] = ()
    evidences: tuple[Evidence, ...] = ()
    error: str | None = None
    new_articles: int = 0


@dataclass(slots=True)
class IngestSummary:
    digest_run_id: str
    coverage: CoverageStats
    results: tuple[SourceIngestResult, ...] = ()
    article_count: int = 0
    new_article_count: int = 0
    evidence_count: int = 0
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    finished_at: datetime = field(default_factory=lambda: datetime.now(UTC))


def ensure_public_http_url(url: str) -> str:
    """Validate that a URL is a public http(s) address without credentials."""

    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"only http(s) URLs are allowed: {url}")
    if parsed.username or parsed.password:
        raise ValueError(f"URL must not embed credentials: {url}")
    host = (parsed.hostname or "").lower()
    if not host:
        raise ValueError(f"URL has no host: {url}")
    if host in ("localhost", "metadata.google.internal"):
        raise ValueError(f"private host is not allowed: {url}")
    for suffix in (".local", ".internal", ".lan", ".localdomain", ".invalid"):
        if host.endswith(suffix):
            raise ValueError(f"private host is not allowed: {url}")
    try:
        ip = ipaddress.ip_address(host.strip("[]"))
    except ValueError:
        return url
    if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast:
        raise ValueError(f"private address is not allowed: {url}")
    if ip.is_unspecified or ip.is_reserved:
        raise ValueError(f"non-routable address is not allowed: {url}")
    return url


def canonicalize_url(url: str, base_url: str) -> str:
    """Resolve, strip fragments and trackers, and normalize a feed link."""

    resolved = urljoin(base_url, url.strip())
    parsed = urlparse(resolved)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"only http(s) article URLs are allowed: {url}")
    query = [
        (key, value)
        for key, value in parse_qsl(parsed.query, keep_blank_values=True)
        if not (key.lower().startswith("utm_") or key.lower() in _TRACKING_PARAMS)
    ]
    host = (parsed.hostname or "").lower()
    port = f":{parsed.port}" if parsed.port not in (None, 80, 443) else ""
    path = parsed.path or "/"
    if len(path) > 1 and path.endswith("/"):
        path = path.rstrip("/")
    normalized = urlunparse(
        (parsed.scheme, f"{host}{port}", path, "", urlencode(query), "")
    )
    return ensure_public_http_url(normalized)


def stable_article_id(source_id: str, canonical_url: str) -> str:
    """Return a stable, idempotent article ID for one canonical URL."""

    digest = hashlib.sha256(canonical_url.strip().lower().encode("utf-8")).hexdigest()
    return f"{source_id}-{digest[:16]}"


def parse_feed_datetime(value: str | None) -> datetime | None:
    """Parse RSS/Atom timestamps into timezone-aware datetimes."""

    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    try:
        parsed = parsedate_to_datetime(text)
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=UTC)
        return parsed
    except (TypeError, ValueError):
        pass
    try:
        iso = text.replace("Z", "+00:00") if text.endswith("Z") else text
        parsed = datetime.fromisoformat(iso)
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=UTC)
        return parsed
    except ValueError:
        return None


def strip_html(text: str, limit: int = 1000) -> str:
    """Remove markup and bound excerpt length without inventing content."""

    cleaned = _HTML_TAG.sub(" ", text or "")
    collapsed = re.sub(r"\s+", " ", cleaned).strip()
    return collapsed[:limit]


def parse_feed(body: bytes, base_url: str) -> tuple[ParsedEntry, ...]:
    """Parse RSS or Atom bytes into normalized entries.

    Malformed entries are quarantined (skipped with a log) instead of failing
    the whole feed; malformed XML raises FeedParseError for the source.
    """

    try:
        root = ET.fromstring(body)
    except ET.ParseError as exc:
        raise FeedParseError(f"malformed feed XML: {exc}") from exc
    tag = root.tag
    if tag == "rss" or tag.endswith("}rss"):
        return _parse_rss_items(root.findall(".//item"), base_url)
    if tag == f"{_ATOM_NS}feed" or tag == "feed":
        items = root.findall(f"{_ATOM_NS}entry") or root.findall("entry")
        return _parse_atom_entries(items, base_url)
    raise FeedParseError(f"unsupported feed root: {tag}")


def _text(element: ET.Element | None) -> str:
    if element is None or element.text is None:
        return ""
    return element.text.strip()


def _find_text(parent: ET.Element, names: tuple[str, ...]) -> str:
    for name in names:
        found = parent.find(name)
        if found is not None and found.text and found.text.strip():
            return found.text.strip()
    return ""


def _parse_rss_items(items: list[ET.Element], base_url: str) -> tuple[ParsedEntry, ...]:
    entries: list[ParsedEntry] = []
    for index, item in enumerate(items[:DEFAULT_MAX_ENTRIES]):
        title = _find_text(item, ("title",))
        link = _find_text(item, ("link",))
        guid = _find_text(item, ("guid",))
        if not title or not link:
            _LOGGER.warning("feed_entry_quarantined", reason="missing-title-or-link")
            continue
        try:
            canonical = canonicalize_url(link, base_url)
        except ValueError as exc:
            _LOGGER.warning("feed_entry_quarantined", reason=str(exc))
            continue
        published = parse_feed_datetime(
            _find_text(item, ("pubDate", "published", "updated"))
        )
        updated = parse_feed_datetime(_find_text(item, ("updated",)))
        description = _find_text(item, ("description", "summary"))
        authors = _find_text(item, ("author",))
        entries.append(
            ParsedEntry(
                url=canonical,
                title=title,
                published_at=published,
                updated_at=updated,
                authors=(authors,) if authors else (),
                excerpt=strip_html(description),
                locator=f"feed-entry:{guid or link or index}",
            )
        )
    return tuple(entries)


def _parse_atom_entries(
    items: list[ET.Element], base_url: str
) -> tuple[ParsedEntry, ...]:
    entries: list[ParsedEntry] = []
    for index, item in enumerate(items[:DEFAULT_MAX_ENTRIES]):
        title = _text(item.find(f"{_ATOM_NS}title")) or _text(item.find("title"))
        link = ""
        for candidate in item.findall(f"{_ATOM_NS}link") or item.findall("link"):
            href = (candidate.get("href") or "").strip()
            rel = (candidate.get("rel") or "alternate").strip()
            if href and rel in ("alternate", "self", ""):
                link = href
                break
        entry_id = _text(item.find(f"{_ATOM_NS}id")) or _text(item.find("id"))
        if not title or not link:
            _LOGGER.warning("feed_entry_quarantined", reason="missing-title-or-link")
            continue
        try:
            canonical = canonicalize_url(link, base_url)
        except ValueError as exc:
            _LOGGER.warning("feed_entry_quarantined", reason=str(exc))
            continue
        published = parse_feed_datetime(
            _text(item.find(f"{_ATOM_NS}published"))
            or _text(item.find("published"))
            or _text(item.find(f"{_ATOM_NS}updated"))
            or _text(item.find("updated"))
        )
        updated = parse_feed_datetime(
            _text(item.find(f"{_ATOM_NS}updated")) or _text(item.find("updated"))
        )
        summary = _text(item.find(f"{_ATOM_NS}summary")) or _text(item.find("summary"))
        content = _text(item.find(f"{_ATOM_NS}content")) or _text(item.find("content"))
        authors: list[str] = []
        for author in item.findall(f"{_ATOM_NS}author") or item.findall("author"):
            name = _text(author.find(f"{_ATOM_NS}name")) or _text(author.find("name"))
            if name:
                authors.append(name)
        entries.append(
            ParsedEntry(
                url=canonical,
                title=title,
                published_at=published,
                updated_at=updated,
                authors=tuple(authors),
                excerpt=strip_html(summary or content),
                locator=f"feed-entry:{entry_id or link or index}",
            )
        )
    return tuple(entries)


def fetch_feed(
    url: str,
    *,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    max_bytes: int = DEFAULT_MAX_BYTES,
    etag: str | None = None,
    last_modified: str | None = None,
) -> FeedContent:
    """Fetch one feed with timeouts, conditional headers, and a size bound."""

    ensure_public_http_url(url)
    headers: dict[str, str] = {"User-Agent": "ai-news-agent/0.1 (+ingestion)"}
    if etag:
        headers["If-None-Match"] = etag
    if last_modified:
        headers["If-Modified-Since"] = last_modified
    try:
        with httpx.stream(
            "GET",
            url,
            headers=headers,
            timeout=timeout,
            follow_redirects=True,
        ) as response:
            if response.status_code == 304:
                return FeedContent(url=url, status_code=304, not_modified=True)
            response.raise_for_status()
            chunks: list[bytes] = []
            total = 0
            for chunk in response.iter_bytes(chunk_size=65536):
                total += len(chunk)
                if total > max_bytes:
                    raise FeedFetchError(f"feed exceeds {max_bytes} byte limit: {url}")
                chunks.append(chunk)
            body = b"".join(chunks)
            lowered = {key.lower(): value for key, value in response.headers.items()}
            return FeedContent(
                url=str(response.url),
                status_code=response.status_code,
                body=body,
                etag=lowered.get("etag"),
                last_modified=lowered.get("last-modified"),
            )
    except httpx.TimeoutException as exc:
        raise FeedFetchError(f"feed timeout after {timeout}s: {url}") from exc
    except httpx.HTTPStatusError as exc:
        raise FeedFetchError(f"feed HTTP {exc.response.status_code}: {url}") from exc
    except httpx.HTTPError as exc:
        raise FeedFetchError(f"feed transport error: {url}: {exc}") from exc


def _is_retryable(message: str) -> bool:
    if "exceeds" in message:
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
    etag: str | None = None,
    last_modified: str | None = None,
    sleep: Sleeper = time.sleep,
) -> FeedContent:
    """Retry timeouts, 429s, and 5xx with backoff; fail fast on other 4xx."""

    last_error: FeedFetchError | None = None
    for attempt in range(max_retries + 1):
        try:
            return fetch_feed(
                url,
                timeout=timeout,
                max_bytes=max_bytes,
                etag=etag,
                last_modified=last_modified,
            )
        except FeedFetchError as exc:
            last_error = exc
            message = str(exc)
            if not _is_retryable(message) or attempt >= max_retries:
                raise
            delay = float(2**attempt)
            sleep(delay)
            _LOGGER.info(
                "feed_fetch_retry", url=url, attempt=attempt + 1, error=message
            )
    raise last_error or FeedFetchError(f"feed fetch failed: {url}")


def _dedupe_entries(entries: tuple[ParsedEntry, ...]) -> tuple[ParsedEntry, ...]:
    seen: set[str] = set()
    unique: list[ParsedEntry] = []
    for entry in entries:
        if entry.url in seen:
            continue
        seen.add(entry.url)
        unique.append(entry)
    return tuple(unique)


def entry_to_records(
    source_id: str,
    entry: ParsedEntry,
    *,
    now: datetime,
) -> tuple[Article, Evidence]:
    """Normalize one feed entry into Article and raw-feed Evidence records."""

    article_id = stable_article_id(source_id, entry.url)
    published_at = entry.published_at or now
    if published_at.tzinfo is None:
        published_at = published_at.replace(tzinfo=UTC)
    article = Article(
        id=article_id,
        source_id=source_id,
        url=entry.url,
        canonical_url=entry.url,
        title=entry.title,
        authors=entry.authors,
        published_at=published_at,
        updated_at=entry.updated_at,
        first_seen_at=now,
        raw_feed_ref=entry.locator or None,
    )
    excerpt = f"{entry.title}. {entry.excerpt}".strip()[:_EVIDENCE_EXCERPT_LIMIT]
    if not excerpt:
        excerpt = entry.title[:_EVIDENCE_EXCERPT_LIMIT]
    content_hash = hashlib.sha256(excerpt.encode("utf-8")).hexdigest()
    evidence = Evidence(
        id=f"{article_id}-feed",
        article_id=article_id,
        kind=EvidenceKind.FEED_ENTRY,
        source_url=entry.url,
        locator=entry.locator or "feed-entry:0",
        excerpt=excerpt,
        content_hash=content_hash,
        captured_at=now,
    )
    return article, evidence


def _get_feed_state(
    store: SQLiteStore, source_id: str
) -> tuple[str | None, str | None, str]:
    try:
        row = store.connection.execute(
            "SELECT etag, last_modified, content_hash "
            "FROM feed_state WHERE source_id = ?",
            (source_id,),
        ).fetchone()
    except sqlite3.Error:
        return None, None, ""
    if row is None:
        return None, None, ""
    return row["etag"], row["last_modified"], row["content_hash"] or ""


def _put_feed_state(
    store: SQLiteStore,
    source_id: str,
    *,
    etag: str | None,
    last_modified: str | None,
    content_hash: str,
    checked_at: datetime,
    success_at: datetime | None,
) -> None:
    store.connection.execute(
        """
        INSERT INTO feed_state
            (source_id, etag, last_modified, content_hash,
             last_checked_at, last_success_at, consecutive_failures)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(source_id) DO UPDATE SET
            etag = excluded.etag,
            last_modified = excluded.last_modified,
            content_hash = excluded.content_hash,
            last_checked_at = excluded.last_checked_at,
            last_success_at = COALESCE(
                excluded.last_success_at, feed_state.last_success_at
            ),
            consecutive_failures = excluded.consecutive_failures
        """,
        (
            source_id,
            etag,
            last_modified,
            content_hash,
            checked_at.isoformat(),
            success_at.isoformat() if success_at else None,
            0 if success_at else 1,
        ),
    )
    store.connection.commit()


def _touch_source(
    store: SQLiteStore, config: SourceConfig, now: datetime, *, success: bool
) -> None:
    record = config.to_source()
    payload = record.model_dump()
    payload["last_checked_at"] = now
    payload["last_success_at"] = now if success else record.last_success_at
    store.save(Source.model_validate(payload))


@dataclass(slots=True)
class _FetchedSource:
    config: SourceConfig
    outcome: SourceOutcome | None
    content: FeedContent | None
    entries: tuple[ParsedEntry, ...]
    error: str | None
    body_hash: str
    etag: str | None
    last_modified: str | None
    prior_hash: str


def _fetch_and_parse_one(
    config: SourceConfig,
    *,
    etag: str | None,
    last_modified: str | None,
    prior_hash: str,
    timeout: float,
    max_bytes: int,
    max_retries: int,
    fetcher: Fetcher | None,
    sleep: Sleeper,
) -> _FetchedSource:
    """Fetch and parse one source without touching the database."""

    adapter = adapter_for_source(config)
    if adapter is not None:
        outcome = adapter.fetch()
        return _FetchedSource(
            config=config,
            outcome=SourceOutcome.UNAVAILABLE,
            content=None,
            entries=(),
            error=outcome.reason,
            body_hash="",
            etag=etag,
            last_modified=last_modified,
            prior_hash=prior_hash,
        )
    try:
        ensure_public_http_url(str(config.route.url))
        if fetcher is not None:
            content = fetcher(
                str(config.route.url),
                timeout=timeout,
                max_bytes=max_bytes,
                max_retries=max_retries,
                etag=etag,
                last_modified=last_modified,
                sleep=sleep,
            )
        else:
            content = fetch_with_retries(
                str(config.route.url),
                timeout=timeout,
                max_bytes=max_bytes,
                max_retries=max_retries,
                etag=etag,
                last_modified=last_modified,
                sleep=sleep,
            )
    except FeedFetchError as exc:
        return _FetchedSource(
            config=config,
            outcome=SourceOutcome.FAILED,
            content=None,
            entries=(),
            error=str(exc),
            body_hash="",
            etag=etag,
            last_modified=last_modified,
            prior_hash=prior_hash,
        )
    except ValueError as exc:
        return _FetchedSource(
            config=config,
            outcome=SourceOutcome.FAILED,
            content=None,
            entries=(),
            error=str(exc),
            body_hash="",
            etag=etag,
            last_modified=last_modified,
            prior_hash=prior_hash,
        )
    if content.not_modified:
        return _FetchedSource(
            config=config,
            outcome=SourceOutcome.UNCHANGED,
            content=content,
            entries=(),
            error=None,
            body_hash=prior_hash,
            etag=etag or content.etag,
            last_modified=last_modified or content.last_modified,
            prior_hash=prior_hash,
        )
    body_hash = hashlib.sha256(content.body).hexdigest()
    if prior_hash and body_hash == prior_hash:
        return _FetchedSource(
            config=config,
            outcome=SourceOutcome.UNCHANGED,
            content=content,
            entries=(),
            error=None,
            body_hash=body_hash,
            etag=content.etag or etag,
            last_modified=content.last_modified or last_modified,
            prior_hash=prior_hash,
        )
    try:
        entries = _dedupe_entries(parse_feed(content.body, str(config.route.url)))
    except FeedParseError as exc:
        return _FetchedSource(
            config=config,
            outcome=SourceOutcome.FAILED,
            content=content,
            entries=(),
            error=str(exc),
            body_hash="",
            etag=content.etag or etag,
            last_modified=content.last_modified or last_modified,
            prior_hash=prior_hash,
        )
    return _FetchedSource(
        config=config,
        outcome=None,
        content=content,
        entries=entries,
        error=None,
        body_hash=body_hash,
        etag=content.etag or etag,
        last_modified=content.last_modified or last_modified,
        prior_hash=prior_hash,
    )


def _persist_fetched(
    store: SQLiteStore, fetched: _FetchedSource, *, now: datetime
) -> SourceIngestResult:
    """Persist one fetched source sequentially on the calling thread."""

    config = fetched.config
    if fetched.outcome == SourceOutcome.UNAVAILABLE:
        _LOGGER.info(
            "ingest_source_unavailable",
            source_id=config.id,
            reason=fetched.error,
        )
        _touch_source(store, config, now, success=False)
        return SourceIngestResult(
            source_id=config.id,
            outcome=SourceOutcome.UNAVAILABLE,
            error=fetched.error,
        )
    if fetched.outcome == SourceOutcome.FAILED:
        event = (
            "feed_parse_failed" if fetched.content is not None else "feed_fetch_failed"
        )
        _LOGGER.warning(event, source_id=config.id, error=fetched.error)
        _touch_source(store, config, now, success=False)
        _put_feed_state(
            store,
            config.id,
            etag=fetched.etag,
            last_modified=fetched.last_modified,
            content_hash=fetched.prior_hash,
            checked_at=now,
            success_at=None,
        )
        return SourceIngestResult(
            source_id=config.id, outcome=SourceOutcome.FAILED, error=fetched.error
        )
    if fetched.outcome == SourceOutcome.UNCHANGED:
        _LOGGER.info("feed_not_modified", source_id=config.id)
        _touch_source(store, config, now, success=True)
        _put_feed_state(
            store,
            config.id,
            etag=fetched.etag,
            last_modified=fetched.last_modified,
            content_hash=fetched.body_hash or fetched.prior_hash,
            checked_at=now,
            success_at=now,
        )
        return SourceIngestResult(source_id=config.id, outcome=SourceOutcome.UNCHANGED)

    articles: list[Article] = []
    evidences: list[Evidence] = []
    new_count = 0
    for entry in fetched.entries:
        article, evidence = entry_to_records(config.id, entry, now=now)
        if store.get(Article, article.id) is not None:
            continue
        store.save(article)
        store.save(evidence)
        articles.append(article)
        evidences.append(evidence)
        new_count += 1
    _touch_source(store, config, now, success=True)
    _put_feed_state(
        store,
        config.id,
        etag=fetched.etag,
        last_modified=fetched.last_modified,
        content_hash=fetched.body_hash,
        checked_at=now,
        success_at=now,
    )
    if not fetched.entries:
        _LOGGER.info("feed_empty", source_id=config.id)
        return SourceIngestResult(source_id=config.id, outcome=SourceOutcome.UNCHANGED)
    if new_count == 0:
        _LOGGER.info("feed_no_new_articles", source_id=config.id)
        return SourceIngestResult(source_id=config.id, outcome=SourceOutcome.UNCHANGED)
    _LOGGER.info(
        "ingest_source_complete",
        source_id=config.id,
        new_articles=new_count,
        total_entries=len(fetched.entries),
    )
    return SourceIngestResult(
        source_id=config.id,
        outcome=SourceOutcome.SUCCESS,
        articles=tuple(articles),
        evidences=tuple(evidences),
        new_articles=new_count,
    )


@ls.traceable(
    name="ingest-source",
    run_type="chain",
    process_inputs=redact,
    process_outputs=redact,
)
def ingest_one_source(
    config: SourceConfig,
    store: SQLiteStore,
    *,
    now: datetime,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    max_bytes: int = DEFAULT_MAX_BYTES,
    max_retries: int = DEFAULT_MAX_RETRIES,
    fetcher: Fetcher | None = None,
    sleep: Sleeper = time.sleep,
) -> SourceIngestResult:
    """Fetch, parse, normalize, and persist one source idempotently."""

    etag, last_modified, prior_hash = _get_feed_state(store, config.id)
    fetched = _fetch_and_parse_one(
        config,
        etag=etag,
        last_modified=last_modified,
        prior_hash=prior_hash,
        timeout=timeout,
        max_bytes=max_bytes,
        max_retries=max_retries,
        fetcher=fetcher,
        sleep=sleep,
    )
    return _persist_fetched(store, fetched, now=now)


@ls.traceable(
    name="ingest-sources",
    run_type="chain",
    process_inputs=redact,
    process_outputs=redact,
)
def ingest_sources(
    configs: tuple[SourceConfig, ...],
    store: SQLiteStore,
    *,
    digest_run_id: str,
    now: datetime | None = None,
    max_workers: int = DEFAULT_MAX_WORKERS,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    max_bytes: int = DEFAULT_MAX_BYTES,
    max_retries: int = DEFAULT_MAX_RETRIES,
    fetcher: Fetcher | None = None,
    sleep: Sleeper = time.sleep,
    policy_version: str = "editorial-policy-2026-09-07",
    code_version: str = "pr03-ingestion",
) -> IngestSummary:
    """Collect every source with bounded concurrency and persist a Run."""

    started = now or datetime.now(UTC)
    workers = max(1, min(max_workers, len(configs) or 1))
    # Read conditional-request state sequentially so workers never touch SQLite.
    states = {config.id: _get_feed_state(store, config.id) for config in configs}

    def _fetch_one(config: SourceConfig) -> _FetchedSource:
        etag, last_modified, prior_hash = states[config.id]
        return _fetch_and_parse_one(
            config,
            etag=etag,
            last_modified=last_modified,
            prior_hash=prior_hash,
            timeout=timeout,
            max_bytes=max_bytes,
            max_retries=max_retries,
            fetcher=fetcher,
            sleep=sleep,
        )

    # Network fetching runs concurrently; all SQLite writes happen below on
    # the calling thread so large TechNode/METR responses cannot fan out DB
    # access across threads.
    if workers == 1 or len(configs) <= 1:
        fetched_all = tuple(_fetch_one(config) for config in configs)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            fetched_all = tuple(pool.map(_fetch_one, configs))
    results = tuple(
        _persist_fetched(store, fetched, now=started) for fetched in fetched_all
    )

    successful = sum(1 for item in results if item.outcome == SourceOutcome.SUCCESS)
    unchanged = sum(1 for item in results if item.outcome == SourceOutcome.UNCHANGED)
    failed = sum(1 for item in results if item.outcome == SourceOutcome.FAILED)
    unavailable = sum(
        1 for item in results if item.outcome == SourceOutcome.UNAVAILABLE
    )
    coverage = CoverageStats(
        attempted=len(configs),
        successful=successful,
        unchanged=unchanged,
        failed=failed,
        unavailable=unavailable,
    )
    finished = datetime.now(UTC)
    status = RunStatus.HEALTHY
    if failed or unavailable:
        status = (
            RunStatus.DEGRADED
            if (successful + unchanged) >= max(1, len(configs) // 2)
            else RunStatus.FAILED
        )
    run = Run(
        id=digest_run_id,
        status=status,
        started_at=started,
        finished_at=finished,
        coverage=coverage,
        policy_version=policy_version,
        code_version=code_version,
        termination_reason=(
            f"ingestion: {successful} successful, {unchanged} unchanged, "
            f"{failed} failed, {unavailable} unavailable"
        ),
    )
    store.save(run)
    article_count = sum(len(item.articles) for item in results)
    _LOGGER.info(
        "ingest_complete",
        digest_run_id=digest_run_id,
        attempted=coverage.attempted,
        successful=successful,
        unchanged=unchanged,
        failed=failed,
        unavailable=unavailable,
        new_articles=sum(item.new_articles for item in results),
    )
    return IngestSummary(
        digest_run_id=digest_run_id,
        coverage=coverage,
        results=results,
        article_count=article_count,
        new_article_count=sum(item.new_articles for item in results),
        evidence_count=article_count,
        started_at=started,
        finished_at=finished,
    )


__all__ = [
    "DEFAULT_MAX_BYTES",
    "DEFAULT_MAX_WORKERS",
    "DEFAULT_TIMEOUT_SECONDS",
    "FeedContent",
    "FeedFetchError",
    "FeedParseError",
    "IngestSummary",
    "ParsedEntry",
    "SourceIngestResult",
    "SourceOutcome",
    "canonicalize_url",
    "ensure_public_http_url",
    "entry_to_records",
    "fetch_feed",
    "fetch_with_retries",
    "ingest_one_source",
    "ingest_sources",
    "parse_feed",
    "parse_feed_datetime",
    "stable_article_id",
    "strip_html",
]
