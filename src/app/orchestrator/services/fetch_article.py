"""fetch_article service — fetch the real article behind a URL.

Reuses the News Agent's tested HTTP plumbing (SSRF-guarded, streaming,
size-capped) plus its OpenGraph extractor, then adds a minimal body-text
cleaner so the research subagent (P4.1) gets a usable plain-text payload.
This is the "go get the real article" path the LinkedIn agent itself
lacks — it relied on Tavily's extract endpoint instead. Building on the
News Agent's own ``http_utils.get_capped`` keeps the SSRF + size guards the
plan's risk row calls out as already-solved (P2.2/P7.4 reuse).

The model returned here is orchestrator-internal — NOT a boundary contract.
The fetch tool that wraps this service writes the artifact to the filesystem
and returns only a compressed summary to the LLM (guiding principle #3);
P4.1's research subagent reads the file back. Promoting ``FetchedArticle``
into ``orchestrator/schemas.py`` is deferred until P4.1 actually depends on
its shape, so this module owns the freedom to evolve fields without breaking
a premature stable surface.
"""
from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field

from app.config import Settings
from app.services.extractor import extract_open_graph_fields
from app.services.http_utils import (
    BlockedHostError,
    ResponseTooLargeError,
    get_capped,
    reject_private_network_requests,
)
from app.services.rss_client import normalize_url

logger = logging.getLogger(__name__)

# Reuse the News Agent extractor's byte budget: same per-request cap, same
# rationale (large article pages balloon past 2MB; stop streaming before the
# body buffers in memory). Deliberately duplicated rather than imported from
# `app.services.extractor._MAX_DOWNLOAD_BYTES` — importing a `_`-prefixed
# symbol across a module boundary couples two unrelated budgets; if the
# extractor ever tunes its own, this one shouldn't silently shift.
_MAX_DOWNLOAD_BYTES = 2_000_000

# Clean body text we persist is capped to keep the orchestrator data dir small
# and the research subagent's context bounded. The full HTML is NOT persisted
# (size + scraping-disclosure risk); this plain-text slice is the artifact.
_MAX_TEXT_CHARS = 20_000

# Tags that contribute noise rather than article content. Stripped before
# `get_text()` so the persisted plain-text slice is signal, not chrome.
# `aside` matters most for AI-news scrapers (sidebar "related stories" rails);
# `form`/`iframe`/`svg` strip newsletter prompts, embeds, and inline graphics.
_NOISE_TAGS = ["script", "style", "noscript", "template", "nav", "header", "footer", "aside", "form", "iframe", "svg"]


class FetchArticleError(Exception):
    """Base for service-level fetch failures. Subclassed so the tool layer can
    translate each into a distinct summary ``status`` for the LLM."""


class BlockedArticleError(FetchArticleError):
    """SSRF guard tripped (private/loopback/metadata host, incl. a redirect hop
    that lands on one)."""


class OversizedArticleError(FetchArticleError):
    """Body exceeded the streaming byte cap before completing."""


class ArticleNotHtmlError(FetchArticleError):
    """Content-Type wasn't text/html — nothing to extract from."""


class FetchedArticle(BaseModel):
    """Artifact written to ``articles/<slug>.json`` and read back by the
    research subagent. Internal to the orchestrator; not exported as a
    boundary contract (see module docstring)."""

    url: str  # the URL the caller asked for
    final_url: str  # URL after redirects
    title: str | None = None
    description: str | None = None
    image_url: str | None = None
    text: str = Field(default="", description="Cleaned plain-text slice (≤ _MAX_TEXT_CHARS).")
    content_type: str = ""
    bytes: int = Field(
        default=0,
        description=(
            "Decoded UTF-8 body length in bytes. Only set on the ok path — "
            "the oversized-cap path raises before the FetchedArticle is built, "
            "so a partial-body bytes value is never persisted."
        ),
    )
    fetched_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    model_config = {"extra": "ignore"}


def slug_for_url(url: str) -> str:
    """Stable, filesystem-safe slug for a fetched-article file. Domain-prefixed
    for human-readability + a short sha256 suffix for collision resistance so
    two URLs on the same domain never overwrite each other."""
    host = (urlparse(url).hostname or "nohost").lower().removeprefix("www.")
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:12]
    # Allow alnum + dots (filesystem-safe) so "example.com" survives intact;
    # anything else collapses to a single dash separator. IP literals collapse
    # colons to dashes (filesystem-safe) and the 12-char sha256 still keeps
    # collisions impossible across same-host URLs.
    safe_host = "".join(c if c.isalnum() or c == "." else "-" for c in host).strip("-.") or "host"
    return f"{safe_host}-{digest}.json"


def _clean_text(html: str) -> str:
    """Strip noise tags from HTML and return a single-line plain text slice.
    BeautifulSoup's ``get_text`` is the cheap path; this is *not* a
    readability extraction (the research subagent uses Tavily/LLM for that)."""
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(_NOISE_TAGS):
        tag.decompose()
    text = " ".join(soup.get_text(separator=" ", strip=True).split())
    return text[:_MAX_TEXT_CHARS]


async def fetch_article(url: str, settings: Settings) -> FetchedArticle:
    """Fetch one article URL with the SSRF guard + streaming cap, extract
    OpenGraph + cleaned body text, and return an artifact ready to persist.

    Raises a ``FetchArticleError`` subclass on each documented failure mode so
    the tool layer (and tests) can switch on the failure type rather than
    parsing error strings. ``normalize_url`` is applied first so a URL scraped
    from RSS arrives in canonical form before the SSRF check inspects the host.
    """
    target = normalize_url(url)
    timeout = httpx.Timeout(settings.request_timeout_seconds)
    headers = {"User-Agent": settings.user_agent}

    try:
        async with httpx.AsyncClient(
            timeout=timeout,
            follow_redirects=True,
            event_hooks={"request": [reject_private_network_requests]},
        ) as client:
            response = await get_capped(client, target, headers, _MAX_DOWNLOAD_BYTES)
    except BlockedHostError as exc:
        raise BlockedArticleError(str(exc)) from exc
    except ResponseTooLargeError as exc:
        raise OversizedArticleError(str(exc)) from exc
    except httpx.HTTPError as exc:
        # Network/transport failures the agent should see as a tool-level error
        # summary rather than an exception class it has to learn.
        raise FetchArticleError(str(exc)) from exc

    content_type = response.headers.get("content-type", "")
    body_text = response.text or ""

    if "text/html" not in content_type:
        raise ArticleNotHtmlError(f"Content-Type {content_type!r} is not text/html")

    # Parse once. OpenGraph + clean text both consume the same body; a parse
    # failure here (lxml missing, malformed HTML) becomes a tool-level error
    # summary rather than an exception escaping into the agent loop.
    try:
        og_title, og_description, og_image = extract_open_graph_fields(body_text)
        text = _clean_text(body_text)
    except Exception as exc:
        # Don't leak the original lxml/bs4 exception type into the tool
        # surface — translate to FetchArticleError so the tool's catch-all
        # turns it into a `status: "error"` summary.
        raise FetchArticleError(f"HTML extraction failed: {exc}") from exc

    return FetchedArticle(
        url=target,
        final_url=normalize_url(str(response.url)),
        content_type=content_type,
        bytes=len(body_text.encode("utf-8")),
        title=og_title,
        description=og_description,
        image_url=og_image,
        text=text,
    )