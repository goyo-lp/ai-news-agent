"""fetch_article — the coordinator's "go get the real article" tool.

Wraps :func:`app.orchestrator.services.fetch_article.fetch_article`. The
research subagent (P4.1) decides when the snippet/summary isn't enough and it
needs the real article; this tool is the only path it takes — never raw httpx
in subagent code — so the SSRF guard, size cap, and content-type gate stay
non-optional.

Structured data lives on the filesystem (guiding principle #3): the article
object is written to ``articles/<slug>.json`` in the orchestrator data dir,
and the tool returns a compressed summary identifying the artifact. The
summary's ``status`` field lets the agent reason about failure modes
(blocked / oversized / not-html / error) without a tool call raising in its
face — a per-URL fetch failure is a normal research outcome, not a run-level
abort.

Target environment: deepagents ``create_deep_agent``, which invokes tools via
``await tool.ainvoke(...)``; the tool is exposed async-only (see news.py).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from app.config import Settings, get_settings
from app.orchestrator.budgets import ResultMemo
from app.orchestrator.services.fetch_article import (
    ArticleNotHtmlError,
    BlockedArticleError,
    FetchedArticle,
    FetchArticleError,
    OversizedArticleError,
    fetch_article,
    normalize_url,
    slug_for_url,
)

logger = logging.getLogger(__name__)

_ARTICLES_SUBDIR = "articles"


def _empty_summary(url: str, status: str, reason: str) -> dict[str, Any]:
    """Failure-branch summary. Same shape as the ok branch with the artifact
    fields nulled so an LLM switching on ``status`` can read ``result["path"]``
    without KeyError — and detect 'no artifact' via ``status != "ok"`` or a
    None path rather than the absent key."""
    return {
        "url": normalize_url(url),
        "final_url": None,
        "title": None,
        "status": status,
        "reason": reason,
        "path": None,
    }


class FetchArticleArgs(BaseModel):
    """Tool input. ``url`` is required — there is no 'best guess' default; the
    agent explicitly decides which article to fetch."""

    url: str = Field(
        ...,
        description=(
            "Absolute http(s) URL of the article to fetch. SSRF guard blocks "
            "private/loopback/metadata hosts, including redirect hops to them."
        ),
    )


def write_fetched_to_state(article: FetchedArticle, data_dir: str) -> Path:
    """Serialize a fetched article to ``<data_dir>/articles/<slug>.json`` and
    return the written path. Creates the dir if missing. Pure (no network):
    testable with a tmp directory. Mirrors ``write_articles_to_state`` so the
    on-disk contract is JSON-shaped and round-trips through ``FetchedArticle``."""
    root = Path(data_dir) / _ARTICLES_SUBDIR
    root.mkdir(parents=True, exist_ok=True)
    path = root / slug_for_url(article.url)
    path.write_text(
        json.dumps(article.model_dump(mode="json"), indent=2, default=str),
        encoding="utf-8",
    )
    logger.info("Wrote fetched article for %s to %s", article.url, path)
    return path


async def _fetch_and_write(url: str, settings: Settings) -> dict[str, Any]:
    """Fetch the article and persist it, translating service-level failures
    into a structured ``status`` field the LLM can branch on. The compressed
    summary caries the URL + a short failure reason — never the article body.

    A catch-all ``Exception`` returns ``status="error"`` so a truly unexpected
    failure (lxml import error, aio runtime hiccup) becomes a normal research
    outcome rather than an exception escaping into the agent loop — the
    documented contract of this tool."""
    try:
        article = await fetch_article(url, settings)
    except BlockedArticleError as exc:
        logger.warning("fetch_article blocked for %s: %s", url, exc)
        return _empty_summary(url, "blocked", str(exc))
    except OversizedArticleError as exc:
        logger.warning("fetch_article oversized for %s: %s", url, exc)
        return _empty_summary(url, "oversized", str(exc))
    except ArticleNotHtmlError as exc:
        logger.warning("fetch_article not_html for %s: %s", url, exc)
        return _empty_summary(url, "not_html", str(exc))
    except FetchArticleError as exc:
        logger.warning("fetch_article failed for %s: %s", url, exc)
        return _empty_summary(url, "error", str(exc))
    except Exception as exc:
        # M1: any non-FetchArticleError exception (e.g. _clean_text raising
        # something the typed branches don't catch) still produces a normal
        # error summary rather than propagating into the agent loop.
        logger.warning("fetch_article unexpected error for %s: %s", url, exc)
        return _empty_summary(url, "error", f"{type(exc).__name__}: {exc}")

    path = write_fetched_to_state(article, settings.orchestrator_data_dir)
    return {
        "url": article.url,
        "final_url": article.final_url,
        "title": article.title,
        "status": "ok",
        "reason": None,
        "path": str(path),
    }


def build_fetch_article_tool(settings: Settings | None = None) -> StructuredTool:
    """Construct the fetch_article LangChain tool. Settings are resolved lazily
    on first call when not supplied (mirrors the news tool).

    The memo is created here and closed over, so its lifetime is the lifetime of
    this tool — one per run, shared across every topic in the research fan-out
    (the spine builds one researcher for the whole run), nothing to reset."""
    bound_settings = settings
    memo = ResultMemo()

    async def _async(url: str) -> str:
        s = bound_settings or get_settings()
        # Key on the normalized URL so the trailing-slash and doi.org-vs-final
        # variants the researcher tries collapse onto one entry.
        key = normalize_url(url)
        replayed = memo.replay(key)
        if replayed is not None:
            logger.info("fetch_article memo hit for %s", key)
            return json.dumps(replayed, default=str)
        result = await _fetch_and_write(url, s)
        memo.put(key, result)
        return json.dumps(result, default=str)

    return StructuredTool.from_function(
        func=None,
        coroutine=_async,
        name="fetch_article",
        description=(
            "Fetch the real article behind a URL — SSRF-guarded, streaming "
            "size-capped — extract its OpenGraph metadata + cleaned body text, "
            "and write the artifact to the orchestrator data dir. Returns a "
            "JSON summary with fields {url, final_url, title, status, reason, "
            "path} in every branch (failure branches null final_url/title/path "
            "and set `reason`). The fetched content is NOT returned; read it "
            "from `path` when `status == \"ok\"`. `status` is one of "
            "ok | blocked | oversized | not_html | error."
        ),
        args_schema=FetchArticleArgs,
    )


fetch_article_tool = build_fetch_article_tool()