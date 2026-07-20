"""technical_rank — the coordinator's deterministic ranker tool.

Reads the curated-articles file produced by :mod:`app.orchestrator.tools.news`
(``articles.json``), re-clusters the articles, scores each cluster for
technical implementation value (with optional Stage-A LLM assessment), and
writes the top-N ranked topics to ``topics.json`` in the orchestrator data dir.
Per the plan's guiding principle #3, the structured topics live on disk; the
tool returns only a compressed summary (count + path), never the topics.

The LLM Stage-A path is optional and runs only when ``OPENROUTER_API_KEY`` is
configured and ``dry_run`` is false; otherwise the heuristic fallback produces
fully deterministic assessments — that's the path P2.1's parity tests pin.

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
from app.orchestrator.schemas import CuratedArticle, TopicCandidate
from app.orchestrator.services.ranking import (
    SourcePolicy,
    curated_articles_to_items,
    rank_topics,
)
from app.orchestrator.services.technical_ranker import TechnicalRanker
from app.orchestrator.tools.news import ARTICLES_FILENAME

logger = logging.getLogger(__name__)

_TOPICS_FILENAME = "topics.json"


class TechnicalRankArgs(BaseModel):
    """Tool input. ``limit`` caps how many ranked topics are written; clamped to
    ``MAX_TOPICS_PER_RUN`` so a model can't bypass the configured ceiling
    (parity with the news tool's ``limit`` clamping)."""

    limit: int | None = Field(
        default=None,
        description=(
            "Optional cap on the number of ranked topics written to topics.json. "
            "Clamped to MAX_TOPICS_PER_RUN. Omit to use the configured default."
        ),
    )


def _read_articles_from_state(data_dir: str) -> list[CuratedArticle]:
    """Load the curated articles written by fetch_curated_ai_news. Raises
    FileNotFoundError (deliberately) so the coordinator surfaces a missing-news
    precondition as a real error rather than treating it as 'zero topics'."""
    path = Path(data_dir) / ARTICLES_FILENAME
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [CuratedArticle.model_validate(item) for item in payload]


def write_topics_to_state(topics: list[TopicCandidate], data_dir: str) -> Path:
    """Serialize ranked topics to ``topics.json`` in the orchestrator data dir
    and return the written path. Creates the dir if missing. Pure (no network):
    testable with a tmp directory. Mirrors ``write_articles_to_state`` so the
    on-disk contract for both tools is identical in shape."""
    root = Path(data_dir)
    root.mkdir(parents=True, exist_ok=True)
    path = root / _TOPICS_FILENAME
    path.write_text(
        json.dumps([t.model_dump(mode="json") for t in topics], indent=2, default=str),
        encoding="utf-8",
    )
    logger.info("Wrote %d ranked topics to %s", len(topics), path)
    return path


def _clamp(limit: int | None, cap: int) -> int:
    """Coerce a caller-supplied topic cap. ``None`` -> configured default;
    ``0`` / negatives -> ``0`` (matches ``rank_topics``' own ``limit <= 0``
    short-circuit so the two layers agree — a model asking for zero topics
    gets zero, not one)."""
    resolved = limit if limit is not None else cap
    return max(0, min(resolved, cap))


async def _rank_and_write(limit: int | None, settings: Settings) -> dict[str, Any]:
    """Run the deterministic ranker over the curated articles file and persist
    the ranked topics. Returns the compressed summary that rides back to the
    LLM — the article/topic payloads stay on the filesystem."""
    articles = _read_articles_from_state(settings.orchestrator_data_dir)
    cap = settings.max_topics_per_run
    limit_used = _clamp(limit, cap)

    policy = SourcePolicy.permissive()
    items = curated_articles_to_items(articles, policy)

    ranker = TechnicalRanker(settings)
    # Dry-run iff the Stage-A path can't run (no API key). This mirrors the
    # heuristic fallback inside `assess_many` and keeps the tool deterministic
    # out-of-the-box — the path P2.1's parity test pins.
    dry_run = not (settings.openrouter_api_key or "").strip()
    assessments = await ranker.assess_many(items, dry_run=dry_run)

    technical_overrides = {k: v.technical_depth for k, v in assessments.items()}
    implementation_overrides = {k: v.implementation_specificity for k, v in assessments.items()}
    hype_overrides = {k: v.hype_score for k, v in assessments.items()}

    topics = rank_topics(
        items,
        limit=limit_used,
        policy=policy,
        technical_overrides=technical_overrides,
        implementation_overrides=implementation_overrides,
        hype_overrides=hype_overrides,
    )
    path = write_topics_to_state(topics, settings.orchestrator_data_dir)
    return {
        "count": len(topics),
        "limit_used": limit_used,
        "path": str(path),
    }


def build_technical_rank_tool(settings: Settings | None = None) -> StructuredTool:
    """Construct the technical_rank LangChain tool.

    Settings are resolved lazily on first call when not supplied (mirrors the
    news tool); passing them explicitly is the seam tests use to inject a fixed
    config without touching the lru_cache."""
    bound_settings = settings

    async def _async(limit: int | None = None) -> str:
        s = bound_settings or get_settings()
        result = await _rank_and_write(limit, s)
        return json.dumps(result, default=str)

    return StructuredTool.from_function(
        func=None,
        coroutine=_async,
        name="technical_rank",
        description=(
            "Read the curated articles file (articles.json produced by "
            "fetch_curated_ai_news), re-cluster the articles, score each cluster "
            "for technical implementation value (optional Stage-A LLM assessment, "
            "heuristic fallback), demote hype- and business-only stories, and "
            "write the top-N ranked topics to topics.json. Returns a JSON summary "
            "with the topic count and the file path — it does NOT return the "
            "topics themselves. Call this after fetch_curated_ai_news. Optional "
            "`limit` caps the topics written and is clamped to MAX_TOPICS_PER_RUN."
        ),
        args_schema=TechnicalRankArgs,
    )


technical_rank_tool = build_technical_rank_tool()