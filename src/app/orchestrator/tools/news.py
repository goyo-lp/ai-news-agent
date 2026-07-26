"""fetch_curated_ai_news — the coordinator's producer tool.

Wraps the repo's own news pipeline (run_curation: ingest -> enrich -> rank ->
summarize, stopping before deliver) and writes the curated, ranked, summarized
articles into the deep agent's filesystem, returning only a compressed summary
(count + path). Per the plan's guiding principle #3, structured data lives in
the filesystem/StateBackend; a subagent's single text report is lossy, so the
full ranked-article list never rides through a subagent's context. This tool is
where that boundary is enforced for the news product.

Target environment: deepagents `create_deep_agent` (coordinator + subagents),
which invokes tools asynchronously, so the tool is exposed as an async
coroutine only. A sync entry would need to run a fresh event loop and would
refuse to nest inside the coordinator's already-running loop; rather than paper
over that with a ThreadPoolExecutor, callers should invoke the tool via
`await tool.ainvoke(...)` (which is what create_deep_agent does).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from app.config import Settings, get_settings
from app.curation import run_curation
from app.orchestrator import state
from app.orchestrator.schemas import CuratedArticle

logger = logging.getLogger(__name__)

class FetchCuratedNewsArgs(BaseModel):
    """Tool input. `limit` caps how many articles the curation pipeline keeps
    for this run; it's clamped to settings.max_articles_per_run so a model can't
    bypass the ceiling the CLI enforces."""

    limit: int | None = Field(
        default=None,
        description=(
            "Optional cap on the number of curated articles to keep this run. "
            "Clamped to MAX_ARTICLES_PER_RUN. Omit to use the configured default."
        ),
    )


def write_articles_to_state(articles: list[CuratedArticle], data_dir: str) -> Path:
    """Serialize curated articles to the orchestrator data dir as a single
    `articles.json` file, returning the written path. Creates the dir if
    missing. Pure (no network): testable with a tmp directory.

    Serializing via the boundary model — not the pipeline-internal Article — is
    what keeps the boundary contract in app.orchestrator.schemas enforceable:
    the file on disk is, by construction, exactly what crosses the boundary."""
    root = Path(data_dir)
    root.mkdir(parents=True, exist_ok=True)
    path = state.articles_path(root)
    payload = [a.model_dump(mode="json") for a in articles]
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    logger.info("Wrote %d curated articles to %s", len(articles), path)
    return path


async def _fetch_and_write(limit: int | None, settings: Settings) -> dict[str, Any]:
    """Async implementation: run the curation pipeline, then persist the
    resulting boundary articles and return a compressed summary. Centralizing
    the summary shape here is what keeps the tool's return contract stable.

    `limit_used` comes straight from run_curation — never reconstructed from
    `settings.max_articles_per_run` — so the reported value is the value
    actually applied, even if a caller injected settings that differ from the
    lru_cache."""
    articles, effective_limit = await run_curation(limit=limit, settings=settings)
    path = write_articles_to_state(articles, settings.orchestrator_data_dir)

    return {
        "count": len(articles),
        "limit_used": effective_limit,
        "path": str(path),
    }


def build_fetch_curated_ai_news_tool(
    settings: Settings | None = None,
) -> StructuredTool:
    """Construct the fetch_curated_ai_news LangChain tool.

    Settings are resolved lazily on first call when not supplied, so the tool
    picks up .env loaded after the tool was built (the common case at process
    startup). Supplying settings explicitly is the seam tests use to inject a
    fixed config without touching the lru_cache. Those same settings are
    threaded into run_curation so the limit reported in the summary is the limit
    the pipeline actually used."""
    bound_settings = settings

    async def _async(limit: int | None = None) -> str:
        s = bound_settings or get_settings()
        result = await _fetch_and_write(limit, s)
        return json.dumps(result, default=str)

    return StructuredTool.from_function(
        func=None,
        coroutine=_async,
        name="fetch_curated_ai_news",
        description=(
            "Run the AI News Agent's curation pipeline (ingest -> enrich -> rank "
            "-> summarize) and persist the curated, ranked, summarized articles "
            "to the orchestrator's data dir as articles.json. Returns a JSON "
            "summary with the article count and the file path — it does NOT "
            "return the articles themselves. Call this once per run to stock "
            "the filesystem before ranking/research/writing. Optional `limit` "
            "caps the number of articles kept and is clamped to MAX_ARTICLES_PER_RUN."
        ),
        args_schema=FetchCuratedNewsArgs,
    )


# Convenience singleton for callers that want the default-configured tool (the
# common case for `create_deep_agent(tools=[...])`). Tests build their own via
# the factory with injected settings instead of importing this.
fetch_curated_ai_news_tool = build_fetch_curated_ai_news_tool()