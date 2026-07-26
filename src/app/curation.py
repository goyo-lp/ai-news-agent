"""The curation pipeline's programmatic entry points.

``run_curation`` runs ingest → enrich → rank → summarize and stops before
deliver, projecting the result onto the boundary contract in
:mod:`app.orchestrator.schemas`. It lives here rather than in :mod:`app.main`
because both the CLI and the orchestrator's ``fetch_curated_ai_news`` tool need
it: with it in the entry module, ``tools/news`` had to import *up* into the CLI,
and the CLI then needed lazy imports to break the resulting cycle. Everything in
this module imports downward only.

``bootstrap_run`` is the shared setup both pipeline entry points perform, kept
here so the two paths cannot drift on run_id / started_at / limit shape.
"""
from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from app.config import Settings, configure_langsmith_env, get_settings
from app.graph.state import AgentState
from app.graph.workflow import build_curation_workflow
from app.orchestrator.schemas import CuratedArticle
from app.schemas.article import parse_articles


def clamp_limit(limit: int | None, settings: Settings) -> int:
    """Coerce a user-supplied limit into [1, settings.max_articles_per_run].
    Shared by run_pipeline and run_curation so programmatic callers of the
    curation tool can't bypass the cap the CLI enforces."""
    resolved = limit if limit is not None else settings.max_articles_per_run
    return max(1, min(resolved, settings.max_articles_per_run))


def initial_state(dry_run: bool, limit: int) -> AgentState:
    """Shared AgentState seed for both pipeline entry points. Centralizing it
    stops the two paths from drifting on run_id/started_at/limit shape."""
    return {
        "run_id": str(uuid4()),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "dry_run": dry_run,
        "limit": limit,
        "errors": [],
    }


def bootstrap_run(
    *,
    dry_run: bool,
    limit: int | None,
    settings: Settings | None = None,
) -> tuple[Settings, int, AgentState]:
    """Shared setup for both pipeline entry points: resolve settings, mirror
    LangSmith env, clamp the limit, and seed the AgentState. Returns the
    effective clamped limit alongside the state so callers (notably the
    curation tool) read the value actually used instead of reconstructing it
    from a possibly-different Settings instance."""
    s = settings if settings is not None else get_settings()
    configure_langsmith_env(s)
    effective_limit = clamp_limit(limit, s)
    return s, effective_limit, initial_state(dry_run=dry_run, limit=effective_limit)


async def run_curation(
    limit: int | None = None,
    settings: Settings | None = None,
) -> tuple[list[CuratedArticle], int]:
    """Run ingest -> enrich -> rank -> summarize and return the resulting
    articles as boundary contracts plus the effective clamped limit, stopping
    before deliver.

    `settings` is the seam the curation tool uses to inject its own resolved
    Settings instance instead of letting run_curation reach back into the
    lru_cache. Returning the effective limit is what stops the tool from
    reconstructing it from a possibly-different Settings instance — the value
    reported in the tool's summary is the value actually applied here.

    The Article -> CuratedArticle projection happens here, at the seam — that's
    what makes the boundary contract in app.orchestrator.schemas enforceable
    rather than decorative."""
    _settings, effective_limit, seed = bootstrap_run(
        dry_run=False, limit=limit, settings=settings
    )

    workflow = build_curation_workflow()
    final_state = await workflow.ainvoke(seed)

    articles = parse_articles(final_state.get("articles_selected"))
    return [CuratedArticle.from_article(a) for a in articles], effective_limit


__all__ = ["bootstrap_run", "clamp_limit", "initial_state", "run_curation"]
