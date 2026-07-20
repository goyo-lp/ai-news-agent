from __future__ import annotations

import argparse
import asyncio
import logging
from datetime import datetime, timezone
from uuid import uuid4

from app.config import Settings, configure_langsmith_env, get_settings
from app.graph.state import AgentState
from app.graph.workflow import build_curation_workflow, build_workflow
from app.logging_setup import setup_logging
from app.orchestrator.schemas import CuratedArticle
from app.schemas.article import parse_articles

logger = logging.getLogger(__name__)


def _clamp_limit(limit: int | None, settings: Settings) -> int:
    """Coerce a user-supplied limit into [1, settings.max_articles_per_run].
    Shared by run_pipeline and run_curation so programmatic callers of the
    curation tool can't bypass the cap the CLI enforces."""
    resolved = limit if limit is not None else settings.max_articles_per_run
    return max(1, min(resolved, settings.max_articles_per_run))


def _initial_state(dry_run: bool, limit: int) -> AgentState:
    """Shared AgentState seed for both pipeline entry points. Centralizing it
    stops the two paths from drifting on run_id/started_at/limit shape."""
    return {
        "run_id": str(uuid4()),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "dry_run": dry_run,
        "limit": limit,
        "errors": [],
    }


def _bootstrap(
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
    effective_limit = _clamp_limit(limit, s)
    return s, effective_limit, _initial_state(dry_run=dry_run, limit=effective_limit)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AI News Agent")
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="Run the full pipeline")
    run_parser.add_argument("--dry-run", action="store_true", help="Run without calling Telegram")
    run_parser.add_argument("--limit", type=int, default=None, help="Max articles to send (<=50)")
    run_parser.add_argument("--verbose", action="store_true", help="Enable debug logs")

    return parser


async def run_pipeline(args: argparse.Namespace) -> int:
    """Run one full pipeline invocation. Exit codes: 0 success, 1 one-or-more
    delivery failures, 2 config error (checked before the graph even builds)."""
    dry_run = bool(args.dry_run)
    settings, _limit, initial_state = _bootstrap(dry_run=dry_run, limit=args.limit)

    missing_fields = settings.missing_required_runtime_fields(dry_run=dry_run)
    if missing_fields:
        joined = ", ".join(missing_fields)
        logger.error("Configuration error: missing required .env values: %s", joined)
        print(f"Configuration error: missing required .env values: {joined}")
        return 2

    workflow = build_workflow()
    final_state = await workflow.ainvoke(initial_state)

    selected_count = len(final_state.get("articles_selected", []))
    deliveries = final_state.get("delivery_results", [])
    attempted_count = len(deliveries)
    failed_count = len([item for item in deliveries if item.get("status") == "error"])
    sent_count = len([item for item in deliveries if item.get("status") in {"sent", "dry_run"}])

    logger.info(
        "Run complete | selected=%s attempted=%s sent=%s failed=%s dry_run=%s",
        selected_count,
        attempted_count,
        sent_count,
        failed_count,
        dry_run,
    )
    if failed_count:
        sample_failures = [
            f"{item.get('article_id')}: {item.get('error', 'unknown')}"
            for item in deliveries
            if item.get("status") == "error"
        ][:3]
        if sample_failures:
            logger.error("Sample delivery errors: %s", " | ".join(sample_failures))

    if final_state.get("errors"):
        logger.warning("Non-fatal errors captured: %s", len(final_state["errors"]))

    print(
        f"Run complete. selected={selected_count} attempted={attempted_count} "
        f"sent={sent_count} failed={failed_count} dry_run={dry_run}"
    )
    return 0 if failed_count == 0 else 1


async def run_curation(
    limit: int | None = None,
    settings: Settings | None = None,
) -> tuple[list[CuratedArticle], int]:
    """Run ingest -> enrich -> rank -> summarize and return the resulting
    articles as boundary contracts plus the effective clamped limit, stopping
    before deliver. Additive sibling of run_pipeline() for the orchestrator's
    fetch_curated_ai_news tool; the CLI `run` command and Telegram delivery
    are unaffected.

    `settings` is the seam the curation tool uses to inject its own resolved
    Settings instance instead of letting run_curation reach back into the
    lru_cache. Returning the effective limit is what stops the tool from
    reconstructing it from a possibly-different Settings instance — the value
    reported in the tool's summary is the value actually applied here.

    The Article -> CuratedArticle projection happens here, at the seam — that's
    what makes the boundary contract in app.orchestrator.schemas enforceable
    rather than decorative."""
    _settings, effective_limit, initial_state = _bootstrap(
        dry_run=False, limit=limit, settings=settings
    )

    workflow = build_curation_workflow()
    final_state = await workflow.ainvoke(initial_state)

    articles = parse_articles(final_state.get("articles_selected"))
    return [CuratedArticle.from_article(a) for a in articles], effective_limit


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command != "run":
        parser.print_help()
        return

    setup_logging(verbose=bool(args.verbose))
    exit_code = asyncio.run(run_pipeline(args))
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
