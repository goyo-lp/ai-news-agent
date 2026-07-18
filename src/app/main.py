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
    settings = get_settings()
    configure_langsmith_env(settings)
    dry_run = bool(args.dry_run)

    missing_fields = settings.missing_required_runtime_fields(dry_run=dry_run)
    if missing_fields:
        joined = ", ".join(missing_fields)
        logger.error("Configuration error: missing required .env values: %s", joined)
        print(f"Configuration error: missing required .env values: {joined}")
        return 2

    limit = _clamp_limit(args.limit, settings)
    initial_state = _initial_state(dry_run=dry_run, limit=limit)

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


async def run_curation(limit: int | None = None) -> list[CuratedArticle]:
    """Run ingest -> enrich -> rank -> summarize and return the resulting
    articles as boundary contracts, stopping before deliver. Additive sibling
    of run_pipeline() for the orchestrator's fetch_curated_ai_news tool; the
    CLI `run` command and Telegram delivery are unaffected.

    The Article -> CuratedArticle projection happens here, at the seam — that's
    what makes the boundary contract in app.orchestrator.schemas enforceable
    rather than decorative."""
    settings = get_settings()
    configure_langsmith_env(settings)
    limit = _clamp_limit(limit, settings)
    initial_state = _initial_state(dry_run=False, limit=limit)

    workflow = build_curation_workflow()
    final_state = await workflow.ainvoke(initial_state)

    articles = parse_articles(final_state.get("articles_selected"))
    return [CuratedArticle.from_article(a) for a in articles]


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
