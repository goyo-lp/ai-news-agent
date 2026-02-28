from __future__ import annotations

import argparse
import asyncio
import logging
from datetime import datetime, timezone
from uuid import uuid4

from app.config import configure_langsmith_env, get_settings
from app.graph.state import AgentState
from app.graph.workflow import build_workflow
from app.logging import setup_logging

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AI News Agent")
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="Run the full pipeline")
    run_parser.add_argument("--dry-run", action="store_true", help="Run without calling Telegram")
    run_parser.add_argument("--limit", type=int, default=None, help="Max articles to send (<=50)")
    run_parser.add_argument("--verbose", action="store_true", help="Enable debug logs")

    return parser


async def run_pipeline(args: argparse.Namespace) -> int:
    settings = get_settings()
    configure_langsmith_env(settings)
    dry_run = bool(args.dry_run)

    missing_fields = settings.missing_required_runtime_fields(dry_run=dry_run)
    if missing_fields:
        joined = ", ".join(missing_fields)
        logger.error("Configuration error: missing required .env values: %s", joined)
        print(f"Configuration error: missing required .env values: {joined}")
        return 2

    limit = args.limit if args.limit is not None else settings.max_articles_per_run
    limit = max(1, min(limit, settings.max_articles_per_run))

    initial_state: AgentState = {
        "run_id": str(uuid4()),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "dry_run": dry_run,
        "limit": limit,
        "errors": [],
    }

    workflow = build_workflow()
    final_state = await workflow.ainvoke(initial_state)

    selected_count = len(final_state.get("articles_top20", []))
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
