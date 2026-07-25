from __future__ import annotations

import argparse
import asyncio
import json
import logging
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx

from app.config import Settings, configure_langsmith_env, get_settings
from app.graph.state import AgentState
from app.graph.workflow import build_curation_workflow, build_workflow
from app.logging_setup import setup_logging
from app.orchestrator import state
from app.orchestrator.schemas import CuratedArticle
from app.orchestrator.tools.export import build_export_report_tool
from app.orchestrator.tools.telegram import build_deliver_telegram_tool
from app.orchestrator.tracing import coordinator_run_config
from app.orchestrator.usage import format_run_usage, track_run_usage
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

    propose_parser = subparsers.add_parser(
        "propose",
        help="Drive the coordinator deep agent to produce LinkedIn post proposals (writes a file bundle)",
    )
    propose_parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite today's existing export bundle instead of refusing",
    )
    propose_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Produce + export proposals but do NOT send them to the LinkedIn Telegram bot",
    )
    propose_parser.add_argument("--verbose", action="store_true", help="Enable debug logs")

    both_parser = subparsers.add_parser(
        "both",
        help="Run the news digest, then produce + deliver LinkedIn proposals",
    )
    both_parser.add_argument("--dry-run", action="store_true", help="Run without calling Telegram")
    both_parser.add_argument("--limit", type=int, default=None, help="Max articles to send (<=50)")
    both_parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite today's existing export bundle instead of refusing",
    )
    both_parser.add_argument("--verbose", action="store_true", help="Enable debug logs")

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


_SEARXNG_COMPOSE_FILE = Path(__file__).resolve().parents[2] / "docker-compose.searxng.yml"
_SEARXNG_DEFAULT_URL = "http://localhost:8080"


def _ensure_searxng(settings: Settings) -> None:
    """Best-effort auto-start of the self-hosted SearXNG instance
    (docker-compose.searxng.yml) so propose's web_search fallback does real
    corroborating searches instead of returning mock results. Only kicks in
    when SEARXNG_BASE_URL is unset — an operator already pointing it at their
    own instance is left alone. Any failure here (no docker, container never
    becomes healthy) just logs a warning; web_search's existing mock-result
    mode is the fallback, not a hard error."""
    if (settings.searxng_base_url or "").strip():
        return

    try:
        subprocess.run(
            ["docker", "compose", "-f", str(_SEARXNG_COMPOSE_FILE), "up", "-d"],
            check=True,
            capture_output=True,
            timeout=60,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        logger.warning("Could not start SearXNG (%s) — web_search will use mock results.", exc)
        return

    with httpx.Client(timeout=2) as client:
        for _ in range(15):
            try:
                if client.get(_SEARXNG_DEFAULT_URL).status_code == 200:
                    settings.searxng_base_url = _SEARXNG_DEFAULT_URL
                    logger.info("SearXNG ready at %s", _SEARXNG_DEFAULT_URL)
                    return
            except httpx.HTTPError:
                pass
            time.sleep(1)

    logger.warning(
        "SearXNG container started but didn't become healthy in time — web_search will use mock results."
    )


def _clear_run_workspace(settings: Settings) -> None:
    """Remove the previous run's regenerated artifacts under
    ``orchestrator_data_dir`` so this ``propose`` run — and its delivery — only
    ever sees its own output. ``drafts/`` is a persistent dir; without this it
    accumulates across days and the delivery loop re-sends prior runs' proposals
    (duplicate posts). Inputs that live outside this dir (style profile, skills,
    delivery history) are untouched."""
    data_dir = Path(settings.orchestrator_data_dir)
    for filename in (state.ARTICLES_FILENAME, state.TOPICS_FILENAME):
        (data_dir / filename).unlink(missing_ok=True)
    for subdir in (state.BRIEFS_DIRNAME, state.DRAFTS_DIRNAME):
        shutil.rmtree(data_dir / subdir, ignore_errors=True)


async def _deliver_proposals(settings: Settings) -> list[dict[str, Any]]:
    """Send every passing draft under ``drafts/`` to the LinkedIn Telegram bot
    via the ``deliver_telegram`` tool. The tool re-verifies each draft's gate
    verdict and refuses any it didn't certify, hard-codes ``bot="linkedin"``, and
    auto-dry-runs when the linkedin profile is unconfigured — so this loop can
    stay dumb: deliver every draft, let the tool gate + route. Returns one
    compressed result dict per draft."""
    drafts_dir = Path(settings.orchestrator_data_dir) / "drafts"
    if not drafts_dir.exists():
        return []
    tool = build_deliver_telegram_tool(settings)
    results: list[dict[str, Any]] = []
    for draft in sorted(drafts_dir.glob("*.json")):
        if draft.name.endswith(".gate.json"):
            continue
        post_id = draft.name[: -len(".json")]
        results.append(json.loads(await tool.ainvoke({"post_id": post_id})))
    return results


async def run_propose(args: argparse.Namespace) -> int:
    """Drive the coordinator deep agent for one run, deliver the passing
    proposals to the LinkedIn Telegram bot, and export the run bundle.

    This is the LinkedIn-proposal path (Decision H: manual CLI). The coordinator
    planner + its research / writer subagents call OpenRouter, so a real key is
    required — the run is gated on it (exit 2) rather than failing mid-flight
    with a 401. The run is wrapped in the P7.2 usage tracker + tagged with the
    P7.1 trace config.

    The per-run workspace is cleared first so delivery only ever sees this run's
    drafts. ``export_report`` is the run's commit record: only when it succeeds
    are the passing drafts sent to the ``linkedin`` bot (Decision C). A same-day
    re-run without ``--force`` therefore refuses AND doesn't re-post; ``--dry-run``
    skips the Telegram send (the LLM still ran and spent).

    Exit codes: 0 success, 1 export refused/failed (nothing delivered),
    2 config error.
    """
    settings = get_settings()
    configure_langsmith_env(settings)

    if not (settings.openrouter_api_key or "").strip():
        logger.error(
            "Configuration error: propose requires OPENROUTER_API_KEY — the "
            "coordinator planner and its subagents call OpenRouter."
        )
        print("Configuration error: missing required .env value: OPENROUTER_API_KEY")
        return 2

    _ensure_searxng(settings)

    # Run-scope the workspace: drafts/ is persistent, so without this a new run's
    # delivery would re-send prior runs' proposals (duplicate posts).
    _clear_run_workspace(settings)

    run_id = str(uuid4())
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    kickoff = (
        f"Today is {today}. Produce up to {settings.max_topics_per_run} "
        "LinkedIn post proposals per the system prompt."
    )

    # Imported lazily to break a cycle: the coordinator pulls in the news tool,
    # which imports run_curation back from this module. The deeper issue is a
    # layering inversion — run_curation is domain logic living in the CLI
    # entrypoint; the clean fix (moving it to a non-entry module so tools/news
    # imports down, not up) belongs to the P8.4 cleanup, not this cutover PR.
    from app.orchestrator.agent import build_coordinator_agent

    agent = build_coordinator_agent(settings)
    with track_run_usage(run_id=run_id, dry_run=False) as usage:
        await agent.ainvoke(
            {"messages": [{"role": "user", "content": kickoff}]},
            coordinator_run_config(run_id=run_id, dry_run=False),
        )

    export_tool = build_export_report_tool(settings)
    export_result = json.loads(await export_tool.ainvoke({"overwrite": bool(args.force)}))

    usage_summary = format_run_usage(usage)
    logger.info("Run usage | %s", usage_summary)
    print(usage_summary)

    # Export is the run's commit record. If it refuses (a same-day bundle
    # already exists and --force wasn't passed) or fails, do NOT deliver — that
    # keeps a same-day re-run from re-posting proposals to the linkedin chat.
    if export_result.get("status") != "ok":
        if export_result.get("reason") == "overwrite_required":
            print(
                f"Today's proposals already exist at {export_result.get('bundle_dir')} "
                "— not re-sending. Re-run with --force to regenerate + re-deliver."
            )
        else:
            print(f"Run complete but export failed: {export_result.get('reason')} — not delivered.")
        return 1

    # Committed run → deliver passing proposals to the LinkedIn bot (Decision C).
    # Done here in the CLI, not by the coordinator, per its DELIVERY contract
    # ("delivery is a separate layer"). --dry-run skips the send.
    if args.dry_run:
        print("(--dry-run: skipped LinkedIn Telegram delivery)")
    else:
        delivered = await _deliver_proposals(settings)
        sent = sum(1 for d in delivered if d.get("status") == "sent")
        dry = sum(1 for d in delivered if d.get("status") == "dry_run")
        print(
            f"Delivered {sent}/{len(delivered)} proposal(s) to the linkedin bot"
            + (f" ({dry} dry-run: linkedin profile unconfigured)" if dry else "")
        )
        for item in delivered:
            if item.get("status") not in {"sent", "dry_run"}:
                print(f"  - {item.get('post_id')}: {item.get('status')} ({item.get('reason')})")

    print(
        f"Proposals exported to {export_result.get('bundle_dir')} "
        f"| counts={export_result.get('counts')}"
    )
    return 0


async def run_both(args: argparse.Namespace) -> int:
    """Run both delivery lanes back to back: the news digest to the `news` bot,
    then the LinkedIn proposal flow to the `linkedin` bot. Each lane keeps its
    own exit-code contract; this returns the worse of the two (0 only if both
    succeed) so `both` fails loudly if either lane does.
    """
    print("=== news digest ===")
    run_exit = await run_pipeline(args)
    print("=== LinkedIn proposals ===")
    propose_exit = await run_propose(args)
    return max(run_exit, propose_exit)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "run":
        setup_logging(verbose=bool(args.verbose))
        raise SystemExit(asyncio.run(run_pipeline(args)))
    if args.command == "propose":
        setup_logging(verbose=bool(args.verbose))
        raise SystemExit(asyncio.run(run_propose(args)))
    if args.command == "both":
        setup_logging(verbose=bool(args.verbose))
        raise SystemExit(asyncio.run(run_both(args)))

    parser.print_help()


if __name__ == "__main__":
    main()
