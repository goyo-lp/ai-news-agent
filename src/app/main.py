"""CLI entry point: parse arguments, dispatch to a lane, report the outcome.

Each lane's domain logic lives elsewhere — curation in :mod:`app.curation`, the
propose pipeline in :mod:`app.orchestrator.spine`, and the surrounding run
lifecycle in :mod:`app.runtime`. What stays here is the argument surface, the
ordering decisions that are policy (notably "export commits the run, and only a
committed run delivers"), and the operator-facing output.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
from dataclasses import dataclass
from uuid import uuid4

from app.config import Settings, configure_langsmith_env, get_settings
from app.curation import bootstrap_run
from app.graph.workflow import build_workflow
from app.logging_setup import setup_logging
from app.orchestrator.schemas import CuratedArticle
from app.orchestrator.spine import SpineResult, run_propose_spine
from app.orchestrator.tools.export import build_export_report_tool
from app.orchestrator.usage import format_run_usage, track_run_usage
from app.schemas.article import parse_articles
from app.runtime import searxng
from app.runtime.delivery import deliver_proposals
from app.runtime.workspace import archive_previous_run, ensure_style_profile

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RunOptions:
    """Inputs to the news digest lane."""

    dry_run: bool = False
    limit: int | None = None


@dataclass(frozen=True)
class ProposeOptions:
    """Inputs to the LinkedIn proposal lane.

    Deliberately has no ``limit``: the digest's ``--limit`` caps how many
    articles get *sent to the news bot* and has no meaning for proposal
    production, so `both` does not thread it across. Proposal volume is governed
    by ``max_topics_per_run``.
    """

    force: bool = False
    skip_delivery: bool = False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AI News Agent")
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="Run the full pipeline")
    run_parser.add_argument("--dry-run", action="store_true", help="Run without calling Telegram")
    run_parser.add_argument("--limit", type=int, default=None, help="Max articles to send (<=50)")
    run_parser.add_argument("--verbose", action="store_true", help="Enable debug logs")

    propose_parser = subparsers.add_parser(
        "propose",
        help=(
            "Produce LinkedIn post proposals via the deterministic spine "
            "(fetch→rank→veto→research→write) and export a file bundle"
        ),
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
    both_parser.add_argument(
        "--limit", type=int, default=None, help="Max articles to send in the digest (<=50)"
    )
    both_parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite today's existing export bundle instead of refusing",
    )
    both_parser.add_argument("--verbose", action="store_true", help="Enable debug logs")

    return parser


async def run_pipeline(options: RunOptions) -> tuple[int, list[CuratedArticle]]:
    """Run one full digest invocation.

    Returns the exit code alongside the curated articles the run produced, so
    ``both`` can hand them to the proposal lane instead of curating twice.
    Exit codes: 0 success, 1 a rank failure or one-or-more delivery failures,
    2 config error (checked before the graph even builds).
    """
    settings, _limit, seed = bootstrap_run(dry_run=options.dry_run, limit=options.limit)

    missing_fields = settings.missing_required_runtime_fields(dry_run=options.dry_run)
    if missing_fields:
        joined = ", ".join(missing_fields)
        logger.error("Configuration error: missing required .env values: %s", joined)
        print(f"Configuration error: missing required .env values: {joined}")
        return 2, []

    workflow = build_workflow()
    final_state = await workflow.ainvoke(seed)

    # A rank that *failed* never sets articles_selected; a rank that legitimately
    # matched nothing sets it to []. Both used to print "selected=0" and exit 0,
    # which is how a 45s timeout in rank passed for "no AI news today" on
    # 2026-07-26. They are different outcomes and must not report the same.
    if "articles_selected" not in final_state:
        logger.error(
            "Rank produced no result (node failed or timed out) — nothing was "
            "delivered. This is a pipeline failure, not an empty news day."
        )
        print("Run failed: the rank stage produced no result; nothing delivered.")
        return 1, []

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
        options.dry_run,
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
        f"sent={sent_count} failed={failed_count} dry_run={options.dry_run}"
    )
    curated = [
        CuratedArticle.from_article(a)
        for a in parse_articles(final_state.get("articles_selected"))
    ]
    return (0 if failed_count == 0 else 1), curated


def _print_spine_summary(result: SpineResult) -> None:
    """The spine's honesty log: what was selected, vetoed, floored out, and
    gated — the numbers the run_report then corroborates."""
    print(
        "Spine: "
        f"selected={len(result.selected)} "
        f"vetoed={len(result.vetoed)} "
        f"floored_out={len(result.skipped_floor)} "
        f"drafts_gated={result.drafts_passed}/{len(result.written)} "
        f"status={result.status}"
    )
    for skipped in result.skipped_floor:
        print(f"  - floored out {skipped.topic_id}: {skipped.reason}")
    for post in result.written:
        if not post.gate_passed:
            print(f"  - {post.post_id}: writer={post.writer_status} gate=not passed")


async def _deliver_and_report(settings: Settings) -> None:
    delivered = await deliver_proposals(settings)
    sent = sum(1 for d in delivered if d.get("status") == "sent")
    dry = sum(1 for d in delivered if d.get("status") == "dry_run")
    print(
        f"Delivered {sent}/{len(delivered)} proposal(s) to the linkedin bot"
        + (f" ({dry} dry-run: linkedin profile unconfigured)" if dry else "")
    )
    for item in delivered:
        if item.get("status") not in {"sent", "dry_run"}:
            print(f"  - {item.get('post_id')}: {item.get('status')} ({item.get('reason')})")


async def run_propose(
    options: ProposeOptions,
    prefetched: list[CuratedArticle] | None = None,
) -> int:
    """Produce LinkedIn post proposals for one run, export the bundle, and
    deliver the passing ones to the LinkedIn Telegram bot.

    ``prefetched`` is the ``both`` seam: the digest lane has already run the
    identical ingest→enrich→rank→summarize pipeline, so its output is reused
    instead of curating a second time. See :func:`run_both`.

    The research/writer subagents call OpenRouter, so a real key is required —
    the run is gated on it (exit 2) rather than failing mid-flight with a 401.

    The per-run workspace is archived first so delivery only ever sees this
    run's drafts. ``export_report`` is the run's commit record: only when it
    succeeds are the passing drafts sent to the ``linkedin`` bot (Decision C).
    A same-day re-run without ``force`` therefore refuses AND doesn't re-post;
    ``skip_delivery`` skips the Telegram send (the LLM still ran and spent).

    Exit codes: 0 success, 1 export refused/failed (nothing delivered),
    2 config error.
    """
    settings = get_settings()
    configure_langsmith_env(settings)

    if not (settings.openrouter_api_key or "").strip():
        logger.error(
            "Configuration error: propose requires OPENROUTER_API_KEY — the "
            "research and writer subagents call OpenRouter."
        )
        print("Configuration error: missing required .env value: OPENROUTER_API_KEY")
        return 2

    await searxng.ensure_available(settings)
    # Run-scope the workspace: drafts/ is persistent, so without this a new
    # run's delivery would re-send prior runs' proposals (duplicate posts).
    archive_previous_run(settings)
    # Voice input: style_profile.json must exist before any writer runs.
    await ensure_style_profile(settings)

    run_id = str(uuid4())
    with track_run_usage(run_id=run_id, dry_run=False) as usage:
        spine_summary = await run_propose_spine(
            settings, run_id=run_id, prefetched=prefetched
        )

    export_tool = build_export_report_tool(settings)
    export_result = json.loads(await export_tool.ainvoke({"overwrite": options.force}))

    usage_summary = format_run_usage(usage)
    logger.info("Run usage | %s", usage_summary)
    print(usage_summary)
    _print_spine_summary(spine_summary)

    # Export is the run's commit record. If it refuses (a same-day bundle
    # already exists and force wasn't passed) or fails, do NOT deliver — that
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
    # Done here, not by the spine, per the DELIVERY contract ("delivery is a
    # separate layer").
    if options.skip_delivery:
        print("(--dry-run: skipped LinkedIn Telegram delivery)")
    else:
        await _deliver_and_report(settings)

    print(
        f"Proposals exported to {export_result.get('bundle_dir')} "
        f"| counts={export_result.get('counts')}"
    )
    return 0


async def run_both(run_options: RunOptions, propose_options: ProposeOptions) -> int:
    """Run both delivery lanes back to back: the news digest to the `news` bot,
    then the LinkedIn proposal flow to the `linkedin` bot. Each lane keeps its
    own exit-code contract; this returns the worse of the two (0 only if both
    succeed) so `both` fails loudly if either lane does.

    The two lanes run the same ingest→enrich→rank→summarize pipeline, so the
    digest's result is handed to the proposal lane rather than recomputed. Two
    passes cost ~2min of duplicated feed fetching and — because the second pass
    re-hits every feed ~90s after the first — were the direct cause of the
    Reddit/HuggingFace/VentureBeat 429s in the 2026-07-26 run.

    Reuse is skipped in two cases, both of which fall back to curating
    independently rather than handing on a set that would mislead the spine:
    when ``--limit`` is set (that flag caps how many articles the *digest*
    sends and deliberately does not govern proposal volume — see
    ``ProposeOptions``), and when the digest produced nothing, since an empty
    hand-off is indistinguishable to the spine from "no news today" and would
    silently cost the proposal lane its whole run.
    """
    print("=== news digest ===")
    run_exit, curated = await run_pipeline(run_options)
    print("=== LinkedIn proposals ===")

    if not curated:
        reusable = None
        logger.info("digest produced no articles; the proposal lane will curate")
    elif run_options.limit is not None:
        reusable = None
        logger.info("--limit caps the digest only; the proposal lane will curate")
    else:
        reusable = curated

    propose_exit = await run_propose(propose_options, prefetched=reusable)
    return max(run_exit, propose_exit)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command not in {"run", "propose", "both"}:
        parser.print_help()
        return

    setup_logging(verbose=bool(args.verbose))

    if args.command == "run":
        exit_code, _curated = asyncio.run(
            run_pipeline(RunOptions(dry_run=args.dry_run, limit=args.limit))
        )
    elif args.command == "propose":
        exit_code = asyncio.run(
            run_propose(ProposeOptions(force=args.force, skip_delivery=args.dry_run))
        )
    else:
        exit_code = asyncio.run(
            run_both(
                RunOptions(dry_run=args.dry_run, limit=args.limit),
                ProposeOptions(force=args.force, skip_delivery=args.dry_run),
            )
        )

    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
