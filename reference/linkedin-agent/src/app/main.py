from __future__ import annotations

import argparse
import asyncio
import json
import logging
from datetime import date, datetime, timezone
from pathlib import Path
from typing import TypedDict
from uuid import uuid4

from app.config import Settings, configure_langsmith_env, get_settings
from app.graph.state import AgentState
from app.graph.workflow import build_workflow
from app.logging import setup_logging
from app.schemas import parse_delivery_results
from app.services.api_usage_tracker import (
    end_run_api_usage,
    format_run_api_usage,
    snapshot_run_api_usage,
    start_run_api_usage,
)
from app.services.style_profile import StyleProfiler

logger = logging.getLogger(__name__)


class PipelineRunResult(TypedDict):
    run_date: str
    exit_code: int
    summary: str
    export_dir: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ai-linked-imposting-agent")
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="Run the full deep-agent pipeline")
    run_parser.add_argument("--dry-run", action="store_true", help="Run with mock/fallback external calls")
    run_parser.add_argument("--date", type=str, default=None, help="Run date in YYYY-MM-DD")
    run_parser.add_argument("--hours-back", type=int, default=None, help="Freshness window in hours")
    run_parser.add_argument("--samples-dir", type=str, default=None, help="Override style samples folder")
    run_parser.add_argument("--no-graphics", action="store_true", help="Disable LangGraphics watch wrapper")
    run_parser.add_argument("--verbose", action="store_true", help="Enable debug logging")

    batch_parser = subparsers.add_parser(
        "run-batch",
        help="Run the full deep-agent pipeline for multiple dates in parallel",
    )
    batch_parser.add_argument(
        "--dates",
        type=str,
        required=True,
        help="Comma-separated YYYY-MM-DD dates (example: 2026-03-01,2026-03-02)",
    )
    batch_parser.add_argument("--dry-run", action="store_true", help="Run with mock/fallback external calls")
    batch_parser.add_argument("--hours-back", type=int, default=None, help="Freshness window in hours")
    batch_parser.add_argument("--samples-dir", type=str, default=None, help="Override style samples folder")
    batch_parser.add_argument(
        "--max-concurrency",
        type=int,
        default=None,
        help="Max concurrent pipeline runs (defaults to PIPELINE_BATCH_CONCURRENCY)",
    )
    batch_parser.add_argument("--no-graphics", action="store_true", help="Disable LangGraphics watch wrapper")
    batch_parser.add_argument("--verbose", action="store_true", help="Enable debug logging")

    style_parser = subparsers.add_parser(
        "build-style-profile",
        help="Build/update style profile artifact from writing samples",
    )
    style_parser.add_argument("--samples-dir", type=str, default=None, help="Style samples folder")

    preview_parser = subparsers.add_parser("preview", help="Print latest generated LinkedIn posts markdown")
    preview_parser.add_argument("--date", type=str, default=None, help="Output date folder in YYYY-MM-DD")

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

    run_date = _resolve_run_date(args.date)
    result = await _run_pipeline_for_date(
        args=args,
        settings=settings,
        run_date=run_date,
        enable_graphics=not bool(args.no_graphics),
    )
    print(result["summary"])
    if result["export_dir"]:
        print(f"Outputs: {result['export_dir']}")
    return result["exit_code"]


async def run_batch_pipeline(args: argparse.Namespace) -> int:
    settings = get_settings()
    configure_langsmith_env(settings)

    dry_run = bool(args.dry_run)
    missing_fields = settings.missing_required_runtime_fields(dry_run=dry_run)
    if missing_fields:
        joined = ", ".join(missing_fields)
        logger.error("Configuration error: missing required .env values: %s", joined)
        print(f"Configuration error: missing required .env values: {joined}")
        return 2

    try:
        dates = _parse_dates_arg(args.dates)
    except ValueError as exc:
        print(str(exc))
        return 2
    max_concurrency = int(args.max_concurrency or settings.pipeline_batch_concurrency)
    max_concurrency = max(1, min(max_concurrency, 16))

    if not bool(args.no_graphics):
        logger.info("Batch mode forces --no-graphics to avoid watcher port conflicts.")

    semaphore = asyncio.Semaphore(max_concurrency)

    async def worker(run_date: str) -> PipelineRunResult:
        async with semaphore:
            return await _run_pipeline_for_date(
                args=args,
                settings=settings,
                run_date=run_date,
                enable_graphics=False,
            )

    results = await asyncio.gather(*(worker(run_date) for run_date in dates))
    results = sorted(results, key=lambda item: str(item["run_date"]))

    for item in results:
        print(str(item["summary"]))
        if item["export_dir"]:
            print(f"Outputs: {item['export_dir']}")

    return 0 if all(item["exit_code"] == 0 for item in results) else 1


async def _run_pipeline_for_date(
    *,
    args: argparse.Namespace,
    settings: Settings,
    run_date: str,
    enable_graphics: bool,
) -> PipelineRunResult:
    dry_run = bool(args.dry_run)
    hours_back = int(args.hours_back or settings.discovery_hours_back)
    hours_back = max(1, min(hours_back, 168))

    initial_state: AgentState = {
        "run_id": str(uuid4()),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "run_date": run_date,
        "dry_run": dry_run,
        "hours_back": hours_back,
        "max_topics": settings.max_topics_per_run,
        "errors": [],
    }

    if args.samples_dir:
        initial_state["style_samples_dir_override"] = str(args.samples_dir)

    workflow = build_workflow(enable_graphics=enable_graphics)
    usage_token = start_run_api_usage(run_id=str(initial_state["run_id"]), dry_run=dry_run)
    try:
        final_state = await workflow.ainvoke(initial_state)
        api_usage_snapshot = snapshot_run_api_usage()
    finally:
        end_run_api_usage(usage_token)

    discovered_count = len(final_state.get("discovered_items", []))
    topics_count = len(final_state.get("ranked_topics", []))
    posts_count = len(final_state.get("linkedin_posts", []))
    delivery_results = parse_delivery_results(final_state.get("delivery_results"))
    sent_count = len([result for result in delivery_results if result.status == "sent"])
    failed_count = len([result for result in delivery_results if result.status == "error"])
    dry_delivery_count = len([result for result in delivery_results if result.status == "dry_run"])
    merged_errors = _merge_error_lists(
        list(final_state.get("errors", [])),
        list(final_state.get("style_profile_errors", [])),
    )
    errors_count = len(merged_errors)
    export_dir = final_state.get("export_dir", "")
    api_usage_summary = format_run_api_usage(api_usage_snapshot)

    logger.info("Run API usage | %s", api_usage_summary)

    logger.info(
        "Run complete | discovered=%s topics=%s posts=%s deliveries_sent=%s deliveries_failed=%s deliveries_dry=%s errors=%s dry_run=%s | %s",
        discovered_count,
        topics_count,
        posts_count,
        sent_count,
        failed_count,
        dry_delivery_count,
        errors_count,
        dry_run,
        api_usage_summary,
    )

    summary = (
        f"[{run_date}] Run complete. discovered={discovered_count} topics={topics_count} "
        f"posts={posts_count} sent={sent_count} failed={failed_count} "
        f"delivery_dry_run={dry_delivery_count} errors={errors_count} dry_run={dry_run} "
        f"{api_usage_summary}"
    )

    exit_code = 0
    if posts_count == 0:
        exit_code = 1
    if not dry_run and failed_count > 0:
        exit_code = 1

    return {
        "run_date": run_date,
        "exit_code": exit_code,
        "summary": summary,
        "export_dir": export_dir,
    }


def build_style_profile(args: argparse.Namespace) -> int:
    settings = get_settings()
    profiler = StyleProfiler(settings)

    samples_dir = Path(args.samples_dir) if args.samples_dir else settings.style_samples_path
    profile = profiler.build_from_directory(samples_dir)
    saved_path = profiler.save_profile(profile)

    print(
        "Style profile built. "
        f"samples={profile.sample_count} sentences={profile.sentence_count} saved={saved_path}"
    )
    return 0


def preview_latest(args: argparse.Namespace) -> int:
    settings = get_settings()

    target_date = _resolve_run_date(args.date) if args.date else _find_latest_output_date(settings.outputs_path)
    if target_date is None:
        print("No outputs found.")
        return 1

    output_dir = settings.outputs_path / target_date
    posts_path = output_dir / "linkedin_posts.md"
    report_path = output_dir / "run_report.json"

    if not posts_path.exists():
        print(f"No linkedin_posts.md found at {posts_path}")
        return 1

    print(posts_path.read_text(encoding="utf-8"))
    if report_path.exists():
        try:
            report_payload = json.loads(report_path.read_text(encoding="utf-8"))
            print("\nRun summary:")
            print(
                f"- discovered={report_payload.get('discovered_count')} "
                f"topics={report_payload.get('topics_count')} posts={report_payload.get('posts_count')} "
                f"sent={report_payload.get('deliveries_sent')} failed={report_payload.get('deliveries_failed')}"
            )
        except Exception:
            pass

    return 0


def _resolve_run_date(raw_value: str | None) -> str:
    if not raw_value:
        return date.today().isoformat()
    try:
        parsed = date.fromisoformat(raw_value)
    except ValueError as exc:
        raise ValueError("Invalid --date format. Use YYYY-MM-DD.") from exc
    return parsed.isoformat()


def _find_latest_output_date(outputs_dir: Path) -> str | None:
    if not outputs_dir.exists():
        return None

    dated_dirs = [path for path in outputs_dir.iterdir() if path.is_dir()]
    if not dated_dirs:
        return None

    latest = sorted(dated_dirs, key=lambda path: path.name)[-1]
    return latest.name


def _parse_dates_arg(raw: str) -> list[str]:
    parts = [piece.strip() for piece in raw.split(",")]
    dates: list[str] = []
    for piece in parts:
        if not piece:
            continue
        resolved = _resolve_run_date(piece)
        if resolved not in dates:
            dates.append(resolved)
    if not dates:
        raise ValueError("No valid dates provided to --dates.")
    return dates


def _merge_error_lists(base: list[str], extras: list[str]) -> list[str]:
    merged: list[str] = []
    for value in [*base, *extras]:
        normalized = " ".join(str(value).split()).strip()
        if not normalized or normalized in merged:
            continue
        merged.append(normalized)
    return merged


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    verbose = bool(getattr(args, "verbose", False))
    setup_logging(verbose=verbose)

    if args.command == "run":
        exit_code = asyncio.run(run_pipeline(args))
        raise SystemExit(exit_code)

    if args.command == "run-batch":
        exit_code = asyncio.run(run_batch_pipeline(args))
        raise SystemExit(exit_code)

    if args.command == "build-style-profile":
        exit_code = build_style_profile(args)
        raise SystemExit(exit_code)

    if args.command == "preview":
        exit_code = preview_latest(args)
        raise SystemExit(exit_code)

    parser.print_help()


if __name__ == "__main__":
    main()
