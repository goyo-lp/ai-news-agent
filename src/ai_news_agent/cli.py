"""Command-line entry points for local operation and diagnostics."""

import json
from collections import Counter
from typing import Annotated

import structlog
import typer

from ai_news_agent.config import Settings
from ai_news_agent.fixtures import load_sample_records
from ai_news_agent.logging import configure_logging
from ai_news_agent.observability import (
    TraceMetadata,
    run_trace_smoke,
    tracing_scope,
)
from ai_news_agent.storage import SQLiteStore

app = typer.Typer(no_args_is_help=True, pretty_exceptions_show_locals=False)


@app.command()
def fixture(
    database: Annotated[
        str,
        typer.Option(help="SQLite path, or :memory: for a temporary database."),
    ] = ":memory:",
) -> None:
    """Validate and persist deterministic sample records without network or keys."""

    settings = Settings()
    configure_logging(
        level=settings.ai_news_log_level,
        output_format=settings.ai_news_log_format,
    )
    logger = structlog.get_logger(__name__)
    records = load_sample_records()
    metadata = TraceMetadata(
        digest_run_id="run-fixture",
        component="fixture",
        environment=settings.ai_news_environment,
    )

    with (
        tracing_scope(settings, metadata, force_disabled=True),
        SQLiteStore(database) as store,
    ):
        migrations = store.migrate()
        for record in records:
            store.save(record)

    counts = Counter(type(record).__name__ for record in records)
    result = {
        "database": database,
        "migrations_applied": list(migrations),
        "record_count": len(records),
        "record_types": dict(sorted(counts.items())),
        "tracing": "disabled",
    }
    logger.info("fixture_complete", **result)
    typer.echo(json.dumps(result, sort_keys=True))


@app.command("trace-smoke")
def trace_smoke(
    digest_run_id: Annotated[
        str,
        typer.Option(help="Shared digest run ID attached to parent and child spans."),
    ] = "manual-trace-smoke",
) -> None:
    """Send an opt-in nested trace to the configured LangSmith project."""

    settings = Settings()
    configure_logging(
        level=settings.ai_news_log_level,
        output_format=settings.ai_news_log_format,
    )
    if not settings.tracing_ready:
        raise typer.BadParameter(
            "set LANGSMITH_TRACING=true and LANGSMITH_API_KEY before live tracing"
        )
    exported = run_trace_smoke(settings, digest_run_id)
    if not exported:
        raise typer.Exit(code=1)
    typer.echo(
        json.dumps(
            {
                "digest_run_id": digest_run_id,
                "project": settings.langsmith_project,
                "status": "exported",
            },
            sort_keys=True,
        )
    )
