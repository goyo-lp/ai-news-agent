"""Command-line entry points for local operation and diagnostics."""

import json
from collections import Counter
from typing import Annotated

import structlog
import typer

from ai_news_agent.clustering import cluster_articles
from ai_news_agent.config import Settings
from ai_news_agent.extraction import (
    DEFAULT_MAX_BYTES as ARTICLE_MAX_BYTES,
)
from ai_news_agent.extraction import (
    retrieve_articles,
)
from ai_news_agent.fixtures import load_sample_records
from ai_news_agent.ingestion import (
    DEFAULT_MAX_BYTES,
    DEFAULT_MAX_WORKERS,
    ingest_sources,
)
from ai_news_agent.logging import configure_logging
from ai_news_agent.observability import (
    TraceMetadata,
    flush_traces,
    run_trace_smoke,
    tracing_scope,
)
from ai_news_agent.schemas import Article, StoryCluster
from ai_news_agent.screening import build_judge, screen_clusters
from ai_news_agent.sources import load_source_configs, load_sources_as_records
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


@app.command()
def ingest(
    config: Annotated[
        str,
        typer.Option(help="Path to config/sources.yaml."),
    ] = "config/sources.yaml",
    database: Annotated[
        str | None,
        typer.Option(help="SQLite path. Defaults to AI_NEWS_DATABASE_PATH."),
    ] = None,
    digest_run_id: Annotated[
        str,
        typer.Option(help="Shared digest run ID recorded on the Run and traces."),
    ] = "manual-ingest",
    max_workers: Annotated[
        int,
        typer.Option(help="Bounded concurrent feed fetches."),
    ] = DEFAULT_MAX_WORKERS,
    timeout: Annotated[
        float,
        typer.Option(help="Per-feed timeout in seconds."),
    ] = 15.0,
    max_bytes: Annotated[
        int,
        typer.Option(help="Maximum feed body in bytes before aborting."),
    ] = DEFAULT_MAX_BYTES,
) -> None:
    """Fetch configured feeds idempotently and record coverage."""

    settings = Settings()
    configure_logging(
        level=settings.ai_news_log_level,
        output_format=settings.ai_news_log_format,
    )
    logger = structlog.get_logger(__name__)
    database_path = database or str(settings.ai_news_database_path)
    source_configs = load_source_configs(config)
    metadata = TraceMetadata(
        digest_run_id=digest_run_id,
        component="ingestion",
        environment=settings.ai_news_environment,
    )

    with tracing_scope(settings, metadata) as trace_client:
        with SQLiteStore(database_path) as store:
            store.migrate()
            for record in load_sources_as_records(config):
                if store.get(type(record), record.id) is None:
                    store.save(record)
            summary = ingest_sources(
                source_configs,
                store,
                digest_run_id=digest_run_id,
                max_workers=max_workers,
                timeout=timeout,
                max_bytes=max_bytes,
            )
        traces_ok = flush_traces(trace_client)

    result = {
        "digest_run_id": digest_run_id,
        "attempted": summary.coverage.attempted,
        "successful": summary.coverage.successful,
        "unchanged": summary.coverage.unchanged,
        "failed": summary.coverage.failed,
        "unavailable": summary.coverage.unavailable,
        "new_articles": summary.new_article_count,
        "database": database_path,
        "tracing": "enabled" if settings.langsmith_tracing else "disabled",
    }
    logger.info("ingest_complete", **result)
    typer.echo(json.dumps(result, sort_keys=True))
    if not traces_ok:
        raise typer.Exit(code=1)


@app.command()
def retrieve(
    database: Annotated[
        str | None,
        typer.Option(help="SQLite path. Defaults to AI_NEWS_DATABASE_PATH."),
    ] = None,
    article: Annotated[
        list[str] | None,
        typer.Option(help="Article ID to retrieve. Repeat for more. Default: all."),
    ] = None,
    limit: Annotated[
        int,
        typer.Option(help="Maximum articles to retrieve."),
    ] = 50,
    timeout: Annotated[
        float,
        typer.Option(help="Per-article timeout in seconds."),
    ] = 15.0,
    max_bytes: Annotated[
        int,
        typer.Option(help="Maximum article body in bytes before aborting."),
    ] = ARTICLE_MAX_BYTES,
) -> None:
    """Retrieve full text for stored articles with polite rate limits."""

    settings = Settings()
    configure_logging(
        level=settings.ai_news_log_level,
        output_format=settings.ai_news_log_format,
    )
    logger = structlog.get_logger(__name__)
    database_path = database or str(settings.ai_news_database_path)
    metadata = TraceMetadata(
        digest_run_id="manual-retrieval",
        component="retrieval",
        environment=settings.ai_news_environment,
    )

    with tracing_scope(settings, metadata) as trace_client:
        with SQLiteStore(database_path) as store:
            store.migrate()
            if article:
                selected = [
                    record
                    for article_id in article
                    if (record := store.get(Article, article_id)) is not None
                ]
            else:
                selected = list(store.iter_records(Article))
            summary = retrieve_articles(
                tuple(selected[:limit]),
                store,
                timeout=timeout,
                max_bytes=max_bytes,
            )
        traces_ok = flush_traces(trace_client)

    result = {
        "attempted": summary.attempted,
        "retrieved": summary.retrieved,
        "cached": summary.cached,
        "insufficient": summary.insufficient,
        "failed": summary.failed,
        "database": database_path,
        "tracing": "enabled" if settings.langsmith_tracing else "disabled",
    }
    logger.info("retrieve_complete", **result)
    typer.echo(json.dumps(result, sort_keys=True))
    if not traces_ok:
        raise typer.Exit(code=1)


@app.command()
def cluster(
    database: Annotated[
        str | None,
        typer.Option(help="SQLite path. Defaults to AI_NEWS_DATABASE_PATH."),
    ] = None,
    limit: Annotated[
        int,
        typer.Option(help="Maximum articles to cluster."),
    ] = 200,
    published: Annotated[
        list[str] | None,
        typer.Option(help="Already-published article ID. Repeat for more."),
    ] = None,
) -> None:
    """Group stored articles into distinct story clusters."""

    settings = Settings()
    configure_logging(
        level=settings.ai_news_log_level,
        output_format=settings.ai_news_log_format,
    )
    logger = structlog.get_logger(__name__)
    database_path = database or str(settings.ai_news_database_path)
    metadata = TraceMetadata(
        digest_run_id="manual-clustering",
        component="clustering",
        environment=settings.ai_news_environment,
    )

    with tracing_scope(settings, metadata) as trace_client:
        with SQLiteStore(database_path) as store:
            store.migrate()
            articles = list(store.iter_records(Article))[:limit]
            summary = cluster_articles(
                tuple(articles),
                store=store,
                published_article_ids=frozenset(published or ()),
            )
        traces_ok = flush_traces(trace_client)

    result = {
        "attempted": summary.attempted,
        "clusters": summary.clusters,
        "duplicates_collapsed": summary.duplicates_collapsed,
        "recycled": summary.recycled,
        "database": database_path,
        "tracing": "enabled" if settings.langsmith_tracing else "disabled",
    }
    logger.info("cluster_complete", **result)
    typer.echo(json.dumps(result, sort_keys=True))
    if not traces_ok:
        raise typer.Exit(code=1)


@app.command()
def screen(
    database: Annotated[
        str | None,
        typer.Option(help="SQLite path. Defaults to AI_NEWS_DATABASE_PATH."),
    ] = None,
    limit: Annotated[
        int,
        typer.Option(help="Maximum clusters to screen."),
    ] = 100,
    max_candidates: Annotated[
        int,
        typer.Option(help="Maximum stories to keep."),
    ] = 50,
    published: Annotated[
        list[str] | None,
        typer.Option(help="Already-published article ID. Repeat for more."),
    ] = None,
    timeout: Annotated[
        float,
        typer.Option(help="Per-judge-call timeout in seconds."),
    ] = 30.0,
) -> None:
    """Shortlist stored story clusters with date rules and a relevance judge."""

    settings = Settings()
    configure_logging(
        level=settings.ai_news_log_level,
        output_format=settings.ai_news_log_format,
    )
    logger = structlog.get_logger(__name__)
    database_path = database or str(settings.ai_news_database_path)
    metadata = TraceMetadata(
        digest_run_id="manual-screening",
        component="screening",
        environment=settings.ai_news_environment,
    )

    with tracing_scope(settings, metadata) as trace_client:
        with SQLiteStore(database_path) as store:
            store.migrate()
            clusters = list(store.iter_records(StoryCluster))[:limit]
            articles = {article.id: article for article in store.iter_records(Article)}
            summary = screen_clusters(
                tuple(clusters),
                articles,
                store,
                judge=build_judge(settings, timeout=timeout),
                max_candidates=max_candidates,
                published_article_ids=frozenset(published or ()),
            )
        traces_ok = flush_traces(trace_client)

    result = {
        "attempted": summary.attempted,
        "shortlisted": summary.shortlisted,
        "borderline": summary.borderline,
        "rejected": summary.rejected,
        "merged": summary.merged,
        "database": database_path,
        "tracing": "enabled" if settings.langsmith_tracing else "disabled",
    }
    logger.info("screen_complete", **result)
    typer.echo(json.dumps(result, sort_keys=True))
    if not traces_ok:
        raise typer.Exit(code=1)


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
