"""LangSmith tracing with shared metadata and defensive redaction."""

from __future__ import annotations

import re
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import langsmith as ls
import structlog
from langsmith import Client
from pydantic import BaseModel, ConfigDict

from ai_news_agent import __version__
from ai_news_agent.config import Settings

_LOGGER = structlog.get_logger(__name__)
_SECRET_KEY = re.compile(
    r"(?:api[_-]?key|authorization|cookie|password|secret|token)", re.IGNORECASE
)
_SECRET_VALUE = re.compile(
    r"(?i)(?:bearer\s+\S+|(?:sk|lsv2|gh[opsu])_[A-Za-z0-9_-]{12,})"
)
_EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")


class TracingConfigurationError(RuntimeError):
    """Raised when tracing is enabled without the required configuration."""


@dataclass(slots=True)
class TraceClient:
    """LangSmith client paired with background export state."""

    client: Client
    export_errors: list[Exception] = field(default_factory=list)


class TraceMetadata(BaseModel):
    """Metadata attached to every span in one digest run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    digest_run_id: str
    component: str
    environment: str
    app_version: str = __version__


def redact(value: Any) -> Any:
    """Recursively remove common credential and personal-data patterns."""

    if isinstance(value, Mapping):
        return {
            str(key): "[REDACTED]" if _SECRET_KEY.search(str(key)) else redact(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple, set)):
        return [redact(item) for item in value]
    if isinstance(value, str):
        return _EMAIL.sub("[REDACTED_EMAIL]", _SECRET_VALUE.sub("[REDACTED]", value))
    return value


def build_langsmith_client(settings: Settings) -> TraceClient:
    """Build a redacting client only after tracing has been explicitly enabled."""

    if not settings.tracing_ready:
        raise TracingConfigurationError(
            "live tracing requires LANGSMITH_TRACING=true and LANGSMITH_API_KEY"
        )

    export_errors: list[Exception] = []

    def record_export_error(exc: Exception) -> None:
        export_errors.append(exc)
        _log_trace_export_error(exc)

    client = Client(
        api_url=settings.langsmith_endpoint,
        api_key=settings.langsmith_api_key.get_secret_value(),
        workspace_id=settings.langsmith_workspace_id,
        hide_inputs=redact,
        hide_outputs=redact,
        hide_metadata=redact,
        tracing_error_callback=record_export_error,
    )
    return TraceClient(client=client, export_errors=export_errors)


@contextmanager
def tracing_scope(
    settings: Settings,
    metadata: TraceMetadata,
    *,
    force_disabled: bool = False,
) -> Iterator[TraceClient | None]:
    """Apply one trace configuration to all nested work in a digest stage."""

    enabled = settings.langsmith_tracing and not force_disabled
    if not enabled:
        with ls.tracing_context(enabled=False):
            yield None
        return

    trace_client = build_langsmith_client(settings)
    safe_metadata = redact(metadata.model_dump())
    with ls.tracing_context(
        enabled=True,
        client=trace_client.client,
        project_name=settings.langsmith_project,
        metadata=safe_metadata,
        tags=["ai-news-agent", metadata.component],
    ):
        yield trace_client


def flush_traces(trace_client: TraceClient | None, *, timeout: float = 10.0) -> bool:
    """Flush buffered spans and make export failures visible in local logs."""

    if trace_client is None:
        return True
    try:
        trace_client.client.flush(timeout=timeout)
    except Exception as exc:  # LangSmith may surface transport-specific errors
        _log_trace_export_error(exc)
        return False
    return not bool(trace_client.export_errors)


@ls.traceable(
    name="trace-smoke-child",
    run_type="chain",
    process_inputs=redact,
    process_outputs=redact,
)
def _trace_smoke_child(payload: dict[str, str]) -> dict[str, str]:
    return {"status": "child-complete", "digest_run_id": payload["digest_run_id"]}


@ls.traceable(
    name="trace-smoke-parent",
    run_type="chain",
    process_inputs=redact,
    process_outputs=redact,
)
def _trace_smoke_parent(payload: dict[str, str]) -> dict[str, str]:
    child = _trace_smoke_child(payload)
    return {"status": "parent-complete", "child_status": child["status"]}


def run_trace_smoke(settings: Settings, digest_run_id: str) -> bool:
    """Emit a parent and child span for opt-in live verification."""

    metadata = TraceMetadata(
        digest_run_id=digest_run_id,
        component="trace-smoke",
        environment=settings.ai_news_environment,
    )
    with tracing_scope(settings, metadata) as client:
        _trace_smoke_parent({"digest_run_id": digest_run_id})
    return flush_traces(client)


def _log_trace_export_error(exc: Exception) -> None:
    _LOGGER.error(
        "langsmith_trace_export_failed",
        error_type=type(exc).__name__,
        error=str(exc),
    )
