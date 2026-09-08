from typing import Any

import pytest

import ai_news_agent.observability as observability
from ai_news_agent.config import Settings
from ai_news_agent.observability import (
    TraceClient,
    TraceMetadata,
    TracingConfigurationError,
    build_langsmith_client,
    flush_traces,
    redact,
    tracing_scope,
)


def test_redact_masks_nested_credentials_and_personal_data() -> None:
    fake_langsmith_token = "lsv2_pt_" + "abcdefghijklmnop"
    value = {
        "api_key": "top-secret",
        "nested": [
            {"authorization": "Bearer abc.def"},
            "email me at person@example.com",
            f"token embedded {fake_langsmith_token}",
        ],
        "safe": "keep me",
    }

    assert redact(value) == {
        "api_key": "[REDACTED]",
        "nested": [
            {"authorization": "[REDACTED]"},
            "email me at [REDACTED_EMAIL]",
            "token embedded [REDACTED]",
        ],
        "safe": "keep me",
    }


def test_disabled_scope_never_builds_client(monkeypatch) -> None:
    def unexpected_client(_settings: Settings) -> None:
        raise AssertionError("disabled tracing must not build a client")

    monkeypatch.setattr(observability, "build_langsmith_client", unexpected_client)
    settings = Settings(_env_file=None)
    metadata = TraceMetadata(
        digest_run_id="run-test", component="test", environment="test"
    )

    with tracing_scope(settings, metadata) as client:
        assert client is None


def test_enabled_tracing_requires_key(monkeypatch) -> None:
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)

    with pytest.raises(TracingConfigurationError, match="LANGSMITH_API_KEY"):
        build_langsmith_client(Settings(_env_file=None))


def test_flush_failure_is_logged(monkeypatch) -> None:
    events: list[tuple[str, dict[str, Any]]] = []

    class CapturingLogger:
        def error(self, event: str, **values: Any) -> None:
            events.append((event, values))

    class BrokenClient:
        def flush(self, *, timeout: float) -> None:
            raise OSError(f"export unavailable after {timeout}s")

    monkeypatch.setattr(observability, "_LOGGER", CapturingLogger())

    trace_client = TraceClient(client=BrokenClient())  # type: ignore[arg-type]

    assert flush_traces(trace_client, timeout=0.5) is False
    assert events == [
        (
            "langsmith_trace_export_failed",
            {
                "error_type": "OSError",
                "error": "export unavailable after 0.5s",
            },
        )
    ]


def test_flush_without_client_is_successful() -> None:
    assert flush_traces(None) is True


def test_flush_detects_background_callback_failure() -> None:
    class ReportedFailureClient:
        def flush(self, *, timeout: float) -> None:
            assert timeout == 10.0

    client = ReportedFailureClient()
    trace_client = TraceClient(  # type: ignore[arg-type]
        client=client,
        export_errors=[OSError("background export failed")],
    )

    assert flush_traces(trace_client) is False
