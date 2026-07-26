"""Session-wide test isolation.

``app.config.get_settings`` is ``@lru_cache``d (one instance per process). If
any test calls it unmocked, it permanently caches a ``Settings()`` built from
the *real* ``.env`` — real API keys included — for the rest of the pytest
process; a later test that drives ``run_pipeline``/``run_propose`` without
overriding ``get_settings`` would then reuse those real credentials.

That was mostly harmless before ``configure_langsmith_env`` was fixed to
actually clear LangSmith's own env-var cache (see ``app.config``): tracing
silently never activated even when "enabled". With that fixed, the same gap
makes tests attempt real network calls to LangSmith using whatever real key
leaked in — slow, and it hammers the real project. Autouse so no test has to
opt in, and it protects tests that don't even know this seam exists.
"""
from __future__ import annotations

from collections.abc import Iterator

import pytest


@pytest.fixture(autouse=True)
def _isolate_langsmith_and_settings_cache(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    from app.config import get_settings
    from langsmith.utils import get_env_var, get_tracer_project

    get_settings.cache_clear()
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    get_env_var.cache_clear()  # type: ignore[attr-defined]
    get_tracer_project.cache_clear()

    yield

    # A test may still construct a live Settings/configure_langsmith_env call
    # with langsmith_tracing=True directly (e.g. testing that function itself)
    # — clear up after so it can't leak into the next test either.
    get_settings.cache_clear()
    get_env_var.cache_clear()  # type: ignore[attr-defined]
    get_tracer_project.cache_clear()
