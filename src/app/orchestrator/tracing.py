"""Coordinator run tracing — Phase 7.1.

LangSmith traces a LangGraph/LangChain run automatically once its env vars are
set (``LANGSMITH_TRACING`` / ``LANGSMITH_API_KEY`` / ``LANGSMITH_PROJECT``).
Mirroring the configured LangSmith settings into the environment is already
owned by ``configure_langsmith_env`` (:mod:`app.config`) and belongs to the run
entry point that starts a coordinator run — not here.

What is orchestrator-specific, and all this module owns, is the run-scoped
``config`` handed to ``agent.ainvoke(state, config)``: a stable run name + tags
so the resulting trace is findable under the project instead of buried among
anonymous run ids. It is a pure ``run_id`` + ``dry_run`` -> config mapping with
no env side effects and no ``Settings`` dependency.
"""
from __future__ import annotations

from langchain_core.runnables import RunnableConfig


def coordinator_run_config(*, run_id: str, dry_run: bool) -> RunnableConfig:
    """Build the LangGraph run ``config`` for one coordinator run.

    Pass the result to ``agent.ainvoke(state, config)``; LangGraph threads the
    ``run_name`` / ``tags`` / ``metadata`` down to LangSmith so the run's whole
    span tree (coordinator + subagents + tools) is named and filterable. The
    ``live`` / ``dry_run`` tag lets the runs that actually spent tokens be
    separated from dry runs.

    Env activation (``LANGSMITH_*``) is deliberately not done here — that is
    ``configure_langsmith_env``'s job at the run entry point.
    """
    return {
        "run_name": f"coordinator-{run_id}",
        "tags": ["orchestrator", "dry_run" if dry_run else "live"],
        "metadata": {"run_id": run_id, "dry_run": dry_run},
    }
