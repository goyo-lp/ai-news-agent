"""Per-run OpenRouter usage + cost tracking — Phase 7.2.

A run spends money in exactly one place: OpenRouter tokens. (SearXNG is
self-hosted and keyless — free — so it's counted for observability, never
costed.) OpenRouter is reached two disjoint ways, so recording funnels through
one primitive from two adapters:

  * the agentic surface — coordinator + both subagents — calls OpenRouter via
    langchain ``ChatOpenAI``; :class:`UsageCallbackHandler` (attached once in
    ``build_openrouter_chat_model``) records every one of those calls.
  * the deterministic services — the Stage-A ranker and the brief verifier —
    call OpenRouter with raw ``httpx``; :func:`record_openrouter_http_response`
    records those from the response JSON.

The two never overlap (httpx vs langchain), so together they see all OpenRouter
traffic with no double counting. Everything accumulates into a
:class:`RunUsage` scoped to a :func:`track_run_usage` block via a ``ContextVar``
— which propagates into the subagents' async task contexts, so their token
spend lands in the same run without threading a handle through every call.
"""
from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult


@dataclass
class ModelUsage:
    """Accumulated OpenRouter usage for one model id within a run."""

    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    # Whether the provider ever returned a cost figure. Distinguishes a genuine
    # $0.00 from "cost not reported" in the run summary — the standard
    # OpenAI-style response omits cost, so a real run can accrue tokens with no
    # cost value at all.
    cost_reported: bool = False


@dataclass
class RunUsage:
    run_id: str
    dry_run: bool
    openrouter: dict[str, ModelUsage] = field(default_factory=dict)
    searxng_calls: int = 0

    def model(self, model: str) -> ModelUsage:
        return self.openrouter.setdefault(model, ModelUsage())


_RUN: ContextVar[RunUsage | None] = ContextVar("orchestrator_run_usage", default=None)


@contextmanager
def track_run_usage(*, run_id: str, dry_run: bool) -> Iterator[RunUsage]:
    """Scope one run's usage. Recording calls made inside the block (including
    inside subagent async tasks, which inherit the ContextVar) accumulate into
    the yielded :class:`RunUsage`; a dry run records nothing (no spend)."""
    run = RunUsage(run_id=run_id, dry_run=dry_run)
    token = _RUN.set(run)
    try:
        yield run
    finally:
        _RUN.reset(token)


def record_openrouter_usage(
    *,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    total_tokens: int,
    cost_usd: float | None,
) -> None:
    """The single OpenRouter recording primitive. No-op outside a run or in a
    dry run. ``total_tokens`` falls back to prompt+completion when the provider
    omits it; ``cost_usd`` is ``None`` when the response carried no cost."""
    run = _RUN.get()
    if run is None or run.dry_run:
        return
    usage = run.model(model)
    usage.calls += 1
    usage.prompt_tokens += prompt_tokens
    usage.completion_tokens += completion_tokens
    usage.total_tokens += total_tokens or (prompt_tokens + completion_tokens)
    if cost_usd is not None:
        usage.cost_usd += cost_usd
        usage.cost_reported = True


def record_openrouter_http_response(*, model: str, payload: Mapping[str, Any]) -> None:
    """Record one raw-httpx OpenRouter chat/completions response (the Stage-A
    ranker + brief verifier). Reads the standard ``usage`` block; cost is taken
    only if the response included it (OpenRouter omits it by default)."""
    usage = payload.get("usage")
    usage = usage if isinstance(usage, Mapping) else {}
    record_openrouter_usage(
        model=model,
        prompt_tokens=_as_int(usage.get("prompt_tokens")),
        completion_tokens=_as_int(usage.get("completion_tokens")),
        total_tokens=_as_int(usage.get("total_tokens")),
        cost_usd=_as_float(usage.get("cost")),
    )


def record_searxng_search() -> None:
    """Count one SearXNG search — observability only (self-hosted, no cost)."""
    run = _RUN.get()
    if run is None or run.dry_run:
        return
    run.searxng_calls += 1


class UsageCallbackHandler(BaseCallbackHandler):
    """Records OpenRouter token usage for every langchain model call (coordinator
    + subagents). Stateless — it writes to the ContextVar-scoped run — so one
    shared instance is attached to all models in ``build_openrouter_chat_model``.
    """

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        output = response.llm_output or {}
        token_usage = output.get("token_usage") or output.get("usage") or {}
        if not isinstance(token_usage, Mapping):
            token_usage = {}
        record_openrouter_usage(
            model=str(output.get("model_name") or output.get("model") or "unknown"),
            prompt_tokens=_as_int(token_usage.get("prompt_tokens")),
            completion_tokens=_as_int(token_usage.get("completion_tokens")),
            total_tokens=_as_int(token_usage.get("total_tokens")),
            # Present only when the provider surfaces cost in the usage block;
            # absent for a standard OpenAI-style response -> None (tokens only).
            cost_usd=_as_float(token_usage.get("cost")),
        )


# One shared handler — stateless, writes only to the active run's ContextVar.
USAGE_CALLBACK_HANDLER = UsageCallbackHandler()


def format_run_usage(run: RunUsage) -> str:
    """One-line end-of-run summary: call counts + token total + estimated cost,
    with a ``cost_source`` note so an unavailable-cost run isn't misread as free.
    """
    calls = sum(m.calls for m in run.openrouter.values())
    total_tokens = sum(m.total_tokens for m in run.openrouter.values())
    cost = sum(m.cost_usd for m in run.openrouter.values())

    if calls == 0:
        cost_source = "no_openrouter_calls"
    elif not any(m.cost_reported for m in run.openrouter.values()):
        cost_source = "cost_unavailable_from_provider"
    else:
        cost_source = "provider_reported"

    return (
        f"run_id={run.run_id} dry_run={run.dry_run} "
        f"openrouter_calls={calls} openrouter_total_tokens={total_tokens} "
        f"searxng_calls={run.searxng_calls} "
        f"estimated_cost_usd={cost:.6f} cost_source={cost_source}"
    )


def _as_int(value: Any) -> int:
    try:
        return int(value) if value is not None else 0
    except (TypeError, ValueError):
        return 0


def _as_float(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None
