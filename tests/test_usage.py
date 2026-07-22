"""P7.2 — per-run OpenRouter usage + cost tracking.

Covers the recording primitive, both adapters (raw-httpx response + the
langchain callback), the SearXNG counter, ContextVar propagation into async
tasks (the subagent case), and the end-of-run summary's cost_source logic.
"""
from __future__ import annotations

import asyncio

from langchain_core.outputs import LLMResult

from app.orchestrator.usage import (
    UsageCallbackHandler,
    format_run_usage,
    record_openrouter_http_response,
    record_openrouter_usage,
    record_searxng_search,
    track_run_usage,
)


def test_records_openrouter_usage_per_model() -> None:
    with track_run_usage(run_id="r", dry_run=False) as run:
        record_openrouter_usage(
            model="deepseek/deepseek-v4-flash",
            prompt_tokens=100,
            completion_tokens=40,
            total_tokens=140,
            cost_usd=0.002,
        )
        record_openrouter_usage(
            model="deepseek/deepseek-v4-flash",
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            cost_usd=None,
        )

    usage = run.openrouter["deepseek/deepseek-v4-flash"]
    assert usage.calls == 2
    assert usage.prompt_tokens == 110
    assert usage.total_tokens == 155
    assert usage.cost_usd == 0.002
    assert usage.cost_reported is True


def test_dry_run_records_nothing() -> None:
    with track_run_usage(run_id="r", dry_run=True) as run:
        record_openrouter_usage(
            model="m", prompt_tokens=1, completion_tokens=1, total_tokens=2, cost_usd=1.0
        )
        record_searxng_search()
    assert run.openrouter == {}
    assert run.searxng_calls == 0


def test_recording_outside_a_run_is_a_noop() -> None:
    # No active run context -> silently ignored, never raises.
    record_openrouter_usage(
        model="m", prompt_tokens=1, completion_tokens=1, total_tokens=2, cost_usd=1.0
    )
    record_searxng_search()


def test_total_tokens_falls_back_to_prompt_plus_completion() -> None:
    with track_run_usage(run_id="r", dry_run=False) as run:
        record_openrouter_usage(
            model="m", prompt_tokens=30, completion_tokens=20, total_tokens=0, cost_usd=None
        )
    assert run.openrouter["m"].total_tokens == 50


def test_http_response_adapter_reads_usage_block_and_cost() -> None:
    payload = {
        "choices": [{"message": {"content": "{}"}}],
        "usage": {"prompt_tokens": 200, "completion_tokens": 50, "total_tokens": 250, "cost": 0.01},
    }
    with track_run_usage(run_id="r", dry_run=False) as run:
        record_openrouter_http_response(model="stage-a", payload=payload)

    usage = run.openrouter["stage-a"]
    assert usage.calls == 1
    assert usage.prompt_tokens == 200
    assert usage.total_tokens == 250
    assert usage.cost_usd == 0.01
    assert usage.cost_reported is True


def test_http_response_adapter_missing_usage_block_records_zero_call() -> None:
    with track_run_usage(run_id="r", dry_run=False) as run:
        record_openrouter_http_response(model="m", payload={"choices": []})
    usage = run.openrouter["m"]
    assert usage.calls == 1
    assert usage.total_tokens == 0
    assert usage.cost_reported is False


def test_callback_handler_records_from_llm_output() -> None:
    handler = UsageCallbackHandler()
    result = LLMResult(
        generations=[[]],
        llm_output={
            "model_name": "coordinator-model",
            "token_usage": {"prompt_tokens": 12, "completion_tokens": 8, "total_tokens": 20},
        },
    )
    with track_run_usage(run_id="r", dry_run=False) as run:
        handler.on_llm_end(result)

    usage = run.openrouter["coordinator-model"]
    assert usage.calls == 1
    assert usage.total_tokens == 20
    assert usage.cost_reported is False  # langchain path carries no cost


def test_callback_handler_captures_cost_when_provider_surfaces_it() -> None:
    """When OpenRouter returns cost in the usage block (usage.include=true), the
    langchain callback records it — not just tokens."""
    handler = UsageCallbackHandler()
    result = LLMResult(
        generations=[[]],
        llm_output={
            "model_name": "m",
            "token_usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10, "cost": 0.003},
        },
    )
    with track_run_usage(run_id="r", dry_run=False) as run:
        handler.on_llm_end(result)
    usage = run.openrouter["m"]
    assert usage.cost_usd == 0.003
    assert usage.cost_reported is True


def test_callback_handler_handles_absent_llm_output() -> None:
    handler = UsageCallbackHandler()
    result = LLMResult(generations=[[]], llm_output=None)
    with track_run_usage(run_id="r", dry_run=False) as run:
        handler.on_llm_end(result)
    # Unknown model, zero tokens, but the call is counted.
    assert run.openrouter["unknown"].calls == 1


def test_searxng_counter_increments() -> None:
    with track_run_usage(run_id="r", dry_run=False) as run:
        record_searxng_search()
        record_searxng_search()
    assert run.searxng_calls == 2


def test_contextvar_propagates_into_async_tasks() -> None:
    """A subagent runs in its own asyncio task; the run context is inherited, so
    its token spend lands in the same RunUsage without threading a handle."""

    async def subagent_work() -> None:
        record_openrouter_usage(
            model="subagent-model", prompt_tokens=5, completion_tokens=5, total_tokens=10, cost_usd=None
        )

    async def driver(run_holder: list) -> None:
        with track_run_usage(run_id="r", dry_run=False) as run:
            run_holder.append(run)
            await asyncio.gather(subagent_work(), subagent_work())

    holder: list = []
    asyncio.run(driver(holder))
    assert holder[0].openrouter["subagent-model"].calls == 2


def test_format_summary_cost_source_variants() -> None:
    # No calls at all.
    with track_run_usage(run_id="r1", dry_run=False) as empty:
        pass
    assert "cost_source=no_openrouter_calls" in format_run_usage(empty)

    # Calls but no provider cost.
    with track_run_usage(run_id="r2", dry_run=False) as no_cost:
        record_openrouter_usage(
            model="m", prompt_tokens=10, completion_tokens=10, total_tokens=20, cost_usd=None
        )
        record_searxng_search()
    summary = format_run_usage(no_cost)
    assert "cost_source=cost_unavailable_from_provider" in summary
    assert "openrouter_calls=1" in summary
    assert "openrouter_total_tokens=20" in summary
    assert "searxng_calls=1" in summary

    # Provider-reported cost.
    with track_run_usage(run_id="r3", dry_run=False) as with_cost:
        record_openrouter_usage(
            model="m", prompt_tokens=10, completion_tokens=10, total_tokens=20, cost_usd=0.005
        )
    assert "cost_source=provider_reported" in format_run_usage(with_cost)
    assert "estimated_cost_usd=0.005000" in format_run_usage(with_cost)
