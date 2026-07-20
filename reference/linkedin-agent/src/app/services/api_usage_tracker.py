from __future__ import annotations

from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ModelUsage:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    cost_value_count: int = 0


@dataclass
class RunApiUsage:
    run_id: str
    dry_run: bool
    openrouter: dict[str, ModelUsage] = field(default_factory=dict)
    tavily_search_calls: int = 0
    tavily_extract_calls: int = 0
    tavily_extract_urls: int = 0
    tavily_search_results: int = 0


_USAGE_CONTEXT: ContextVar[RunApiUsage | None] = ContextVar("api_usage_run", default=None)


def start_run_api_usage(run_id: str, dry_run: bool) -> Token[RunApiUsage | None]:
    return _USAGE_CONTEXT.set(RunApiUsage(run_id=run_id, dry_run=dry_run))


def end_run_api_usage(token: Token[RunApiUsage | None]) -> None:
    _USAGE_CONTEXT.reset(token)


def snapshot_run_api_usage() -> dict[str, Any]:
    usage = _USAGE_CONTEXT.get()
    if usage is None:
        return _empty_snapshot()

    model_payload: dict[str, Any] = {}
    openrouter_calls = 0
    openrouter_prompt_tokens = 0
    openrouter_completion_tokens = 0
    openrouter_total_tokens = 0
    openrouter_cost_usd = 0.0
    openrouter_cost_value_count = 0

    for model, model_usage in usage.openrouter.items():
        model_payload[model] = {
            "calls": model_usage.calls,
            "prompt_tokens": model_usage.prompt_tokens,
            "completion_tokens": model_usage.completion_tokens,
            "total_tokens": model_usage.total_tokens,
            "estimated_cost_usd": round(model_usage.cost_usd, 8),
            "cost_values": model_usage.cost_value_count,
        }
        openrouter_calls += model_usage.calls
        openrouter_prompt_tokens += model_usage.prompt_tokens
        openrouter_completion_tokens += model_usage.completion_tokens
        openrouter_total_tokens += model_usage.total_tokens
        openrouter_cost_usd += model_usage.cost_usd
        openrouter_cost_value_count += model_usage.cost_value_count

    return {
        "run_id": usage.run_id,
        "dry_run": usage.dry_run,
        "openrouter": {
            "calls": openrouter_calls,
            "prompt_tokens": openrouter_prompt_tokens,
            "completion_tokens": openrouter_completion_tokens,
            "total_tokens": openrouter_total_tokens,
            "estimated_cost_usd": round(openrouter_cost_usd, 8),
            "cost_values": openrouter_cost_value_count,
            "models": model_payload,
        },
        "tavily": {
            "search_calls": usage.tavily_search_calls,
            "extract_calls": usage.tavily_extract_calls,
            "extract_urls": usage.tavily_extract_urls,
            "search_results": usage.tavily_search_results,
        },
    }


def format_run_api_usage(snapshot: dict[str, Any]) -> str:
    openrouter = snapshot.get("openrouter") if isinstance(snapshot, dict) else {}
    tavily = snapshot.get("tavily") if isinstance(snapshot, dict) else {}

    openrouter_calls = _as_int((openrouter or {}).get("calls"), 0)
    tavily_search_calls = _as_int((tavily or {}).get("search_calls"), 0)
    tavily_extract_calls = _as_int((tavily or {}).get("extract_calls"), 0)
    total_tokens = _as_int((openrouter or {}).get("total_tokens"), 0)
    estimated_cost = _as_float((openrouter or {}).get("estimated_cost_usd"), 0.0)
    cost_values = _as_int((openrouter or {}).get("cost_values"), 0)

    cost_suffix = "cost_source=response_usage"
    if openrouter_calls == 0:
        cost_suffix = "cost_source=no_openrouter_calls"
    elif cost_values == 0:
        cost_suffix = "cost_source=unavailable_from_provider"

    return (
        f"openrouter_calls={openrouter_calls} "
        f"tavily_search_calls={tavily_search_calls} "
        f"tavily_extract_calls={tavily_extract_calls} "
        f"openrouter_total_tokens={total_tokens} "
        f"estimated_cost_usd={estimated_cost:.6f} "
        f"{cost_suffix}"
    )


def record_tavily_search(*, results_count: int) -> None:
    usage = _USAGE_CONTEXT.get()
    if usage is None or usage.dry_run:
        return
    usage.tavily_search_calls += 1
    usage.tavily_search_results += max(results_count, 0)


def record_tavily_extract(*, url_count: int) -> None:
    usage = _USAGE_CONTEXT.get()
    if usage is None or usage.dry_run:
        return
    usage.tavily_extract_calls += 1
    usage.tavily_extract_urls += max(url_count, 0)


def record_openrouter_http_response(*, model: str, payload: dict[str, Any]) -> None:
    usage = _USAGE_CONTEXT.get()
    if usage is None or usage.dry_run:
        return

    model_usage = usage.openrouter.setdefault(model, ModelUsage())
    model_usage.calls += 1

    prompt_tokens, completion_tokens, total_tokens, maybe_cost = _extract_usage(payload)
    model_usage.prompt_tokens += prompt_tokens
    model_usage.completion_tokens += completion_tokens
    model_usage.total_tokens += total_tokens
    if maybe_cost is not None:
        model_usage.cost_usd += maybe_cost
        model_usage.cost_value_count += 1


def record_openrouter_tool_call(*, model: str, payload: Any | None = None) -> None:
    usage = _USAGE_CONTEXT.get()
    if usage is None or usage.dry_run:
        return

    model_usage = usage.openrouter.setdefault(model, ModelUsage())
    model_usage.calls += 1

    if payload is None:
        return

    prompt_tokens, completion_tokens, total_tokens, maybe_cost = _extract_usage(payload)
    model_usage.prompt_tokens += prompt_tokens
    model_usage.completion_tokens += completion_tokens
    model_usage.total_tokens += total_tokens
    if maybe_cost is not None:
        model_usage.cost_usd += maybe_cost
        model_usage.cost_value_count += 1


def _extract_usage(payload: Any) -> tuple[int, int, int, float | None]:
    usage_payload: dict[str, Any] = {}
    maybe_cost: float | None = None

    if isinstance(payload, dict):
        raw_usage = payload.get("usage")
        if isinstance(raw_usage, dict):
            usage_payload = raw_usage
        maybe_cost = _extract_cost(payload)

    prompt_tokens = _as_int(
        usage_payload.get("prompt_tokens")
        if usage_payload
        else None,
        _as_int(usage_payload.get("input_tokens") if usage_payload else None, 0),
    )
    completion_tokens = _as_int(
        usage_payload.get("completion_tokens")
        if usage_payload
        else None,
        _as_int(usage_payload.get("output_tokens") if usage_payload else None, 0),
    )
    total_tokens = _as_int(usage_payload.get("total_tokens") if usage_payload else None, 0)
    if total_tokens == 0:
        total_tokens = prompt_tokens + completion_tokens

    return prompt_tokens, completion_tokens, total_tokens, maybe_cost


def _extract_cost(payload: dict[str, Any]) -> float | None:
    direct_keys = ("cost", "total_cost", "estimated_cost")
    for key in direct_keys:
        value = _as_optional_float(payload.get(key))
        if value is not None:
            return value

    usage = payload.get("usage")
    if isinstance(usage, dict):
        for key in direct_keys:
            value = _as_optional_float(usage.get(key))
            if value is not None:
                return value

    return None


def _as_int(value: Any, default: int) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except Exception:
        return default


def _as_float(value: Any, default: float) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _as_optional_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _empty_snapshot() -> dict[str, Any]:
    return {
        "run_id": "unknown",
        "dry_run": False,
        "openrouter": {
            "calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "estimated_cost_usd": 0.0,
            "cost_values": 0,
            "models": {},
        },
        "tavily": {
            "search_calls": 0,
            "extract_calls": 0,
            "extract_urls": 0,
            "search_results": 0,
        },
    }
