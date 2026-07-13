"""Composable middleware run before and after each OpenRouter chat completion call.

DeepSeek V4 Flash is a reasoning model. With its default effort ("high") it emits
a think-block before the final answer. Without stripping, that reasoning leaks into
summaries and breaks the 3-sentence post-processor.

A middleware is an async callable (payload, call_next) -> content_str. It may mutate
payload before calling call_next (the before hook) and transform the returned content
afterwards (the after hook). Middleware are composed as an onion: the first registered
middleware before runs first and its after runs last.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Awaitable, Callable, Protocol

logger = logging.getLogger(__name__)

OPEN = chr(60) + "think" + chr(62)
CLOSE = chr(60) + "/think" + chr(62)

# Matches a full reasoning block (non-greedy, DOTALL, case-insensitive).
_THINK_BLOCK = re.compile(re.escape(OPEN) + r".*?" + re.escape(CLOSE), re.IGNORECASE | re.DOTALL)
# Defensive: a trailing unclosed block (when max_tokens truncates the model mid-reasoning).
_UNCLOSED_THINK = re.compile(re.escape(OPEN) + r".*$", re.IGNORECASE | re.DOTALL)


def strip_think_blocks(text: str) -> str:
    """Remove all reasoning traces from text."""
    cleaned = _THINK_BLOCK.sub("", text)
    cleaned = _UNCLOSED_THINK.sub("", cleaned)
    return cleaned.strip()


class OpenRouterCall(Protocol):
    """Innermost call function that POSTs the payload to OpenRouter."""

    def __call__(self, payload: dict) -> Awaitable[str]: ...


Middleware = Callable[[dict, "OpenRouterCall"], Awaitable[str]]


async def strip_reasoning_middleware(payload: dict, call_next: OpenRouterCall) -> str:
    """After-call middleware that strips reasoning traces from the response."""
    content = await call_next(payload)
    return strip_think_blocks(content)


def reasoning_effort_middleware(effort: str | None) -> Middleware:
    """Build a before-call middleware that sets OpenRouter reasoning.effort.

    Pass effort=None to omit the field entirely (use the model default). For DeepSeek
    V4 Flash, supported efforts are "xhigh" and "high".
    """

    async def middleware(payload: dict, call_next: OpenRouterCall) -> str:
        if effort:
            payload = {**payload, "reasoning": {"effort": effort}}
        return await call_next(payload)

    return middleware


def lookup_reasoning_effort(value: Any) -> str | None:
    """Coerce a Settings value into a reasoning-effort string or None."""
    if value is None:
        return None
    text = str(value).strip().lower()
    return text or None


class MiddlewareChain:
    """Compose a list of middleware around an inner call function."""

    def __init__(self, middlewares: list[Middleware] | None = None) -> None:
        self._middlewares: list[Middleware] = list(middlewares or [])

    def add(self, middleware: Middleware) -> None:
        self._middlewares.append(middleware)

    async def execute(
        self,
        call: Callable[[dict[str, Any]], Awaitable[str]],
        payload: dict[str, Any],
    ) -> str:
        middlewares = self._middlewares

        async def next_call(idx: int, p: dict[str, Any]) -> str:
            if idx >= len(middlewares):
                return await call(p)
            return await middlewares[idx](p, lambda np: next_call(idx + 1, np))

        return await next_call(0, payload)
