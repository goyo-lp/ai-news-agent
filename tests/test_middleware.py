import asyncio

from app.services.middleware import (
    MiddlewareChain,
    reasoning_effort_middleware,
    strip_think_blocks,
    strip_reasoning_middleware,
)


OPEN = chr(60) + "think" + chr(62)
CLOSE = chr(60) + "/think" + chr(62)


def _leak() -> str:
    return OPEN + "The user wants 3 sentences. Let me explain." + CLOSE + " Final. Two. Three."


def test_strip_think_blocks_removes_reasoning_traces() -> None:
    out = strip_think_blocks(_leak())
    assert OPEN not in out
    assert "Let me explain" not in out
    assert out.startswith("Final.")


def test_strip_think_blocks_handles_unclosed_block() -> None:
    truncated = OPEN + "partial reasoning without close tag"
    out = strip_think_blocks(truncated)
    assert out == ""


def test_strip_think_blocks_passes_through_clean_text() -> None:
    clean = "Apple released iOS 19. It adds on-device LLM. Battery life improved."
    assert strip_think_blocks(clean) == clean


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def test_reasoning_effort_middleware_injects_effort() -> None:
    captured = {}

    async def base_call(payload):
        captured.update(payload)
        return "ok"

    chain = MiddlewareChain([reasoning_effort_middleware(effort="high")])
    out = _run(chain.execute(base_call, {"messages": []}))
    assert out == "ok"
    assert captured["reasoning"] == {"effort": "high"}


def test_reasoning_effort_middleware_omits_field_when_none() -> None:
    captured = {}

    async def base_call(payload):
        captured.update(payload)
        return "ok"

    chain = MiddlewareChain([reasoning_effort_middleware(effort=None)])
    _run(chain.execute(base_call, {"messages": []}))
    assert "reasoning" not in captured


def test_strip_reasoning_middleware_strips_response() -> None:
    async def base_call(payload):
        return _leak()

    chain = MiddlewareChain([strip_reasoning_middleware])
    out = _run(chain.execute(base_call, {"messages": []}))
    assert OPEN not in out
    assert out.startswith("Final.")


def test_chain_runs_before_in_order_then_after_in_reverse() -> None:
    order = []

    async def mw_a(payload, call_next):
        order.append("a-before")
        r = await call_next(payload)
        order.append("a-after")
        return r

    async def mw_b(payload, call_next):
        order.append("b-before")
        r = await call_next(payload)
        order.append("b-after")
        return r

    async def base_call(payload):
        order.append("call")
        return "ok"

    chain = MiddlewareChain([mw_a, mw_b])
    _run(chain.execute(base_call, {}))
    assert order == ["a-before", "b-before", "call", "b-after", "a-after"]


def test_chain_combines_effort_and_strip() -> None:
    captured = {}

    async def base_call(payload):
        captured.update(payload)
        return _leak()

    chain = MiddlewareChain([
        reasoning_effort_middleware(effort="high"),
        strip_reasoning_middleware,
    ])
    out = _run(chain.execute(base_call, {"messages": []}))
    assert captured["reasoning"] == {"effort": "high"}
    assert OPEN not in out
    assert out.startswith("Final.")
