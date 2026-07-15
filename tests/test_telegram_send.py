from __future__ import annotations

import httpx

from app.config import Settings
from app.services.telegram_client import TelegramClient


def _client() -> TelegramClient:
    return TelegramClient(
        Settings(
            _env_file=None,
            telegram_bot_token="t",
            telegram_chat_id="c",
            request_timeout_seconds=10,
        )
    )


async def test_post_with_retry_success() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"ok": True, "result": {"message_id": 5}},
        )

    telegram = _client()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        result = await telegram._post_with_retry(http, "sendPhoto", {"chat_id": "c"})

    assert result["ok"] is True
    assert result["result"]["message_id"] == 5


async def test_post_with_retry_recovers_from_429() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(
                429,
                json={"ok": False, "parameters": {"retry_after": 0}},
            )
        return httpx.Response(200, json={"ok": True, "result": {"message_id": 7}})

    telegram = _client()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        result = await telegram._post_with_retry(http, "sendMessage", {"chat_id": "c"})

    assert calls["n"] >= 2
    assert result["ok"] is True


async def test_post_with_retry_surfaces_connect_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("boom")

    telegram = _client()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        result = await telegram._post_with_retry(http, "sendMessage", {"chat_id": "c"})

    assert result["ok"] is False
    assert "boom" in str(result.get("description", ""))


async def test_post_with_retry_malformed_response_returns_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"ok": False, "description": "bad"})

    telegram = _client()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        result = await telegram._post_with_retry(http, "sendMessage", {"chat_id": "c"})

    assert result["ok"] is False
    assert "bad" in str(result.get("description", ""))


async def test_post_with_retry_redacts_token_from_exception_message() -> None:
    token = "supersecrettoken123"

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(f"failed for url https://api.telegram.org/bot{token}/sendMessage")

    telegram = TelegramClient(
        Settings(
            _env_file=None,
            telegram_bot_token=token,
            telegram_chat_id="c",
            request_timeout_seconds=10,
        )
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        result = await telegram._post_with_retry(http, "sendMessage", {"chat_id": "c"})

    description = str(result.get("description", ""))
    assert result["ok"] is False
    assert token not in description
    assert "***" in description


async def test_post_with_retry_exhausts_three_attempts() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        raise httpx.ConnectError("boom")

    telegram = _client()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        result = await telegram._post_with_retry(http, "sendMessage", {"chat_id": "c"})

    assert calls["n"] == 3
    assert result["ok"] is False
