"""Telegram client send-path tests.

Two test groups:

1. **Existing retry / redaction behavior** (test_post_with_retry_*) —
   updated for P6.1 to drive the new ``_post_with_retry`` signature
   which accepts a ``BotProfile`` instead of reading credentials from
   the legacy top-level Settings fields directly. The wire-shape
   contract (3 attempts, 429 retry, token redaction on exception +
   on a 4xx body description, connect-error propagation) is unchanged.

2. **Multi-bot routing** (test_send_article_*), P6.1's actual scope:
   - dry-run preview names the target ``bot``
   - absent / incomplete news profile surfaces as ``status="error"``
     (never silently routes to a different chat)
   - absent linkedin profile surfaces as ``status="error"`` the same way
   - send_articles routes to the named bot's chat_id (not the legacy
     default) when both profiles are configured distinctly
   - news-path parity: with only the legacy top-level credentials
     configured, ``bot="news"`` resolves via the back-compat fallback
     and send succeeds
"""
from __future__ import annotations

from typing import Any

import httpx
import pytest

from app.config import BotProfile, Settings
from app.schemas.article import Article
from app.services.telegram_client import TelegramClient


def _profile(name: str = "news", token: str = "t", chat_id: str = "c") -> BotProfile:
    return BotProfile(name=name, token=token, chat_id=chat_id)  # type: ignore[arg-type]


def _client(**overrides: Any) -> TelegramClient:
    base: dict[str, Any] = dict(
        _env_file=None,
        telegram_bot_token="t",
        telegram_chat_id="c",
        request_timeout_seconds=10,
    )
    base.update(overrides)
    return TelegramClient(Settings(**base))


def _article(article_id: str = "a1") -> Article:
    return Article(
        id=article_id,
        source_name="Test Source",
        source_rss="https://example.com/feed.xml",
        title="Test title",
        url="https://example.com/test",
        summary="Test summary",
    )


# --------------------------------------------------------------------------- #
# Retry + redaction (existing behavior preserved across the new signature)
# --------------------------------------------------------------------------- #


async def test_post_with_retry_success() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"ok": True, "result": {"message_id": 5}},
        )

    telegram = _client()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        result = await telegram._post_with_retry(
            http, "sendPhoto", {"chat_id": "c"}, _profile()
        )

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
        result = await telegram._post_with_retry(
            http, "sendMessage", {"chat_id": "c"}, _profile()
        )

    assert calls["n"] >= 2
    assert result["ok"] is True


async def test_post_with_retry_surfaces_connect_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("boom")

    telegram = _client()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        result = await telegram._post_with_retry(
            http, "sendMessage", {"chat_id": "c"}, _profile()
        )

    assert result["ok"] is False
    assert "boom" in str(result.get("description", ""))


async def test_post_with_retry_malformed_response_returns_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"ok": False, "description": "bad"})

    telegram = _client()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        result = await telegram._post_with_retry(
            http, "sendMessage", {"chat_id": "c"}, _profile()
        )

    assert result["ok"] is False
    assert "bad" in str(result.get("description", ""))


async def test_post_with_retry_redacts_token_from_exception_message() -> None:
    token = "supersecrettoken123"

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(f"failed for url https://api.telegram.org/bot{token}/sendMessage")

    telegram = _client(telegram_bot_token=token)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        result = await telegram._post_with_retry(
            http,
            "sendMessage",
            {"chat_id": "c"},
            _profile(token=token),
        )

    description = str(result.get("description", ""))
    assert result["ok"] is False
    assert token not in description
    assert "***" in description


async def test_post_with_retry_redacts_token_from_4xx_description_body() -> None:
    """Telegram 4xx errors sometimes echo the request URL in the
    description; the client must redact the token in that case too —
    not only on connect exceptions. P6.1 added this guard (previously
    the redaction was exception-path-only)."""
    token = "tok_does_leak_via_4xx_body"

    def handler(request: httpx.Request) -> httpx.Response:
        # Simulate Telegram echoing the request URL in a 400 description.
        return httpx.Response(
            400,
            json={
                "ok": False,
                "description": f"Bad Request: chat not found for "
                f"https://api.telegram.org/bot{token}/sendMessage",
            },
        )

    telegram = _client(telegram_bot_token=token)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        # Three attempts, all return the same 400 — _post_with_retry returns
        # the data dict on the final attempt.
        result = await telegram._post_with_retry(
            http,
            "sendMessage",
            {"chat_id": "c"},
            _profile(token=token),
        )

    assert result["ok"] is False
    description = str(result.get("description", ""))
    assert token not in description, (
        f"token leaked via 4xx description body: {description!r}"
    )
    assert "***" in description


async def test_post_with_retry_exhausts_three_attempts() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        raise httpx.ConnectError("boom")

    telegram = _client()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        result = await telegram._post_with_retry(
            http, "sendMessage", {"chat_id": "c"}, _profile()
        )

    assert calls["n"] == 3
    assert result["ok"] is False


# --------------------------------------------------------------------------- #
# Multi-bot routing (P6.1)
# --------------------------------------------------------------------------- #


async def test_dry_run_preview_names_target_bot_news() -> None:
    """A dry-run preview carries the target bot name so previews (the
    coordinator's ``deliver_telegram`` tool surface, P6.2) name the
    destination — acceptance: 'dry-run previews show bot=linkedin + target
    chat'. News parity here: previews show bot=news."""
    telegram = _client()
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"ok": True, "result": {"message_id": 1}}))) as http:
        result = await telegram.send_article(http, _article("a1"), dry_run=True, bot="news")
    assert result["status"] == "dry_run"
    assert result["bot"] == "news"


async def test_dry_run_preview_names_target_bot_linkedin() -> None:
    """Mirror of the news test, for the linkedin profile — pins that the
    bot name lands in the preview for *both* supported profiles, not only
    the news default."""
    telegram = _client()
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"ok": True, "result": {"message_id": 1}}))) as http:
        result = await telegram.send_article(http, _article("a1"), dry_run=True, bot="linkedin")
    assert result["status"] == "dry_run"
    assert result["bot"] == "linkedin"


async def test_send_article_reports_error_when_news_profile_unconfigured() -> None:
    """No legacy creds, no explicit news profile env vars → the news
    profile is None → send_article returns status=error with the bot
    name in the report. The run never silently falls through to a
    configured linkedin chat.

    The mock handler counts calls so the test asserts BOTH the error
    report shape AND that no network attempt was made — a future
    refactor of the early-return path that accidentally falls through
    to the network would still produce status=error here (because the
    profile has no token, and the network call would fail), but the
    `calls["n"] == 0` assertion catches the silent-fallthrough before
    it can ship as a regression."""
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(200, json={"ok": True, "result": {"message_id": 1}})

    telegram = TelegramClient(
        Settings(
            _env_file=None,
            telegram_bot_token=None,
            telegram_chat_id=None,
            telegram_news_bot_token=None,
            telegram_news_chat_id=None,
            telegram_linkedin_bot_token=None,
            telegram_linkedin_chat_id=None,
        )
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        result = await telegram.send_article(http, _article(), dry_run=False, bot="news")
    assert result["status"] == "error"
    assert result["bot"] == "news"
    assert "news" in result["error"].lower()
    assert calls["n"] == 0, (
        f"expected zero network attempts (profile is None); got {calls['n']}"
    )


async def test_send_article_reports_error_when_linkedin_profile_unconfigured() -> None:
    """The linkedin profile is optional until that bot exists (P1.2
    acceptance: dry-run needs no new required keys). A non-dry-run send
    when it isn't configured surfaces status=error naming the linkedin
    bot — NOT a silent fallthrough to the news chat (the risk row in
    the plan).

    The counting handler pins that no network call landed (same logic
    as the news test above — a future refactor that leaks the path is
    caught here too)."""
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(200, json={"ok": True, "result": {"message_id": 1}})

    telegram = TelegramClient(
        Settings(
            _env_file=None,
            telegram_bot_token="news-tok",
            telegram_chat_id="news-chat",
            telegram_linkedin_bot_token=None,
            telegram_linkedin_chat_id=None,
        )
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        result = await telegram.send_article(http, _article(), dry_run=False, bot="linkedin")
    assert result["status"] == "error"
    assert result["bot"] == "linkedin"
    assert "linkedin" in result["error"].lower()
    assert calls["n"] == 0, (
        f"expected zero network attempts (linkedin profile is None); got {calls['n']}"
    )


async def test_send_article_routes_to_linkedin_chat_when_configured() -> None:
    """End-to-end-ish: when both news (via legacy fallback) + linkedin
    profiles are configured distinctly, sending bot=linkedin lands at
    the linkedin chat_id — never the news chat_id. The mock records the
    payload the client posted."""
    posted: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        import json as _json

        posted.append(_json.loads(request.content.decode("utf-8")))
        return httpx.Response(200, json={"ok": True, "result": {"message_id": 9}})

    telegram = TelegramClient(
        Settings(
            _env_file=None,
            telegram_bot_token="news-tok",
            telegram_chat_id="news-chat",
            telegram_linkedin_bot_token="li-tok",
            telegram_linkedin_chat_id="li-chat",
            request_timeout_seconds=10,
        )
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        result = await telegram.send_article(http, _article(), dry_run=False, bot="linkedin")

    assert result["status"] == "sent"
    assert result["bot"] == "linkedin"
    assert result["message_id"] == 9
    # Sent to the linkedin chat, never the news chat.
    assert posted, "no payload was posted"
    assert posted[-1]["chat_id"] == "li-chat"
    assert posted[-1]["chat_id"] != "news-chat"


async def test_send_article_news_path_parity_via_legacy_fallback() -> None:
    """The existing news-digest ``run`` path is unchanged by P6.1: with
    only the legacy top-level credentials set (no explicit
    ``telegram_news_*`` env vars), ``bot="news"`` resolves via the
    back-compat fallback in Settings.bot_profile and send succeeds."""
    posted: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        import json as _json

        posted.append(_json.loads(request.content.decode("utf-8")))
        return httpx.Response(200, json={"ok": True, "result": {"message_id": 3}})

    telegram = TelegramClient(
        Settings(
            _env_file=None,
            telegram_bot_token="legacy-tok",
            telegram_chat_id="legacy-chat",
            request_timeout_seconds=10,
        )
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        result = await telegram.send_article(http, _article(), dry_run=False, bot="news")

    assert result["status"] == "sent"
    assert result["bot"] == "news"
    assert posted[-1]["chat_id"] == "legacy-chat"


async def test_send_articles_no_bot_kwarg_defaults_to_news(monkeypatch: pytest.MonkeyPatch) -> None:
    """``send_articles`` ``bot`` kwarg defaults to ``"news"`` so the
    existing deliver_node call (which we updated to pass ``bot="news"``
    explicitly) would also work with the default — parity safety net.

    Patches ``_post_with_retry`` to record the payloads + return success so
    we can assert the calls landed at the news chat_id without opening a
    real httpx transport (``send_articles`` opens its own client)."""
    posted: list[tuple[str, dict[str, Any], BotProfile]] = []

    async def _stub_post(client: httpx.AsyncClient, method: str, payload: dict[str, Any], profile: BotProfile) -> dict[str, Any]:
        posted.append((method, payload, profile))
        return {"ok": True, "result": {"message_id": 11}}

    telegram = TelegramClient(
        Settings(
            _env_file=None,
            telegram_bot_token="legacy-tok",
            telegram_chat_id="legacy-chat",
            request_timeout_seconds=10,
        )
    )
    monkeypatch.setattr(telegram, "_post_with_retry", _stub_post)
    # No bot kwarg — should default to "news".
    results = await telegram.send_articles([_article("a1"), _article("a2")], dry_run=False)

    assert len(results) == 2
    assert all(r["status"] == "sent" for r in results)
    assert all(r["bot"] == "news" for r in results)
    # Each call landed at the legacy-chat (news profile via back-compat).
    assert all(p[2].chat_id == "legacy-chat" for p in posted)
    assert all(p[2].name == "news" for p in posted)


async def test_send_articles_dry_run_previews_carry_bot_name() -> None:
    """A dry-run batch preview carries the bot name in every result so
    P6.2's surface-level preview (one summary across many proposals)
    doesn't have to re-resolve config to know the target. Dry-run never
    touches the network — no monkeypatch needed."""
    telegram = _client()
    results = await telegram.send_articles([_article("a1")], dry_run=True, bot="linkedin")
    assert len(results) == 1
    assert results[0]["bot"] == "linkedin"
    assert results[0]["status"] == "dry_run"