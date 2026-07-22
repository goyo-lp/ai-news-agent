from __future__ import annotations

import asyncio
import html
import logging
from typing import Any
from urllib.parse import urlparse

import httpx

from app.config import BotName, BotProfile, Settings
from app.schemas.article import Article

logger = logging.getLogger(__name__)

TELEGRAM_CAPTION_LIMIT = 1024
TELEGRAM_TEXT_LIMIT = 4096

_ALLOWED_URL_SCHEMES = {"http", "https"}


def _safe_href(url: str) -> str:
    """Return url unchanged if it's http(s); otherwise '#', so a javascript:/data:/
    etc. scheme scraped from an article never becomes a clickable link."""
    return url if urlparse(url).scheme.lower() in _ALLOWED_URL_SCHEMES else "#"


def _truncate_text(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    if limit <= 3:
        return value[:limit]
    return value[: limit - 3].rstrip() + "..."


def _escape_fields(url: str, title: str, summary: str) -> tuple[str, str, str]:
    return (
        html.escape(_safe_href(url), quote=True),
        html.escape(title, quote=False),
        html.escape(summary, quote=False),
    )


def build_telegram_caption(url: str, title: str, summary: str, limit: int = TELEGRAM_CAPTION_LIMIT) -> str:
    """Build a photo-caption HTML string, reserving up to 1/3 of `limit` for the
    title (clamped 40-200 chars) and giving the rest to the summary."""
    safe_url, safe_title, safe_summary = _escape_fields(url, title, summary)

    max_title_len = max(40, min(200, limit // 3))
    safe_title = _truncate_text(safe_title, max_title_len)

    prefix = f'<a href="{safe_url}">{safe_title}</a>\n\n'
    available = max(limit - len(prefix), 0)
    body = _truncate_text(safe_summary, available)
    caption = prefix + body
    return caption[:limit]


def build_telegram_text(url: str, title: str, summary: str, limit: int = TELEGRAM_TEXT_LIMIT) -> str:
    safe_url, safe_title, safe_summary = _escape_fields(url, title, summary)

    text = f'<a href="{safe_url}">{safe_title}</a>\n\n{safe_summary}'
    return _truncate_text(text, limit)


class TelegramClient:
    """Telegram delivery client for the AI News Agent.

    Multi-bot routing (P6.1): the client accepts a named ``bot`` profile
    (``"news"`` or ``"linkedin"``) per call. Profile resolution semantics
    (incl. legacy top-level ``telegram_bot_token`` / ``telegram_chat_id``
    back-compat fallback for the ``"news"`` profile) live in
    :meth:`Settings.bot_profile`; this client just consumes the resolved
    :class:`BotProfile`. An unconfigured/absent profile returns ``None``
    from there (config-only signal, never raises) and this client
    reports it as ``status="error"`` with a clear reason — never
    silently routes to the wrong chat.

    Per-call ``bot="news"`` is preserved as the news-digest path's
    argument; per-call ``bot="linkedin"`` is the LinkedIn-bot path the
    coordinator's ``deliver_telegram`` tool (P6.2) invokes.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def send_articles(
        self,
        articles: list[Article],
        dry_run: bool,
        *,
        bot: BotName = "news",
    ) -> list[dict[str, Any]]:
        """Sequentially deliver ``articles`` to the named bot's chat.

        Deliberately sequential, not gathered: Telegram rate-limits per
        chat, and digest ordering matters (story 1 should land before
        story 2 in-channel). Parallelizing risks 429 thrashing and
        out-of-order delivery. ``bot`` selects the named profile; the
        news-digest ``run`` path passes ``bot="news"`` (the default, so
        its existing calls without the kwarg are unchanged)."""
        timeout = httpx.Timeout(self.settings.request_timeout_seconds)
        async with httpx.AsyncClient(timeout=timeout) as client:
            results: list[dict[str, Any]] = []
            for article in articles:
                result = await self.send_article(client, article, dry_run=dry_run, bot=bot)
                results.append(result)
            return results

    async def send_article(
        self,
        client: httpx.AsyncClient,
        article: Article,
        dry_run: bool,
        *,
        bot: BotName = "news",
    ) -> dict[str, Any]:
        title = article.effective_title
        summary = article.summary or "Summary unavailable."

        caption = build_telegram_caption(article.url, title, summary)
        text_message = build_telegram_text(article.url, title, summary)

        # Dry-run preview carries the bot name so previews name the target bot
        # (P6.1 + P6.2 acceptance). Never hexdumps the token.
        if dry_run:
            return {
                "article_id": article.id,
                "status": "dry_run",
                "bot": bot,
                "mode": "photo" if article.image_url else "text",
                "preview": caption if article.image_url else text_message,
            }

        profile = self.settings.bot_profile(bot)
        if profile is None or not profile.is_complete():
            return {
                "article_id": article.id,
                "status": "error",
                "bot": bot,
                "error": f"Telegram bot profile {bot!r} is not configured "
                "(token + chat_id required).",
            }

        if article.image_url:
            result = await self._send_photo(client, article, caption, profile)
            if result is not None:
                return result
            logger.warning("Photo send failed for %s, falling back to text", article.id)

        return await self._send_text(client, article, text_message, profile)

    async def _send_photo(
        self,
        client: httpx.AsyncClient,
        article: Article,
        caption: str,
        profile: BotProfile,
    ) -> dict[str, Any] | None:
        """Send as a photo message; None means the caller should fall back to text."""
        payload = {
            "chat_id": profile.chat_id,
            "photo": article.image_url,
            "caption": caption,
            "parse_mode": self.settings.telegram_parse_mode,
        }
        sent = await self._post_with_retry(client, "sendPhoto", payload, profile)
        if sent.get("ok"):
            return {
                "article_id": article.id,
                "status": "sent",
                "bot": profile.name,
                "mode": "photo",
                "message_id": sent["result"].get("message_id"),
            }
        return None

    async def _send_text(
        self,
        client: httpx.AsyncClient,
        article: Article,
        text_message: str,
        profile: BotProfile,
    ) -> dict[str, Any]:
        payload = {
            "chat_id": profile.chat_id,
            "text": text_message,
            "parse_mode": self.settings.telegram_parse_mode,
            "disable_web_page_preview": False,
        }
        sent = await self._post_with_retry(client, "sendMessage", payload, profile)
        if sent.get("ok"):
            return {
                "article_id": article.id,
                "status": "sent",
                "bot": profile.name,
                "mode": "text",
                "message_id": sent["result"].get("message_id"),
            }

        return {
            "article_id": article.id,
            "status": "error",
            "bot": profile.name,
            "error": sent.get("description", "Telegram send failed."),
        }

    async def _post_with_retry(
        self,
        client: httpx.AsyncClient,
        method: str,
        payload: dict[str, Any],
        profile: BotProfile,
    ) -> dict[str, Any]:
        token = profile.token
        if not token:
            return {"ok": False, "description": "Missing bot token."}

        url = f"https://api.telegram.org/bot{token}/{method}"
        attempts = 3

        for attempt in range(1, attempts + 1):
            try:
                response = await client.post(url, json=payload)
                data: dict[str, Any] = response.json()
                if response.status_code == 429:
                    retry_after = int(data.get("parameters", {}).get("retry_after", 2))
                    await asyncio.sleep(retry_after)
                    continue
                if response.is_success and data.get("ok"):
                    return data
                if attempt < attempts:
                    await asyncio.sleep(attempt)
                    continue
                # The bot token lives in `url` (Telegram's API requires it in
                # the path, not a header); redact the token from the returned
                # description in case a Telegram 4xx body ever echoes the URL
                # or a future httpx exception type stringifies the request URL.
                if isinstance(data.get("description"), str):
                    data["description"] = data["description"].replace(token, "***")
                return data
            except Exception as exc:
                if attempt < attempts:
                    await asyncio.sleep(attempt)
                    continue
                # The bot token lives in `url` (Telegram's API requires it in
                # the path, not a header); redact it in case a future httpx
                # exception type ever stringifies the request URL.
                message = str(exc).replace(token, "***")
                logger.warning("Telegram %s failed after %s attempts: %s", method, attempts, message)
                return {"ok": False, "description": message}

        return {"ok": False, "description": "Unknown send failure."}
