from __future__ import annotations

import asyncio
import html
import logging
from typing import Any
from urllib.parse import urlparse

import httpx

from app.config import Settings
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
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def send_articles(self, articles: list[Article], dry_run: bool) -> list[dict[str, Any]]:
        # Deliberately sequential, not gathered: Telegram rate-limits per chat, and
        # digest ordering matters (story 1 should land before story 2 in-channel).
        # Parallelizing risks 429 thrashing and out-of-order delivery.
        timeout = httpx.Timeout(self.settings.request_timeout_seconds)
        async with httpx.AsyncClient(timeout=timeout) as client:
            results: list[dict[str, Any]] = []
            for article in articles:
                result = await self.send_article(client, article, dry_run=dry_run)
                results.append(result)
            return results

    async def send_article(
        self,
        client: httpx.AsyncClient,
        article: Article,
        dry_run: bool,
    ) -> dict[str, Any]:
        title = article.effective_title
        summary = article.summary or "Summary unavailable."

        caption = build_telegram_caption(article.url, title, summary)
        text_message = build_telegram_text(article.url, title, summary)

        if dry_run:
            return {
                "article_id": article.id,
                "status": "dry_run",
                "mode": "photo" if article.image_url else "text",
                "preview": caption if article.image_url else text_message,
            }

        if not self.settings.telegram_bot_token or not self.settings.telegram_chat_id:
            return {
                "article_id": article.id,
                "status": "error",
                "error": "Telegram credentials are missing.",
            }

        if article.image_url:
            result = await self._send_photo(client, article, caption)
            if result is not None:
                return result
            logger.warning("Photo send failed for %s, falling back to text", article.id)

        return await self._send_text(client, article, text_message)

    async def _send_photo(
        self,
        client: httpx.AsyncClient,
        article: Article,
        caption: str,
    ) -> dict[str, Any] | None:
        """Send as a photo message; None means the caller should fall back to text."""
        payload = {
            "chat_id": self.settings.telegram_chat_id,
            "photo": article.image_url,
            "caption": caption,
            "parse_mode": self.settings.telegram_parse_mode,
        }
        sent = await self._post_with_retry(client, "sendPhoto", payload)
        if sent.get("ok"):
            return {
                "article_id": article.id,
                "status": "sent",
                "mode": "photo",
                "message_id": sent["result"].get("message_id"),
            }
        return None

    async def _send_text(
        self,
        client: httpx.AsyncClient,
        article: Article,
        text_message: str,
    ) -> dict[str, Any]:
        payload = {
            "chat_id": self.settings.telegram_chat_id,
            "text": text_message,
            "parse_mode": self.settings.telegram_parse_mode,
            "disable_web_page_preview": False,
        }
        sent = await self._post_with_retry(client, "sendMessage", payload)
        if sent.get("ok"):
            return {
                "article_id": article.id,
                "status": "sent",
                "mode": "text",
                "message_id": sent["result"].get("message_id"),
            }

        return {
            "article_id": article.id,
            "status": "error",
            "error": sent.get("description", "Telegram send failed."),
        }

    async def _post_with_retry(
        self,
        client: httpx.AsyncClient,
        method: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        token = self.settings.telegram_bot_token
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
                return data
            except Exception as exc:
                if attempt < attempts:
                    await asyncio.sleep(attempt)
                    continue
                # The bot token lives in `url` (Telegram's API requires it in the
                # path, not a header); redact it in case a future httpx exception
                # type ever stringifies the request URL.
                message = str(exc).replace(token, "***")
                logger.warning("Telegram %s failed after %s attempts: %s", method, attempts, message)
                return {"ok": False, "description": message}

        return {"ok": False, "description": "Unknown send failure."}
