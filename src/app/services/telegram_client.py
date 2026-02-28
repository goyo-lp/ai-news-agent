from __future__ import annotations

import asyncio
import html
import logging
from typing import Any

import httpx

from app.config import Settings
from app.schemas.article import Article

logger = logging.getLogger(__name__)

TELEGRAM_CAPTION_LIMIT = 1024
TELEGRAM_TEXT_LIMIT = 4096


def _truncate_text(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    if limit <= 3:
        return value[:limit]
    return value[: limit - 3].rstrip() + "..."


def build_telegram_caption(url: str, title: str, summary: str, limit: int = TELEGRAM_CAPTION_LIMIT) -> str:
    safe_url = html.escape(url, quote=True)
    safe_title = html.escape(title, quote=False)
    safe_summary = html.escape(summary, quote=False)

    max_title_len = max(40, min(200, limit // 3))
    safe_title = _truncate_text(safe_title, max_title_len)

    prefix = f'<a href="{safe_url}">{safe_title}</a>\n\n'
    available = max(limit - len(prefix), 0)
    body = _truncate_text(safe_summary, available)
    caption = prefix + body
    return caption[:limit]


def build_telegram_text(url: str, title: str, summary: str, limit: int = TELEGRAM_TEXT_LIMIT) -> str:
    safe_url = html.escape(url, quote=True)
    safe_title = html.escape(title, quote=False)
    safe_summary = html.escape(summary, quote=False)

    text = f'<a href="{safe_url}">{safe_title}</a>\n\n{safe_summary}'
    return _truncate_text(text, limit)


class TelegramClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def send_articles(self, articles: list[Article], dry_run: bool) -> list[dict[str, Any]]:
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

            logger.warning("Photo send failed for %s, falling back to text", article.id)

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
                data = response.json()
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
                return {"ok": False, "description": str(exc)}

        return {"ok": False, "description": "Unknown send failure."}
