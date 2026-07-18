from __future__ import annotations

import asyncio
import html
import logging
import re

import httpx

from app.config import Settings
from app.schemas import DeliveryResult, LinkedInPost

logger = logging.getLogger(__name__)

TELEGRAM_MESSAGE_LIMIT = 4096


class TelegramClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def deliver_posts(self, posts: list[LinkedInPost], dry_run: bool) -> list[DeliveryResult]:
        if not posts:
            return []

        if dry_run or not (self.settings.telegram_bot_token and self.settings.telegram_chat_id):
            return [
                DeliveryResult(post_id=post.post_id, status="dry_run")
                for post in posts
            ]

        timeout = httpx.Timeout(self.settings.request_timeout_seconds)
        semaphore = asyncio.Semaphore(max(1, self.settings.telegram_send_concurrency))
        async with httpx.AsyncClient(timeout=timeout) as client:
            async def worker(idx: int, post: LinkedInPost) -> tuple[int, DeliveryResult]:
                async with semaphore:
                    try:
                        message = build_telegram_message(post)
                        message_id = await self._send_message(client, message)
                        return idx, DeliveryResult(
                            post_id=post.post_id,
                            status="sent",
                            message_id=message_id,
                        )
                    except Exception as exc:
                        logger.warning("Telegram delivery failed for %s: %s", post.post_id, exc)
                        return idx, DeliveryResult(
                            post_id=post.post_id,
                            status="error",
                            error=str(exc),
                        )

            gathered = await asyncio.gather(*(worker(idx, post) for idx, post in enumerate(posts)))
            return [result for _, result in sorted(gathered, key=lambda item: item[0])]

    async def _send_message(self, client: httpx.AsyncClient, message: str) -> int | None:
        payload = {
            "chat_id": self.settings.telegram_chat_id,
            "text": message,
            "parse_mode": self.settings.telegram_parse_mode,
            "disable_web_page_preview": True,
        }

        url = f"https://api.telegram.org/bot{self.settings.telegram_bot_token}/sendMessage"
        response = await client.post(url, json=payload)
        response.raise_for_status()

        data = response.json()
        if not isinstance(data, dict) or not bool(data.get("ok")):
            raise RuntimeError(f"Telegram API returned error: {data}")

        result = data.get("result")
        if isinstance(result, dict):
            message_id = result.get("message_id")
            if isinstance(message_id, int):
                return message_id
        return None


def build_telegram_message(post: LinkedInPost) -> str:
    title = html.escape(post.headline.strip() or "AI update")
    body_text = _normalize_message_body(post.body)
    body = html.escape(body_text)

    hashtags = post.hashtags if post.hashtags else ["#AI", "#MachineLearning", "#Innovation"]
    clean_hashtags = " ".join(_normalize_hashtag(tag) for tag in hashtags if tag.strip())

    message = (
        f"<b>{title}</b>\n\n"
        f"{body}\n\n"
        "<b>Recommended hashtags:</b>\n"
        f"{html.escape(clean_hashtags)}"
    )

    if len(message) <= TELEGRAM_MESSAGE_LIMIT:
        return message

    # Keep title + hashtag footer and trim the body to stay within Telegram limits.
    footer = f"\n\n<b>Recommended hashtags:</b>\n{html.escape(clean_hashtags)}"
    title_block = f"<b>{title}</b>\n\n"
    max_body_len = max(0, TELEGRAM_MESSAGE_LIMIT - len(title_block) - len(footer) - 3)
    trimmed_body = body[:max_body_len].rstrip()
    if max_body_len > 0:
        trimmed_body += "..."

    return f"{title_block}{trimmed_body}{footer}"


def _normalize_hashtag(tag: str) -> str:
    normalized = "#" + tag.strip().lstrip("#")
    return normalized.replace(" ", "")


def _normalize_message_body(text: str) -> str:
    raw = str(text).strip()
    if not raw:
        return "Technical update."

    # Respect user-provided paragraph breaks when present.
    blocks = [segment.strip() for segment in re.split(r"\n\s*\n+", raw) if segment.strip()]
    paragraphs: list[str] = []
    for block in blocks:
        normalized = " ".join(block.split()).strip()
        if not normalized:
            continue
        paragraphs.extend(_split_into_paragraphs(normalized))

    compact = "\n\n".join(paragraphs)
    return compact or "Technical update."


def _split_into_paragraphs(text: str) -> list[str]:
    sentences = [segment.strip() for segment in re.split(r"(?<=[.!?])\s+", text) if segment.strip()]
    if len(sentences) <= 2:
        return [text]

    # Keep Telegram output readable by grouping roughly two sentences per paragraph.
    grouped: list[str] = []
    for idx in range(0, len(sentences), 2):
        grouped.append(" ".join(sentences[idx : idx + 2]).strip())
    return grouped
