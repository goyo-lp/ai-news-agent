from __future__ import annotations

import asyncio
import logging
import re
from datetime import timezone

import httpx

from app.config import Settings
from app.schemas.article import Article

logger = logging.getLogger(__name__)

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> list[str]:
    cleaned = " ".join(text.strip().split())
    if not cleaned:
        return []
    return [part.strip() for part in _SENTENCE_SPLIT.split(cleaned) if part.strip()]


def enforce_sentence_count(text: str, count: int = 3) -> str:
    sentences = split_sentences(text)
    if len(sentences) >= count:
        return " ".join(sentences[:count])

    fallbacks = [
        "This update is relevant to current AI developments.",
        "The linked source provides additional technical and business context.",
        "Read the full article for complete details and implications.",
    ]

    while len(sentences) < count:
        sentences.append(fallbacks[len(sentences) % len(fallbacks)])

    return " ".join(sentences[:count])


class OpenRouterClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def summarize_articles(self, articles: list[Article], dry_run: bool) -> list[Article]:
        semaphore = asyncio.Semaphore(min(self.settings.http_concurrency, 4))

        async with httpx.AsyncClient(timeout=self.settings.request_timeout_seconds) as client:
            async def worker(article: Article) -> Article:
                async with semaphore:
                    summary = await self.summarize_article(client, article, dry_run=dry_run)
                    updated = article.model_copy(deep=True)
                    updated.summary = summary
                    return updated

            return await asyncio.gather(*(worker(article) for article in articles))

    async def summarize_article(
        self,
        client: httpx.AsyncClient,
        article: Article,
        dry_run: bool,
    ) -> str:
        if dry_run or not self.settings.openrouter_api_key:
            return self._fallback_summary(article)

        prompt = self._build_prompt(article)
        headers = {
            "Authorization": f"Bearer {self.settings.openrouter_api_key}",
            "Content-Type": "application/json",
        }
        if self.settings.openrouter_site_url:
            headers["HTTP-Referer"] = self.settings.openrouter_site_url
        if self.settings.openrouter_app_name:
            headers["X-Title"] = self.settings.openrouter_app_name

        payload = {
            "model": self.settings.openrouter_model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You summarize AI news for a Telegram digest. "
                        "Return exactly 3 concise sentences."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
            "max_tokens": 220,
        }

        try:
            first_pass = await self._request_summary(client, headers, payload)
            if len(split_sentences(first_pass)) >= 3:
                return enforce_sentence_count(first_pass, count=3)

            # Retry once with an explicit output reminder if the first response is malformed.
            retry_payload = dict(payload)
            retry_payload["messages"] = [
                *payload["messages"],
                {
                    "role": "user",
                    "content": "Rewrite your answer as exactly 3 sentences.",
                },
            ]
            second_pass = await self._request_summary(client, headers, retry_payload)
            return enforce_sentence_count(second_pass, count=3)
        except Exception as exc:
            logger.warning("OpenRouter call failed for %s: %s", article.id, exc)
            return self._fallback_summary(article)

    async def _request_summary(
        self,
        client: httpx.AsyncClient,
        headers: dict[str, str],
        payload: dict[str, object],
    ) -> str:
        response = await client.post(
            f"{self.settings.openrouter_base_url}/chat/completions",
            headers=headers,
            json=payload,
        )
        response.raise_for_status()
        return str(response.json()["choices"][0]["message"]["content"])

    def _build_prompt(self, article: Article) -> str:
        published = (
            article.published_at.astimezone(timezone.utc).isoformat()
            if article.published_at is not None
            else "unknown"
        )
        context = article.effective_summary_source
        return (
            f"Title: {article.effective_title}\n"
            f"Source: {article.source_name}\n"
            f"Published: {published}\n"
            f"URL: {article.url}\n"
            f"Context: {context}\n"
            "Write exactly 3 sentences focused on key facts and why this matters."
        )

    def _fallback_summary(self, article: Article) -> str:
        context = article.effective_summary_source.strip()
        title_sentence = f"{article.effective_title} is a notable AI update from {article.source_name}."
        context_sentence = (
            f"Key context: {context[:180].rstrip('.')}."
            if context
            else "The story appears relevant to current AI research and product developments."
        )
        action_sentence = "Open the link to review full details, claims, and technical context."
        return enforce_sentence_count(
            f"{title_sentence} {context_sentence} {action_sentence}",
            count=3,
        )
