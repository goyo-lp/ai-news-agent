from __future__ import annotations

import asyncio
import json
import logging
import re
from datetime import timezone

import httpx

from app.config import Settings
from app.schemas.article import Article
from app.services.middleware import (
    MiddlewareChain,
    Middleware,
    lookup_reasoning_effort,
    reasoning_effort_middleware,
    strip_reasoning_middleware,
)

logger = logging.getLogger(__name__)

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)

# Patterns that indicate the model leaked chain-of-thought reasoning into the
# output. Matched against the start of the response (case-insensitive).
_REASONING_START = re.compile(
    r"^\s*(the user(user wants|is asking|asked)?|user wants|i need|i'll|i will|i should|"
    r"let me|let's|here (is|are)|to summarize|first|sure|okay|i can|i could|i think)\b",
    re.IGNORECASE,
)


def parse_relevance_scores(text: str, valid_keys: set[str]) -> dict[str, float]:
    """Parse a JSON object of 0-100 scores into {key: 0.0-1.0}, dropping anything malformed."""
    match = _JSON_OBJECT.search(text)
    if not match:
        return {}
    try:
        raw = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}
    if not isinstance(raw, dict):
        return {}

    scores: dict[str, float] = {}
    for key, value in raw.items():
        if key not in valid_keys:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        scores[key] = max(0.0, min(number / 100.0, 1.0))
    return scores


def split_sentences(text: str) -> list[str]:
    cleaned = " ".join(text.strip().split())
    if not cleaned:
        return []
    return [part.strip() for part in _SENTENCE_SPLIT.split(cleaned) if part.strip()]


def looks_like_reasoning(text: str) -> bool:
    """True when the response starts with chain-of-thought / meta-commentary."""
    return bool(_REASONING_START.match(text.strip()))


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
    def __init__(self, settings: Settings, middlewares: list[Middleware] | None = None) -> None:
        self.settings = settings
        if middlewares is None:
            effort = lookup_reasoning_effort(settings.openrouter_reasoning_effort)
            middlewares = [
                reasoning_effort_middleware(effort=effort),
                strip_reasoning_middleware,
            ]
        self._chain = MiddlewareChain(middlewares)

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
        headers = self._build_headers()

        payload = {
            "model": self.settings.openrouter_model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You summarize AI news for a Telegram digest. "
                        "Return exactly 3 short factual sentences (aim for under 20 words each) "
                        "about what the article reports and why it matters. Strip filler and "
                        "hedge words. Do not name or allude to the publisher, source, or author. "
                        "Output ONLY the 3 sentences — no reasoning, no preamble, no labels, no "
                        "meta-commentary about the task."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
            "max_tokens": 10000,
        }

        try:
            first_pass = await self._request_summary(client, headers, payload)
            first_pass = first_pass.strip()
            if len(split_sentences(first_pass)) >= 3 and not looks_like_reasoning(first_pass):
                return enforce_sentence_count(first_pass, count=3)

            # Retry once with a stronger output reminder if the first response was
            # malformed or leaked chain-of-thought reasoning into the output.
            retry_payload = dict(payload)
            retry_payload["messages"] = [
                *payload["messages"],
                {
                    "role": "assistant",
                    "content": first_pass,
                },
                {
                    "role": "user",
                    "content": (
                        "Your previous response leaked reasoning or was malformed. "
                        "Start fresh and write ONLY 3 factual sentences about the article's "
                        "content. Do not mention the publisher. Do not explain what you are "
                        "doing. Output only the 3 sentences."
                    ),
                },
            ]
            second_pass = await self._request_summary(client, headers, retry_payload)
            second_pass = second_pass.strip()
            if looks_like_reasoning(second_pass):
                logger.warning("OpenRouter kept leaking reasoning for %s; falling back", article.id)
                return self._fallback_summary(article)
            return enforce_sentence_count(second_pass, count=3)
        except Exception as exc:
            logger.warning("OpenRouter call failed for %s: %s", article.id, exc)
            return self._fallback_summary(article)

    async def score_articles_relevance(
        self,
        articles: list[Article],
        dry_run: bool,
    ) -> dict[str, float]:
        """One batched call scoring each article's relevance, as {article_id: 0.0-1.0}.

        Returns an empty dict when skipped (dry-run, no key) or on any failure,
        which callers treat as "keep the deterministic ranking".
        """
        if dry_run or not self.settings.openrouter_api_key or not articles:
            return {}

        index_to_id = {str(idx + 1): article.id for idx, article in enumerate(articles)}
        lines = []
        for idx, article in enumerate(articles):
            context = article.effective_summary_source.strip()[:200]
            line = f"{idx + 1}. {article.effective_title}"
            if context:
                line += f" — {context}"
            lines.append(line)

        prompt = (
            "Rate each AI news item from 0 to 100 for relevance to a daily digest that tracks: "
            "frontier AI company updates (OpenAI, Anthropic, Google/Gemini, xAI, Meta, Mistral, "
            "DeepSeek, Cursor), new model releases including open-source/open-weights models, "
            "AI startup funding rounds, major product launches, and trending open-source/GitHub "
            "projects. Penalize academic paper abstracts, event roundups, tutorials, opinion "
            "pieces, and podcasts.\n\n"
            + "\n".join(lines)
            + '\n\nReturn only a JSON object mapping item number to score, e.g. {"1": 85, "2": 20}.'
        )

        payload = {
            "model": self.settings.openrouter_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": 10000,
        }

        try:
            async with httpx.AsyncClient(timeout=self.settings.request_timeout_seconds) as client:
                content = await self._request_summary(client, self._build_headers(), payload)
        except Exception as exc:
            logger.warning("OpenRouter relevance scoring failed: %s", exc)
            return {}

        by_index = parse_relevance_scores(content, set(index_to_id))
        return {index_to_id[key]: value for key, value in by_index.items()}

    def _build_headers(self) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.settings.openrouter_api_key}",
            "Content-Type": "application/json",
        }
        if self.settings.openrouter_site_url:
            headers["HTTP-Referer"] = self.settings.openrouter_site_url
        if self.settings.openrouter_app_name:
            headers["X-Title"] = self.settings.openrouter_app_name
        return headers

    async def _request_summary(
        self,
        client: httpx.AsyncClient,
        headers: dict[str, str],
        payload: dict[str, object],
    ) -> str:
        base_url = self.settings.openrouter_base_url

        async def base_call(p: dict) -> str:
            response = await client.post(
                f"{base_url}/chat/completions",
                headers=headers,
                json=p,
            )
            response.raise_for_status()
            return str(response.json()["choices"][0]["message"]["content"])

        return await self._chain.execute(base_call, dict(payload))

    def _build_prompt(self, article: Article) -> str:
        published = (
            article.published_at.astimezone(timezone.utc).isoformat()
            if article.published_at is not None
            else "unknown"
        )
        context = article.effective_summary_source
        return (
            f"Title: {article.effective_title}\n"
            f"Published: {published}\n"
            f"URL: {article.url}\n"
            f"Context: {context}\n"
            "Summarize what this article reports and why it matters, in exactly 3 short "
            "sentences (under 20 words each). Keep each sentence tight and factual. "
            "Focus on the article's content only — do not mention the publisher or "
            "who wrote it. Output only the 3 sentences."
        )

    def _fallback_summary(self, article: Article) -> str:
        context = article.effective_summary_source.strip()
        title_sentence = f"{article.effective_title} is a notable AI update."
        context_sentence = (
            f"Key context: {context[:120].rstrip('.')}."
            if context
            else "Relevant to current AI research and product developments."
        )
        action_sentence = "See the link for full details."
        return enforce_sentence_count(
            f"{title_sentence} {context_sentence} {action_sentence}",
            count=3,
        )
