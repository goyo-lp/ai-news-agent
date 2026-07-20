from __future__ import annotations

import asyncio
from contextlib import suppress
import json
import logging
import re
from typing import Any, cast

import httpx

from app.config import Settings
from app.schemas import LinkedInPost, ResearchBrief, StyleProfile
from app.services.api_usage_tracker import record_openrouter_http_response

logger = logging.getLogger(__name__)

_TARGET_POSTS = 5
_MIN_POST_WORDS = 105
_MAX_POST_WORDS = 182


class PostGenerator:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def generate_posts(
        self,
        briefs: list[ResearchBrief],
        style_profile: StyleProfile,
        dry_run: bool,
    ) -> list[LinkedInPost]:
        if not briefs:
            return []

        target_count = min(_TARGET_POSTS, len(briefs))

        if (self.settings.openrouter_api_key or "").strip() and not dry_run:
            fallback_task = asyncio.create_task(
                self._fallback_posts_async(briefs, style_profile, target_count=target_count)
            )
            generation_tasks: list[asyncio.Task[list[LinkedInPost]]] = [
                asyncio.create_task(
                    self._generate_with_llm_model(
                        briefs,
                        style_profile,
                        target_count=target_count,
                        model=self.settings.openrouter_model,
                    )
                )
            ]

            secondary_model = (self.settings.openrouter_post_secondary_model or "").strip()
            if secondary_model:
                generation_tasks.append(
                    asyncio.create_task(
                        self._generate_with_llm_model(
                            briefs,
                            style_profile,
                            target_count=target_count,
                            model=secondary_model,
                        )
                    )
                )

            generated = await asyncio.gather(*generation_tasks, return_exceptions=True)
            candidates: list[list[LinkedInPost]] = []
            for result in generated:
                if isinstance(result, BaseException):
                    logger.warning("OpenRouter post generation candidate failed: %s", result)
                    continue
                validated = self._validate_posts(result, briefs, target_count=target_count)
                if validated:
                    candidates.append(validated)

            if candidates:
                ranked = sorted(candidates, key=lambda items: _average_confidence(items), reverse=True)
                fallback_task.cancel()
                with suppress(asyncio.CancelledError):
                    await fallback_task
                return ranked[0]

            return await fallback_task

        return self._fallback_posts(briefs, style_profile, target_count=target_count)

    async def _generate_with_llm_model(
        self,
        briefs: list[ResearchBrief],
        style_profile: StyleProfile,
        target_count: int,
        *,
        model: str,
    ) -> list[LinkedInPost]:
        payload = {
            "model": model,
            "temperature": 0.35,
            "max_tokens": 2400,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You write reflective LinkedIn posts about AI developments in plain language. "
                        "Assume a smart but non-technical audience. "
                        "Explain ideas simply and keep jargon to a minimum. "
                        "No sales tone, no motivational hype, no pretender voice, no CTA. "
                        "Never mention browsing limitations, scraping failures, or inability to access pages. "
                        "Return strict JSON only."
                    ),
                },
                {
                    "role": "user",
                    "content": self._build_prompt(briefs, style_profile, target_count=target_count),
                },
            ],
        }

        headers = {
            "Authorization": f"Bearer {self.settings.openrouter_api_key}",
            "Content-Type": "application/json",
        }
        if self.settings.openrouter_site_url:
            headers["HTTP-Referer"] = self.settings.openrouter_site_url
        if self.settings.openrouter_app_name:
            headers["X-Title"] = self.settings.openrouter_app_name

        async with httpx.AsyncClient(timeout=self.settings.request_timeout_seconds) as client:
            response = await client.post(
                f"{self.settings.openrouter_base_url}/chat/completions",
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
            response_payload = response.json()
            record_openrouter_http_response(model=model, payload=cast(dict[str, Any], response_payload))
            content = str(response_payload["choices"][0]["message"]["content"])

        parsed = _parse_json_payload(content)
        raw_posts = parsed.get("posts") if isinstance(parsed, dict) else None
        if not isinstance(raw_posts, list):
            raise ValueError("LLM response missing posts array")

        posts: list[LinkedInPost] = []
        for idx, raw_post in enumerate(raw_posts[:target_count]):
            if not isinstance(raw_post, dict):
                continue

            hashtags = [str(item).strip() for item in raw_post.get("hashtags", []) if str(item).strip()]
            citation_urls = [
                str(item).strip()
                for item in raw_post.get("citation_urls", [])
                if str(item).strip()
            ]
            supporting_topic_ids = [
                str(item).strip()
                for item in raw_post.get("supporting_topic_ids", [])
                if str(item).strip()
            ]

            posts.append(
                LinkedInPost(
                    post_id=f"post-{idx + 1}",
                    angle="technical_reflection",
                    headline=str(raw_post.get("headline") or f"Technical reflection {idx + 1}"),
                    body=str(raw_post.get("body") or "").strip(),
                    hashtags=_normalize_hashtags(hashtags),
                    supporting_topic_ids=supporting_topic_ids,
                    citation_urls=citation_urls,
                    confidence=0.85,
                )
            )

        return posts

    async def _fallback_posts_async(
        self,
        briefs: list[ResearchBrief],
        style_profile: StyleProfile,
        target_count: int,
    ) -> list[LinkedInPost]:
        return self._fallback_posts(briefs, style_profile, target_count=target_count)

    def _build_prompt(self, briefs: list[ResearchBrief], style_profile: StyleProfile, target_count: int) -> str:
        briefs_payload = [brief.model_dump(mode="json") for brief in briefs[:target_count]]
        style_payload = style_profile.model_dump(mode="json")
        return (
            f"Write exactly {target_count} LinkedIn posts from these AI research briefs.\n"
            "Constraints:\n"
            "- Each post maps to exactly one brief/topic (no synthesis post).\n"
            "- Reflective and practical tone: write as if thinking through the implications personally.\n"
            "- Write in very simple English for a general business audience.\n"
            "- Use short sentences and common words.\n"
            "- If you use a technical term, explain it in plain words in the same sentence.\n"
            "- Avoid sales language, motivational slogans, fake authority, and CTA prompts.\n"
            "- Never mention model/tool limits, scraping blocks, or inability to access articles.\n"
            "- Keep each post between 105 and 182 words.\n"
            "- Post structure: strong opening line, plain-language explanation, practical takeaway/risk.\n"
            "- Include a concise headline, body, and 3-5 hashtags.\n"
            "- If verification status is not fully verified, acknowledge uncertainty conservatively without sounding defensive.\n"
            "- Return strict JSON: {\"posts\": [{headline, body, hashtags, supporting_topic_ids, citation_urls}, ...]}\n\n"
            f"Style profile:\n{json.dumps(style_payload, indent=2)}\n\n"
            f"Research briefs:\n{json.dumps(briefs_payload, indent=2)}"
        )

    def _fallback_posts(
        self,
        briefs: list[ResearchBrief],
        style_profile: StyleProfile,
        target_count: int,
    ) -> list[LinkedInPost]:
        posts: list[LinkedInPost] = []
        selected = briefs[:target_count]

        for idx, brief in enumerate(selected, start=1):
            status_note = ""
            if brief.verification_status != "verified":
                status_note = " Public evidence is still limited, so I treat this as a directional signal."

            headline = _build_headline(brief)
            key_points = brief.key_points[:3]
            key_point_text = " ".join(f"- {point}" for point in key_points) if key_points else ""
            body = (
                f"{_reflective_open(style_profile)} {brief.summary}\n\n"
                f"In simple terms, this means: {brief.technical_significance.lower()} "
                f"For day-to-day teams, {brief.business_impact.lower()}\n\n"
                "Specific details I'm tracking: "
                f"{key_point_text if key_point_text else 'the implementation notes and evaluation setup in the cited sources.'} "
                f"{brief.why_now}{status_note}"
            )

            posts.append(
                LinkedInPost(
                    post_id=f"post-{idx}",
                    angle="technical_reflection",
                    headline=headline,
                    body=body,
                    hashtags=_normalize_hashtags(["#AI", "#AIAgents", "#MachineLearning", "#EnterpriseAI"]),
                    supporting_topic_ids=[brief.topic_id],
                    citation_urls=[citation.url for citation in brief.citations[:3]],
                    confidence=0.7,
                )
            )

        return self._validate_posts(posts, briefs, target_count=target_count)

    def _validate_posts(
        self,
        posts: list[LinkedInPost],
        briefs: list[ResearchBrief],
        target_count: int,
    ) -> list[LinkedInPost]:
        if not posts:
            return []

        valid_topic_ids = {brief.topic_id for brief in briefs}
        normalized: list[LinkedInPost] = []

        for idx, post in enumerate(posts[:target_count]):
            body = _clean_post_body(post.body)
            if not body:
                continue
            body = _enforce_word_window(body, min_words=_MIN_POST_WORDS, max_words=_MAX_POST_WORDS)

            hashtags = post.hashtags[:5]
            if len(hashtags) < 3:
                hashtags = _normalize_hashtags(hashtags + ["#AI", "#AIAgents", "#MachineLearning"])

            supporting = [topic_id for topic_id in post.supporting_topic_ids if topic_id in valid_topic_ids]
            if not supporting:
                fallback_id = briefs[min(idx, len(briefs) - 1)].topic_id
                supporting = [fallback_id]

            # Enforce one-topic-per-post mapping.
            supporting = supporting[:1]

            citations = [url for url in post.citation_urls if url.startswith("http")]
            if not citations:
                fallback_brief = briefs[min(idx, len(briefs) - 1)]
                citations = [citation.url for citation in fallback_brief.citations[:2]]

            normalized.append(
                LinkedInPost(
                    post_id=f"post-{idx + 1}",
                    angle="technical_reflection",
                    headline=post.headline.strip() or f"Technical reflection {idx + 1}",
                    body=body,
                    hashtags=_normalize_hashtags(hashtags)[:5],
                    supporting_topic_ids=supporting,
                    citation_urls=citations[:5],
                    confidence=post.confidence,
                )
            )

        return normalized


def _normalize_hashtags(hashtags: list[str]) -> list[str]:
    normalized: list[str] = []
    for tag in hashtags:
        trimmed = tag.strip()
        if not trimmed:
            continue
        if not trimmed.startswith("#"):
            trimmed = f"#{trimmed}"
        trimmed = re.sub(r"\s+", "", trimmed)
        if trimmed and trimmed not in normalized:
            normalized.append(trimmed)
    return normalized


def _build_headline(brief: ResearchBrief) -> str:
    stem = brief.headline.strip().rstrip(".")
    if stem.endswith("?"):
        return stem
    return f"A technical note on: {stem}"


def _reflective_open(style_profile: StyleProfile) -> str:
    opener = "Lately"
    for candidate in style_profile.common_openers:
        normalized = " ".join(str(candidate).split()).strip(" -,:;.")
        if len(normalized) < 3:
            continue
        if len(normalized.split()) < 3:
            continue
        lowered = normalized.lower()
        if lowered in {"a", "an", "the", "after a", "after an", "after the"}:
            continue
        opener = normalized
        break
    return f"{opener[0].upper() + opener[1:]}, I keep coming back to this development."


def _strip_salesy_language(text: str) -> str:
    replacements = {
        "game changer": "notable shift",
        "must-have": "worth evaluating",
        "revolutionary": "meaningful",
        "unlock": "enable",
        "10x": "significant",
        "can't miss": "relevant",
        "act now": "monitor closely",
    }

    cleaned = text
    for source, target in replacements.items():
        cleaned = re.sub(source, target, cleaned, flags=re.IGNORECASE)

    # Remove direct CTA style endings.
    cleaned = re.sub(r"\b(what do you think\??|curious to hear your take\.?|let me know\.?|thoughts\?)$", "", cleaned, flags=re.IGNORECASE)
    return " ".join(cleaned.split())


def _strip_access_failure_language(text: str) -> str:
    replacement = "Public evidence is still limited and should be treated as directional."
    patterns = [
        r"\b(i|we)\s+(can(?:not|'t)|could(?:\s+not|n't)|did(?:\s+not|n't))\s+(access|verify|find|retrieve)[^.!?]*[.!?]?",
        r"\bno\s+(abstract|methodology|full\s+text|authors)\b[^.!?]*[.!?]?",
        r"\bfrom\s+what\s+is\s+publicly\s+indexed\b[^.!?]*[.!?]?",
    ]

    cleaned = text
    for pattern in patterns:
        cleaned = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE)
    return cleaned


def _clean_post_body(text: str) -> str:
    lines = [re.sub(r"\s+", " ", line).strip() for line in str(text).splitlines()]
    merged = "\n".join(line for line in lines if line)
    merged = _strip_salesy_language(merged)
    merged = _strip_access_failure_language(merged)
    merged = _simplify_jargon_for_general_audience(merged)
    return " ".join(merged.split())


def _enforce_word_window(text: str, *, min_words: int, max_words: int) -> str:
    words = text.split()
    if not words:
        return text
    if len(words) > max_words:
        return " ".join(words[:max_words]).rstrip(".,;:") + "."
    if len(words) >= min_words:
        return text

    pad = (
        "I see this as an early signal. I will keep watching for clearer results, "
        "real examples, and practical limits in follow-up coverage."
    )
    padded = f"{text} {pad}"
    padded_words = padded.split()
    if len(padded_words) > max_words:
        padded = " ".join(padded_words[:max_words]).rstrip(".,;:") + "."
    return padded


def _simplify_jargon_for_general_audience(text: str) -> str:
    replacements = {
        "implementation": "real-world use",
        "orchestration": "coordination",
        "inference": "model use",
        "latency": "response time",
        "throughput": "how much work it can handle",
        "benchmark": "test result",
        "multimodal": "works with text, images, or audio",
        "retrieval": "finding the right information",
        "context window": "working memory",
        "architecture": "system design",
        "evaluation": "testing",
        "tradeoffs": "pros and cons",
        "directional signal": "early signal",
    }

    simplified = text
    for source, target in replacements.items():
        simplified = re.sub(rf"\\b{re.escape(source)}\\b", target, simplified, flags=re.IGNORECASE)
    simplified = simplified.replace(";", ".")
    simplified = re.sub(r"\s+", " ", simplified).strip()
    return simplified


def _parse_json_payload(content: str) -> dict[str, Any]:
    stripped = content.strip()
    if stripped.startswith("{"):
        parsed = json.loads(stripped)
        if isinstance(parsed, dict):
            return cast(dict[str, Any], parsed)
        raise ValueError("JSON root must be an object")

    match = re.search(r"\{[\s\S]*\}", stripped)
    if not match:
        raise ValueError("No JSON object found in response")
    parsed = json.loads(match.group(0))
    if isinstance(parsed, dict):
        return cast(dict[str, Any], parsed)
    raise ValueError("JSON root must be an object")


def _average_confidence(posts: list[LinkedInPost]) -> float:
    if not posts:
        return 0.0
    return sum(post.confidence for post in posts) / len(posts)
