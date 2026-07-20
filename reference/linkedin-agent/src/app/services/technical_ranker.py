from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass
from typing import Any, cast

import httpx

from app.config import Settings
from app.schemas import DiscoveredItem
from app.services.api_usage_tracker import record_openrouter_http_response

logger = logging.getLogger(__name__)


@dataclass
class TechnicalAssessment:
    technical_depth: float
    implementation_specificity: float
    hype_score: float
    notes: str


class TechnicalRanker:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def assess_many(
        self,
        items: list[DiscoveredItem],
        dry_run: bool,
    ) -> dict[str, TechnicalAssessment]:
        if not items:
            return {}

        if dry_run or not (self.settings.openrouter_api_key or "").strip():
            return {item.id: self._heuristic_assessment(item) for item in items}

        timeout = httpx.Timeout(self.settings.request_timeout_seconds)
        limits = httpx.Limits(max_connections=8, max_keepalive_connections=4)
        assessments: dict[str, TechnicalAssessment] = {}

        async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
            async def worker(item: DiscoveredItem) -> tuple[str, TechnicalAssessment]:
                try:
                    assessment = await self._assess_with_optional_ensemble(client, item)
                    return item.id, assessment
                except Exception as exc:
                    logger.warning("Technical assessment failed for %s: %s", item.id, exc)
                    return item.id, self._heuristic_assessment(item)

            gathered = await asyncio.gather(*(worker(item) for item in items))

        for item_id, assessment in gathered:
            assessments[item_id] = assessment

        return assessments

    async def _assess_with_optional_ensemble(
        self,
        client: httpx.AsyncClient,
        item: DiscoveredItem,
    ) -> TechnicalAssessment:
        primary_model = self.settings.openrouter_stage_a_model
        secondary_model = (self.settings.openrouter_stage_a_secondary_model or "").strip() or None
        if not secondary_model:
            return await self._assess_one(client, item, model=primary_model)

        first_task = self._assess_one(client, item, model=primary_model)
        second_task = self._assess_one(client, item, model=secondary_model)
        primary, secondary = await asyncio.gather(first_task, second_task)

        return TechnicalAssessment(
            technical_depth=_clip01((primary.technical_depth + secondary.technical_depth) / 2.0),
            implementation_specificity=_clip01(
                (primary.implementation_specificity + secondary.implementation_specificity) / 2.0
            ),
            hype_score=_clip01(max(primary.hype_score, secondary.hype_score)),
            notes=f"ensemble({primary_model},{secondary_model})",
        )

    async def _assess_one(
        self,
        client: httpx.AsyncClient,
        item: DiscoveredItem,
        *,
        model: str,
    ) -> TechnicalAssessment:
        prompt = _build_assessment_prompt(item)
        payload = {
            "model": model,
            "temperature": 0.1,
            "max_tokens": 300,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You classify whether an AI article has technical implementation depth. "
                        "Do not reward hype. Return strict JSON only."
                    ),
                },
                {"role": "user", "content": prompt},
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
        return TechnicalAssessment(
            technical_depth=_clip01(float(parsed.get("technical_depth") or 0.0)),
            implementation_specificity=_clip01(float(parsed.get("implementation_specificity") or 0.0)),
            hype_score=_clip01(float(parsed.get("hype_score") or 0.0)),
            notes=str(parsed.get("notes") or "").strip(),
        )

    def _heuristic_assessment(self, item: DiscoveredItem) -> TechnicalAssessment:
        text = f"{item.title} {item.snippet or ''} {item.raw_content or ''}".lower()

        technical_markers = {
            "architecture",
            "benchmark",
            "evaluation",
            "inference",
            "latency",
            "framework",
            "tooling",
            "protocol",
            "sdk",
            "api",
            "agent",
            "agents",
            "reasoning",
            "context",
            "memory",
            "retrieval",
            "training",
            "deployment",
            "observability",
            "orchestration",
            "a2a",
            "enterprise tooling",
            "workflow",
        }
        hype_markers = {
            "game changer",
            "revolutionary",
            "must-have",
            "massive",
            "break the internet",
            "unbelievable",
        }

        tech_hits = sum(1 for token in technical_markers if token in text)
        hype_hits = sum(1 for token in hype_markers if token in text)

        implementation_specificity = min(1.0, 0.12 * tech_hits)
        technical_depth = min(1.0, 0.2 + 0.1 * tech_hits)
        hype_score = min(1.0, 0.25 * hype_hits)

        return TechnicalAssessment(
            technical_depth=technical_depth,
            implementation_specificity=implementation_specificity,
            hype_score=hype_score,
            notes="heuristic",
        )


def _build_assessment_prompt(item: DiscoveredItem) -> str:
    return (
        "Score this article candidate for technical implementation value.\n"
        "Do not rely on fixed buzzword lists; infer whether the content contains concrete technical methods, architecture details, or implementation insights.\n"
        "Return strict JSON with keys: technical_depth (0..1), implementation_specificity (0..1), hype_score (0..1), notes (short string).\n\n"
        f"Title: {item.title}\n"
        f"Snippet: {item.snippet or ''}\n"
        f"Content: {(item.raw_content or '')[:1200]}"
    )


def _parse_json_payload(content: str) -> dict[str, Any]:
    stripped = content.strip()
    if stripped.startswith("{"):
        parsed = json.loads(stripped)
        if isinstance(parsed, dict):
            return cast(dict[str, Any], parsed)
        raise ValueError("JSON root must be an object")

    match = re.search(r"\{[\s\S]*\}", stripped)
    if not match:
        raise ValueError("No JSON object found in scorer response")

    parsed = json.loads(match.group(0))
    if isinstance(parsed, dict):
        return cast(dict[str, Any], parsed)
    raise ValueError("JSON root must be an object")


def _clip01(value: float) -> float:
    return max(0.0, min(value, 1.0))
