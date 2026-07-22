"""Technical-depth assessment ported from the reference LinkedIn agent
(``reference/linkedin-agent/src/app/services/technical_ranker.py``).

Two paths:
  * Heuristic — deterministic, no network, used in dry-run and whenever the
    OpenRouter API key is missing. Hype demotion via ``hype_markers``.
  * Stage-A LLM — calls OpenRouter with the configured ``openrouter_stage_a_model``
    (optional ensemble with ``openrouter_stage_a_secondary_model``).

Intentional divergences from the reference (three; all documented here so a
future maintainer doesn't "restore" reference behavior and silently shift
hype-detection):
  1. Usage recording goes through the host's ``orchestrator.usage`` module
     (``record_openrouter_http_response``, wired at the httpx response, P7.2),
     not the reference's ``api_usage_tracker``.
  2. The heuristic shares the ranker's keyword sets (``_TECH_KEYWORDS`` /
     ``_BUSINESS_KEYWORDS`` / ``_LOW_SIGNAL_KEYWORDS``) with
     :mod:`ranking`'s scorer, instead of maintaining its own copy of the
     reference's ``technical_markers`` set. The two sets differ materially
     (the reference includes ``evaluation``, ``latency``, ``tooling``,
     ``protocol``, ``context``, ``memory``, ``retrieval``, ``observability``,
     ``orchestration``, ``a2a``, ``workflow``; this port instead gets
     ``model``, ``launch``, ``release``, ``multimodal``, ``chip``,
     ``accelerator``, ``open``, ``source`` from the shared set). The intent is
     lockstep scoring/assessing semantics within the orchestrator; if parity
     with the LinkedIn reference matters more than internal consistency, the
     reference's set should be ported explicitly here.
  3. On per-item LLM failure the worker falls back to the heuristic and marks
     it ``notes="heuristic-fallback: <ExcType>"`` (the reference silently
     dropped to bare ``notes="heuristic"``, indistinguishable from dry-run).
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass
from typing import Any, cast

import httpx

from app.config import Settings
from app.orchestrator.services.ranking import DiscoveredItem, _TECH_KEYWORDS
from app.orchestrator.usage import record_openrouter_http_response

logger = logging.getLogger(__name__)


@dataclass
class TechnicalAssessment:
    technical_depth: float
    implementation_specificity: float
    hype_score: float
    notes: str


# Single shared marker sets — these are the language a "hype without technical
# detail" detector listens for. Kept here (next to the assessor) rather than in
# ranking.py so future tuning doesn't entangle them with the clustering code.
_HYPE_MARKERS = {
    "game changer",
    "revolutionary",
    "must-have",
    "massive",
    "break the internet",
    "unbelievable",
}


class TechnicalRanker:
    """Assess a batch of ``DiscoveredItem`` for technical implementation value.

    Constructed with a ``Settings`` so the assessor picks up the env-driven
    Stage A model the moment one is configured. ``assess_many`` returns
    ``{item_id: TechnicalAssessment}`` keyed on the item's ``id`` — the same key
    :func:`rank_topics` emits as ``TechnicalOverride`` keys."""

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
                    # Mark the fallback distinctly so an operator reading
                    # downstream artifacts can tell "ran in dry-run" from
                    # "OpenRouter failed N times" — same assessment values,
                    # different `notes`.
                    fallback = self._heuristic_assessment(item)
                    return item.id, TechnicalAssessment(
                        technical_depth=fallback.technical_depth,
                        implementation_specificity=fallback.implementation_specificity,
                        hype_score=fallback.hype_score,
                        notes=f"heuristic-fallback: {type(exc).__name__}",
                    )

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
        secondary_model = (
            self.settings.openrouter_stage_a_secondary_model or ""
        ).strip() or None
        if not secondary_model:
            return await self._assess_one(client, item, model=primary_model)

        primary, secondary = await asyncio.gather(
            self._assess_one(client, item, model=primary_model),
            self._assess_one(client, item, model=secondary_model),
        )
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
            # Ask OpenRouter to include cost in the usage block so the P7.2
            # tracker records real spend, not just tokens.
            "usage": {"include": True},
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You classify whether an AI article has technical "
                        "implementation depth. Do not reward hype. Return "
                        "strict JSON only."
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
        record_openrouter_http_response(model=model, payload=response_payload)
        content = str(response_payload["choices"][0]["message"]["content"])
        parsed = _parse_json_payload(content)
        return TechnicalAssessment(
            technical_depth=_clip01(float(parsed.get("technical_depth") or 0.0)),
            implementation_specificity=_clip01(
                float(parsed.get("implementation_specificity") or 0.0)
            ),
            hype_score=_clip01(float(parsed.get("hype_score") or 0.0)),
            notes=str(parsed.get("notes") or "").strip(),
        )

    def _heuristic_assessment(self, item: DiscoveredItem) -> TechnicalAssessment:
        """Deterministic keyword-based fallback — used in dry-run, when the API
        key is missing, or as a per-item fallback when the LLM path raises.

        The technical markers are the same keyword sets shared with
        :mod:`ranking`'s relevance scorer, so the heuristic and the scorer stay
        in lockstep on what counts as 'technical'. ``hype_markers`` are the
        hype-without-technical-detail lever: each hit adds 0.25 to the hype
        score, which :func:`_score_item_with_overrides` then subtracts
        (-0.18 * hype_score)."""
        text = f"{item.title} {item.snippet or ''} {item.raw_content or ''}".lower()

        tech_hits = sum(1 for token in _TECH_KEYWORDS if token in text)
        hype_hits = sum(1 for token in _HYPE_MARKERS if token in text)

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
        "Do not rely on fixed buzzword lists; infer whether the content contains "
        "concrete technical methods, architecture details, or implementation "
        "insights.\n"
        "Return strict JSON with keys: technical_depth (0..1), "
        "implementation_specificity (0..1), hype_score (0..1), notes (short string).\n\n"
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


__all__ = ["TechnicalAssessment", "TechnicalRanker"]