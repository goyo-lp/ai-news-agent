from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, cast

import httpx

from app.config import Settings
from app.schemas import ResearchBrief
from app.services.api_usage_tracker import record_openrouter_http_response
from app.services.tavily_client import TavilyClient

logger = logging.getLogger(__name__)


class BriefVerifier:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.tavily_client = TavilyClient(settings)

    async def verify_briefs(
        self,
        briefs: list[ResearchBrief],
        hours_back: int,
        dry_run: bool,
    ) -> tuple[list[ResearchBrief], list[str]]:
        if not briefs:
            return [], []

        timeout = httpx.Timeout(self.settings.request_timeout_seconds)
        verified: list[tuple[int, ResearchBrief]] = []
        errors: list[str] = []
        semaphore = asyncio.Semaphore(max(1, self.settings.verification_concurrency))

        async with httpx.AsyncClient(timeout=timeout) as client:
            async def worker(idx: int, brief: ResearchBrief) -> tuple[int, ResearchBrief, str | None]:
                async with semaphore:
                    try:
                        verified_brief = await self._verify_one(
                            client=client,
                            brief=brief,
                            hours_back=hours_back,
                            dry_run=dry_run,
                        )
                        return idx, verified_brief, None
                    except Exception as exc:
                        error = f"Verification failed for {brief.topic_id}: {exc}"
                        logger.warning(error)
                        fallback = brief.model_copy(deep=True)
                        fallback.verification_status = "insufficient_evidence"
                        fallback.verification_confidence = 0.25
                        fallback.verification_notes = [
                            "Verification failed, falling back to cautious language.",
                        ]
                        fallback.summary = _make_cautious(fallback.summary)
                        return idx, fallback, error

            gathered = await asyncio.gather(
                *(worker(idx, brief) for idx, brief in enumerate(briefs)),
            )

        for idx, brief, maybe_error in sorted(gathered, key=lambda item: item[0]):
            verified.append((idx, brief))
            if maybe_error:
                errors.append(maybe_error)

        return [brief for _, brief in verified], errors

    async def _verify_one(
        self,
        client: httpx.AsyncClient,
        brief: ResearchBrief,
        hours_back: int,
        dry_run: bool,
    ) -> ResearchBrief:
        base_urls: list[str] = []
        for citation in brief.citations:
            if citation.url not in base_urls:
                base_urls.append(citation.url)
            if len(base_urls) >= max(1, self.settings.verification_sources_per_topic):
                break

        queries = _build_verifier_queries(brief)
        search_tasks = [
            self.tavily_client.search_news(
                client=client,
                query=query,
                hours_back=hours_back,
                max_results=max(2, self.settings.verification_sources_per_topic),
                dry_run=dry_run,
            )
            for query in queries
        ]
        base_extract_task = self.tavily_client.extract_contents(
            client=client,
            urls=base_urls,
            dry_run=dry_run,
        )
        gathered_searches = await asyncio.gather(*search_tasks, return_exceptions=True) if search_tasks else []
        base_extracted = await base_extract_task

        corroborating: list[Any] = []
        for maybe_items in gathered_searches:
            if isinstance(maybe_items, Exception):
                logger.warning("Verifier evidence search failed for %s: %s", brief.topic_id, maybe_items)
                continue
            corroborating.extend(cast(list[Any], maybe_items))

        deduped_corroborating = _dedupe_discovered(corroborating)

        evidence_urls = list(base_urls)
        evidence_urls.extend(item.url for item in deduped_corroborating if item.url not in evidence_urls)
        evidence_urls = evidence_urls[: max(3, self.settings.verification_sources_per_topic + 1)]

        delta_urls = [url for url in evidence_urls if url not in base_urls]
        delta_extracted = await self.tavily_client.extract_contents(
            client=client,
            urls=delta_urls,
            dry_run=dry_run,
        )
        extracted = {**base_extracted, **delta_extracted}

        evidence_texts: list[str] = []
        snippet_by_url = {
            item.url: " ".join((item.raw_content or item.snippet or "").split()).strip()
            for item in deduped_corroborating
        }
        for url in evidence_urls:
            content = " ".join(extracted.get(url, "").split()).strip()
            if not content:
                content = snippet_by_url.get(url, "")
            if content:
                evidence_texts.append(f"URL: {url}\n{content[:1200]}")

        if dry_run or not (self.settings.openrouter_api_key or "").strip():
            fallback = brief.model_copy(deep=True)
            fallback.verification_status = "partially_verified"
            fallback.verification_confidence = 0.6 if evidence_texts else 0.4
            fallback.verification_notes = _merge_notes(
                fallback.verification_notes,
                [
                    "Dry-run or missing verifier model key; verification executed with heuristic fallback.",
                    f"Evidence URLs checked: {len(evidence_urls)}",
                ],
            )
            fallback.summary = _make_cautious(fallback.summary)
            return fallback

        prompt = _build_verification_prompt(brief, evidence_texts)
        parsed = await self._run_verifier_models(client=client, prompt=prompt)

        verified = brief.model_copy(deep=True)
        verified.verification_status = str(parsed.get("verdict") or "partially_verified")
        verified.verification_confidence = float(parsed.get("confidence") or 0.5)

        verified.summary = _sanitize_access_failure_language(
            str(parsed.get("corrected_summary") or verified.summary).strip()
        )
        verified.technical_significance = _sanitize_access_failure_language(
            str(parsed.get("corrected_technical_significance") or verified.technical_significance).strip()
        )
        verified.business_impact = _sanitize_access_failure_language(
            str(parsed.get("corrected_business_impact") or verified.business_impact).strip()
        )
        verified.why_now = _sanitize_access_failure_language(
            str(parsed.get("corrected_why_now") or verified.why_now).strip()
        )

        existing_notes = list(verified.verification_notes)
        notes_payload = parsed.get("notes")
        if isinstance(notes_payload, list):
            parsed_notes = [str(item).strip() for item in notes_payload if str(item).strip()]
            verified.verification_notes = _merge_notes(existing_notes, parsed_notes)
        else:
            verified.verification_notes = _merge_notes(existing_notes, ["Verifier returned no notes."])

        if verified.verification_status in {"insufficient_evidence", "partially_verified"}:
            verified.summary = _make_cautious(verified.summary)

        return verified

    async def _run_verifier_models(self, client: httpx.AsyncClient, prompt: str) -> dict[str, Any]:
        primary_model = self.settings.openrouter_model
        secondary_model = (self.settings.openrouter_verifier_secondary_model or "").strip() or None
        if secondary_model is None:
            return await self._verify_with_model(client=client, prompt=prompt, model=primary_model)

        primary_task = self._verify_with_model(client=client, prompt=prompt, model=primary_model)
        secondary_task = self._verify_with_model(client=client, prompt=prompt, model=secondary_model)
        primary, secondary = await asyncio.gather(primary_task, secondary_task)
        return _reconcile_verifier_payloads(
            primary=primary,
            secondary=secondary,
            primary_model=primary_model,
            secondary_model=secondary_model,
        )

    async def _verify_with_model(
        self,
        *,
        client: httpx.AsyncClient,
        prompt: str,
        model: str,
    ) -> dict[str, Any]:
        payload = {
            "model": model,
            "temperature": 0.1,
            "max_tokens": 900,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a strict technical fact-checking assistant. "
                        "Never overclaim. If evidence is weak, reduce confidence and rewrite conservatively."
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
        return _parse_json_payload(content)


def _build_verification_prompt(brief: ResearchBrief, evidence_texts: list[str]) -> str:
    evidence_blob = "\n\n".join(evidence_texts[:4]) if evidence_texts else "No extracted evidence available."
    brief_payload = json.dumps(brief.model_dump(mode="json"), indent=2)
    return (
        "Verify this technical brief against the evidence.\n"
        "Return strict JSON with keys:\n"
        "verdict (verified|partially_verified|insufficient_evidence),\n"
        "confidence (0..1),\n"
        "corrected_summary, corrected_technical_significance, corrected_business_impact, corrected_why_now,\n"
        "notes (array of concise verification notes).\n"
        "Never mention scraping/browsing limitations or inability to access pages.\n"
        "Keep language reflective and neutral; avoid hype or sales tone.\n\n"
        f"Brief:\n{brief_payload}\n\n"
        f"Evidence:\n{evidence_blob}"
    )


def _make_cautious(text: str) -> str:
    sentence = " ".join(text.split())
    if sentence.lower().startswith("based on current evidence"):
        return sentence
    return f"Based on current evidence, {sentence[0].lower() + sentence[1:] if sentence else sentence}"


def _parse_json_payload(content: str) -> dict[str, Any]:
    stripped = content.strip()
    if stripped.startswith("{"):
        parsed = json.loads(stripped)
        if isinstance(parsed, dict):
            return cast(dict[str, Any], parsed)
        raise ValueError("JSON root must be an object")

    match = re.search(r"\{[\s\S]*\}", stripped)
    if not match:
        raise ValueError("No JSON object found in verifier response")

    parsed = json.loads(match.group(0))
    if isinstance(parsed, dict):
        return cast(dict[str, Any], parsed)
    raise ValueError("JSON root must be an object")


def _merge_notes(base: list[str], additions: list[str]) -> list[str]:
    merged: list[str] = []
    for item in [*base, *additions]:
        normalized = " ".join(str(item).split()).strip()
        normalized = _sanitize_access_failure_language(normalized)
        if not normalized or normalized in merged:
            continue
        merged.append(normalized)
    return merged


def _build_verifier_queries(brief: ResearchBrief) -> list[str]:
    raw_queries = [
        brief.headline,
        f"{brief.headline} technical implementation details",
        f"{brief.headline} architecture benchmark evidence",
        f"{brief.headline} production limitations failure modes",
    ]

    deduped: list[str] = []
    seen: set[str] = set()
    for query in raw_queries:
        normalized = " ".join(query.split()).strip()
        if not normalized:
            continue
        lowered = normalized.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        deduped.append(normalized)
        if len(deduped) >= 4:
            break
    return deduped


def _dedupe_discovered(items: list[Any]) -> list[Any]:
    deduped: list[Any] = []
    seen_urls: set[str] = set()
    for item in items:
        url = str(getattr(item, "url", "")).strip()
        if not url or url in seen_urls:
            continue
        seen_urls.add(url)
        deduped.append(item)
    return deduped


def _sanitize_access_failure_language(text: str) -> str:
    sentence = " ".join(text.split()).strip()
    if not sentence:
        return sentence

    replacement = "Public evidence is still limited and should be treated as directional."
    patterns = [
        r"\b(i|we)\s+(can(?:not|'t)|could(?:\s+not|n't)|did(?:\s+not|n't))\s+(access|verify|find|retrieve)[^.!?]*[.!?]?",
        r"\bno\s+(abstract|methodology|full\s+text|authors)\b[^.!?]*[.!?]?",
        r"\bfrom\s+what\s+is\s+publicly\s+indexed\b[^.!?]*[.!?]?",
    ]

    cleaned = sentence
    for pattern in patterns:
        cleaned = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE)
    cleaned = " ".join(cleaned.split()).strip()
    return cleaned or replacement


def _reconcile_verifier_payloads(
    *,
    primary: dict[str, Any],
    secondary: dict[str, Any],
    primary_model: str,
    secondary_model: str,
) -> dict[str, Any]:
    status_rank = {
        "verified": 0,
        "partially_verified": 1,
        "insufficient_evidence": 2,
    }

    primary_status = str(primary.get("verdict") or "partially_verified")
    secondary_status = str(secondary.get("verdict") or "partially_verified")
    primary_conf = _clip01(float(primary.get("confidence") or 0.5))
    secondary_conf = _clip01(float(secondary.get("confidence") or 0.5))

    primary_tuple = (status_rank.get(primary_status, 99), primary_conf)
    secondary_tuple = (status_rank.get(secondary_status, 99), secondary_conf)
    strictest = primary if primary_tuple >= secondary_tuple else secondary

    notes: list[str] = []
    for payload in (primary, secondary):
        raw_notes = payload.get("notes")
        if isinstance(raw_notes, list):
            notes.extend(str(item).strip() for item in raw_notes if str(item).strip())
    notes = _merge_notes(
        [],
        notes + [f"Verifier ensemble used models: {primary_model}, {secondary_model}."],
    )

    merged = dict(strictest)
    merged["confidence"] = min(primary_conf, secondary_conf)
    merged["notes"] = notes
    return merged


def _clip01(value: float) -> float:
    return max(0.0, min(value, 1.0))
