from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, Literal, cast

import httpx
from pydantic import BaseModel, Field, SecretStr, ValidationError

from app.config import Settings
from app.schemas import ResearchBrief
from app.services.api_usage_tracker import record_openrouter_tool_call
from app.services.tavily_client import TavilyClient

logger = logging.getLogger(__name__)

_create_deep_agent: Any = None
_chat_openai_cls: Any = None

try:
    from deepagents import create_deep_agent as _imported_create_deep_agent
    from langchain_openai import ChatOpenAI as _imported_chat_openai_cls
except Exception:  # pragma: no cover
    pass
else:
    _create_deep_agent = _imported_create_deep_agent
    _chat_openai_cls = _imported_chat_openai_cls


class DeepAgentFinding(BaseModel):
    verification_status: Literal["verified", "partially_verified", "insufficient_evidence"]
    verification_confidence: float = Field(ge=0.0, le=1.0)
    corrected_summary: str
    corrected_technical_significance: str
    corrected_business_impact: str
    corrected_why_now: str
    technical_implementation_notes: list[str] = Field(default_factory=list)
    verification_notes: list[str] = Field(default_factory=list)


class DeepAgentInvestigator:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.tavily_client = TavilyClient(settings)
        self._agent = self._build_agent()

    async def investigate_briefs(
        self,
        briefs: list[ResearchBrief],
        *,
        hours_back: int,
        dry_run: bool,
    ) -> tuple[list[ResearchBrief], list[str]]:
        if not briefs:
            return [], []

        if not self._should_run(dry_run=dry_run):
            return briefs, []

        if self._agent is None:
            return briefs, ["Deep agent unavailable, skipping adaptive investigation."]

        timeout = httpx.Timeout(self.settings.request_timeout_seconds)
        investigated: list[tuple[int, ResearchBrief]] = []
        errors: list[str] = []
        semaphore = asyncio.Semaphore(max(1, self.settings.adaptive_investigation_concurrency))

        async with httpx.AsyncClient(timeout=timeout) as http_client:
            async def worker(idx: int, brief: ResearchBrief) -> tuple[int, ResearchBrief, str | None]:
                async with semaphore:
                    try:
                        updated = await asyncio.wait_for(
                            self._investigate_one(
                                http_client=http_client,
                                brief=brief,
                                hours_back=hours_back,
                                dry_run=dry_run,
                            ),
                            timeout=float(self.settings.deep_agent_timeout_seconds),
                        )
                        return idx, updated, None
                    except asyncio.TimeoutError:
                        error = (
                            f"Deep agent timeout for {brief.topic_id} "
                            f"after {self.settings.deep_agent_timeout_seconds}s."
                        )
                        logger.warning(error)
                        return idx, brief, error
                    except Exception as exc:
                        error = f"Deep agent investigation failed for {brief.topic_id}: {exc}"
                        logger.warning(error)
                        return idx, brief, error

            gathered = await asyncio.gather(
                *(worker(idx, brief) for idx, brief in enumerate(briefs)),
            )

        for idx, brief, maybe_error in sorted(gathered, key=lambda item: item[0]):
            investigated.append((idx, brief))
            if maybe_error:
                errors.append(maybe_error)

        return [brief for _, brief in investigated], errors

    def _should_run(self, *, dry_run: bool) -> bool:
        if dry_run:
            return False
        if not self.settings.deep_agent_enabled:
            return False
        if not (self.settings.openrouter_api_key or "").strip():
            return False
        return True

    def _build_agent(self) -> Any:
        if _create_deep_agent is None or _chat_openai_cls is None:
            return None

        headers: dict[str, str] = {}
        if self.settings.openrouter_site_url:
            headers["HTTP-Referer"] = self.settings.openrouter_site_url
        if self.settings.openrouter_app_name:
            headers["X-Title"] = self.settings.openrouter_app_name

        model_kwargs: dict[str, Any] = {}
        if headers:
            model_kwargs["default_headers"] = headers

        model = _chat_openai_cls(
            model=self.settings.deep_agent_model,
            api_key=SecretStr(self.settings.openrouter_api_key or ""),
            base_url=self.settings.openrouter_base_url,
            temperature=0.0,
            timeout=float(self.settings.deep_agent_timeout_seconds),
            max_retries=1,
            **model_kwargs,
        )

        subagents: list[dict[str, Any]] = [
            {
                "name": "technical_novelty_skill",
                "description": (
                    "Extract concrete technical mechanisms, methods, benchmarks, architecture details, "
                    "and implementation caveats."
                ),
                "system_prompt": (
                    "You are a technical novelty analyst. Focus only on concrete implementation details, "
                    "evaluation evidence, and engineering tradeoffs. Ignore hype, business fluff, and "
                    "marketing phrasing."
                ),
            },
            {
                "name": "claim_verification_skill",
                "description": (
                    "Check whether claims are supported by supplied evidence snippets and classify confidence."
                ),
                "system_prompt": (
                    "You are a strict claim verifier. Mark weakly supported claims as uncertain. "
                    "Never overstate certainty. Track contradictions across sources."
                ),
            },
            {
                "name": "reflective_editor_skill",
                "description": (
                    "Rewrite into reflective technical language with no sales pitch or motivational posturing."
                ),
                "system_prompt": (
                    "You are a reflective technical editor. Keep tone personal and analytical, "
                    "without sales calls, fake authority, or hype."
                ),
            },
        ]

        return _create_deep_agent(
            model=model,
            tools=[],
            system_prompt=_MAIN_DEEP_AGENT_SYSTEM_PROMPT,
            subagents=cast(Any, subagents),
            checkpointer=False,
            name="adaptive_technical_investigator",
        )

    async def _investigate_one(
        self,
        *,
        http_client: httpx.AsyncClient,
        brief: ResearchBrief,
        hours_back: int,
        dry_run: bool,
    ) -> ResearchBrief:
        evidence_urls, evidence_texts = await self._collect_evidence(
            http_client=http_client,
            brief=brief,
            hours_back=hours_back,
            dry_run=dry_run,
        )

        prompt = _build_deep_agent_prompt(
            brief=brief,
            evidence_texts=evidence_texts,
        )
        result = await self._agent.ainvoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ]
            }
        )
        record_openrouter_tool_call(model=self.settings.deep_agent_model, payload=result)
        finding = self._parse_finding(result)
        return _apply_finding(brief=brief, finding=finding, evidence_count=len(evidence_urls))

    async def _collect_evidence(
        self,
        *,
        http_client: httpx.AsyncClient,
        brief: ResearchBrief,
        hours_back: int,
        dry_run: bool,
    ) -> tuple[list[str], list[str]]:
        base_urls: list[str] = []
        for citation in brief.citations:
            if citation.url not in base_urls:
                base_urls.append(citation.url)
            if len(base_urls) >= max(1, self.settings.deep_agent_max_evidence_sources):
                break

        queries = _build_evidence_queries(brief)
        search_tasks = [
            self.tavily_client.search_news(
                client=http_client,
                query=query,
                hours_back=hours_back,
                max_results=max(2, self.settings.deep_agent_max_evidence_sources),
                dry_run=dry_run,
            )
            for query in queries
        ]
        base_extract_task = self.tavily_client.extract_contents(
            client=http_client,
            urls=base_urls,
            dry_run=dry_run,
        )
        gathered_searches = await asyncio.gather(*search_tasks, return_exceptions=True) if search_tasks else []
        base_extracted = await base_extract_task

        corroborating: list[Any] = []
        for maybe_items in gathered_searches:
            if isinstance(maybe_items, Exception):
                logger.warning("Adaptive evidence search failed for %s: %s", brief.topic_id, maybe_items)
                continue
            corroborating.extend(cast(list[Any], maybe_items))

        deduped_corroborating = _dedupe_discovered(corroborating)

        evidence_urls = list(base_urls)
        evidence_urls.extend(item.url for item in deduped_corroborating if item.url not in evidence_urls)
        evidence_urls = evidence_urls[: max(2, self.settings.deep_agent_max_evidence_sources)]

        delta_urls = [url for url in evidence_urls if url not in base_urls]
        delta_extracted = await self.tavily_client.extract_contents(
            client=http_client,
            urls=delta_urls,
            dry_run=dry_run,
        )
        extracted = {**base_extracted, **delta_extracted}

        evidence_texts: list[str] = []
        snippet_by_url = {
            item.url: " ".join((item.raw_content or item.snippet or "").split()).strip()
            for item in deduped_corroborating
        }
        max_chars = max(400, self.settings.deep_agent_max_evidence_chars)
        for url in evidence_urls:
            content = " ".join(extracted.get(url, "").split()).strip()
            if not content:
                content = snippet_by_url.get(url, "")
            if content:
                evidence_texts.append(f"URL: {url}\n{content[:max_chars]}")
        return evidence_urls, evidence_texts

    def _parse_finding(self, payload: Any) -> DeepAgentFinding:
        if isinstance(payload, dict):
            structured_response = payload.get("structured_response")
            if isinstance(structured_response, dict):
                return DeepAgentFinding.model_validate(structured_response)

        text_payload = _extract_last_message_text(payload)
        json_payload = _parse_json_payload(text_payload)
        try:
            return DeepAgentFinding.model_validate(json_payload)
        except ValidationError as exc:
            raise ValueError(f"Deep agent JSON validation failed: {exc}") from exc


def _build_deep_agent_prompt(brief: ResearchBrief, evidence_texts: list[str]) -> str:
    evidence_blob = "\n\n".join(evidence_texts[:6]) if evidence_texts else "No evidence snippets available."
    brief_payload = json.dumps(brief.model_dump(mode="json"), indent=2)
    return (
        "Perform adaptive technical investigation on this AI brief.\n"
        "Use your subagent skills to: "
        "1) extract technical novelty, 2) verify claims against evidence, 3) rewrite reflectively.\n"
        "Return strict JSON only with exactly these keys:\n"
        "- verification_status (verified|partially_verified|insufficient_evidence)\n"
        "- verification_confidence (0..1)\n"
        "- corrected_summary\n"
        "- corrected_technical_significance\n"
        "- corrected_business_impact\n"
        "- corrected_why_now\n"
        "- technical_implementation_notes (array of short strings)\n"
        "- verification_notes (array of short strings)\n"
        "Never mention inability to browse, scrape blocks, missing abstract pages, or tool limitations.\n"
        "If evidence is weak, use neutral phrasing: public evidence is still limited.\n"
        "Do not include markdown or explanations outside JSON.\n\n"
        f"Brief:\n{brief_payload}\n\n"
        f"Evidence:\n{evidence_blob}"
    )


def _apply_finding(brief: ResearchBrief, finding: DeepAgentFinding, evidence_count: int) -> ResearchBrief:
    updated = brief.model_copy(deep=True)
    updated.summary = _sanitize_access_failure_language(" ".join(finding.corrected_summary.split()))
    updated.technical_significance = _sanitize_access_failure_language(
        " ".join(finding.corrected_technical_significance.split())
    )
    updated.business_impact = _sanitize_access_failure_language(
        " ".join(finding.corrected_business_impact.split())
    )
    updated.why_now = _sanitize_access_failure_language(" ".join(finding.corrected_why_now.split()))
    updated.verification_status = finding.verification_status
    updated.verification_confidence = finding.verification_confidence

    merged_points = _merge_notes(
        base=updated.key_points,
        additions=finding.technical_implementation_notes,
        limit=7,
    )
    updated.key_points = merged_points

    merged_notes = _merge_notes(
        base=updated.verification_notes,
        additions=finding.verification_notes + [f"Adaptive investigation checked {evidence_count} sources."],
        limit=8,
    )
    updated.verification_notes = merged_notes

    if updated.verification_status in {"partially_verified", "insufficient_evidence"}:
        updated.summary = _make_cautious(updated.summary)

    return updated


def _merge_notes(base: list[str], additions: list[str], limit: int) -> list[str]:
    merged: list[str] = []
    for item in [*base, *additions]:
        normalized = " ".join(str(item).split()).strip()
        if not normalized:
            continue
        normalized = _sanitize_access_failure_language(normalized)
        if normalized in merged:
            continue
        merged.append(normalized)
        if len(merged) >= limit:
            break
    return merged


def _make_cautious(text: str) -> str:
    sentence = " ".join(text.split())
    if not sentence:
        return sentence
    if sentence.lower().startswith("based on current evidence"):
        return sentence
    return f"Based on current evidence, {sentence[0].lower() + sentence[1:]}"


def _extract_last_message_text(payload: Any) -> str:
    if not isinstance(payload, dict):
        return str(payload)

    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        return json.dumps(payload)

    last_message = messages[-1]
    content: Any = None
    if isinstance(last_message, dict):
        content = last_message.get("content")
    else:
        content = getattr(last_message, "content", None)

    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, str):
                chunks.append(item)
                continue
            if isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if isinstance(text, str):
                    chunks.append(text)
        return "\n".join(chunks)

    return str(content)


def _parse_json_payload(content: str) -> dict[str, Any]:
    stripped = content.strip()
    if stripped.startswith("{"):
        parsed = json.loads(stripped)
        if isinstance(parsed, dict):
            return cast(dict[str, Any], parsed)
        raise ValueError("JSON root must be an object")

    match = re.search(r"\{[\s\S]*\}", stripped)
    if not match:
        raise ValueError("No JSON object found in deep agent response")

    parsed = json.loads(match.group(0))
    if isinstance(parsed, dict):
        return cast(dict[str, Any], parsed)
    raise ValueError("JSON root must be an object")


def _build_evidence_queries(brief: ResearchBrief) -> list[str]:
    raw_queries = [
        brief.headline,
        f"{brief.headline} technical implementation details",
        f"{brief.headline} architecture benchmark limitations",
        f"{brief.headline} deployment constraints evaluation",
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


_MAIN_DEEP_AGENT_SYSTEM_PROMPT = (
    "You are the adaptive technical investigation layer for an AI news pipeline. "
    "You must avoid hype and sales framing. "
    "Your job is to verify claims conservatively and improve technical specificity. "
    "Never mention tool access limitations. "
    "When uncertainty exists, reduce confidence and state that public evidence is still emerging."
)
