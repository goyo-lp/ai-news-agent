from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, cast

import httpx

from app.config import Settings, get_settings
from app.graph.state import AgentState
from app.schemas import Citation, DiscoveredItem, RankedTopic, ResearchBrief, parse_ranked_topics, serialize_models
from app.services.api_usage_tracker import record_openrouter_http_response
from app.services.tavily_client import TavilyClient
from app.services.tracing import traceable

logger = logging.getLogger(__name__)

_MAX_CITATIONS = 6
_MAX_EVIDENCE_URLS = 8
_MAX_EVIDENCE_SNIPPETS = 6
_ARXIV_ID_RE = re.compile(r"\b\d{4}\.\d{4,5}(?:v\d+)?\b", re.IGNORECASE)


@traceable(name="deep_research_top5_node")
async def deep_research_top5_node(state: AgentState) -> AgentState:
    settings = get_settings()
    dry_run = bool(state.get("dry_run", False))
    hours_back = int(state.get("hours_back", settings.discovery_hours_back))

    ranked_topics = parse_ranked_topics(state.get("ranked_topics"))
    client = TavilyClient(settings)

    briefs: list[ResearchBrief] = []
    errors: list[str] = []

    timeout = httpx.Timeout(settings.request_timeout_seconds)
    semaphore = asyncio.Semaphore(max(1, settings.deep_research_topic_concurrency))

    async with httpx.AsyncClient(timeout=timeout) as http_client:

        async def worker(idx: int, topic: RankedTopic) -> tuple[int, ResearchBrief | None, list[str]]:
            async with semaphore:
                topic_errors: list[str] = []
                try:
                    brief, local_errors = await _build_topic_brief(
                        http_client=http_client,
                        tavily_client=client,
                        settings=settings,
                        topic=topic,
                        hours_back=hours_back,
                        dry_run=dry_run,
                    )
                    topic_errors.extend(local_errors)
                    return idx, brief, topic_errors
                except Exception as exc:
                    error = f"Deep research failed for topic {topic.topic_id}: {exc}"
                    logger.warning(error)
                    topic_errors.append(error)
                    return idx, None, topic_errors

        gathered = await asyncio.gather(*(worker(idx, topic) for idx, topic in enumerate(ranked_topics)))

    for _, maybe_brief, topic_errors in sorted(gathered, key=lambda item: item[0]):
        if maybe_brief is not None:
            briefs.append(maybe_brief)
        errors.extend(topic_errors)

    next_state = cast(AgentState, dict(state))
    serialized = serialize_models(briefs)
    next_state["research_briefs"] = serialized
    next_state["deep_research_briefs"] = serialized

    existing_errors = list(next_state.get("errors", []))
    existing_errors.extend(errors)
    next_state["errors"] = existing_errors

    logger.info("Deep research complete: %s briefs", len(briefs))
    return next_state


async def _build_topic_brief(
    *,
    http_client: httpx.AsyncClient,
    tavily_client: TavilyClient,
    settings: Settings,
    topic: RankedTopic,
    hours_back: int,
    dry_run: bool,
) -> tuple[ResearchBrief, list[str]]:
    errors: list[str] = []
    seed_urls = _dedupe_urls([topic.primary_url, *topic.supporting_urls])[: _MAX_EVIDENCE_URLS]

    llm_queries_task = asyncio.create_task(
        _plan_queries_with_llm(
            http_client=http_client,
            settings=settings,
            topic=topic,
            dry_run=dry_run,
        )
    )
    seed_extract_task = asyncio.create_task(
        tavily_client.extract_contents(
            client=http_client,
            urls=seed_urls,
            dry_run=dry_run,
        )
    )

    planned_queries = await llm_queries_task
    queries = _build_topic_queries(topic, llm_queries=planned_queries)

    search_tasks = [
        asyncio.create_task(
            tavily_client.search_news(
                client=http_client,
                query=query,
                hours_back=hours_back,
                max_results=4,
                dry_run=dry_run,
            )
        )
        for query in queries
    ]

    corroborating_nested = await asyncio.gather(*search_tasks, return_exceptions=True) if search_tasks else []
    seed_extracted = await seed_extract_task

    corroborating: list[DiscoveredItem] = []
    for idx, maybe_items in enumerate(corroborating_nested):
        if isinstance(maybe_items, Exception):
            errors.append(f"Search fallback failed for topic {topic.topic_id} query[{idx}]: {maybe_items}")
            continue
        corroborating.extend(cast(list[DiscoveredItem], maybe_items))

    corroborating = _dedupe_discovered_by_url(corroborating)

    evidence_urls = _dedupe_urls([*seed_urls, *[item.url for item in corroborating]])[:_MAX_EVIDENCE_URLS]
    delta_urls = [url for url in evidence_urls if url not in seed_urls]
    delta_extracted = await tavily_client.extract_contents(
        client=http_client,
        urls=delta_urls,
        dry_run=dry_run,
    )
    extracted = {**seed_extracted, **delta_extracted}

    evidence_texts = _build_evidence_texts(
        topic=topic,
        evidence_urls=evidence_urls,
        extracted=extracted,
        corroborating=corroborating,
    )

    citations = _build_citations(
        topic=topic,
        corroborating=corroborating,
        max_citations=_MAX_CITATIONS,
    )

    llm_brief_payload = await _synthesize_brief_with_llm(
        http_client=http_client,
        settings=settings,
        topic=topic,
        citations=citations,
        evidence_texts=evidence_texts,
        dry_run=dry_run,
    )

    if llm_brief_payload:
        brief = _build_brief_from_payload(topic=topic, citations=citations, payload=llm_brief_payload)
    else:
        brief = _build_brief_fallback(topic=topic, evidence_texts=evidence_texts, citations=citations)

    if not evidence_texts:
        errors.append(f"Limited extracted evidence for topic {topic.topic_id}; used cautious fallback briefing.")

    brief.verification_notes = _merge_notes(
        brief.verification_notes,
        [
            f"Seed URLs scanned: {len(seed_urls)}",
            f"Fallback web queries executed: {len(queries)}",
            f"Total evidence URLs considered: {len(evidence_urls)}",
        ],
        limit=8,
    )

    return brief, errors


def _build_topic_queries(topic: RankedTopic, llm_queries: list[str]) -> list[str]:
    title = " ".join(topic.title.split())
    query_candidates: list[str] = [
        title,
        f"{title} technical implementation details",
        f"{title} benchmarks methodology results",
        f"{title} architecture workflow integration",
        f"{title} production deployment constraints",
    ]

    arxiv_id = _extract_arxiv_id(topic.title, topic.primary_url)
    if arxiv_id:
        query_candidates.extend(
            [
                f"arXiv {arxiv_id} summary technical details",
                f"{arxiv_id} model method benchmark",
            ]
        )

    for llm_query in llm_queries:
        if llm_query.strip():
            query_candidates.append(llm_query.strip())

    deduped: list[str] = []
    for query in query_candidates:
        normalized = " ".join(query.split()).strip()
        if not normalized:
            continue
        if normalized.lower() in {item.lower() for item in deduped}:
            continue
        deduped.append(normalized)
        if len(deduped) >= 6:
            break
    return deduped


async def _plan_queries_with_llm(
    *,
    http_client: httpx.AsyncClient,
    settings: Settings,
    topic: RankedTopic,
    dry_run: bool,
) -> list[str]:
    if dry_run or not (settings.openrouter_api_key or "").strip():
        return []

    payload = {
        "model": settings.openrouter_model,
        "temperature": 0.2,
        "max_tokens": 260,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You generate focused web search queries for technical AI research discovery. "
                    "Return strict JSON only."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Given this topic, generate up to 3 query strings that can surface concrete "
                    "technical implementation details, benchmarks, architecture notes, or failure modes. "
                    "Avoid marketing phrasing.\n"
                    "Return JSON: {\"queries\": [\"...\", \"...\"]}.\n"
                    f"Topic title: {topic.title}\n"
                    f"Hint: {topic.summary_hint}\n"
                    f"Primary URL: {topic.primary_url}"
                ),
            },
        ],
    }

    headers = {
        "Authorization": f"Bearer {settings.openrouter_api_key}",
        "Content-Type": "application/json",
    }
    if settings.openrouter_site_url:
        headers["HTTP-Referer"] = settings.openrouter_site_url
    if settings.openrouter_app_name:
        headers["X-Title"] = settings.openrouter_app_name

    try:
        response = await http_client.post(
            f"{settings.openrouter_base_url}/chat/completions",
            headers=headers,
            json=payload,
        )
        response.raise_for_status()
        response_payload = response.json()
        record_openrouter_http_response(
            model=settings.openrouter_model,
            payload=cast(dict[str, Any], response_payload),
        )
        content = str(response_payload["choices"][0]["message"]["content"])
        parsed = _parse_json_payload(content)
    except Exception as exc:
        logger.warning("LLM query planner failed for topic %s: %s", topic.topic_id, exc)
        return []

    raw_queries = parsed.get("queries")
    if not isinstance(raw_queries, list):
        return []
    queries = [str(item).strip() for item in raw_queries if str(item).strip()]
    return queries[:3]


def _build_evidence_texts(
    *,
    topic: RankedTopic,
    evidence_urls: list[str],
    extracted: dict[str, str],
    corroborating: list[DiscoveredItem],
) -> list[str]:
    snippet_by_url = {item.url: (item.raw_content or item.snippet or "").strip() for item in corroborating}

    evidence_texts: list[str] = []
    for url in evidence_urls:
        content = " ".join(extracted.get(url, "").split()).strip()
        if not content:
            content = " ".join(snippet_by_url.get(url, "").split()).strip()
        if not content and url == topic.primary_url:
            content = " ".join(topic.summary_hint.split()).strip()
        if not content:
            continue

        trimmed = content[:1600].strip()
        evidence_texts.append(f"URL: {url}\n{trimmed}")
        if len(evidence_texts) >= _MAX_EVIDENCE_SNIPPETS:
            break

    return evidence_texts


def _build_citations(
    *,
    topic: RankedTopic,
    corroborating: list[DiscoveredItem],
    max_citations: int,
) -> list[Citation]:
    citations: list[Citation] = [
        Citation(
            title=topic.title,
            url=topic.primary_url,
            domain=topic.primary_domain,
            published_at=topic.published_at,
        )
    ]

    seen_urls = {topic.primary_url}
    for item in corroborating:
        if item.url in seen_urls:
            continue
        seen_urls.add(item.url)
        citations.append(
            Citation(
                title=item.title,
                url=item.url,
                domain=item.domain,
                published_at=item.published_at,
            )
        )
        if len(citations) >= max(2, max_citations):
            break

    return citations


async def _synthesize_brief_with_llm(
    *,
    http_client: httpx.AsyncClient,
    settings: Settings,
    topic: RankedTopic,
    citations: list[Citation],
    evidence_texts: list[str],
    dry_run: bool,
) -> dict[str, Any] | None:
    if dry_run or not (settings.openrouter_api_key or "").strip():
        return None

    evidence_blob = "\n\n".join(evidence_texts[:_MAX_EVIDENCE_SNIPPETS]) if evidence_texts else "No extracted evidence snippets available."
    citations_blob = "\n".join(f"- {citation.title} ({citation.url})" for citation in citations[:_MAX_CITATIONS])

    payload = {
        "model": settings.openrouter_model,
        "temperature": 0.25,
        "max_tokens": 1300,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You write technical AI research briefs for engineers. "
                    "Never mention your own browsing limitations, scraping issues, or inability to access pages. "
                    "If evidence is limited, state that public evidence is still limited in neutral language. "
                    "Return strict JSON only."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Create a concise research brief for this topic.\n"
                    "Return JSON with keys exactly:\n"
                    "summary, technical_significance, business_impact, why_now, key_points, risks.\n"
                    "Rules:\n"
                    "- Focus on technical implementation details, methods, architecture, benchmarks, and tradeoffs.\n"
                    "- No marketing tone and no sales calls to action.\n"
                    "- key_points: 3-6 short bullets.\n"
                    "- risks: 2-4 short bullets.\n\n"
                    f"Topic: {topic.title}\n"
                    f"Signal rationale: {topic.rationale}\n"
                    f"Cluster size: {topic.cluster_size}\n"
                    f"Citations:\n{citations_blob}\n\n"
                    f"Evidence:\n{evidence_blob}"
                ),
            },
        ],
    }

    headers = {
        "Authorization": f"Bearer {settings.openrouter_api_key}",
        "Content-Type": "application/json",
    }
    if settings.openrouter_site_url:
        headers["HTTP-Referer"] = settings.openrouter_site_url
    if settings.openrouter_app_name:
        headers["X-Title"] = settings.openrouter_app_name

    try:
        response = await http_client.post(
            f"{settings.openrouter_base_url}/chat/completions",
            headers=headers,
            json=payload,
        )
        response.raise_for_status()
        response_payload = response.json()
        record_openrouter_http_response(
            model=settings.openrouter_model,
            payload=cast(dict[str, Any], response_payload),
        )
        content = str(response_payload["choices"][0]["message"]["content"])
        parsed = _parse_json_payload(content)
        if not isinstance(parsed, dict):
            return None
        return parsed
    except Exception as exc:
        logger.warning("LLM brief synthesis failed for topic %s: %s", topic.topic_id, exc)
        return None


def _build_brief_from_payload(
    *,
    topic: RankedTopic,
    citations: list[Citation],
    payload: dict[str, Any],
) -> ResearchBrief:
    key_points_raw = payload.get("key_points")
    risks_raw = payload.get("risks")

    key_points = [str(item).strip() for item in key_points_raw if str(item).strip()] if isinstance(key_points_raw, list) else []
    risks = [str(item).strip() for item in risks_raw if str(item).strip()] if isinstance(risks_raw, list) else []

    summary = _sanitize_access_fail_language(str(payload.get("summary") or "").strip())
    technical_significance = _sanitize_access_fail_language(
        str(payload.get("technical_significance") or "").strip()
    )
    business_impact = _sanitize_access_fail_language(str(payload.get("business_impact") or "").strip())
    why_now = _sanitize_access_fail_language(str(payload.get("why_now") or "").strip())

    if not summary:
        summary = _fallback_summary(topic)
    if not technical_significance:
        technical_significance = _infer_technical_significance(topic.title, "")
    if not business_impact:
        business_impact = _infer_business_impact(topic.title, "")
    if not why_now:
        why_now = _infer_why_now(topic)

    if len(key_points) < 2:
        key_points = _merge_notes(
            [
                f"Primary source domain: {topic.primary_domain}",
                f"Signal quality: {topic.rationale}",
            ],
            key_points,
            limit=6,
        )

    if len(risks) < 2:
        risks = _merge_notes(
            risks,
            [
                "Public evidence may still be incomplete and can shift with fuller benchmarks.",
                "Production implications should be validated against real workload constraints.",
            ],
            limit=4,
        )

    return ResearchBrief(
        topic_id=topic.topic_id,
        headline=topic.title,
        summary=summary,
        technical_significance=technical_significance,
        business_impact=business_impact,
        why_now=why_now,
        key_points=key_points[:6],
        risks=risks[:4],
        citations=citations,
    )


def _build_brief_fallback(
    *,
    topic: RankedTopic,
    evidence_texts: list[str],
    citations: list[Citation],
) -> ResearchBrief:
    content = " ".join((evidence_texts[0] if evidence_texts else topic.summary_hint).split())
    content = re.sub(r"^URL:\s*\S+\s*", "", content).strip()
    if len(content) > 420:
        content = content[:420].rstrip() + "..."

    technical_significance = _infer_technical_significance(topic.title, content)
    business_impact = _infer_business_impact(topic.title, content)
    why_now = _infer_why_now(topic)

    key_points = _merge_notes(
        [
            f"Primary source domain: {topic.primary_domain}",
            f"Signal quality: {topic.rationale}",
        ],
        [content] if content else [],
        limit=6,
    )

    risks = [
        "Public evidence is still evolving; treat current claims as directional.",
        "Benchmark and deployment outcomes may change as implementation details are released.",
    ]

    summary = _fallback_summary(topic)

    return ResearchBrief(
        topic_id=topic.topic_id,
        headline=topic.title,
        summary=summary,
        technical_significance=technical_significance,
        business_impact=business_impact,
        why_now=why_now,
        key_points=key_points,
        risks=risks,
        citations=citations,
    )


def _fallback_summary(topic: RankedTopic) -> str:
    return (
        f"{topic.title}. Current coverage suggests a technically relevant development with "
        f"{topic.cluster_size} related source signal(s); this is worth tracking as implementation "
        "details continue to surface."
    )


def _infer_technical_significance(title: str, content: str) -> str:
    combined = f"{title} {content}".lower()
    if any(token in combined for token in {"model", "reasoning", "benchmark", "multimodal", "retrieval", "rag", "memory"}):
        return "Indicates concrete progress in model behavior, retrieval strategy, or evaluation depth."
    if any(token in combined for token in {"chip", "inference", "latency", "throughput", "serving"}):
        return "Highlights serving and infrastructure tradeoffs that directly affect production performance."
    if any(token in combined for token in {"agent", "workflow", "orchestration", "tooling", "a2a"}):
        return "Focuses on agent orchestration and system design decisions with implementation implications."
    return "Represents an implementation-level AI development with practical engineering implications."


def _infer_business_impact(title: str, content: str) -> str:
    combined = f"{title} {content}".lower()
    if any(token in combined for token in {"enterprise", "deployment", "customer", "platform", "integration"}):
        return "Likely to influence enterprise roadmap decisions around integration and deployment strategy."
    if any(token in combined for token in {"startup", "funding", "series", "acquisition", "partnership"}):
        return "Signals market movement that can reshape product priorities and competitive timing."
    return "Creates a practical decision point for teams evaluating AI implementation bets this quarter."


def _infer_why_now(topic: RankedTopic) -> str:
    if topic.published_at is None:
        return "Recent coverage concentration makes this a timely signal to monitor while evidence matures."

    return (
        f"Published on {topic.published_at.date().isoformat()} and reinforced by related coverage, "
        "making it timely for near-term engineering planning."
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
        raise ValueError("No JSON object found in response")
    parsed = json.loads(match.group(0))
    if isinstance(parsed, dict):
        return cast(dict[str, Any], parsed)
    raise ValueError("JSON root must be an object")


def _sanitize_access_fail_language(text: str) -> str:
    normalized = " ".join(text.split()).strip()
    if not normalized:
        return normalized

    replacement = "Public evidence is still limited and should be treated as directional."
    patterns = [
        r"\b(i|we)\s+(can(?:not|'t)|could(?:\s+not|n't)|did(?:\s+not|n't))\s+(access|verify|find|retrieve)[^.!?]*[.!?]?",
        r"\bno\s+(abstract|methodology|full\s+text|authors)\b[^.!?]*[.!?]?",
        r"\bfrom\s+what\s+is\s+publicly\s+indexed\b[^.!?]*[.!?]?",
        r"\bregistration\s+is\s+still\s+pending\b[^.!?]*[.!?]?",
    ]

    cleaned = normalized
    for pattern in patterns:
        cleaned = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE)

    cleaned = " ".join(cleaned.split()).strip()
    if not cleaned:
        return replacement
    return cleaned


def _extract_arxiv_id(*values: str) -> str | None:
    for value in values:
        match = _ARXIV_ID_RE.search(value or "")
        if match:
            return match.group(0)
    return None


def _dedupe_urls(urls: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for url in urls:
        normalized = " ".join(str(url).split()).strip()
        if not normalized:
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped


def _dedupe_discovered_by_url(items: list[DiscoveredItem]) -> list[DiscoveredItem]:
    deduped: list[DiscoveredItem] = []
    seen: set[str] = set()
    for item in items:
        if item.url in seen:
            continue
        seen.add(item.url)
        deduped.append(item)
    return deduped


def _merge_notes(base: list[str], additions: list[str], limit: int) -> list[str]:
    merged: list[str] = []
    for item in [*base, *additions]:
        normalized = " ".join(str(item).split()).strip()
        if not normalized:
            continue
        normalized = _sanitize_access_fail_language(normalized)
        if normalized in merged:
            continue
        merged.append(normalized)
        if len(merged) >= limit:
            break
    return merged
