"""Editor relevance veto — one batched LLM editorial pass over the ranked
candidate pool, before any research spend lands.

Motivation (2026-07-25 production run): the deterministic ranker + the LLM
coordinator's selection shipped an Android-ADB platform story and a Ruff
formatter release note from an "AI news" agent — both well-scored by the
technical ranker, both off-topic for an AI-industry LinkedIn audience. The
ranker's keyword model can't tell "AI-adjacent dev tooling" from "random
developer tooling"; one cheap editorial call can.

Design:
  * ONE batched call for the whole pool (≤ ~40 rows) — not per-topic — so
    the veto costs one round-trip and one small prompt, not N.
  * Fail-open everywhere: a veto error, a parse failure, or a missing
    verdict for an index all resolve to *keep*. An advisory filter must
    never kill a run or silently drop everything.
  * Dry-run / no API key -> no-op (every candidate kept), same contract as
    the Stage-A ranker and the verifier.
  * Usage is recorded through the raw-httpx adapter so the veto's spend
    lands in the run's usage report like the ranker/verifier's.
"""
from __future__ import annotations

import json
import logging
import re

import httpx

from app.config import Settings
from app.orchestrator.schemas import TopicCandidate
from app.orchestrator.usage import record_openrouter_http_response

logger = logging.getLogger(__name__)

_HINT_CHARS = 200
_MAX_CANDIDATES = 40

_SYSTEM = (
    "You are the editor of a LinkedIn account covering the AI industry — "
    "models, products, startups, funding, enterprise adoption, technical "
    "breakthroughs, AI policy. You decide which candidate stories are worth "
    "a post to that audience."
)

_INSTRUCTIONS = """For each numbered candidate below, decide KEEP or DROP for a professional AI-industry audience.

KEEP: model/product/feature launches, AI company funding/deals/partnerships,
technical breakthroughs and results, enterprise AI adoption, AI policy with
industry impact, and AI-adjacent developer tooling (coding assistants,
agents, eval/observability frameworks, AI infra).

DROP: pure dev-tooling release notes with no AI angle (formatters, linters,
build tools), OS/platform news unrelated to AI, generic event roundups or
newsletter digests, rumor/opinion with no concrete development, and
non-AI hardware.

Respond with JSON ONLY, one verdict per candidate:
{"verdicts": [{"i": <candidate number>, "keep": <true|false>, "reason": "<≤12 words>"}]}

CANDIDATES:
"""


def _format_candidates(topics: list[TopicCandidate]) -> str:
    lines: list[str] = []
    for i, t in enumerate(topics, start=1):
        hint = " ".join(t.summary_hint.split())[:_HINT_CHARS]
        lines.append(f"{i}. {t.title}\n   domain: {t.primary_domain} | {hint}")
    return "\n".join(lines)


def parse_verdicts(text: str, count: int) -> dict[int, tuple[bool, str]]:
    """Extract ``{index: (keep, reason)}`` from the model's reply. Robust to
    prose around the JSON block (grabs the outermost brace span); a malformed
    payload yields an empty dict — every caller treats a missing index as
    keep (fail-open)."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return {}
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}
    raw = payload.get("verdicts")
    if not isinstance(raw, list):
        return {}
    verdicts: dict[int, tuple[bool, str]] = {}
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        try:
            idx = int(entry.get("i"))
        except (TypeError, ValueError):
            continue
        if not 1 <= idx <= count:
            continue
        keep = entry.get("keep")
        reason = str(entry.get("reason") or "").strip()
        if isinstance(keep, bool):
            verdicts[idx] = (keep, reason)
    return verdicts


async def veto_irrelevant_topics(
    topics: list[TopicCandidate],
    settings: Settings,
    *,
    dry_run: bool,
) -> tuple[list[TopicCandidate], list[dict[str, str]]]:
    """Run the editorial veto over the candidate pool.

    Returns ``(survivors, vetoes)`` where ``vetoes`` is a list of
    ``{topic_id, title, reason}`` dicts for dropped candidates (for the run
    summary). Order is preserved. Fail-open: any error returns the full
    pool with empty vetoes."""
    if not topics or not settings.editor_veto_enabled or dry_run:
        return topics, []

    candidates = topics[:_MAX_CANDIDATES]
    model = settings.openrouter_editor_model or settings.openrouter_coordinator_model
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": _INSTRUCTIONS + _format_candidates(candidates)},
        ],
        "temperature": 0,
        "max_tokens": 2000,
        "usage": {"include": True},
    }
    headers = {"Authorization": f"Bearer {settings.openrouter_api_key}"}

    try:
        async with httpx.AsyncClient(timeout=settings.request_timeout_seconds * 2) as client:
            response = await client.post(
                f"{settings.openrouter_base_url.rstrip('/')}/chat/completions",
                json=payload,
                headers=headers,
            )
            response.raise_for_status()
            data = response.json()
    except Exception as exc:
        logger.warning("editor veto call failed (%s) — keeping full pool", exc)
        return topics, []

    record_openrouter_http_response(model=model, payload=data)

    text = ""
    try:
        text = str(data["choices"][0]["message"]["content"] or "")
    except (KeyError, IndexError, TypeError):
        logger.warning("editor veto: no content in response — keeping full pool")
        return topics, []

    verdicts = parse_verdicts(text, len(candidates))
    if not verdicts:
        logger.warning("editor veto: no parseable verdicts — keeping full pool")
        return topics, []

    survivors: list[TopicCandidate] = []
    vetoes: list[dict[str, str]] = []
    for i, topic in enumerate(candidates, start=1):
        keep, reason = verdicts.get(i, (True, ""))
        if keep:
            survivors.append(topic)
        else:
            vetoes.append({"topic_id": topic.topic_id, "title": topic.title, "reason": reason})
            logger.info("editor veto DROP: %s — %s", topic.title, reason)
    # Candidates beyond the batch cap (shouldn't happen) are kept untouched.
    survivors.extend(topics[_MAX_CANDIDATES:])
    return survivors, vetoes


__all__ = ["parse_verdicts", "veto_irrelevant_topics"]
