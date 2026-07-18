from __future__ import annotations

import logging

from app.graph.state import AgentState
from app.schemas import ResearchBrief, parse_research_briefs, serialize_models
from app.services.tracing import traceable

logger = logging.getLogger(__name__)

_STATUS_RANK = {
    "verified": 0,
    "partially_verified": 1,
    "insufficient_evidence": 2,
}


@traceable(name="merge_briefs_node")
async def merge_briefs_node(state: AgentState) -> AgentState:
    baseline = parse_research_briefs(state.get("deep_research_briefs") or state.get("research_briefs"))
    adaptive = parse_research_briefs(state.get("adaptive_briefs"))
    verified = parse_research_briefs(state.get("verified_briefs"))

    adaptive_by_topic = {brief.topic_id: brief for brief in adaptive}
    verified_by_topic = {brief.topic_id: brief for brief in verified}

    ordered_topic_ids: list[str] = []
    for brief in baseline:
        if brief.topic_id not in ordered_topic_ids:
            ordered_topic_ids.append(brief.topic_id)
    for brief in adaptive:
        if brief.topic_id not in ordered_topic_ids:
            ordered_topic_ids.append(brief.topic_id)
    for brief in verified:
        if brief.topic_id not in ordered_topic_ids:
            ordered_topic_ids.append(brief.topic_id)

    merged_briefs: list[ResearchBrief] = []
    for topic_id in ordered_topic_ids:
        base = next((item for item in baseline if item.topic_id == topic_id), None)
        adaptive_brief = adaptive_by_topic.get(topic_id)
        verified_brief = verified_by_topic.get(topic_id)
        merged_briefs.append(_merge_one(base, adaptive_brief, verified_brief))

    errors = list(state.get("errors", []))
    errors.extend(state.get("adaptive_errors", []))
    errors.extend(state.get("verify_errors", []))
    merged_errors = _dedupe(errors)

    logger.info("Merged briefs complete: %s briefs", len(merged_briefs))
    return {
        "research_briefs": serialize_models(merged_briefs),
        "errors": merged_errors,
    }


def _merge_one(
    baseline: ResearchBrief | None,
    adaptive: ResearchBrief | None,
    verified: ResearchBrief | None,
) -> ResearchBrief:
    if adaptive is None and verified is None:
        if baseline is not None:
            return baseline
        raise ValueError("Cannot merge empty brief tuple")

    primary = adaptive or verified
    assert primary is not None
    merged = primary.model_copy(deep=True)

    if baseline is not None:
        # Preserve original citations and risks when enrichment branches are sparse.
        if not merged.citations:
            merged.citations = baseline.citations
        if not merged.risks:
            merged.risks = baseline.risks

    if verified is None or adaptive is None:
        return merged

    strictest = max(
        (adaptive, verified),
        key=lambda brief: _STATUS_RANK.get(brief.verification_status, 99),
    )
    merged.verification_status = strictest.verification_status
    merged.verification_confidence = min(
        _clip01(adaptive.verification_confidence),
        _clip01(verified.verification_confidence),
    )

    merged.verification_notes = _dedupe(
        [
            *adaptive.verification_notes,
            *verified.verification_notes,
            "Merged adaptive and deterministic verification branches.",
        ]
    )[:8]

    merged.key_points = _dedupe([*adaptive.key_points, *verified.key_points])[:7]

    if merged.verification_status in {"partially_verified", "insufficient_evidence"}:
        merged.summary = _make_cautious(merged.summary)
    return merged


def _dedupe(values: list[str]) -> list[str]:
    output: list[str] = []
    for value in values:
        normalized = " ".join(str(value).split()).strip()
        if not normalized:
            continue
        if normalized in output:
            continue
        output.append(normalized)
    return output


def _make_cautious(text: str) -> str:
    sentence = " ".join(text.split())
    if not sentence:
        return sentence
    if sentence.lower().startswith("based on current evidence"):
        return sentence
    return f"Based on current evidence, {sentence[0].lower() + sentence[1:]}"


def _clip01(value: float) -> float:
    return max(0.0, min(value, 1.0))
