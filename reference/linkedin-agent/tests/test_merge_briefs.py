from __future__ import annotations

from datetime import datetime, timezone

import pytest

from app.nodes.merge_briefs import merge_briefs_node
from app.schemas import Citation, ResearchBrief


def _brief(topic_id: str, status: str, confidence: float, summary: str) -> ResearchBrief:
    return ResearchBrief(
        topic_id=topic_id,
        headline="Topic",
        summary=summary,
        technical_significance="Technical detail",
        business_impact="Business detail",
        why_now="Why now",
        citations=[
            Citation(
                title="Source",
                url="https://example.com/source",
                domain="example.com",
                published_at=datetime.now(timezone.utc),
            )
        ],
        verification_status=status,
        verification_confidence=confidence,
        verification_notes=["note-a"],
    )


@pytest.mark.asyncio
async def test_merge_briefs_prefers_stricter_verification_status() -> None:
    adaptive = _brief("t1", "verified", 0.9, "Adaptive summary")
    deterministic = _brief("t1", "partially_verified", 0.6, "Deterministic summary")

    final_state = await merge_briefs_node(
        {
            "deep_research_briefs": [adaptive.model_dump(mode="json")],
            "adaptive_briefs": [adaptive.model_dump(mode="json")],
            "verified_briefs": [deterministic.model_dump(mode="json")],
            "errors": [],
            "adaptive_errors": [],
            "verify_errors": [],
        }
    )

    merged_payload = final_state["research_briefs"][0]
    merged = ResearchBrief.model_validate(merged_payload)
    assert merged.verification_status == "partially_verified"
    assert merged.verification_confidence == pytest.approx(0.6)
    assert merged.summary.lower().startswith("based on current evidence")
