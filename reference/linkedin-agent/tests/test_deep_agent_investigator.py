from __future__ import annotations

from datetime import datetime, timezone

import pytest

from app.config import Settings
from app.schemas import Citation, ResearchBrief
from app.services.deep_agent_investigator import DeepAgentFinding, DeepAgentInvestigator, _apply_finding


def _brief() -> ResearchBrief:
    return ResearchBrief(
        topic_id="topic-1",
        headline="A new agentic orchestration method",
        summary="The release introduces a new orchestration runtime.",
        technical_significance="Initial technical significance.",
        business_impact="Initial business impact.",
        why_now="Initial timing context.",
        key_points=["Primary source: example.com"],
        risks=["Early implementation details may change."],
        citations=[
            Citation(
                title="Source",
                url="https://example.com/source",
                domain="example.com",
                published_at=datetime.now(timezone.utc),
            )
        ],
    )


@pytest.mark.asyncio
async def test_investigator_skips_on_dry_run() -> None:
    settings = Settings(_env_file=None, deep_agent_enabled=True, openrouter_api_key=None)
    investigator = DeepAgentInvestigator(settings)

    brief = _brief()
    investigated, errors = await investigator.investigate_briefs(
        briefs=[brief],
        hours_back=24,
        dry_run=True,
    )

    assert not errors
    assert len(investigated) == 1
    assert investigated[0].model_dump(mode="json") == brief.model_dump(mode="json")


def test_apply_finding_merges_notes_and_adds_caution_for_partial_verification() -> None:
    brief = _brief()
    finding = DeepAgentFinding(
        verification_status="partially_verified",
        verification_confidence=0.55,
        corrected_summary="Evidence suggests measurable gains in tooling reliability.",
        corrected_technical_significance="It describes implementation-level orchestration changes.",
        corrected_business_impact="It may improve deployment confidence in production settings.",
        corrected_why_now="Coverage converged this week across multiple engineering sources.",
        technical_implementation_notes=["Uses explicit task routing and retry control."],
        verification_notes=["Benchmarks are promising but limited in scope."],
    )

    updated = _apply_finding(brief=brief, finding=finding, evidence_count=4)

    assert updated.verification_status == "partially_verified"
    assert updated.verification_confidence == 0.55
    assert updated.summary.startswith("Based on current evidence")
    assert "Uses explicit task routing and retry control." in updated.key_points
    assert any("Adaptive investigation checked 4 sources." == note for note in updated.verification_notes)
