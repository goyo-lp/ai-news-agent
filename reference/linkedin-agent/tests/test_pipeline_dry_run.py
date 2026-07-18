from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pytest

from app.config import get_settings
from app.graph.workflow import build_workflow


@pytest.mark.asyncio
async def test_pipeline_dry_run_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    outputs_dir = tmp_path / "outputs"
    samples_dir = tmp_path / "style_samples"
    style_profile_file = tmp_path / "style_profile.json"
    trusted_sources_file = Path(__file__).resolve().parents[1] / "data" / "trusted-sources.yaml"

    samples_dir.mkdir(parents=True, exist_ok=True)
    (samples_dir / "sample.txt").write_text(
        "I focus on practical AI execution. What are you seeing in your team?",
        encoding="utf-8",
    )

    monkeypatch.setenv("OUTPUTS_DIR", str(outputs_dir))
    monkeypatch.setenv("STYLE_SAMPLES_DIR", str(samples_dir))
    monkeypatch.setenv("STYLE_PROFILE_FILE", str(style_profile_file))
    monkeypatch.setenv("SOURCES_FILE", str(Path(__file__).resolve().parents[1] / "data" / "news-sources.yaml"))
    monkeypatch.setenv("TRUSTED_SOURCES_FILE", str(trusted_sources_file))
    monkeypatch.setenv("LANGGRAPHICS_ENABLED", "false")

    get_settings.cache_clear()
    try:
        workflow = build_workflow(enable_graphics=False)
        final_state = await workflow.ainvoke(
            {
                "run_id": str(uuid4()),
                "started_at": datetime.now(timezone.utc).isoformat(),
                "run_date": "2026-03-02",
                "dry_run": True,
                "hours_back": 48,
                "max_topics": 5,
                "errors": [],
            }
        )
    finally:
        get_settings.cache_clear()

    assert len(final_state.get("ranked_topics", [])) > 0
    assert len(final_state.get("research_briefs", [])) > 0
    assert len(final_state.get("linkedin_posts", [])) == 5
    assert len(final_state.get("delivery_results", [])) == 5
    assert final_state.get("export_dir")

    run_dir = outputs_dir / "2026-03-02"
    assert (run_dir / "top_50_articles.json").exists()
    assert (run_dir / "technical_candidates.json").exists()
    assert (run_dir / "adaptive_briefs.json").exists()
    assert (run_dir / "research_briefs.json").exists()
    assert (run_dir / "linkedin_posts.md").exists()
    assert (run_dir / "run_report.json").exists()

    report = json.loads((run_dir / "run_report.json").read_text(encoding="utf-8"))
    assert report["deliveries_attempted"] == 5
    assert report["deliveries_failed"] == 0
