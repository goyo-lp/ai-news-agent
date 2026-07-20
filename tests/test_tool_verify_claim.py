from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from app.config import Settings
from app.orchestrator.schemas import Citation, ResearchBrief
from app.orchestrator.services import brief_verifier as svc_mod
from app.orchestrator.tools.verify_claim import (
    build_verify_claim_tool,
    read_brief_from_state,
    verify_claim_tool,
    write_verified_brief_to_state,
)


def _settings(tmp_path: Path, *, api_key: str | None = None) -> Settings:
    return Settings(
        _env_file=None,
        orchestrator_data_dir=str(tmp_path),
        openrouter_api_key=api_key,
        openrouter_verifier_secondary_model=None,
        verification_sources_per_topic=3,
        verification_concurrency=4,
        request_timeout_seconds=10,
    )


def _brief(topic_id: str = "t1") -> ResearchBrief:
    return ResearchBrief(
        topic_id=topic_id,
        headline="A Big Announcement",
        summary="What happened and why it matters.",
        technical_significance="Architecture improves throughput.",
        business_impact="Cuts inference cost.",
        why_now="Ships this week.",
        citations=[Citation(title="Source", url="https://example.com/a", domain="example.com")],
    )


def _patch_openrouter_httpx(monkeypatch: pytest.MonkeyPatch, response: dict[str, Any]) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": json.dumps(response)}}]},
        )

    real = httpx.AsyncClient

    def _factory(*args: object, **kwargs: object) -> httpx.AsyncClient:
        kwargs["transport"] = httpx.MockTransport(handler)
        return real(**kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(svc_mod.httpx, "AsyncClient", _factory)


def test_write_verified_brief_to_state_round_trips(tmp_path: Path) -> None:
    brief = _brief("t1").model_copy(
        update={
            "verification_status": "verified",
            "verification_confidence": 0.9,
            "verification_notes": ["Confirmed."],
        }
    )
    path = write_verified_brief_to_state(brief, str(tmp_path))

    assert path.parent == tmp_path / "briefs"
    assert path.name == "t1.verified.json"
    on_disk = json.loads(path.read_text(encoding="utf-8"))
    assert on_disk["verification_status"] == "verified"
    assert ResearchBrief.model_validate(on_disk) == brief


def test_write_verified_brief_rejects_path_traversal_topic_id(tmp_path: Path) -> None:
    """Defense-in-depth: a topic_id that could escape the briefs/ subdir
    (slashes, `..`) is rejected at write time, not just at read time."""
    bad = _brief("../escape")
    with pytest.raises(ValueError):
        write_verified_brief_to_state(bad, str(tmp_path))


def test_read_brief_from_state_round_trips(tmp_path: Path) -> None:
    brief = _brief("t1")
    (tmp_path / "briefs").mkdir(parents=True)
    (tmp_path / "briefs" / "t1.json").write_text(
        json.dumps(brief.model_dump(mode="json"), indent=2, default=str), encoding="utf-8"
    )

    loaded = read_brief_from_state(str(tmp_path), "t1")
    assert loaded == brief


def test_read_brief_from_state_raises_on_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_brief_from_state(str(tmp_path), "nope")


def test_read_brief_rejects_path_traversal_topic_id(tmp_path: Path) -> None:
    """Same defense at the read path: `topic_id` never escapes the briefs/ dir."""
    with pytest.raises(ValueError):
        read_brief_from_state(str(tmp_path), "../escape")


async def test_tool_dry_run_writes_verified_brief_and_compressed_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings(tmp_path, api_key=None)
    tool = build_verify_claim_tool(settings)

    # Seed the brief on disk first (the research subagent's write).
    brief = _brief("t1")
    write_path = tmp_path / "briefs" / "t1.json"
    write_path.parent.mkdir(parents=True, exist_ok=True)
    write_path.write_text(json.dumps(brief.model_dump(mode="json"), default=str), encoding="utf-8")

    result = json.loads(await tool.ainvoke({"topic_id": "t1"}))

    assert set(result) == {
        "topic_id",
        "verification_status",
        "verification_confidence",
        "notes_count",
        "status",
        "reason",
        "path",
    }
    assert result["status"] == "ok"
    assert result["reason"] is None
    assert result["topic_id"] == "t1"
    assert result["verification_status"] == "partially_verified"
    assert result["verification_confidence"] >= 0.6
    assert result["notes_count"] >= 1
    assert Path(result["path"]).name == "t1.verified.json"

    # The compressed summary does NOT carry the brief body — it's on disk.
    assert "summary" not in result
    assert "headline" not in result

    # The verified brief round-trips from disk through the boundary model.
    artifact = json.loads(Path(result["path"]).read_text(encoding="utf-8"))
    assert ResearchBrief.model_validate(artifact).verification_status == "partially_verified"


async def test_tool_real_call_returns_verifier_verdict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings(tmp_path, api_key="sk-test")
    verifier_payload = {
        "verdict": "verified",
        "confidence": 0.88,
        "corrected_summary": "Based on current evidence, the launch is clean.",
        "corrected_technical_significance": "Documented architecture.",
        "corrected_business_impact": "Material cost improvement.",
        "corrected_why_now": "Ships this week.",
        "notes": ["Two sources confirm.", "Benchmarks reproducible."],
    }
    _patch_openrouter_httpx(monkeypatch, verifier_payload)

    tool = build_verify_claim_tool(settings)
    brief = _brief("t1")
    (tmp_path / "briefs").mkdir(parents=True, exist_ok=True)
    (tmp_path / "briefs" / "t1.json").write_text(
        json.dumps(brief.model_dump(mode="json"), default=str), encoding="utf-8"
    )

    result = json.loads(await tool.ainvoke({"topic_id": "t1", "hours_back": 24}))

    assert result["status"] == "ok"
    assert result["verification_status"] == "verified"
    assert abs(result["verification_confidence"] - 0.88) < 0.01
    assert result["notes_count"] == 2


async def test_tool_missing_brief_returns_error_not_raising(tmp_path: Path) -> None:
    """A missing brief is a precondition error — the tool returns a structured
    `failed`/`error` summary rather than raising into the agent loop."""
    settings = _settings(tmp_path, api_key=None)
    tool = build_verify_claim_tool(settings)

    result = json.loads(await tool.ainvoke({"topic_id": "missing"}))

    assert result["status"] == "error"
    assert result["verification_status"] == "failed"
    assert result["topic_id"] == "missing"
    assert "not found" in result["reason"].lower()
    assert result["path"] is None


async def test_tool_empty_topic_id_returns_error_without_writing(tmp_path: Path) -> None:
    settings = _settings(tmp_path, api_key=None)
    tool = build_verify_claim_tool(settings)

    result = json.loads(await tool.ainvoke({"topic_id": "   "}))
    assert result["status"] == "error"
    assert result["verification_status"] == "failed"
    assert "required" in result["reason"]
    assert not (tmp_path / "briefs").exists()


async def test_tool_path_traversal_topic_id_returns_error_without_writing(
    tmp_path: Path,
) -> None:
    """Defense-in-depth: a model-supplied `../etc/passwd`-style topic_id is
    rejected at the read-path guard before any file IO happens."""
    settings = _settings(tmp_path, api_key=None)
    tool = build_verify_claim_tool(settings)

    result = json.loads(await tool.ainvoke({"topic_id": "../secret"}))

    assert result["status"] == "error"
    assert result["path"] is None
    assert "invalid topic_id" in result["reason"].lower()

def test_default_singleton_is_a_structured_tool() -> None:
    assert verify_claim_tool.name == "verify_claim"
    assert verify_claim_tool.args_schema is not None
