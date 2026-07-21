from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.config import Settings
from app.orchestrator.schemas import PostProposal
from app.orchestrator.services.quality_gate import (
    QualityResult,
    check_proposal,
)
from app.orchestrator.tools.quality import (
    build_quality_gate_tool,
    quality_gate_tool,
    read_proposal_from_state,
    write_gate_to_state,
)


def _settings(tmp_path: Path) -> Settings:
    return Settings(_env_file=None, orchestrator_data_dir=str(tmp_path))


def _words(n: int, theme: str = "the writer keeps watching for clearer results from this early signal which matters across the next several weeks for adopters and observers alike") -> str:
    base = theme.split()
    if len(base) >= n:
        return " ".join(base[:n])
    reps = (n + len(base) - 1) // len(base)
    return " ".join((" ".join(base * reps)).split()[:n])


def _proposal(
    *,
    body: str = "",
    topic_ids: list[str] | None = None,
    hashtags: list[str] | None = None,
    post_id: str = "post-1",
) -> PostProposal:
    return PostProposal(
        post_id=post_id,
        angle="reflection",
        headline="A technical note on: AI architecture",
        body=body,
        hashtags=hashtags if hashtags is not None else ["#AI", "#AIAgents", "#MachineLearning"],
        supporting_topic_ids=topic_ids if topic_ids is not None else ["t1"],
        citation_urls=["https://example.com/a"],
    )


def _write_draft(tmp_path: Path, post_id: str, proposal: PostProposal) -> None:
    drafts = tmp_path / "drafts"
    drafts.mkdir(parents=True, exist_ok=True)
    (drafts / f"{post_id}.json").write_text(
        json.dumps(proposal.model_dump(mode="json"), default=str),
        encoding="utf-8",
    )


# --- write/read helpers -----------------------------------------------------


def test_write_gate_to_state_round_trips(tmp_path: Path) -> None:
    result = QualityResult(
        passed=True,
        word_count=120,
        hashtags_count=3,
        single_topic=True,
        has_hype=False,
        hype_markers=[],
        cleaned_body="cleaned.",
        reasons=[],
    )
    path = write_gate_to_state(result, "post-1", str(tmp_path))

    assert path.parent == tmp_path / "drafts"
    assert path.name == "post-1.gate.json"
    on_disk = json.loads(path.read_text(encoding="utf-8"))
    assert on_disk["passed"] is True
    assert on_disk["word_count"] == 120
    assert on_disk["cleaned_body"] == "cleaned."


def test_write_gate_rejects_path_traversal_post_id(tmp_path: Path) -> None:
    result = QualityResult(
        passed=False,
        word_count=0,
        hashtags_count=0,
        single_topic=False,
        has_hype=False,
        cleaned_body="",
        reasons=["bad"],
    )
    with pytest.raises(ValueError):
        write_gate_to_state(result, "../escape", str(tmp_path))


def test_read_proposal_from_state_round_trips(tmp_path: Path) -> None:
    proposal = _proposal(body=_words(120))
    _write_draft(tmp_path, "post-1", proposal)

    loaded = read_proposal_from_state(str(tmp_path), "post-1")
    assert loaded == proposal


def test_read_proposal_raises_on_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_proposal_from_state(str(tmp_path), "nope")


def test_read_proposal_rejects_path_traversal_post_id(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        read_proposal_from_state(str(tmp_path), "../etc")


# --- tool -------------------------------------------------------------------


async def test_tool_passes_clean_draft_and_writes_verdict(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    tool = build_quality_gate_tool(settings)
    _write_draft(tmp_path, "post-1", _proposal(body=_words(120)))

    result = json.loads(await tool.ainvoke({"post_id": "post-1"}))

    assert set(result) == {
        "post_id",
        "passed",
        "word_count",
        "hashtags_count",
        "single_topic",
        "has_hype",
        "reasons_count",
        "status",
        "reason",
        "path",
    }
    assert result["status"] == "passed"
    assert result["passed"] is True
    assert result["reason"] is None
    assert 105 <= result["word_count"] <= 182
    assert result["hashtags_count"] == 3
    assert result["single_topic"] is True
    assert result["has_hype"] is False
    assert result["reasons_count"] == 0
    assert Path(result["path"]).name == "post-1.gate.json"

    # The summary does NOT carry the body or the full reasons list.
    assert "cleaned_body" not in result
    assert "reasons" not in result
    # The verdict artifact is on disk and carries the cleaned body + reasons.
    artifact = json.loads(Path(result["path"]).read_text(encoding="utf-8"))
    assert artifact["passed"] is True
    assert "cleaned_body" in artifact
    assert artifact["reasons"] == []


async def test_tool_fails_too_short_draft_and_reports_reason_count(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    tool = build_quality_gate_tool(settings)
    _write_draft(tmp_path, "post-1", _proposal(body=_words(50)))

    result = json.loads(await tool.ainvoke({"post_id": "post-1"}))

    assert result["status"] == "failed"
    assert result["passed"] is False
    assert result["reasons_count"] >= 1
    assert result["has_hype"] is False
    # Verdict artifact carries the failure-reason detail.
    artifact = json.loads(Path(result["path"]).read_text(encoding="utf-8"))
    assert any("too short" in r for r in artifact["reasons"])


async def test_tool_fails_hype_draft_and_reports_reason_count(tmp_path: Path) -> None:
    """Use hype markers that survive ``_strip_salesy_language`` (NOT in
    `_SALESY_REPLACEMENTS`) so they reach the cleaner-then-detect pipeline
    intact and actually trip `has_hype`. ``unbelievable`` and
    ``break the internet`` fit; ``revolutionary`` would get reworded to
    ``meaningful`` and never trip detection."""
    settings = _settings(tmp_path)
    tool = build_quality_gate_tool(settings)
    body = " ".join(["unbelievable", "launch", "break", "the", "internet", "today"] * 30)
    _write_draft(tmp_path, "post-1", _proposal(body=body))

    result = json.loads(await tool.ainvoke({"post_id": "post-1"}))

    assert result["status"] == "failed"
    assert result["has_hype"] is True
    assert result["reasons_count"] >= 1


async def test_tool_missing_draft_returns_error_status(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    tool = build_quality_gate_tool(settings)

    result = json.loads(await tool.ainvoke({"post_id": "missing"}))

    assert result["status"] == "error"
    assert result["passed"] is False
    assert result["path"] is None
    assert "not found" in result["reason"].lower()


async def test_tool_empty_post_id_returns_error_without_writing(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    tool = build_quality_gate_tool(settings)

    result = json.loads(await tool.ainvoke({"post_id": "   "}))

    assert result["status"] == "error"
    assert "required" in result["reason"]
    assert not (tmp_path / "drafts").exists()


async def test_tool_path_traversal_post_id_returns_error_without_writing(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    tool = build_quality_gate_tool(settings)

    result = json.loads(await tool.ainvoke({"post_id": "../secret"}))

    assert result["status"] == "error"
    assert result["path"] is None
    assert "invalid post_id" in result["reason"].lower()


def test_default_singleton_is_a_structured_tool() -> None:
    assert quality_gate_tool.name == "quality_gate"
    assert quality_gate_tool.args_schema is not None


# --- service integration: round trip through the tool ------------------------


async def test_tool_and_service_agree_on_clean_in_window_draft(
    tmp_path: Path,
) -> None:
    """The tool's compressed summary and the service's full QualityResult must
    agree on verdict + metrics — the tool wraps the service, doesn't re-derive
    anything the service already computed."""
    settings = _settings(tmp_path)
    tool = build_quality_gate_tool(settings)
    proposal = _proposal(body=_words(120))
    _write_draft(tmp_path, "post-1", proposal)

    tool_result = json.loads(await tool.ainvoke({"post_id": "post-1"}))
    service_result = check_proposal(proposal)

    assert tool_result["passed"] == service_result.passed
    assert tool_result["word_count"] == service_result.word_count
    assert tool_result["hashtags_count"] == service_result.hashtags_count
    assert tool_result["single_topic"] == service_result.single_topic
    assert tool_result["has_hype"] == service_result.has_hype
    assert tool_result["reasons_count"] == len(service_result.reasons)