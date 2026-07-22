"""P8.1 — full dry-run E2E snapshot.

Drives the real deepagents coordinator deterministically (scripted fake model,
no OpenRouter) through the run-order the prompt prescribes — fetch → rank →
write brief+verified → write draft → gate — then runs ``export_report`` to
produce the per-run bundle, and pins the deterministic bundle files against a
committed golden. Two contracts:

  1. **Stable snapshot** — ``posts.md`` + ``briefs.json`` match the committed
     fixture byte-for-byte. A change in the rendered post format, the brief
     schema serialization, or the pipeline wiring surfaces here as a diff.
  2. **Reruns match** — two independent runs (fresh data dirs) produce
     byte-identical ``posts.md`` + ``briefs.json``. Catches any hidden
     nondeterminism (dict ordering, timestamps leaking into the artifacts).

``run_report.json`` embeds absolute data-dir paths, so only its
path-independent ``counts`` block is pinned (structured summary), not the whole
file.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage

from app.orchestrator.agent import build_coordinator_agent
from app.orchestrator.tools.export import build_export_report_tool
from e2e_support import (
    ScriptedModel,
    abs_path,
    fixture_article,
    mock_run_curation_one_article,
    settings_for,
    stub_brief_json,
    stub_draft_json,
    tool_call,
)

_FIXTURE_DIR = Path(__file__).parent / "fixtures" / "e2e_snapshot"
_FIXTURE_DATE = "2026-07-21"
_TOPIC_ID = "topic-a"
_POST_ID = "post-1"


def _run_pipeline_and_export(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[str, str, dict[str, object]]:
    """Drive the scripted coordinator + export_report against a data dir rooted
    at ``tmp_path``. Returns ``(posts_md, briefs_json, run_report)``."""
    mock_run_curation_one_article(fixture_article(_TOPIC_ID), monkeypatch)
    settings = settings_for(
        tmp_path, max_topics_per_run=1, outputs_dir=str(tmp_path / "outputs")
    )
    script = [
        tool_call("fetch_curated_ai_news", {}),
        tool_call("technical_rank", {}),
        tool_call("write_file", {
            "file_path": abs_path(tmp_path, "briefs", f"{_TOPIC_ID}.json"),
            "content": stub_brief_json(_TOPIC_ID),
        }),
        tool_call("write_file", {
            "file_path": abs_path(tmp_path, "briefs", f"{_TOPIC_ID}.verified.json"),
            "content": stub_brief_json(_TOPIC_ID),
        }),
        tool_call("write_file", {
            "file_path": abs_path(tmp_path, "drafts", f"{_POST_ID}.json"),
            "content": stub_draft_json(_POST_ID, _TOPIC_ID),
        }),
        tool_call("quality_gate", {"post_id": _POST_ID}),
        AIMessage(content="Done."),
    ]
    agent = build_coordinator_agent(settings, model=ScriptedModel(script=script))
    asyncio.run(agent.ainvoke({"messages": [{"role": "user", "content": "Run for today."}]}))

    export_tool = build_export_report_tool(settings)
    result = json.loads(asyncio.run(export_tool.ainvoke({"date": _FIXTURE_DATE})))
    assert result["status"] == "ok", result

    bundle = Path(result["bundle_dir"])
    posts_md = (bundle / "posts.md").read_text(encoding="utf-8")
    briefs_json = (bundle / "briefs.json").read_text(encoding="utf-8")
    run_report = json.loads((bundle / "run_report.json").read_text(encoding="utf-8"))
    return posts_md, briefs_json, run_report


def test_snapshot_matches_committed_golden(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The rendered posts.md + serialized briefs.json match the committed
    golden byte-for-byte — the stable-snapshot contract."""
    posts_md, briefs_json, _ = _run_pipeline_and_export(tmp_path, monkeypatch)

    expected_posts = (_FIXTURE_DIR / "posts.md").read_text(encoding="utf-8")
    expected_briefs = (_FIXTURE_DIR / "briefs.json").read_text(encoding="utf-8")

    assert posts_md == expected_posts
    assert briefs_json == expected_briefs


def test_snapshot_run_report_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The run report's path-independent counts block is the structured summary
    an operator branches on. Pin it; the absolute-path fields are tmp-specific
    and deliberately not snapshotted."""
    _, _, run_report = _run_pipeline_and_export(tmp_path, monkeypatch)

    assert run_report["date"] == _FIXTURE_DATE
    assert run_report["counts"] == {
        "topics": 1,
        "briefs": 1,
        "drafts": 1,
        "drafts_passed": 1,
        "drafts_failed": 0,
        "drafts_gate_absent": 0,
    }


def test_reruns_are_byte_identical(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two independent runs in fresh data dirs produce byte-identical
    artifacts — the determinism contract that makes the snapshot meaningful."""
    posts_a, briefs_a, _ = _run_pipeline_and_export(tmp_path / "run_a", monkeypatch)
    posts_b, briefs_b, _ = _run_pipeline_and_export(tmp_path / "run_b", monkeypatch)

    assert posts_a == posts_b
    assert briefs_a == briefs_b
