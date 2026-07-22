"""P8.3 — the `propose` CLI cutover.

`propose` drives the coordinator deep agent, then exports the per-run bundle.
These tests drive it end-to-end with the shared scripted-model harness (no
OpenRouter) via a monkeypatched ``build_coordinator_agent``, and pin the
config-gate on the OpenRouter key.
"""
from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage

from app import main
from app.config import Settings
from app.orchestrator import agent as agent_module
from app.orchestrator.agent import build_coordinator_agent
from e2e_support import (
    ScriptedModel,
    abs_path,
    fixture_article,
    mock_run_curation_one_article,
    stub_brief_json,
    stub_draft_json,
    tool_call,
)


def _settings(tmp_path: Path, **overrides: object) -> Settings:
    kwargs: dict[str, object] = {
        "orchestrator_data_dir": str(tmp_path),
        "outputs_dir": str(tmp_path / "outputs"),
        "openrouter_api_key": "test-key",  # passes the propose config-gate
        "openrouter_coordinator_model": "coordinator-sentinel",
        "max_topics_per_run": 1,
    }
    kwargs.update(overrides)
    return Settings(_env_file=None, **kwargs)  # type: ignore[arg-type]


def _full_run_script(tmp_path: Path, topic_id: str = "topic-a", post_id: str = "post-1") -> list[AIMessage]:
    return [
        tool_call("fetch_curated_ai_news", {}),
        tool_call("technical_rank", {}),
        tool_call("write_file", {
            "file_path": abs_path(tmp_path, "briefs", f"{topic_id}.verified.json"),
            "content": stub_brief_json(topic_id),
        }),
        tool_call("write_file", {
            "file_path": abs_path(tmp_path, "drafts", f"{post_id}.json"),
            "content": stub_draft_json(post_id, topic_id),
        }),
        tool_call("quality_gate", {"post_id": post_id}),
        AIMessage(content="Done."),
    ]


def _wire_scripted_propose(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, settings: Settings, script: list[AIMessage]
) -> None:
    """Point `propose` at the tmp settings + a scripted coordinator (no network)."""
    monkeypatch.setattr(main, "get_settings", lambda: settings)
    mock_run_curation_one_article(fixture_article("topic-a"), monkeypatch)

    def _fake_build(s: Settings) -> object:
        return build_coordinator_agent(s, model=ScriptedModel(script=script))

    # run_propose imports build_coordinator_agent lazily from its source module
    # (to avoid a circular import), so patch it there, not on `main`.
    monkeypatch.setattr(agent_module, "build_coordinator_agent", _fake_build)


def test_propose_drives_agent_and_exports_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    settings = _settings(tmp_path)
    _wire_scripted_propose(tmp_path, monkeypatch, settings, _full_run_script(tmp_path))

    exit_code = asyncio.run(main.run_propose(argparse.Namespace(verbose=False, force=False)))

    assert exit_code == 0
    # The coordinator produced a passing draft + gate verdict on disk.
    assert (tmp_path / "drafts" / "post-1.json").exists()
    assert (tmp_path / "drafts" / "post-1.gate.json").exists()
    # export_report wrote the bundle.
    bundle = tmp_path / "outputs"
    posts = list(bundle.glob("*/posts.md"))
    assert len(posts) == 1
    # The usage summary + bundle path were printed.
    out = capsys.readouterr().out
    assert "openrouter_calls=" in out
    assert "Proposals exported to" in out


def test_propose_requires_openrouter_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """No OpenRouter key -> config error (exit 2) before the agent is built,
    rather than a mid-run 401. The agent factory must never be called."""
    settings = _settings(tmp_path, openrouter_api_key="")
    monkeypatch.setattr(main, "get_settings", lambda: settings)

    def _must_not_build(_s: Settings) -> object:
        raise AssertionError("coordinator must not be built without a key")

    monkeypatch.setattr(agent_module, "build_coordinator_agent", _must_not_build)

    exit_code = asyncio.run(main.run_propose(argparse.Namespace(verbose=False, force=False)))

    assert exit_code == 2
    assert "OPENROUTER_API_KEY" in capsys.readouterr().out


def test_propose_refuses_to_clobber_same_day_bundle_without_force(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A second same-day run must not silently destroy the first run's
    (paid-for) bundle: export refuses, propose exits 1 and tells the operator to
    use --force. `--force` then overwrites deliberately. Exercises the real
    export guard, not a stub."""
    settings = _settings(tmp_path)

    # First run succeeds and writes today's bundle.
    _wire_scripted_propose(tmp_path, monkeypatch, settings, _full_run_script(tmp_path))
    assert asyncio.run(main.run_propose(argparse.Namespace(verbose=False, force=False))) == 0
    capsys.readouterr()  # drain

    # Second run, no --force: export refuses -> exit 1 with the --force hint.
    _wire_scripted_propose(tmp_path, monkeypatch, settings, _full_run_script(tmp_path))
    assert asyncio.run(main.run_propose(argparse.Namespace(verbose=False, force=False))) == 1
    assert "--force" in capsys.readouterr().out

    # Third run, --force: overwrites deliberately -> exit 0.
    _wire_scripted_propose(tmp_path, monkeypatch, settings, _full_run_script(tmp_path))
    assert asyncio.run(main.run_propose(argparse.Namespace(verbose=False, force=True))) == 0
    assert "Proposals exported to" in capsys.readouterr().out
