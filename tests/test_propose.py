"""P8.3 — the `propose` CLI cutover.

`propose` drives the coordinator deep agent, then exports the per-run bundle.
These tests drive it end-to-end with the shared scripted-model harness (no
OpenRouter) via a monkeypatched ``build_coordinator_agent``, and pin the
config-gate on the OpenRouter key.
"""
from __future__ import annotations

import argparse
import asyncio
import json
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

    async def _noop_searxng(_s: Settings) -> None:
        return None

    monkeypatch.setattr(main, "_ensure_searxng", _noop_searxng)
    monkeypatch.setattr(main, "_ensure_style_profile", _noop_searxng)

    # Scripted-model drafts bypass submit_draft, so they lack provenance
    # signatures, and the script doesn't write briefs for citation-fidelity
    # checks. Bypass these checks for propose-level CLI tests.
    import app.orchestrator.tools.telegram as telegram_mod
    import app.orchestrator.tools.export as export_mod
    import app.orchestrator.tools.quality as quality_mod
    import app.orchestrator.services.evidence_floor as floor_mod

    monkeypatch.setattr(telegram_mod, "verify_draft", lambda _raw, _settings: True)
    monkeypatch.setattr(telegram_mod, "meets_evidence_floor", lambda _b, _s: (True, ""))
    monkeypatch.setattr(telegram_mod, "_check_evidence_floor", lambda _s, _p: (True, ""))
    monkeypatch.setattr(export_mod, "verify_draft", lambda _raw, _settings: True)
    monkeypatch.setattr(export_mod, "meets_evidence_floor", lambda _b, _s: (True, ""))
    monkeypatch.setattr(floor_mod, "meets_evidence_floor", lambda _b, _s: (True, ""))
    monkeypatch.setattr(quality_mod, "citation_fidelity_reasons", lambda _p, _b: [])
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

    exit_code = asyncio.run(main.run_propose(argparse.Namespace(verbose=False, force=False, dry_run=False, legacy_coordinator=True)))

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

    exit_code = asyncio.run(main.run_propose(argparse.Namespace(verbose=False, force=False, dry_run=False)))

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
    sends = _capture_sends(monkeypatch)

    # First run succeeds and writes today's bundle.
    _wire_scripted_propose(tmp_path, monkeypatch, settings, _full_run_script(tmp_path))
    assert asyncio.run(main.run_propose(argparse.Namespace(verbose=False, force=False, dry_run=False, legacy_coordinator=True))) == 0
    capsys.readouterr()  # drain
    after_first_run = len(sends)

    # Second run, no --force: export refuses -> exit 1 with the --force hint, and
    # delivery is unreachable, so the refused run posts nothing.
    _wire_scripted_propose(tmp_path, monkeypatch, settings, _full_run_script(tmp_path))
    assert asyncio.run(main.run_propose(argparse.Namespace(verbose=False, force=False, dry_run=False, legacy_coordinator=True))) == 1
    assert "--force" in capsys.readouterr().out
    assert len(sends) == after_first_run  # refused run sent nothing

    # Third run, --force: overwrites deliberately -> exit 0.
    _wire_scripted_propose(tmp_path, monkeypatch, settings, _full_run_script(tmp_path))
    assert asyncio.run(main.run_propose(argparse.Namespace(verbose=False, force=True, dry_run=False, legacy_coordinator=True))) == 0
    assert "Proposals exported to" in capsys.readouterr().out


def test_propose_delivers_passing_draft_to_linkedin_bot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """With the linkedin profile configured, propose sends each passing draft to
    the linkedin bot (Decision C) via the deliver_telegram tool."""
    settings = _settings(
        tmp_path,
        telegram_linkedin_bot_token="ln-token",
        telegram_linkedin_chat_id="123456",
    )
    _wire_scripted_propose(tmp_path, monkeypatch, settings, _full_run_script(tmp_path))

    sends: list[dict[str, object]] = []

    async def _fake_send(self, text, *, bot="news", dry_run=False, disable_web_page_preview=True):  # type: ignore[no-untyped-def]
        sends.append({"bot": bot, "dry_run": dry_run, "chars": len(text)})
        return {"status": "sent", "message_id": 42}

    monkeypatch.setattr(
        "app.services.telegram_client.TelegramClient.send_message", _fake_send
    )

    exit_code = asyncio.run(main.run_propose(argparse.Namespace(verbose=False, force=False, dry_run=False, legacy_coordinator=True)))

    assert exit_code == 0
    assert len(sends) == 1
    assert sends[0]["bot"] == "linkedin"
    assert sends[0]["dry_run"] is False
    assert "Delivered 1/1 proposal(s) to the linkedin bot" in capsys.readouterr().out


def test_propose_dry_run_skips_telegram_delivery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """--dry-run produces + exports proposals but never calls Telegram, even
    when the linkedin profile is configured."""
    settings = _settings(
        tmp_path,
        telegram_linkedin_bot_token="ln-token",
        telegram_linkedin_chat_id="123456",
    )
    _wire_scripted_propose(tmp_path, monkeypatch, settings, _full_run_script(tmp_path))

    calls: list[int] = []

    async def _fake_send(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        calls.append(1)
        return {"status": "sent", "message_id": 1}

    monkeypatch.setattr(
        "app.services.telegram_client.TelegramClient.send_message", _fake_send
    )

    exit_code = asyncio.run(main.run_propose(argparse.Namespace(verbose=False, force=False, dry_run=True, legacy_coordinator=True)))

    assert exit_code == 0
    assert calls == []  # no Telegram send
    assert "skipped LinkedIn Telegram delivery" in capsys.readouterr().out


def _short_draft_json(post_id: str, topic_id: str) -> str:
    """A schema-valid PostProposal whose body is far too short to clear the
    gate (word count << 105) — quality_gate writes passed=False for it."""
    return json.dumps(
        {
            "post_id": post_id,
            "angle": "short",
            "headline": "A draft that fails the gate",
            "body": "Too short to pass the quality gate.",
            "hashtags": ["#a", "#b", "#c"],
            "supporting_topic_ids": [topic_id],
            "citation_urls": [f"https://example.com/{topic_id}"],
            "confidence": 0.5,
        }
    )


def _capture_sends(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    sends: list[str] = []

    async def _fake_send(self, text, *, bot="news", dry_run=False, disable_web_page_preview=True):  # type: ignore[no-untyped-def]
        sends.append(bot)
        return {"status": "sent", "message_id": len(sends)}

    monkeypatch.setattr("app.services.telegram_client.TelegramClient.send_message", _fake_send)
    return sends


def test_propose_clears_stale_drafts_before_delivery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A draft left over from a previous run must NOT be re-delivered — the
    workspace is cleared at the start of the run, so delivery only sees this
    run's output. This is the duplicate-post guard."""
    settings = _settings(
        tmp_path, telegram_linkedin_bot_token="ln", telegram_linkedin_chat_id="1"
    )
    # A stale draft + passing gate from a prior run.
    drafts = tmp_path / "drafts"
    drafts.mkdir(parents=True)
    (drafts / "stale-post.json").write_text(stub_draft_json("stale-post", "old-topic"))
    (drafts / "stale-post.gate.json").write_text(json.dumps({"passed": True}))

    _wire_scripted_propose(tmp_path, monkeypatch, settings, _full_run_script(tmp_path))
    sends = _capture_sends(monkeypatch)

    exit_code = asyncio.run(main.run_propose(argparse.Namespace(verbose=False, force=False, dry_run=False, legacy_coordinator=True)))

    assert exit_code == 0
    # Only this run's draft (post-1) was sent; the stale one was wiped, not sent.
    assert len(sends) == 1
    assert not (drafts / "stale-post.json").exists()


def test_propose_delivers_only_gate_passed_drafts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """With one passing and one gate-failing draft, only the passing one is
    sent; the failing one is refused by the deliver_telegram gate check and
    reported."""
    settings = _settings(
        tmp_path, telegram_linkedin_bot_token="ln", telegram_linkedin_chat_id="1"
    )
    topic_id = "topic-a"
    script = [
        tool_call("fetch_curated_ai_news", {}),
        tool_call("technical_rank", {}),
        tool_call("write_file", {
            "file_path": abs_path(tmp_path, "drafts", "post-1.json"),
            "content": stub_draft_json("post-1", topic_id),
        }),
        tool_call("quality_gate", {"post_id": "post-1"}),
        tool_call("write_file", {
            "file_path": abs_path(tmp_path, "drafts", "post-2.json"),
            "content": _short_draft_json("post-2", topic_id),
        }),
        tool_call("quality_gate", {"post_id": "post-2"}),
        AIMessage(content="Done."),
    ]
    _wire_scripted_propose(tmp_path, monkeypatch, settings, script)
    sends = _capture_sends(monkeypatch)

    exit_code = asyncio.run(main.run_propose(argparse.Namespace(verbose=False, force=False, dry_run=False, legacy_coordinator=True)))

    assert exit_code == 0
    assert len(sends) == 1  # only the passing draft
    out = capsys.readouterr().out
    assert "Delivered 1/2 proposal(s)" in out
    assert "post-2" in out and "gate_not_passed" in out
