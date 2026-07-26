"""The `propose` CLI contract.

`propose` drives the deterministic spine, exports the per-run bundle, and — only
when the export commits — delivers passing drafts to the LinkedIn bot. These
tests drive the real CLI entry point against stub research/writer subagents (no
OpenRouter), and pin the config gate, the export/delivery ordering, `--force`,
`--dry-run` and the stale-draft guard.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from app import main
from app.config import Settings
from support import (
    SHORT_BODY,
    StubResearchAgent,
    StubWriterAgent,
    disable_editor_veto,
    fixture_article,
    force_offline_ranking,
    mock_run_curation,
    settings_for,
)


def _settings(tmp_path: Path, **overrides: object) -> Settings:
    """Propose needs a non-empty key to clear the config gate; the ranker is
    pinned offline separately so nothing reaches the network."""
    return settings_for(tmp_path, **{"openrouter_api_key": "test-key", **overrides})


def _options(**overrides: object) -> main.ProposeOptions:
    base: dict[str, object] = {"force": False, "skip_delivery": False}
    base.update(overrides)
    return main.ProposeOptions(**base)  # type: ignore[arg-type]


def _wire_propose(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
    *,
    writer: StubWriterAgent | None = None,
    research: StubResearchAgent | None = None,
) -> None:
    """Point `propose` at the tmp settings and stub every external boundary:
    SearXNG autostart, the style-profile resolve, curation, the ranker's LLM,
    the editorial veto, and the two subagents."""
    monkeypatch.setattr(main, "get_settings", lambda: settings)

    async def _noop(_s: Settings) -> None:
        return None

    monkeypatch.setattr(main.searxng, "ensure_available", _noop)
    monkeypatch.setattr(main, "ensure_style_profile", _noop)

    force_offline_ranking(monkeypatch)
    disable_editor_veto(monkeypatch)
    mock_run_curation([fixture_article("topic-a")], monkeypatch)

    from app.orchestrator import spine as spine_mod

    monkeypatch.setattr(
        spine_mod, "build_research_agent", lambda s: research or StubResearchAgent(s)
    )
    monkeypatch.setattr(
        spine_mod, "build_writer_agent", lambda s: writer or StubWriterAgent(s)
    )


def _capture_sends(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, object]]:
    sends: list[dict[str, object]] = []

    async def _fake_send(self, text, *, bot="news", dry_run=False, disable_web_page_preview=True):  # type: ignore[no-untyped-def]
        sends.append({"bot": bot, "dry_run": dry_run, "chars": len(text)})
        return {"status": "sent", "message_id": len(sends) + 1}

    monkeypatch.setattr("app.services.telegram_client.TelegramClient.send_message", _fake_send)
    return sends


def test_propose_runs_spine_and_exports_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    settings = _settings(tmp_path)
    _wire_propose(tmp_path, monkeypatch, settings)

    exit_code = asyncio.run(main.run_propose(_options()))

    assert exit_code == 0
    # The spine produced a draft + gate verdict on disk.
    drafts = list((tmp_path / "drafts").glob("post-*.json"))
    assert len([d for d in drafts if not d.name.endswith(".gate.json")]) == 1
    # export_report wrote the bundle.
    assert len(list((tmp_path / "outputs").glob("*/posts.md"))) == 1
    out = capsys.readouterr().out
    assert "openrouter_calls=" in out
    assert "Proposals exported to" in out
    # The spine's honesty line reports what it did.
    assert "Spine: selected=1" in out
    assert "drafts_gated=1/1" in out


def test_propose_requires_openrouter_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """No OpenRouter key -> config error (exit 2) before the spine runs, rather
    than a mid-run 401."""
    settings = _settings(tmp_path, openrouter_api_key="")
    monkeypatch.setattr(main, "get_settings", lambda: settings)

    from app.orchestrator import spine as spine_mod

    def _must_not_build(_s: Settings) -> object:
        raise AssertionError("subagents must not be built without a key")

    monkeypatch.setattr(spine_mod, "build_research_agent", _must_not_build)
    monkeypatch.setattr(spine_mod, "build_writer_agent", _must_not_build)

    exit_code = asyncio.run(main.run_propose(_options()))

    assert exit_code == 2
    assert "OPENROUTER_API_KEY" in capsys.readouterr().out


def test_propose_refuses_to_clobber_same_day_bundle_without_force(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A second same-day run must not silently destroy the first run's
    (paid-for) bundle: export refuses, propose exits 1 and tells the operator to
    use --force. Delivery is unreachable on a refused run, so it posts nothing.
    Exercises the real export guard, not a stub."""
    settings = _settings(tmp_path)
    sends = _capture_sends(monkeypatch)

    _wire_propose(tmp_path, monkeypatch, settings)
    assert asyncio.run(main.run_propose(_options())) == 0
    capsys.readouterr()  # drain
    after_first_run = len(sends)

    _wire_propose(tmp_path, monkeypatch, settings)
    assert asyncio.run(main.run_propose(_options())) == 1
    assert "--force" in capsys.readouterr().out
    assert len(sends) == after_first_run  # refused run sent nothing

    _wire_propose(tmp_path, monkeypatch, settings)
    assert asyncio.run(main.run_propose(_options(force=True))) == 0
    assert "Proposals exported to" in capsys.readouterr().out


def test_propose_delivers_passing_draft_to_linkedin_bot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """With the linkedin profile configured, propose sends each passing draft to
    the linkedin bot (Decision C) via the deliver_telegram tool."""
    settings = _settings(
        tmp_path, telegram_linkedin_bot_token="ln-token", telegram_linkedin_chat_id="123456"
    )
    _wire_propose(tmp_path, monkeypatch, settings)
    sends = _capture_sends(monkeypatch)

    exit_code = asyncio.run(main.run_propose(_options()))

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
        tmp_path, telegram_linkedin_bot_token="ln-token", telegram_linkedin_chat_id="123456"
    )
    _wire_propose(tmp_path, monkeypatch, settings)
    sends = _capture_sends(monkeypatch)

    exit_code = asyncio.run(main.run_propose(_options(skip_delivery=True)))

    assert exit_code == 0
    assert sends == []
    assert "skipped LinkedIn Telegram delivery" in capsys.readouterr().out


def test_propose_archives_stale_drafts_before_delivery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A draft left over from a previous run must NOT be re-delivered — the
    workspace is archived at the start of the run, so delivery only ever sees
    this run's output. This is the duplicate-post guard."""
    settings = _settings(
        tmp_path, telegram_linkedin_bot_token="ln", telegram_linkedin_chat_id="1"
    )
    drafts = tmp_path / "drafts"
    drafts.mkdir(parents=True)
    (drafts / "stale-post.json").write_text("{}")
    (drafts / "stale-post.gate.json").write_text(json.dumps({"passed": True}))

    _wire_propose(tmp_path, monkeypatch, settings)
    sends = _capture_sends(monkeypatch)

    exit_code = asyncio.run(main.run_propose(_options()))

    assert exit_code == 0
    assert len(sends) == 1  # only this run's draft
    assert not (drafts / "stale-post.json").exists()
    # Archived, not deleted — a crashed run keeps its forensics.
    assert list((tmp_path / ".archive").glob("*/drafts/stale-post.json"))


def test_propose_delivers_only_gate_passed_drafts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A gate-failing draft is refused by the deliver_telegram gate check and
    reported, while the run itself still exports and exits 0."""
    settings = _settings(
        tmp_path, telegram_linkedin_bot_token="ln", telegram_linkedin_chat_id="1"
    )
    from app.orchestrator.spine import post_id_for
    from datetime import datetime, timezone

    article = fixture_article("topic-a")
    date_slug = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    failing_id = post_id_for(article.title, date_slug, set())

    _wire_propose(
        tmp_path,
        monkeypatch,
        settings,
        writer=StubWriterAgent(settings, body_for={failing_id: SHORT_BODY}),
    )
    sends = _capture_sends(monkeypatch)

    exit_code = asyncio.run(main.run_propose(_options()))

    assert exit_code == 0
    assert sends == []  # the only draft failed the gate
    out = capsys.readouterr().out
    assert "Delivered 0/1 proposal(s)" in out
    assert "gate_not_passed" in out
