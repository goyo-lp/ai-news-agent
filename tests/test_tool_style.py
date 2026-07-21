"""P3.2 — style-profile loader: service (build/load/default) + tool (load /
rebuild paths, state artifact, compressed summary, async-only)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.config import Settings
from app.orchestrator.schemas import StyleProfile
from app.orchestrator.services import style_profile as svc
from app.orchestrator.tools.style import (
    STYLE_PROFILE_FILENAME,
    build_load_style_profile_tool,
    load_style_profile_tool,
    write_profile_to_state,
)


def _settings(tmp_path: Path, *, seed: Path | None = None, samples: Path | None = None) -> Settings:
    return Settings(
        _env_file=None,
        orchestrator_data_dir=str(tmp_path / "state"),
        style_profile_file=str(seed if seed is not None else tmp_path / "seed.json"),
        style_samples_dir=str(samples if samples is not None else tmp_path / "samples"),
    )


def _write_samples(samples_dir: Path) -> None:
    samples_dir.mkdir(parents=True, exist_ok=True)
    (samples_dir / "a.md").write_text(
        "I tested a new AI workflow today. It reduced iteration time by 30%.",
        encoding="utf-8",
    )
    (samples_dir / "b.md").write_text(
        "What do you think about practical AI adoption in product teams? Curious to hear.",
        encoding="utf-8",
    )


# --------------------------------------------------------------------------- #
# service
# --------------------------------------------------------------------------- #


def test_build_from_texts_counts_and_traits() -> None:
    profile = svc.build_from_texts(
        [
            "I tested a new AI workflow today. It reduced iteration time by 30%.",
            "What do you think about practical AI adoption in product teams?",
        ]
    )
    assert profile.sample_count == 2
    assert profile.sentence_count >= 3
    assert profile.avg_sentence_words > 0
    assert profile.tone_traits


def test_build_from_texts_empty_falls_back_to_default() -> None:
    assert svc.build_from_texts([]) == svc.default_profile()
    assert svc.build_from_texts(["  ", ""]) == svc.default_profile()


def test_build_from_directory(tmp_path: Path) -> None:
    _write_samples(tmp_path)
    profile = svc.build_from_directory(tmp_path)
    assert profile.sample_count == 2


def test_build_from_directory_missing_dir_is_default(tmp_path: Path) -> None:
    assert svc.build_from_directory(tmp_path / "nope") == svc.default_profile()


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    profile = svc.build_from_texts(["AI systems ship fast. Curious what teams build."])
    path = svc.save_profile(profile, tmp_path / "nested" / "p.json")
    assert path.exists()
    assert svc.load_saved_profile(path) == profile


def test_load_saved_profile_missing_returns_none(tmp_path: Path) -> None:
    assert svc.load_saved_profile(tmp_path / "absent.json") is None


def test_load_saved_profile_corrupt_returns_none(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("{not valid json", encoding="utf-8")
    assert svc.load_saved_profile(bad) is None


# --------------------------------------------------------------------------- #
# tool
# --------------------------------------------------------------------------- #


async def test_tool_loads_existing_seed(tmp_path: Path) -> None:
    seed = tmp_path / "seed.json"
    svc.save_profile(svc.build_from_texts(["AI ships. Curious what teams build daily."]), seed)
    tool = build_load_style_profile_tool(_settings(tmp_path, seed=seed))

    summary = json.loads(await tool.ainvoke({"rebuild": False}))

    assert summary["status"] == "ok"
    assert summary["source"] == "loaded"
    state_path = Path(summary["path"])
    assert state_path.name == STYLE_PROFILE_FILENAME
    # the state artifact matches the seed the writer will read
    assert svc.load_saved_profile(state_path) == svc.load_saved_profile(seed)


async def test_tool_missing_seed_uses_default(tmp_path: Path) -> None:
    tool = build_load_style_profile_tool(_settings(tmp_path))  # seed does not exist
    summary = json.loads(await tool.ainvoke({"rebuild": False}))

    assert summary["source"] == "default"
    assert svc.load_saved_profile(Path(summary["path"])) == svc.default_profile()


async def test_tool_rebuild_from_samples_writes_state(tmp_path: Path) -> None:
    samples = tmp_path / "samples"
    _write_samples(samples)
    seed = tmp_path / "seed.json"
    settings = _settings(tmp_path, seed=seed, samples=samples)
    tool = build_load_style_profile_tool(settings)

    summary = json.loads(await tool.ainvoke({"rebuild": True}))

    assert summary["source"] == "rebuilt"
    assert summary["sample_count"] == 2
    # rebuild derives from the samples and writes only the state artifact...
    rebuilt = svc.build_from_directory(samples)
    assert svc.load_saved_profile(Path(summary["path"])) == rebuilt
    # ...and never touches the read-only seed (a runtime call must not mutate a
    # git-tracked file).
    assert not seed.exists()


async def test_tool_summary_omits_full_vocabulary(tmp_path: Path) -> None:
    samples = tmp_path / "samples"
    _write_samples(samples)
    tool = build_load_style_profile_tool(_settings(tmp_path, samples=samples))
    summary = json.loads(await tool.ainvoke({"rebuild": True}))

    assert "vocabulary" not in summary
    assert "common_openers" not in summary
    assert isinstance(summary["vocabulary_size"], int)


async def test_tool_error_surfaces_as_status_error(tmp_path: Path) -> None:
    # A file where the state dir must be created makes mkdir raise; the tool
    # must report status=error, not propagate into the agent loop.
    blocker = tmp_path / "state"
    blocker.write_text("i am a file, not a dir", encoding="utf-8")
    tool = build_load_style_profile_tool(_settings(tmp_path))

    summary = json.loads(await tool.ainvoke({"rebuild": False}))
    assert summary["status"] == "error"
    assert summary["path"] is None
    assert summary["reason"]


def test_write_profile_to_state_creates_missing_dir(tmp_path: Path) -> None:
    profile = StyleProfile(sample_count=3, tone_traits=["direct"])
    path = write_profile_to_state(profile, str(tmp_path / "deep" / "state"))
    assert path.exists()
    assert json.loads(path.read_text())["sample_count"] == 3


def test_tool_sync_invoke_raises_not_implemented() -> None:
    with pytest.raises(NotImplementedError):
        load_style_profile_tool.invoke({"rebuild": False})
