"""P3.1 — the linkedin-voice skill must load through deepagents' own parser and
stay in sync with the deterministic gate the writer subagent has to pass.

The gate (app.orchestrator.services.quality_gate) is the enforcement; the skill
is the instruction. If they drift — a hype marker added to the gate but not the
skill, or a word window that no longer matches — the writer gets told one thing
and judged by another. These tests pin the overlap so that drift breaks CI here
rather than silently at run time.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from app.orchestrator.services.quality_gate import (
    _HYPE_MARKERS,
    _MAX_POST_WORDS,
    _MIN_POST_WORDS,
)

# The skill lives at the repo root, not under src/, so it can be a skills source
# path for the writer subagent (P4.2). Resolve it from this test file's location.
_SKILL_DIR = Path(__file__).resolve().parents[1] / "skills" / "linkedin-voice"
_SKILL_MD = _SKILL_DIR / "SKILL.md"


def _skill_text() -> str:
    return _SKILL_MD.read_text(encoding="utf-8")


def test_skill_md_exists() -> None:
    assert _SKILL_MD.is_file(), f"missing skill file at {_SKILL_MD}"


def test_skill_parses_via_deepagents_parser() -> None:
    """Loads through the exact parser SkillsMiddleware uses — so a frontmatter
    mistake fails here instead of silently dropping the skill at run time."""
    from deepagents.middleware.skills import _parse_skill_metadata

    meta = _parse_skill_metadata(
        content=_skill_text(),
        skill_path=str(_SKILL_MD),
        directory_name=_SKILL_DIR.name,
    )

    assert meta is not None, "skill frontmatter failed to parse"
    assert meta["name"] == "linkedin-voice"
    assert meta["description"].strip()

    # Pin the spec's name↔directory invariant through the real validator, not
    # just the literal frontmatter string: a rename of the dir or the `name`
    # field that breaks the match (which makes the middleware skip the skill)
    # fails here.
    from deepagents.middleware.skills import _validate_skill_name

    is_valid, error = _validate_skill_name(meta["name"], _SKILL_DIR.name)
    assert is_valid, error


def test_skill_lists_every_gate_hype_marker() -> None:
    """The skill tells the writer which phrasings fail the gate. If the gate's
    marker set grows, the skill must list the new marker or the writer will keep
    emitting it and looping on retry.

    Apostrophes are normalized away on both sides: the gate carries an
    apostrophe-less duplicate (`cant miss` alongside `can't miss`) to catch
    stripped input, but the skill only needs to name the human-readable form."""

    def _norm(s: str) -> str:
        return s.lower().replace("'", "").replace("’", "")

    text = _norm(_skill_text())
    missing = [marker for marker in _HYPE_MARKERS if _norm(marker) not in text]
    assert not missing, f"skill omits hype markers the gate rejects: {missing}"


def test_skill_states_the_gate_word_window() -> None:
    """The 105–182 window is the writer's single most load-bearing constraint;
    if the gate's bounds change, the skill's stated numbers must follow."""
    text = _skill_text()
    assert str(_MIN_POST_WORDS) in text
    assert str(_MAX_POST_WORDS) in text


@pytest.mark.parametrize("token", ["one topic", "hashtag"])
def test_skill_names_the_other_gate_checks(token: str) -> None:
    """Single-topic and the hashtag floor are the remaining gate checks; the
    skill should surface them so the writer designs for them up front."""
    assert token in _skill_text().lower()
