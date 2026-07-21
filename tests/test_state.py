"""P5.1 — orchestrator state backend + filesystem data conventions.

Pins two independent contracts:

1. **Path convention parity** — the helpers in ``app.orchestrator.state``
   produce *exactly* the same paths the existing inline-string tools
   (``news.py``, ``fetch.py``, ``technical_rank.py``, ``verify_claim.py``,
   ``quality.py``, ``style.py``) build. If any tool's inline string drifts
   from this central convention, the parity test catches it.

2. **Backend round-trip + cross-tool visibility** — ``build_orchestrator_backend``
   mounts a real ``FilesystemBackend`` rooted at ``orchestrator_data_dir``
   so a brief written via the deepagents built-in ``write_file`` (subagent
   path) is visible to a stdlib read at ``<data_dir>/briefs/<topic>.json``
   (custom-tool path). This is the contract the research/writer subagents
   committed to in P4 and the coordinator finally mounts here.

3. **Path-traversal guard** — the ``*_path`` builders that take an externally
   supplied id reject separators / ``..`` / empty, mirroring the inline
   guards the custom tools already enforce.

Per guiding principle #3, structured data lives on the filesystem/backend,
never rides a subagent's chat reply — this module is *where* on the
filesystem it lives, and these tests pin that the "where" is what every
consumer agrees on.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from deepagents.backends.filesystem import FilesystemBackend

from app.config import Settings
from app.orchestrator.state import (
    ARTICLES_FILENAME,
    BRIEFS_DIRNAME,
    DRAFTS_DIRNAME,
    STYLE_PROFILE_FILENAME,
    TOPICS_FILENAME,
    articles_path,
    brief_path,
    build_orchestrator_backend,
    draft_path,
    gate_path,
    style_profile_path,
    topics_path,
    verified_brief_path,
)


def _settings(tmp_path: Path) -> Settings:
    """Build a Settings whose orchestrator data dir points at the tmp dir so
    tests don't write into the repo's real data tree."""
    return Settings(_env_file=None, orchestrator_data_dir=str(tmp_path))


# --------------------------------------------------------------------------- #
# Path-convention parity: helpers must match the inline strings tools build.
# --------------------------------------------------------------------------- #


def test_articles_path_matches_news_tool_filename() -> None:
    """`articles_path` and `news.ARTICLES_FILENAME` must produce the same
    filename — technical_rank imports the constant and reads that file."""
    from app.orchestrator.tools.news import ARTICLES_FILENAME as tools_name

    assert ARTICLES_FILENAME == tools_name == "articles.json"
    assert articles_path("/d").name == ARTICLES_FILENAME


def test_topics_path_matches_technical_rank_filename() -> None:
    """`topics_path` and technical_rank's inline `_TOPICS_FILENAME` must
    agree — drift would split the producer/consumer pair."""
    from app.orchestrator.tools.technical_rank import _TOPICS_FILENAME

    assert TOPICS_FILENAME == _TOPICS_FILENAME == "topics.json"
    assert topics_path("/d").name == TOPICS_FILENAME


def test_style_profile_path_matches_style_tool_filename() -> None:
    """`style_profile_path` and `style.STYLE_PROFILE_FILENAME` must agree —
    the writer subagent reads that exact file under the data dir."""
    from app.orchestrator.tools.style import STYLE_PROFILE_FILENAME as tools_name

    assert STYLE_PROFILE_FILENAME == tools_name == "style_profile.json"
    assert style_profile_path("/d").name == STYLE_PROFILE_FILENAME


def test_brief_path_matches_verify_claim_subdir_convention() -> None:
    """`brief_path` and `verified_brief_path` live under the same `briefs/`
    subdir verify_claim already writes to, with the same per-topic filename
    pattern (``<topic_id>.json`` and ``<topic_id>.verified.json``)."""
    from app.orchestrator.tools.verify_claim import (
        _BRIEFS_SUBDIR,
        _BRIEF_SUFFIX,
        _VERIFIED_SUFFIX,
    )

    assert BRIEFS_DIRNAME == _BRIEFS_SUBDIR == "briefs"

    bp = brief_path("climate-gpt", "/d")
    assert bp == Path("/d") / BRIEFS_DIRNAME / f"climate-gpt{_BRIEF_SUFFIX}"

    vbp = verified_brief_path("climate-gpt", "/d")
    assert vbp == Path("/d") / BRIEFS_DIRNAME / f"climate-gpt{_VERIFIED_SUFFIX}"

    # Same parent, distinct filenames — verify's contract is "verified copy
    # alongside the original", not "overwrite".
    assert bp.parent == vbp.parent
    assert bp.name != vbp.name


def test_draft_and_gate_paths_match_quality_tool_convention() -> None:
    """`draft_path` / `gate_path` and the quality_gate tool's inline paths
    must agree — coordinator-level delivery reads the same files the writer
    subagent's quality_gate call writes.

    Imports the quality tool's own constants (`_DRAFTS_SUBDIR`,
    `_DRAFT_SUFFIX`, `_GATE_SUFFIX`) and pins the literal equality both
    directions. Catches a rename in quality.py (e.g. ``_DRAFTS_SUBDIR =
    "draft"`` typo, ``_GATE_SUFFIX = ".quality.json"``) that a one-direction
    test against `state.DRAFTS_DIRNAME` would never surface."""
    from app.orchestrator.tools.quality import (
        _DRAFTS_SUBDIR,
        _DRAFT_SUFFIX,
        _GATE_SUFFIX,
    )

    assert DRAFTS_DIRNAME == _DRAFTS_SUBDIR == "drafts"
    assert _DRAFT_SUFFIX == ".json"
    assert _GATE_SUFFIX == ".gate.json"

    bp = draft_path("post-42", "/d")
    gp = gate_path("post-42", "/d")
    assert bp == Path("/d") / _DRAFTS_SUBDIR / f"post-42{_DRAFT_SUFFIX}"
    assert gp == Path("/d") / _DRAFTS_SUBDIR / f"post-42{_GATE_SUFFIX}"
    assert bp.parent == gp.parent
    assert bp.name != gp.name  # gate never overwrites the draft


# --------------------------------------------------------------------------- #
# Path-traversal guard
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "bad",
    [
        "",
        "a/b",
        "..",
        ".",
        "tra/versal",
        "endslash/",
        "/leading",
    ],
)
def test_brief_path_rejects_traversal_or_separator_in_topic_id(bad: str) -> None:
    with pytest.raises(ValueError, match="topic_id"):
        brief_path(bad, "/d")


@pytest.mark.parametrize(
    "bad",
    [
        "",
        "a/b",
        "..",
        ".",
        "tra/versal",
        "endslash/",
        "/leading",
    ],
)
def test_verified_brief_path_rejects_traversal_or_separator_in_topic_id(bad: str) -> None:
    with pytest.raises(ValueError, match="topic_id"):
        verified_brief_path(bad, "/d")


@pytest.mark.parametrize(
    "bad",
    [
        "",
        "a/b",
        "..",
        ".",
        "tra/versal",
        "endslash/",
        "/leading",
    ],
)
def test_draft_path_rejects_traversal_or_separator_in_post_id(bad: str) -> None:
    with pytest.raises(ValueError, match="post_id"):
        draft_path(bad, "/d")


@pytest.mark.parametrize("bad", ["", "a/b", "..", ".", "/leading"])
def test_gate_path_rejects_traversal_or_separator_in_post_id(bad: str) -> None:
    with pytest.raises(ValueError, match="post_id"):
        gate_path(bad, "/d")


def test_path_helpers_accept_normal_slugs() -> None:
    """Sanity: slugs the model would actually hand these tools pass the guard.
    Pins the valid-form boundary so a future tightening of the guard knows what
    still has to work — including hyphenated ids and the literal slugs other
    tests in this file use (so a future guard change that breaks them surfaces
    here, not as a misleading failure in a peer test)."""
    assert brief_path("topic-1", "/d").name == "topic-1.json"
    assert verified_brief_path("topic_2", "/d").name == "topic_2.verified.json"
    assert draft_path("postid", "/d").name == "postid.json"
    assert gate_path("postid", "/d").name == "postid.gate.json"
    # The hyphenated shape the model would actually emit (a typical topic_id):
    assert brief_path("stmt-2026-q1", "/d").name == "stmt-2026-q1.json"
    assert gate_path("stmt-2026-q1-draft", "/d").name == "stmt-2026-q1-draft.gate.json"


def test_path_helpers_pass_whitespace_for_inline_guard_parity() -> None:
    """The inline guards in verify_claim and quality (`if "/" in topic_id or
    topic_id in {"", ".", ".."}`) do NOT reject whitespace-only ids. The
    module helper is parity-bound to that accepted set (see
    ``_validate_slug``'s docstring) — accepting whitespace keeps the helper
    symmetric to the inline guard so both sides write the same filename.
    Pinning this behavior protects the cross-tool visibility contract: a
    well-meaning future "tightening" that rejects whitespace here would
    silently break parity by producing a different filename than the custom
    tool writes under the same input."""
    assert brief_path(" foo ", "/d").name == " foo .json"
    assert verified_brief_path(" foo ", "/d").name == " foo .verified.json"
    assert draft_path(" foo ", "/d").name == " foo .json"
    assert gate_path(" foo ", "/d").name == " foo .gate.json"


# --------------------------------------------------------------------------- #
# Backend construction + cross-tool visibility
# --------------------------------------------------------------------------- #


def test_build_orchestrator_backend_is_filesystem_backend_rooted_at_data_dir(
    tmp_path: Path,
) -> None:
    """The mounted backend is a FilesystemBackend rooted at
    orchestrator_data_dir — not StateBackend. StateBackend would isolate
    subagent write_file payloads to LangGraph state, breaking the
    custom-tool / subagent visibility contract the research+writer
    subagents (P4) already document as a precondition. This test pins the
    concrete backend class + root."""
    s = _settings(tmp_path)
    backend = build_orchestrator_backend(s)
    assert isinstance(backend, FilesystemBackend)
    assert Path(backend.cwd).resolve() == tmp_path.resolve()


def test_build_orchestrator_backend_creates_root_if_missing(tmp_path: Path) -> None:
    """The coordinator's first action is typically `write_todos` / a call to
    `fetch_curated_ai_news`; the root must exist before any write lands. The
    backend constructor itself doesn't mkdir — we do, mirroring the tools'
    own `mkdir(parents=True, exist_ok=True)`."""
    root = tmp_path / "nested" / "orchestrator"
    s = Settings(_env_file=None, orchestrator_data_dir=str(root))
    assert not root.exists()
    build_orchestrator_backend(s)
    assert root.exists() and root.is_dir()


def test_build_orchestrator_backend_lazy_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """When settings are not supplied, the factory falls back to the
    process-wide ``get_settings()`` lru_cache (same seam the tool factories
    use). We pin that path by setting ``OPENROUTER_API_KEY`` to a sentinel
    and replacing the lru_cache with one rooted at our tmp dir."""
    from app.config import get_settings

    monkeypatch.setenv("ORCHESTRATOR_DATA_DIR", str(tmp_path))
    get_settings.cache_clear()
    try:
        backend = build_orchestrator_backend()
        assert Path(backend.cwd).resolve() == tmp_path.resolve()
    finally:
        get_settings.cache_clear()
        monkeypatch.delenv("ORCHESTRATOR_DATA_DIR", raising=False)


def test_backend_round_trip_persists_across_calls(tmp_path: Path) -> None:
    """A file written via the deepagents built-in ``awrite`` survives a
    fresh ``aread`` from the *same* backend (turn-2 of a run). This is the
    guiding-principle-#3 guarantee that structured data persists across
    turns — never has to ride a chat reply."""
    backend = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=False)
    asyncio.run(backend.awrite("artifacts/notes.txt", "phase5 round-trip"))
    result = asyncio.run(backend.aread("artifacts/notes.txt"))
    assert result.error is None
    assert result.file_data["content"] == "phase5 round-trip"
    # Also on disk (visible to stdlib custom tools):
    on_disk = (tmp_path / "artifacts" / "notes.txt").read_text(encoding="utf-8")
    assert on_disk == "phase5 round-trip"


def test_backend_write_visible_to_stdlib_read_cross_tool(tmp_path: Path) -> None:
    """The critical seam: a brief written via ``backend.awrite`` (subagent's
    ``write_file`` path) is visible to a stdlib ``Path.read_text`` at the
    location ``brief_path`` returns (verify_claim's read path). This is the
    "structured data lives in the filesystem, both sides see the same file"
    contract Phase 5 ships; without it, verify_claim would always report
    'brief not found'."""
    s = _settings(tmp_path)
    backend = build_orchestrator_backend(s)

    topic_id = "stmt-2026-q1"
    brief_payload = {"topic_id": topic_id, "headline": "Stable test headline"}
    # Subagent path: deepagents' built-in write_file routes through the backend
    # with a path relative to its root.
    rel = str(brief_path(topic_id, tmp_path).relative_to(tmp_path))
    asyncio.run(backend.awrite(rel, json.dumps(brief_payload)))

    # Custom-tool path: stdlib read at the absolute path state.brief_path returns
    absolute = brief_path(topic_id, tmp_path)
    assert absolute.exists()
    parsed = json.loads(absolute.read_text(encoding="utf-8"))
    assert parsed["topic_id"] == topic_id
    assert parsed["headline"] == "Stable test headline"


def test_stdlib_write_visible_to_backend_read_cross_tool(tmp_path: Path) -> None:
    """The reverse seam: a verdict written via stdlib (verify_claim's write
    path) is visible to a subsequent backend ``aread`` (the writer subagent's
    read_file path). Pins symmetry — visibility is bidirectional, not write-
    only."""
    s = _settings(tmp_path)
    backend = build_orchestrator_backend(s)

    topic_id = "stmt-2026-q1"
    absolute = verified_brief_path(topic_id, tmp_path)
    absolute.parent.mkdir(parents=True, exist_ok=True)
    absolute.write_text(
        json.dumps({"topic_id": topic_id, "verification_status": "verified"}),
        encoding="utf-8",
    )

    rel = str(absolute.relative_to(tmp_path))
    result = asyncio.run(backend.aread(rel))
    assert result.error is None
    parsed = json.loads(result.file_data["content"])
    assert parsed["verification_status"] == "verified"


# Principle #3 ("structured data lives on disk, subagent chat replies carry
# only compressed pointers, never the bulk payload") is *enforced* at the
# tool layer (each tool returns a compressed JSON summary, never the body)
# and the prompt layer (P5.3). It is not, and cannot be, a state.py
# invariant — state.py only owns *where on the filesystem* data lives. A
# prior version of this file carried a "documentation-as-test" asserting
# properties of fabricated test data; removed because it exercised nothing
# actually in this module. See prompts.py (P5.3) + each tool's summary shape
# for the actual principle-#3 enforcement points.