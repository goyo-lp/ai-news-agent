"""Orchestrator state backend + filesystem data conventions.

Phase 5 entry. The migration plan describes the lamp as "configure
StateBackend + file conventions"; the mounted backend here is
**FilesystemBackend**, not ``StateBackend``.

The full divergence story (why a real filesystem backend is load-bearing for
subagent <-> custom-tool visibility) lives in the research/writer subagent
docstrings from P4 — this module only adds what was missing:

1. ``build_orchestrator_backend(settings)`` — single place that turns
   ``settings.orchestrator_data_dir`` into a ``FilesystemBackend`` rooted
   there (``virtual_mode=False`` so the writer subagent's absolute
   ``skills_source`` path the writer passes still resolves through to the
   real repo, same contract P4.2's writer subagent already relies on).
2. Filename / subdir constants + path helpers (``articles_path``,
   ``topics_path``, ``brief_path``, ``verified_brief_path``, ``draft_path``,
   ``gate_path``, ``style_profile_path``) — the forward-going source of
   truth for *where on the filesystem* structured data lives (guiding
   principle #3). The parity tests pin agreement with the inline strings
   the existing tools (news, technical_rank, verify_claim, quality, style)
   already build, so the inline strings can't silently drift; adopting the
   helpers per-tool is incremental and out of this PR's LOC budget.
3. ``_validate_slug`` — single path-traversal guard reused by every
   ``*_path`` builder that takes an externally supplied id (topic_id /
   post_id). The custom tools already guard inline; this module is the
   forward-going single source so a new helper can't forget it. The guard
   mirrors the inline rules in ``verify_claim`` / ``quality`` exactly; see
   its docstring for the parity guarantee.

Scope-out (honest): the fetch_article tool writes its per-URL artifact to
``<data_dir>/articles/<slug>.json`` (``fetch._ARTICLES_SUBDIR = "articles"``)
— a distinct subdir from ``articles.json`` (the curated snapshot). That
subdir is part of the convention, but no helper + no parity test exists for
it yet; adopting it lives with a future refactor of ``fetch.py`` onto these
helpers, not in this PR's scope. Documented here so the omission isn't
inferred."""
from __future__ import annotations

from pathlib import Path

from deepagents.backends.filesystem import FilesystemBackend

from app.config import Settings, get_settings

# --------------------------------------------------------------------------- #
# Filename / subdir convention (single source of truth)
# --------------------------------------------------------------------------- #
# These names are the literal filenames the deterministic tools already write.
# Centralizing them here lets the parity test catch any drift between an
# inline-string tool and this convention; a future refactor can swap each tool's
# inline string for ``state.X`` one-by-one without breaking the contract.

ARTICLES_FILENAME = "articles.json"
TOPICS_FILENAME = "topics.json"
STYLE_PROFILE_FILENAME = "style_profile.json"

BRIEFS_DIRNAME = "briefs"
DRAFTS_DIRNAME = "drafts"


def _validate_slug(slug: str, *, kind: str) -> str:
    """Reject any externally supplied id that could escape its dir.

    The custom tools (verify_claim, quality_gate) each guard this inline; this
    helper is the forward-going single source so a new ``*_path`` builder can't
    forget it.

    Parity: mirrors the inline guards in ``verify_claim._brief_file_path`` and
    ``quality._draft_path`` *literally* — no normalization, no whitespace
    stripping. A slug with leading/trailing whitespace passes both the inline
    guard and this one; the resulting filename carries that whitespace on both
    sides (symmetric, both sides see the same file). Adding ``.strip()`` here
    would make this helper asymmetric to the inline guards and silently
    produce a different filename than the custom tools (e.g.
    ``" foo "`` -> ``"foo.json"`` here while the inline guard writes
    ``" foo .json"``). That asymmetry would *break* the cross-tool visibility
    contract, so the helper does not strip.
    """
    if not slug or slug in {"", ".", ".."}:
        raise ValueError(f"{kind} slug is empty or a path metacharacter: {slug!r}")
    if "/" in slug:
        raise ValueError(f"{kind} slug must not contain '/': {slug!r}")
    # ``..`` as a *component* (e.g. ``foo/..``) is caught by the ``/`` rule
    # above; ``..`` as a *bare slug* is caught by the explicit set. This
    # leaves ``.`` and ``..`` already covered and keeps the helper's accepted
    # set exactly equal to the inline guards' accepted set.
    return slug


def articles_path(data_dir: str | Path) -> Path:
    """``<data_dir>/articles.json`` — the curated snapshot fetch_curated_ai_news
    writes and technical_rank reads."""
    return Path(data_dir) / ARTICLES_FILENAME


def topics_path(data_dir: str | Path) -> Path:
    """``<data_dir>/topics.json`` — ranked candidates technical_rank writes and
    the coordinator loops over to delegate research."""
    return Path(data_dir) / TOPICS_FILENAME


def style_profile_path(data_dir: str | Path) -> Path:
    """``<data_dir>/style_profile.json`` — the seed style profile the writer
    subagent reads. Mirrors ``STYLE_PROFILE_FILENAME`` in tools/style.py."""
    return Path(data_dir) / STYLE_PROFILE_FILENAME


def brief_path(topic_id: str, data_dir: str | Path) -> Path:
    """``<data_dir>/briefs/<topic_id>.json`` — the authored ResearchBrief the
    research subagent writes with ``write_file`` and verify_claim reads via
    stdlib. Path-traversal-guarded at the topic_id."""
    return Path(data_dir) / BRIEFS_DIRNAME / f"{_validate_slug(topic_id, kind='topic_id')}.json"


def verified_brief_path(topic_id: str, data_dir: str | Path) -> Path:
    """``<data_dir>/briefs/<topic_id>.verified.json`` — the post-verification
    copy verify_claim writes and the writer subagent reads. Same path-traversal
    guard as brief_path."""
    return (
        Path(data_dir)
        / BRIEFS_DIRNAME
        / f"{_validate_slug(topic_id, kind='topic_id')}.verified.json"
    )


def draft_path(post_id: str, data_dir: str | Path) -> Path:
    """``<data_dir>/drafts/<post_id>.json`` — the drafted PostProposal the
    writer subagent writes and quality_gate reads. Path-traversal-guarded at
    the post_id."""
    return Path(data_dir) / DRAFTS_DIRNAME / f"{_validate_slug(post_id, kind='post_id')}.json"


def gate_path(post_id: str, data_dir: str | Path) -> Path:
    """``<data_dir>/drafts/<post_id>.gate.json`` — the quality_gate verdict the
    writer subagent branches on (and the coordinator surfaces). Same
    path-traversal guard as draft_path."""
    return (
        Path(data_dir)
        / DRAFTS_DIRNAME
        / f"{_validate_slug(post_id, kind='post_id')}.gate.json"
    )


# --------------------------------------------------------------------------- #
# Backend
# --------------------------------------------------------------------------- #


def build_orchestrator_backend(settings: Settings | None = None) -> FilesystemBackend:
    """Construct the ``FilesystemBackend`` the coordinator mounts.

    Rooted at ``settings.orchestrator_data_dir`` so the deepagents built-in
    file tools (``write_file`` / ``read_file`` / ``edit_file``) operate on the
    *same* files the deterministic custom tools (fetch_curated_ai_news,
    verify_claim, quality_gate, ...) access via stdlib. ``virtual_mode=False``
    is intentional and load-bearing: the writer subagent passes an *absolute*
    ``skills_source`` path to SkillsMiddleware (resolved from
    ``settings.skills_dir`` so the ``linkedin-voice`` skill loads from the
    real repo regardless of the data-dir root). Under ``virtual_mode=True``
    absolute paths are blocked; under ``virtual_mode=False`` they pass through
    as-is, which is the contract P4.2's writer subagent already relies on.

    Creates the root dir if missing (mirroring the tools' own ``mkdir(parents=
    True, exist_ok=True)`` convention) — the coordinator's first action is
    typically ``write_todos`` + a ``fetch_curated_ai_news`` call, and the dir
    has to exist before ``write_todos`` lands its first file.

    Settings resolve lazily when not supplied, mirroring the tool factories.
    """
    s = settings or get_settings()
    root = Path(s.orchestrator_data_dir)
    root.mkdir(parents=True, exist_ok=True)
    return FilesystemBackend(root_dir=str(root), virtual_mode=False)