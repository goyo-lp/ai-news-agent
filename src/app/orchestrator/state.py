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
   ``gate_path``, ``style_profile_path``) — the single source of truth for
   *where on the filesystem* structured data lives (guiding principle #3).
   Every tool builds its paths through these; none carries its own copy of a
   filename string.
3. ``validate_slug`` — the one path-traversal guard, used by every ``*_path``
   builder that takes an externally supplied id (topic_id / post_id), so a new
   builder cannot forget it.
4. ``read_brief`` — resolve-and-parse for a topic's brief, preferring the
   verified copy. Three tools (submit_draft, quality_gate, deliver_telegram)
   need it; keeping it here means the "verified wins, unverified is the
   fallback" rule lives with the convention that defines both paths.

Scope-out (honest): the fetch_article tool writes its per-URL artifact to
``<data_dir>/articles/<slug>.json`` (``fetch._ARTICLES_SUBDIR = "articles"``)
— a distinct subdir from ``articles.json`` (the curated snapshot). That subdir
is part of the convention but has no helper here yet; adopting it lives with a
future refactor of ``fetch.py``. Documented so the omission isn't inferred."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from deepagents.backends.filesystem import FilesystemBackend

from app.config import Settings, get_settings

if TYPE_CHECKING:
    from app.orchestrator.schemas import ResearchBrief

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Filename / subdir convention (single source of truth)
# --------------------------------------------------------------------------- #
# The literal filenames the deterministic tools read and write. Defined once
# here and referenced everywhere else, so a rename is a one-line change.

ARTICLES_FILENAME = "articles.json"
TOPICS_FILENAME = "topics.json"
STYLE_PROFILE_FILENAME = "style_profile.json"

BRIEFS_DIRNAME = "briefs"
DRAFTS_DIRNAME = "drafts"


def validate_slug(slug: str, *, kind: str) -> str:
    """Reject any externally supplied id that could escape its dir.

    Deliberately does no normalization and no whitespace stripping: a slug is
    used verbatim to build a filename, so every caller must see the same file
    for the same input. Stripping here would make a ``" foo "`` id resolve to
    ``foo.json`` for one caller and ``" foo .json"`` for another, breaking the
    cross-tool visibility contract that the shared data dir depends on.
    """
    if not slug or slug in {"", ".", ".."}:
        raise ValueError(f"{kind} slug is empty or a path metacharacter: {slug!r}")
    if "/" in slug:
        raise ValueError(f"{kind} slug must not contain '/': {slug!r}")
    # ``..`` as a *component* (e.g. ``foo/..``) is caught by the ``/`` rule
    # above; ``..`` as a *bare slug* is caught by the explicit set.
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
    """``<data_dir>/style_profile.json`` — the style profile the writer subagent
    reads."""
    return Path(data_dir) / STYLE_PROFILE_FILENAME


def brief_path(topic_id: str, data_dir: str | Path) -> Path:
    """``<data_dir>/briefs/<topic_id>.json`` — the authored ResearchBrief the
    research subagent writes with ``write_file`` and verify_claim reads via
    stdlib. Path-traversal-guarded at the topic_id."""
    return Path(data_dir) / BRIEFS_DIRNAME / f"{validate_slug(topic_id, kind='topic_id')}.json"


def verified_brief_path(topic_id: str, data_dir: str | Path) -> Path:
    """``<data_dir>/briefs/<topic_id>.verified.json`` — the post-verification
    copy verify_claim writes and the writer subagent reads. Same path-traversal
    guard as brief_path."""
    return (
        Path(data_dir)
        / BRIEFS_DIRNAME
        / f"{validate_slug(topic_id, kind='topic_id')}.verified.json"
    )


def draft_path(post_id: str, data_dir: str | Path) -> Path:
    """``<data_dir>/drafts/<post_id>.json`` — the drafted PostProposal the
    writer subagent writes and quality_gate reads. Path-traversal-guarded at
    the post_id."""
    return Path(data_dir) / DRAFTS_DIRNAME / f"{validate_slug(post_id, kind='post_id')}.json"


def gate_path(post_id: str, data_dir: str | Path) -> Path:
    """``<data_dir>/drafts/<post_id>.gate.json`` — the quality_gate verdict the
    writer subagent branches on (and the coordinator surfaces). Same
    path-traversal guard as draft_path."""
    return (
        Path(data_dir)
        / DRAFTS_DIRNAME
        / f"{validate_slug(post_id, kind='post_id')}.gate.json"
    )


def read_brief(topic_id: str, data_dir: str | Path) -> "ResearchBrief | None":
    """Load the brief behind a topic, preferring the verified copy.

    The unverified copy is the deliberate fallback rather than a miss, so a
    caller refusing the brief can name the actual status (``unverified``)
    instead of reporting a bare "missing". Returns ``None`` when neither file
    exists, when the id is invalid, or when the payload doesn't parse — every
    caller treats all three as "no usable evidence" and fails closed.
    """
    from app.orchestrator.schemas import ResearchBrief

    try:
        candidates = (
            verified_brief_path(topic_id, data_dir),
            brief_path(topic_id, data_dir),
        )
    except ValueError:
        return None
    for path in candidates:
        if path.exists():
            try:
                return ResearchBrief.model_validate_json(path.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.warning("brief load failed at %s: %s", path, exc)
                return None
    return None


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

    Path semantics (resolved in P8.3): deepagents' built-in ``write_file``
    middleware calls ``validate_path``, which normalizes a relative path like
    ``briefs/<topic>.json`` to absolute-with-leading-slash (``/briefs/…``) —
    which, under ``virtual_mode=False``, resolves at the OS root and fails on a
    read-only filesystem. P8.3 took fix (b): the P4 research/writer subagent
    prompts now interpolate ``orchestrator_data_dir`` (absolute) so LLM-driven
    ``write_file`` / ``read_file`` calls emit absolute paths under the mount —
    matching the explicit-absolute-path approach the P5.3 e2e dry run already
    proved works through this backend. ``virtual_mode`` stays ``False`` so the
    writer's absolute ``skills_source`` still resolves (see above).
    """
    s = settings or get_settings()
    root = Path(s.orchestrator_data_dir)
    root.mkdir(parents=True, exist_ok=True)
    return FilesystemBackend(root_dir=str(root), virtual_mode=False)