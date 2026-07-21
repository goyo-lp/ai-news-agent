"""load_style_profile — make the author's voice fingerprint available to the
writer subagent (P4.2).

The writer reads two voice inputs: the ``linkedin-voice`` skill (P3.1, static
instructions) and a :class:`StyleProfile` (data-derived: sentence length,
common openers, vocabulary, tone). This tool resolves the profile and writes
it to the orchestrator's data dir as ``style_profile.json`` — the single state
artifact the writer reads — then returns a compressed summary.

Two paths, chosen by the ``rebuild`` arg:

  * **load** (default): read the seed at ``settings.style_profile_file``; if it
    is absent or unparseable, fall back to a built-in default profile. Cheap,
    no sample scanning — the common per-run path.
  * **rebuild**: re-derive the profile from the writing samples under
    ``settings.style_samples_dir`` (instead of trusting the seed), then write
    it to state. Use when the samples change or the seed is stale.

The seed at ``settings.style_profile_file`` is read-only *input*: neither path
mutates it. That keeps a runtime tool call from editing a git-tracked file
(surprising, and a hard failure under a read-only deploy) — regenerating the
committed seed is a deliberate offline step, not an agent side effect. The one
output is the state artifact under ``orchestrator_data_dir``.

Per guiding principle #3 the profile is the artifact on disk; the tool returns
only counts + tone traits (the small, useful bits), never the full vocabulary /
openers lists. Path resolution (Settings -> Path) lives here so the
style_profile service stays Settings-free.

Target environment: deepagents ``create_deep_agent`` — async-only tool, same
sibling pattern as news / quality / verify_claim.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from app.config import Settings, get_settings
from app.orchestrator.schemas import StyleProfile
from app.orchestrator.services import style_profile as style_service

logger = logging.getLogger(__name__)

STYLE_PROFILE_FILENAME = "style_profile.json"


class LoadStyleProfileArgs(BaseModel):
    """Tool input. ``rebuild`` re-derives the profile from the writing samples
    and persists it back to the seed file; the default (load) path just reads
    the existing seed."""

    rebuild: bool = Field(
        default=False,
        description=(
            "When true, re-derive the style profile from the writing samples "
            "dir and persist it to the seed file. When false (default), load "
            "the existing seed profile, falling back to a default if absent."
        ),
    )


def write_profile_to_state(profile: StyleProfile, data_dir: str) -> Path:
    """Serialize the resolved profile to ``<data_dir>/style_profile.json`` — the
    state artifact the writer reads — and return the path. Delegates the
    serialization to the service's canonical ``save_profile`` so the
    StyleProfile->JSON encoding lives in exactly one place; the tool owns only
    the path convention + log. Mirrors the news.py / quality.py state writers."""
    path = style_service.save_profile(
        profile, Path(data_dir) / STYLE_PROFILE_FILENAME
    )
    logger.info(
        "Wrote style profile (samples=%d) to %s", profile.sample_count, path
    )
    return path


def _resolve_profile(rebuild: bool, settings: Settings) -> tuple[StyleProfile, str]:
    """Resolve the profile and its source label ("rebuilt" | "loaded" |
    "default") without touching state. The seed file is read-only input — this
    never writes it; the single output is the state artifact, written by the
    caller."""
    if rebuild:
        profile = style_service.build_from_directory(Path(settings.style_samples_dir))
        return profile, "rebuilt"

    loaded = style_service.load_saved_profile(Path(settings.style_profile_file))
    if loaded is not None:
        return loaded, "loaded"
    return style_service.default_profile(), "default"


async def _load_and_write(rebuild: bool, settings: Settings) -> dict[str, Any]:
    """Resolve the profile, persist it to state, and return a compressed
    summary. Any failure surfaces as ``status=error`` so the LLM sees a
    structured result, not a raise into the agent loop."""
    failure: dict[str, Any] = {
        "source": "rebuilt" if rebuild else "loaded",
        "sample_count": 0,
        "sentence_count": 0,
        "vocabulary_size": 0,
        "tone_traits": [],
        "status": "error",
        "reason": "",
        "path": None,
    }
    try:
        profile, source = _resolve_profile(rebuild, settings)
        path = write_profile_to_state(profile, settings.orchestrator_data_dir)
    except Exception as exc:
        logger.warning("load_style_profile failed (rebuild=%s): %s", rebuild, exc)
        failure["reason"] = f"{type(exc).__name__}: {exc}"
        return failure

    return {
        "source": source,
        "sample_count": profile.sample_count,
        "sentence_count": profile.sentence_count,
        "vocabulary_size": len(profile.vocabulary),
        "tone_traits": profile.tone_traits,
        "status": "ok",
        "reason": None,
        "path": str(path),
    }


def build_load_style_profile_tool(settings: Settings | None = None) -> StructuredTool:
    """Construct the load_style_profile LangChain tool. Settings resolve lazily
    on first call when not supplied (picks up .env loaded after build);
    supplying them explicitly is the seam tests use."""
    bound_settings = settings

    async def _async(rebuild: bool = False) -> str:
        s = bound_settings or get_settings()
        result = await _load_and_write(rebuild, s)
        return json.dumps(result, default=str)

    return StructuredTool.from_function(
        func=None,
        coroutine=_async,
        name="load_style_profile",
        description=(
            "Resolve the author's writing-voice StyleProfile and write it to "
            "the orchestrator data dir as style_profile.json for the writer "
            "subagent. Default (rebuild=false) loads the existing seed profile, "
            "falling back to a built-in default if none exists. rebuild=true "
            "re-derives it from the writing-samples dir instead of the seed. "
            "The seed file is read-only; the profile is written to state either "
            "way. "
            "Returns a JSON summary {source, sample_count, sentence_count, "
            "vocabulary_size, tone_traits, status, reason, path} — the full "
            "vocabulary/openers live in the file at `path`. `source` is "
            "`loaded`, `rebuilt`, or `default`; `status` is `ok` or `error`."
        ),
        args_schema=LoadStyleProfileArgs,
    )


load_style_profile_tool = build_load_style_profile_tool()
