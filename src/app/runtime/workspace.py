"""Per-run workspace preparation under ``orchestrator_data_dir``.

A propose run regenerates its artifacts in place, and ``drafts/`` is a
persistent directory — so without run-scoping, a new run's delivery loop would
re-send the previous run's proposals as duplicate posts. These helpers give the
run a clean workspace and guarantee the writer's voice input exists before any
writer subagent starts.
"""
from __future__ import annotations

import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path

from app.config import Settings
from app.orchestrator import state
from app.orchestrator.tools.style import resolve_style_profile

logger = logging.getLogger(__name__)


def archive_previous_run(settings: Settings) -> Path | None:
    """Move the previous run's regenerated artifacts out of the way so this run
    — and its delivery — only ever sees its own output.

    Archive, not delete (2026-07-26 hardening): an earlier ``_clear`` version
    wiped the workspace *before* the run, so a crashed run left neither an
    export nor the prior state to inspect — trace review of exactly such a crash
    is what motivated this change. Artifacts move to
    ``<data_dir>/.archive/<utc-timestamp>/`` (cheap same-filesystem rename).
    Inputs that live outside this dir (style seed, skills, delivery history) are
    untouched; the generated ``style_profile.json`` inside it deliberately
    stays. Returns the archive dir when anything was moved."""
    data_dir = Path(settings.orchestrator_data_dir)
    targets = [
        data_dir / state.ARTICLES_FILENAME,
        data_dir / state.TOPICS_FILENAME,
        data_dir / state.BRIEFS_DIRNAME,
        data_dir / state.DRAFTS_DIRNAME,
        data_dir / "web",
    ]
    existing = [t for t in targets if t.exists()]
    if not existing:
        return None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    archive_dir = data_dir / ".archive" / stamp
    archive_dir.mkdir(parents=True, exist_ok=False)
    for target in existing:
        shutil.move(str(target), archive_dir / target.name)
    logger.info("Archived prior run workspace to %s", archive_dir)
    return archive_dir


async def ensure_style_profile(settings: Settings) -> None:
    """Guarantee ``style_profile.json`` exists in the data dir before the writer
    runs. The load_style_profile tool used to be orphaned — nothing in any live
    path called it, so the writer was told to read a file that never existed.
    Now the run resolves it up front (seed profile, else the built-in default;
    ``rebuild`` stays an offline operator action)."""
    result = await resolve_style_profile(rebuild=False, settings=settings)
    if result.get("status") == "ok":
        logger.info("Style profile ready (source=%s)", result.get("source"))
    else:
        logger.warning("Style profile resolution failed: %s", result.get("reason"))


__all__ = ["archive_previous_run", "ensure_style_profile"]
