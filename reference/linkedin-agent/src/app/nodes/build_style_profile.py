from __future__ import annotations

import logging
from pathlib import Path
from app.config import get_settings
from app.graph.state import AgentState
from app.services.style_profile import StyleProfiler
from app.services.tracing import traceable

logger = logging.getLogger(__name__)


@traceable(name="build_style_profile_node")
async def build_style_profile_node(state: AgentState) -> AgentState:
    settings = get_settings()

    override = str(state.get("style_samples_dir_override") or "").strip()
    samples_path = Path(override) if override else settings.style_samples_path

    profiler = StyleProfiler(settings)
    profile = profiler.build_from_directory(samples_path)
    saved_path = profiler.save_profile(profile)

    style_errors: list[str] = []
    if profile.sample_count == 0:
        style_errors.append(f"No style samples found in '{samples_path}'. Using default style profile.")

    logger.info(
        "Style profile ready: samples=%s, saved=%s",
        profile.sample_count,
        saved_path,
    )
    return {
        "style_profile": profile.model_dump(mode="json"),
        "style_profile_errors": style_errors,
    }
