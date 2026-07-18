from __future__ import annotations

from typing import Any

from langgraph.graph import END, StateGraph

from app.config import get_settings
from app.graph.state import AgentState
from app.nodes.adaptive_investigation import adaptive_investigation_node
from app.nodes.build_style_profile import build_style_profile_node
from app.nodes.deep_research_top5 import deep_research_top5_node
from app.nodes.deliver_telegram import deliver_telegram_node
from app.nodes.discover_news import discover_news_node
from app.nodes.export_artifacts import export_artifacts_node
from app.nodes.export_report import export_report_node
from app.nodes.generate_posts import generate_posts_node
from app.nodes.merge_briefs import merge_briefs_node
from app.nodes.normalize_and_dedupe import normalize_and_dedupe_node
from app.nodes.quality_gate import quality_gate_node
from app.nodes.rank_and_cluster import rank_and_cluster_node
from app.nodes.verify_briefs import verify_briefs_node
from app.services.langgraphics_assets import ensure_langgraphics_static_assets


def build_workflow(enable_graphics: bool | None = None) -> Any:
    settings = get_settings()
    graph = StateGraph(AgentState)

    graph.add_node("discover_news", discover_news_node)
    graph.add_node("normalize_and_dedupe", normalize_and_dedupe_node)
    graph.add_node("rank_and_cluster", rank_and_cluster_node)
    graph.add_node("deep_research_top5", deep_research_top5_node)
    graph.add_node("adaptive_investigation", adaptive_investigation_node)
    graph.add_node("verify_briefs", verify_briefs_node)
    graph.add_node("merge_briefs", merge_briefs_node)
    graph.add_node("build_style_profile", build_style_profile_node)
    graph.add_node("generate_posts", generate_posts_node)
    graph.add_node("quality_gate", quality_gate_node)
    graph.add_node("export_artifacts", export_artifacts_node)
    graph.add_node("deliver_telegram", deliver_telegram_node)
    graph.add_node("export_report", export_report_node)

    graph.set_entry_point("discover_news")
    graph.add_edge("discover_news", "normalize_and_dedupe")
    graph.add_edge("normalize_and_dedupe", "rank_and_cluster")
    graph.add_edge("rank_and_cluster", "deep_research_top5")
    graph.add_edge("deep_research_top5", "build_style_profile")
    graph.add_edge("deep_research_top5", "adaptive_investigation")
    graph.add_edge("deep_research_top5", "verify_briefs")
    graph.add_edge(["adaptive_investigation", "verify_briefs"], "merge_briefs")
    graph.add_edge(["merge_briefs", "build_style_profile"], "generate_posts")
    graph.add_edge("generate_posts", "quality_gate")
    graph.add_edge("quality_gate", "deliver_telegram")
    graph.add_edge("quality_gate", "export_artifacts")
    graph.add_edge(["deliver_telegram", "export_artifacts"], "export_report")
    graph.add_edge("export_report", END)

    compiled = graph.compile()

    graphics_enabled = settings.langgraphics_enabled if enable_graphics is None else enable_graphics
    if graphics_enabled:
        from langgraphics import watch  # type: ignore[import-untyped]

        ensure_langgraphics_static_assets()
        return watch(
            compiled,
            host=settings.langgraphics_host,
            port=settings.langgraphics_port,
            ws_port=settings.langgraphics_ws_port,
            open_browser=settings.langgraphics_open_browser,
            direction=settings.langgraphics_direction,  # type: ignore[arg-type]
            mode=settings.langgraphics_mode,  # type: ignore[arg-type]
            inspect=settings.langgraphics_inspect,  # type: ignore[arg-type]
            theme=settings.langgraphics_theme,  # type: ignore[arg-type]
        )

    return compiled
