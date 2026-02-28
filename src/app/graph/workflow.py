from __future__ import annotations

from langgraph.graph import END, StateGraph

from app.config import get_settings
from app.graph.state import AgentState
from app.nodes.deliver import deliver_node
from app.nodes.enrich import enrich_node
from app.nodes.ingest import ingest_node
from app.nodes.rank import rank_node
from app.nodes.summarize import summarize_node
from app.services.langgraphics_assets import ensure_langgraphics_static_assets


def build_workflow():
    settings = get_settings()
    graph = StateGraph(AgentState)

    graph.add_node("ingest", ingest_node)
    graph.add_node("enrich", enrich_node)
    graph.add_node("rank", rank_node)
    graph.add_node("summarize", summarize_node)
    graph.add_node("deliver", deliver_node)

    graph.set_entry_point("ingest")
    graph.add_edge("ingest", "enrich")
    graph.add_edge("enrich", "rank")
    graph.add_edge("rank", "summarize")
    graph.add_edge("summarize", "deliver")
    graph.add_edge("deliver", END)

    compiled = graph.compile()

    if settings.langgraphics_enabled:
        from langgraphics import watch

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
