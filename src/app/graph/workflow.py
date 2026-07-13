from __future__ import annotations

from langgraph.graph import END, StateGraph

from app.graph.state import AgentState
from app.nodes.deliver import deliver_node
from app.nodes.enrich import enrich_node
from app.nodes.ingest import ingest_node
from app.nodes.rank import rank_node
from app.nodes.summarize import summarize_node


def route_after_rank(state: AgentState) -> str:
    """Skip summarize/deliver when nothing survived the rank filters."""
    return "summarize" if state.get("articles_selected") else END


def build_workflow():
    graph = StateGraph(AgentState)

    graph.add_node("ingest", ingest_node)
    graph.add_node("enrich", enrich_node)
    graph.add_node("rank", rank_node)
    graph.add_node("summarize", summarize_node)
    graph.add_node("deliver", deliver_node)

    graph.set_entry_point("ingest")
    graph.add_edge("ingest", "enrich")
    graph.add_edge("enrich", "rank")
    graph.add_conditional_edges("rank", route_after_rank, {"summarize": "summarize", END: END})
    graph.add_edge("summarize", "deliver")
    graph.add_edge("deliver", END)

    return graph.compile()
