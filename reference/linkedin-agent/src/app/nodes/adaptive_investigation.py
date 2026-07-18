from __future__ import annotations

import logging
from app.config import get_settings
from app.graph.state import AgentState
from app.schemas import parse_research_briefs, serialize_models
from app.services.deep_agent_investigator import DeepAgentInvestigator
from app.services.tracing import traceable

logger = logging.getLogger(__name__)


@traceable(name="adaptive_investigation_node")
async def adaptive_investigation_node(state: AgentState) -> AgentState:
    settings = get_settings()
    dry_run = bool(state.get("dry_run", False))
    hours_back = int(state.get("hours_back", settings.discovery_hours_back))

    briefs = parse_research_briefs(state.get("deep_research_briefs") or state.get("research_briefs"))
    investigator = DeepAgentInvestigator(settings)
    investigated_briefs, errors = await investigator.investigate_briefs(
        briefs=briefs,
        hours_back=hours_back,
        dry_run=dry_run,
    )

    serialized = serialize_models(investigated_briefs)

    logger.info("Adaptive investigation complete: %s briefs", len(investigated_briefs))
    return {
        "adaptive_briefs": serialized,
        "adaptive_errors": errors,
    }
