from __future__ import annotations

import logging
from app.config import get_settings
from app.graph.state import AgentState
from app.schemas import parse_research_briefs, serialize_models
from app.services.brief_verifier import BriefVerifier
from app.services.tracing import traceable

logger = logging.getLogger(__name__)


@traceable(name="verify_briefs_node")
async def verify_briefs_node(state: AgentState) -> AgentState:
    settings = get_settings()
    dry_run = bool(state.get("dry_run", False))
    hours_back = int(state.get("hours_back", settings.discovery_hours_back))

    briefs = parse_research_briefs(state.get("deep_research_briefs") or state.get("research_briefs"))
    verifier = BriefVerifier(settings)
    verified_briefs, errors = await verifier.verify_briefs(
        briefs=briefs,
        hours_back=hours_back,
        dry_run=dry_run,
    )

    logger.info("Verification complete: %s briefs", len(verified_briefs))
    return {
        "verified_briefs": serialize_models(verified_briefs),
        "verify_errors": errors,
    }
