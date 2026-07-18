from __future__ import annotations

import logging
from typing import cast

from app.graph.state import AgentState
from app.schemas import parse_discovered, serialize_models
from app.services.tracing import traceable
from app.services.url_utils import dedupe_items

logger = logging.getLogger(__name__)


@traceable(name="normalize_and_dedupe_node")
async def normalize_and_dedupe_node(state: AgentState) -> AgentState:
    discovered = parse_discovered(state.get("discovered_items"))
    normalized = dedupe_items(discovered)

    next_state = cast(AgentState, dict(state))
    next_state["normalized_items"] = serialize_models(normalized)

    logger.info(
        "Normalization complete: %s -> %s items",
        len(discovered),
        len(normalized),
    )
    return next_state
