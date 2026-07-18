from __future__ import annotations

import logging
from typing import cast

from app.config import get_settings
from app.graph.state import AgentState
from app.schemas import parse_discovered, serialize_models
from app.services.scoring import rank_topics
from app.services.source_policy import SourcePolicy
from app.services.technical_ranker import TechnicalRanker
from app.services.tracing import traceable

logger = logging.getLogger(__name__)


@traceable(name="rank_and_cluster_node")
async def rank_and_cluster_node(state: AgentState) -> AgentState:
    settings = get_settings()

    normalized_items = parse_discovered(state.get("normalized_items"))
    max_topics = int(state.get("max_topics", settings.max_topics_per_run))
    max_topics = max(1, min(max_topics, settings.max_topics_per_run))
    dry_run = bool(state.get("dry_run", False))

    policy = SourcePolicy.from_file(settings.trusted_sources_path)

    ranker = TechnicalRanker(settings)
    assessments = await ranker.assess_many(normalized_items, dry_run=dry_run)
    technical_overrides = {item_id: value.technical_depth for item_id, value in assessments.items()}
    implementation_overrides = {
        item_id: value.implementation_specificity for item_id, value in assessments.items()
    }
    hype_overrides = {item_id: value.hype_score for item_id, value in assessments.items()}

    ranked_topics = rank_topics(
        normalized_items,
        limit=max_topics,
        policy=policy,
        technical_overrides=technical_overrides,
        implementation_overrides=implementation_overrides,
        hype_overrides=hype_overrides,
    )

    next_state = cast(AgentState, dict(state))
    next_state["ranked_topics"] = serialize_models(ranked_topics)

    logger.info("Ranking complete: selected %s technical topics", len(ranked_topics))
    return next_state
