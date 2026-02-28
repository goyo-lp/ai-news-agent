from __future__ import annotations

import logging

from app.config import get_settings
from app.graph.state import AgentState
from app.schemas.article import FetchRules, SourceConfig, parse_articles, serialize_articles
from app.services.extractor import OpenGraphExtractor
from app.services.tracing import traceable

logger = logging.getLogger(__name__)


@traceable(name="enrich_node")
async def enrich_node(state: AgentState) -> AgentState:
    settings = get_settings()

    raw_articles = parse_articles(state.get("articles_raw"))
    source_configs = [SourceConfig.model_validate(item) for item in state.get("sources", [])]
    defaults = FetchRules.model_validate(state.get("fetch_defaults", {}))

    source_rules = {
        source.name: source.merged_rules(defaults)
        for source in source_configs
    }

    extractor = OpenGraphExtractor(settings)
    enriched, errors = await extractor.enrich_articles(raw_articles, source_rules)

    next_state: AgentState = dict(state)
    next_state["articles_enriched"] = serialize_articles(enriched)

    existing_errors = list(next_state.get("errors", []))
    existing_errors.extend(errors)
    next_state["errors"] = existing_errors

    logger.info("Enrichment complete: %s items", len(enriched))
    return next_state
