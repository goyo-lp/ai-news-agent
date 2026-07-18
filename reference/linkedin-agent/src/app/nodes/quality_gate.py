from __future__ import annotations

import logging
from difflib import SequenceMatcher

from app.graph.state import AgentState
from app.schemas import LinkedInPost, ResearchBrief, parse_linkedin_posts, parse_research_briefs, serialize_models
from app.services.tracing import traceable

logger = logging.getLogger(__name__)

_TARGET_POSTS = 5


@traceable(name="quality_gate_node")
async def quality_gate_node(state: AgentState) -> AgentState:
    posts = parse_linkedin_posts(state.get("generated_posts") or state.get("linkedin_posts"))
    briefs = parse_research_briefs(state.get("research_briefs"))

    checks: list[str] = []
    target_count = min(_TARGET_POSTS, max(len(briefs), 1))

    if len(posts) < target_count and briefs:
        checks.append(f"Post count below {target_count}; padding with deterministic reflective variants.")
        while len(posts) < target_count:
            source_brief = briefs[min(len(posts), len(briefs) - 1)]
            posts.append(_fallback_post_from_brief(source_brief, idx=len(posts) + 1))

    if len(posts) > target_count:
        checks.append(f"More than {target_count} posts generated; truncating.")
        posts = posts[:target_count]

    for idx in range(len(posts)):
        for jdx in range(idx + 1, len(posts)):
            similarity = SequenceMatcher(None, posts[idx].body, posts[jdx].body).ratio()
            if similarity > 0.9:
                checks.append(
                    f"High similarity between post-{idx + 1} and post-{jdx + 1}; forcing differentiation sentence."
                )
                posts[jdx].body += " Additional reflection: implementation tradeoffs matter more than headline claims."

    valid_topic_ids = {brief.topic_id for brief in briefs}
    for idx, post in enumerate(posts, start=1):
        post.angle = "technical_reflection"

        if len(post.hashtags) < 3:
            checks.append(f"post-{idx} had too few hashtags; backfilling defaults.")
            post.hashtags = list(dict.fromkeys(post.hashtags + ["#AI", "#AIAgents", "#MachineLearning"]))[:5]
        if len(post.hashtags) > 5:
            checks.append(f"post-{idx} had too many hashtags; trimming to 5.")
            post.hashtags = post.hashtags[:5]

        if not post.supporting_topic_ids and briefs:
            checks.append(f"post-{idx} missing topic mapping; backfilling.")
            post.supporting_topic_ids = [briefs[min(idx - 1, len(briefs) - 1)].topic_id]

        post.supporting_topic_ids = [
            topic_id for topic_id in post.supporting_topic_ids if topic_id in valid_topic_ids
        ][:1]
        if not post.supporting_topic_ids and briefs:
            post.supporting_topic_ids = [briefs[min(idx - 1, len(briefs) - 1)].topic_id]

        if not post.citation_urls and briefs:
            checks.append(f"post-{idx} missing citations; backfilling from brief.")
            source_brief = briefs[min(idx - 1, len(briefs) - 1)]
            post.citation_urls = [citation.url for citation in source_brief.citations[:2]]

    logger.info("Quality gate complete: %s checks", len(checks))
    return {
        "linkedin_posts": serialize_models(posts),
        "quality_checks": checks,
    }


def _fallback_post_from_brief(brief: ResearchBrief, idx: int) -> LinkedInPost:
    cautious_prefix = ""
    if brief.verification_status != "verified":
        cautious_prefix = "Evidence is still emerging, so I'm treating this carefully. "

    return LinkedInPost(
        post_id=f"post-{idx}",
        angle="technical_reflection",
        headline=f"A technical note on: {brief.headline}",
        body=(
            f"{cautious_prefix}{brief.summary} "
            f"What I find most concrete here is {brief.technical_significance.lower()}"
        ),
        hashtags=["#AI", "#AIAgents", "#MachineLearning", "#EnterpriseAI"],
        supporting_topic_ids=[brief.topic_id],
        citation_urls=[citation.url for citation in brief.citations[:2]],
        confidence=0.5,
    )
