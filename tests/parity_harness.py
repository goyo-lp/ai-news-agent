"""Parity harness for the ported deterministic logic — Phase 8.2.

The migration is a fresh rebuild alongside a retained news pipeline, not an
in-place graph swap, so there is no runnable "old" pipeline in-host to diff
against (the vendored ``reference/`` is a services-only source copy, still
Tavily-coupled). Parity is therefore a **golden characterization** of the two
deterministic ported surfaces — the quality gate and the technical ranker's
hype-demotion — locking their behavior on a shared fixture set so a future
change that regresses ranking order or the gate verdict surfaces as a diff.

``build_parity_report()`` is pure + deterministic and is the single source both
``tests/test_parity.py`` (asserts it against the committed golden) and
``scripts/parity.py`` (prints it for a human to eyeball before cutover) consume.

Determinism note: the ranker's absolute scores bucket recency by wall-clock,
but every fixture here has ``published_at=None``, which short-circuits recency
to a constant — so the exercised path is wall-clock-*independent*. The harness
captures the ranked *order* + which items are hype-demoted (never the raw float
scores). If a fixture is ever given a real ``published_at``, that constant no
longer holds and order can drift across a recency bucket boundary.
"""
from __future__ import annotations

import asyncio
from dataclasses import asdict
from typing import Any

from app.config import Settings
from app.orchestrator.schemas import CuratedArticle, PostProposal
from app.orchestrator.services.quality_gate import QualityResult, check_proposal
from app.orchestrator.services.ranking import (
    SourcePolicy,
    curated_articles_to_items,
    rank_topics,
)
from app.orchestrator.services.technical_ranker import TechnicalRanker
from support import CLEAN_GATE_BODY


def _proposal(
    *,
    post_id: str,
    body: str = CLEAN_GATE_BODY,
    hashtags: list[str] | None = None,
    supporting_topic_ids: list[str] | None = None,
) -> PostProposal:
    return PostProposal(
        post_id=post_id,
        angle="steady improvement",
        headline="Stable Diffusion 3.5's quiet attention refactor",
        body=body,
        hashtags=hashtags if hashtags is not None else ["#stablediffusion", "#attention", "#inference"],
        supporting_topic_ids=supporting_topic_ids if supporting_topic_ids is not None else ["topic-a"],
        citation_urls=["https://example.com/topic-a"],
        confidence=0.8,
    )


# Gate fixtures — a pass plus one per failure branch of check_proposal.
_GATE_FIXTURES: dict[str, PostProposal] = {
    "clean_pass": _proposal(post_id="clean_pass"),
    "too_short": _proposal(post_id="too_short", body="A short body that is well under the floor."),
    "too_long": _proposal(post_id="too_long", body=" ".join(["word"] * 400)),
    "hype_present": _proposal(post_id="hype_present", body=CLEAN_GATE_BODY + " This is revolutionary."),
    "multi_topic": _proposal(post_id="multi_topic", supporting_topic_ids=["topic-a", "topic-b"]),
    "too_few_hashtags": _proposal(post_id="too_few_hashtags", hashtags=["#one"]),
    "no_hashtags": _proposal(post_id="no_hashtags", hashtags=[]),
}


def _article(*, id: str, title: str, summary: str) -> CuratedArticle:
    """A CuratedArticle with uniform corroboration/recency inputs so ranking
    order is driven by the technical-vs-hype signal, not by cluster size or
    publish time."""
    return CuratedArticle(
        id=id,
        source_name="Test Source",
        title=title,
        url=f"https://example.com/{id}",
        summary=summary,
        score=0.5,
        duplicate_count=1,
        cluster_size=1,
    )


# Ranking fixtures — a technical article, a hype-only article, and a
# business-only (funding) article. The ported ranker's whole job is to float
# the technical story above the hype/business ones ("demote hype", plan P2.1).
_RANK_FIXTURES: list[CuratedArticle] = [
    _article(
        id="technical",
        title="New attention kernel cuts transformer inference latency",
        summary=(
            "A benchmark-backed implementation change to the attention kernel "
            "reduces latency and memory on-device with reproducible throughput "
            "numbers."
        ),
    ),
    _article(
        id="hype",
        title="This revolutionary AI will change everything, experts say",
        summary="A game-changing breakthrough that is a paradigm shift for the industry.",
    ),
    _article(
        id="funding",
        title="AI startup raises $200M Series C at a $2B valuation",
        summary="The funding round values the company at two billion dollars in new capital.",
    ),
]


def _gate_result_to_dict(result: QualityResult) -> dict[str, Any]:
    """The verdict-relevant fields of a gate result — the cleaned_body is
    derived and long, so it is deliberately excluded from the parity golden."""
    payload = asdict(result)
    payload.pop("cleaned_body", None)
    payload["hype_markers"] = sorted(payload["hype_markers"])
    return payload


def _run_ranking() -> dict[str, Any]:
    """Rank the fixtures via the deterministic heuristic path (no OpenRouter)
    and return the wall-clock-stable outcome: the ranked topic order + the
    hype-demotion assertion."""
    settings = Settings(_env_file=None, openrouter_api_key="")
    policy = SourcePolicy.permissive()
    items = curated_articles_to_items(_RANK_FIXTURES, policy)

    ranker = TechnicalRanker(settings)
    assessments = asyncio.run(ranker.assess_many(items, dry_run=True))
    topics = rank_topics(
        items,
        limit=len(_RANK_FIXTURES),
        policy=policy,
        technical_overrides={k: v.technical_depth for k, v in assessments.items()},
        implementation_overrides={k: v.implementation_specificity for k, v in assessments.items()},
        hype_overrides={k: v.hype_score for k, v in assessments.items()},
    )
    order = [t.topic_id for t in topics]
    # Every fixture is permissive-policy-allowed and limit covers all of them,
    # so all three ids are always present in `order`.
    return {
        "ranked_order": order,
        # The load-bearing behavioral guarantee of the ported ranker.
        "technical_ranks_above_hype": order.index("technical") < order.index("hype"),
        "technical_ranks_above_funding": order.index("technical") < order.index("funding"),
    }


def build_parity_report() -> dict[str, Any]:
    """Assemble the full parity report — the golden the regression test locks
    and the script prints. Pure + deterministic."""
    return {
        "quality_gate": {
            name: _gate_result_to_dict(check_proposal(proposal))
            for name, proposal in _GATE_FIXTURES.items()
        },
        "technical_rank": _run_ranking(),
    }
