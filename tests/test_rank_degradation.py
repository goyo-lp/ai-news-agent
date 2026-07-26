"""The rank stage's failure contract: a slow LLM re-rank must cost ordering
quality, never the digest.

On 2026-07-26 it cost the digest. The relevance call ran past the rank node's
45s graph timeout, LangGraph killed the node, `articles_selected` was never
set, and the run printed `selected=0` and exited 0 — indistinguishable from a
day with no AI news. 40 ranked articles were on the floor and nothing reached
Telegram. These tests pin both halves of the fix: rank degrades on its own, and
a rank that genuinely fails is no longer reported as success.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pytest

from app.config import Settings
from app.graph.workflow import build_curation_workflow
from app.nodes.rank import rank_node
from app.schemas.article import Article, serialize_articles
from app.services.openrouter_client import OpenRouterClient


def _articles(count: int) -> list[Article]:
    now = datetime.now(timezone.utc)
    return [
        Article(
            id=f"a{i}",
            source_name=f"Source {i}",
            source_rss=f"https://example.com/{i}/feed",
            title=f"OpenAI releases frontier model {i}",
            url=f"https://example.com/{i}",
            published_at=now,
            description=f"A distinct frontier launch number {i}.",
        )
        for i in range(count)
    ]


def _state(articles: list[Article]) -> dict:
    return {
        "run_id": "test",
        "started_at": "",
        "dry_run": False,
        "limit": 20,
        "errors": [],
        "articles_enriched": serialize_articles(articles),
    }


@pytest.fixture(autouse=True)
def _offline(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """No network, and a history file per test."""
    monkeypatch.setenv("HISTORY_FILE", str(tmp_path / "history.json"))
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")


def _settings_with_rerank_budget(seconds: int) -> Settings:
    return Settings(llm_rerank_timeout_seconds=seconds)


def test_slow_rerank_degrades_to_the_deterministic_ranking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The blend is an enrichment. When it overruns its budget, rank still
    returns the deterministically-ranked articles."""
    articles = _articles(6)

    async def never_returns(self, candidates, dry_run):  # type: ignore[no-untyped-def]
        await asyncio.sleep(30)
        return {}, None

    monkeypatch.setattr(OpenRouterClient, "score_articles_relevance", never_returns)
    monkeypatch.setattr(
        "app.nodes.rank.get_settings", lambda: _settings_with_rerank_budget(1)
    )

    result = asyncio.run(rank_node(_state(articles)))

    assert len(result["articles_selected"]) > 0
    assert any("timed out" in e for e in result["errors"])


def test_rerank_timeout_is_bounded_by_its_own_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The bound is the rerank budget, not the graph node timeout — the whole
    point is that rank finishes long before LangGraph would kill it."""

    async def never_returns(self, candidates, dry_run):  # type: ignore[no-untyped-def]
        await asyncio.sleep(30)
        return {}, None

    monkeypatch.setattr(OpenRouterClient, "score_articles_relevance", never_returns)
    monkeypatch.setattr(
        "app.nodes.rank.get_settings", lambda: _settings_with_rerank_budget(1)
    )

    async def timed() -> float:
        loop = asyncio.get_running_loop()
        start = loop.time()
        await rank_node(_state(_articles(4)))
        return loop.time() - start

    assert asyncio.run(timed()) < 10


def test_a_slow_rerank_no_longer_empties_the_graphs_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End to end through the real graph: the exact 2026-07-26 shape — a
    relevance call slower than the old 45s node timeout — now yields articles
    instead of an unset `articles_selected`."""

    async def slow(self, candidates, dry_run):  # type: ignore[no-untyped-def]
        await asyncio.sleep(30)
        return {}, None

    async def passthrough_ingest(state):  # type: ignore[no-untyped-def]
        return {**state, "articles_raw": state["articles_enriched"]}

    async def passthrough_enrich(state):  # type: ignore[no-untyped-def]
        return dict(state)

    async def passthrough_summarize(state):  # type: ignore[no-untyped-def]
        return dict(state)

    monkeypatch.setattr(OpenRouterClient, "score_articles_relevance", slow)
    monkeypatch.setattr(
        "app.nodes.rank.get_settings", lambda: _settings_with_rerank_budget(1)
    )
    # Patch the names the graph builder closes over, not the defining modules:
    # workflow.py imports these at module load, so patching app.nodes.* would
    # leave the real nodes wired into the graph.
    monkeypatch.setattr("app.graph.workflow.ingest_node", passthrough_ingest)
    monkeypatch.setattr("app.graph.workflow.enrich_node", passthrough_enrich)
    monkeypatch.setattr("app.graph.workflow.summarize_node", passthrough_summarize)

    final = asyncio.run(build_curation_workflow().ainvoke(_state(_articles(6))))

    assert "articles_selected" in final
    assert len(final["articles_selected"]) > 0
