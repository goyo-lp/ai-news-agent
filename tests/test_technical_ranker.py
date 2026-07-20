from __future__ import annotations

from datetime import datetime, timedelta, timezone

from app.orchestrator.schemas import CuratedArticle
from app.orchestrator.services.ranking import (
    DiscoveredItem,
    SourcePolicy,
    cluster_items,
    curated_articles_to_items,
    domain_from_url,
    rank_topics,
)
from app.orchestrator.services.technical_ranker import TechnicalRanker
from app.config import Settings


def _curated(
    article_id: str,
    title: str,
    *,
    url: str = "https://example.com/post",
    summary: str | None = None,
    summary_context: str = "",
    published_at: datetime | None = None,
) -> CuratedArticle:
    return CuratedArticle(
        id=article_id,
        source_name="Test Source",
        title=title,
        url=url,
        published_at=published_at,
        summary_context=summary_context,
        summary=summary,
        score=0.5,
    )


def _settings() -> Settings:
    # No OPENROUTER_API_KEY -> heuristic path -> fully deterministic.
    return Settings(_env_file=None, openrouter_api_key=None)


# --- ranking.py -------------------------------------------------------------


def test_domain_from_url_strips_www_and_lowercases() -> None:
    assert domain_from_url("https://WWW.Example.com/path") == "example.com"
    assert domain_from_url("https://blog.openai.com/x") == "blog.openai.com"
    assert domain_from_url("not-a-url") == ""


def test_curated_articles_to_items_projects_boundary_fields() -> None:
    curated = _curated(
        "a1",
        "OpenAI launches new model",
        url="https://www.example.com/post",
        summary_context="og description",
        summary="A concise summary.",
    )
    items = curated_articles_to_items([curated], SourcePolicy.permissive())

    assert len(items) == 1
    item = items[0]
    assert item.id == "a1"
    assert item.domain == "example.com"
    assert item.snippet == "og description"
    assert item.raw_content == "A concise summary."
    assert item.source_tier == 1  # permissive: every domain is tier-1
    assert item.source_weight == 1.0


def test_rank_topics_demotes_hype_without_technical_detail() -> None:
    """P2.1 acceptance: stories heavy on hype phrasing with no technical
    keywords must rank below a concrete technical story."""
    now = datetime.now(timezone.utc)
    hype = _curated(
        "hype",
        "Revolutionary game changer that will break the internet",
        url="https://hype.example.com/x",
        summary_context="massive unbelievable must-have announcement",
        published_at=now,
    )
    technical = _curated(
        "tech",
        "Architecture benchmark inference latency improved",
        url="https://tech.example.com/y",
        summary_context="benchmark inference latency framework deployment",
        published_at=now,
    )

    settings = _settings()
    items = curated_articles_to_items([hype, technical], SourcePolicy.permissive())
    ranker = TechnicalRanker(settings)
    # Exercise the private deterministic path directly for a stable, LLM-free
    # test (assess_many is async; the heuristic is what we want to pin here).
    per_item = {it.id: ranker._heuristic_assessment(it) for it in items}

    topics = rank_topics(
        items,
        limit=2,
        policy=SourcePolicy.permissive(),
        technical_overrides={k: v.technical_depth for k, v in per_item.items()},
        implementation_overrides={k: v.implementation_specificity for k, v in per_item.items()},
        hype_overrides={k: v.hype_score for k, v in per_item.items()},
    )

    assert topics[0].topic_id == "tech"
    assert topics[1].topic_id == "hype"
    assert topics[0].score > topics[1].score
    # The hype story's assessment actually carries hype > 0 (the -0.18 lever).
    assert per_item["hype"].hype_score > 0.0
    assert per_item["tech"].hype_score == 0.0


def test_rank_topics_demotes_business_only_without_technical_detail() -> None:
    """A funding/deal story with no technical depth or implementation
    specificity is penalized by ``_business_only_penalty`` even when its
    relevance score is high."""
    now = datetime.now(timezone.utc)
    funding = _curated(
        "fund",
        "AI startup raises series B funding round valuation",
        url="https://news.example.com/funding",
        summary_context="funding raised series valuation partnership acquisition",
        published_at=now,
    )
    tech = _curated(
        "tech",
        "New inference architecture benchmark training",
        url="https://tech.example.com/y",
        summary_context="architecture benchmark inference training framework",
        published_at=now,
    )

    settings = _settings()
    items = curated_articles_to_items([funding, tech], SourcePolicy.permissive())
    ranker = TechnicalRanker(settings)
    per_item = {it.id: ranker._heuristic_assessment(it) for it in items}

    topics = rank_topics(
        items,
        limit=2,
        policy=SourcePolicy.permissive(),
        technical_overrides={k: v.technical_depth for k, v in per_item.items()},
        implementation_overrides={k: v.implementation_specificity for k, v in per_item.items()},
        hype_overrides={k: v.hype_score for k, v in per_item.items()},
    )

    assert topics[0].topic_id == "tech"
    assert topics[1].topic_id == "fund"


def test_rank_topics_deterministic_ordering() -> None:
    """Same input -> same ordering, byte-for-byte. P2.1's parity-test anchor."""
    now = datetime.now(timezone.utc)
    a = _curated("a", "New inference architecture benchmark training", url="https://a.example.com/", published_at=now)
    b = _curated("b", "Model agents framework sdk release", url="https://b.example.com/", published_at=now - timedelta(hours=2))
    c = _curated("c", "Roundup webinar podcast opinion guide", url="https://c.example.com/", published_at=now)

    settings = _settings()
    items = curated_articles_to_items([a, b, c], SourcePolicy.permissive())
    ranker = TechnicalRanker(settings)
    per_item = {it.id: ranker._heuristic_assessment(it) for it in items}

    run1 = rank_topics(
        items,
        limit=3,
        policy=SourcePolicy.permissive(),
        technical_overrides={k: v.technical_depth for k, v in per_item.items()},
        implementation_overrides={k: v.implementation_specificity for k, v in per_item.items()},
        hype_overrides={k: v.hype_score for k, v in per_item.items()},
    )
    # re-run from a fresh model_copy (rank_topics deep-copies internally) to
    # prove no hidden mutation crosses between runs.
    run2 = rank_topics(
        items,
        limit=3,
        policy=SourcePolicy.permissive(),
        technical_overrides={k: v.technical_depth for k, v in per_item.items()},
        implementation_overrides={k: v.implementation_specificity for k, v in per_item.items()},
        hype_overrides={k: v.hype_score for k, v in per_item.items()},
    )

    assert [t.topic_id for t in run1] == [t.topic_id for t in run2]
    assert [t.score for t in run1] == [t.score for t in run2]
    # low-signal story ranks last by construction (relevance demoted).
    assert run1[-1].topic_id == "c"


def test_rank_topics_respects_limit() -> None:
    """Five genuinely-different stories cap to `limit`. The titles are
    intentionally character-divergent so clustering produces five singleton
    clusters, not one mega-cluster."""
    now = datetime.now(timezone.utc)
    distinct = [
        ("a", "Zeta architecture benchmark inference training"),
        ("b", "Nuance reasoning multimodal model evaluation"),
        ("c", "Frontier chip accelerator deployment latency"),
        ("d", "Sandbox agents framework tooling protocol"),
        ("e", "Quantized weights quantization kernels throughput"),
    ]
    items = curated_articles_to_items(
        [_curated(i, t, url=f"https://{i}.example.com/", published_at=now) for i, t in distinct],
        SourcePolicy.permissive(),
    )
    ranker = TechnicalRanker(_settings())
    per_item = {it.id: ranker._heuristic_assessment(it) for it in items}

    topics = rank_topics(
        items, limit=3, policy=SourcePolicy.permissive(),
        technical_overrides={k: v.technical_depth for k, v in per_item.items()},
        implementation_overrides={k: v.implementation_specificity for k, v in per_item.items()},
        hype_overrides={k: v.hype_score for k, v in per_item.items()},
    )

    assert len(topics) == 3
    assert [t.score for t in topics] == sorted([t.score for t in topics], reverse=True)


def test_rank_topics_empty_input() -> None:
    assert rank_topics([], limit=5, policy=SourcePolicy.permissive()) == []
    assert rank_topics([], limit=0, policy=SourcePolicy.permissive()) == []


def test_cluster_items_groups_same_story() -> None:
    now = datetime.now(timezone.utc)
    a = DiscoveredItem(id="a", title="OpenAI launches new model", url="https://a.example.com/x", domain="a.example.com", published_at=now)
    b = DiscoveredItem(id="b", title="OpenAI launches new model at event", url="https://b.example.com/y", domain="b.example.com", published_at=now)
    c = DiscoveredItem(id="c", title="Unrelated chip benchmark", url="https://c.example.com/z", domain="c.example.com", published_at=now)

    clusters = cluster_items([a, b, c])
    assert len(clusters) == 2
    same_story = next(c for c in clusters if c.id == "a")
    assert {m.id for m in same_story.members} == {"a", "b"}


def test_source_policy_permissive_allows_everything() -> None:
    policy = SourcePolicy.permissive()
    assert policy.cluster_allowed({"random.example.com"}) is True
    assert policy.tier_for("random.example.com") == 1
    assert policy.weight_for("random.example.com") == 1.0


# --- technical_ranker.py ----------------------------------------------------


def test_heuristic_assessment_marks_technical_and_hype() -> None:
    settings = _settings()
    ranker = TechnicalRanker(settings)
    tech = DiscoveredItem(id="t", title="Architecture benchmark inference", url="https://x.example.com/", domain="x.example.com")
    hype = DiscoveredItem(id="h", title="Revolutionary game changer massive", url="https://y.example.com/", domain="y.example.com")

    tech_a = ranker._heuristic_assessment(tech)
    hype_a = ranker._heuristic_assessment(hype)

    assert tech_a.technical_depth > 0.2
    assert tech_a.hype_score == 0.0
    assert hype_a.hype_score > 0.0
    # `notes='heuristic'` marks the fallback path vs the LLM path.
    assert tech_a.notes == "heuristic"


async def test_assess_many_uses_heuristic_when_no_api_key() -> None:
    """Deterministic path: with OPENROUTER_API_KEY unset, assess_many returns
    heuristic assessments keyed by item id, with zero network calls."""
    settings = Settings(_env_file=None, openrouter_api_key=None)
    ranker = TechnicalRanker(settings)
    items = [
        DiscoveredItem(id="a", title="benchmark inference", url="https://a.example.com/", domain="a.example.com"),
        DiscoveredItem(id="b", title="revolutionary hype", url="https://b.example.com/", domain="b.example.com"),
    ]
    result = await ranker.assess_many(items, dry_run=False)
    assert set(result) == {"a", "b"}
    assert result["a"].notes == "heuristic"
    assert result["b"].hype_score > 0.0


async def test_assess_many_dry_run_forces_heuristic_even_with_api_key() -> None:
    settings = Settings(_env_file=None, openrouter_api_key="sk-test")
    ranker = TechnicalRanker(settings)
    items = [DiscoveredItem(id="a", title="benchmark", url="https://a.example.com/", domain="a.example.com")]
    result = await ranker.assess_many(items, dry_run=True)
    assert result["a"].notes == "heuristic"


async def test_assess_many_falls_back_to_heuristic_on_llm_failure(monkeypatch) -> None:
    """M1: the per-item worker catches Stage-A LLM failures and falls back to
    the heuristic — flagged distinctly via ``notes="heuristic-fallback:
    <ExcType>"`` so an operator can tell dry-run from runtime error.

    Replaces ``httpx.AsyncClient`` on the ranker module so any POST inside the
    Stage-A path raises ``httpx.ConnectError`` on contact."""
    import httpx

    import app.orchestrator.services.technical_ranker as tr_mod

    class _BrokenAsyncClient:
        def __init__(self, *args: object, **kwargs: object) -> None: ...

        async def __aenter__(self) -> "_BrokenAsyncClient":
            return self

        async def __aexit__(self, *exc: object) -> None: ...

        async def post(self, url: str, **kwargs: object) -> httpx.Response:
            raise httpx.ConnectError("upstream down")

    monkeypatch.setattr(tr_mod.httpx, "AsyncClient", _BrokenAsyncClient)

    settings = Settings(
        _env_file=None,
        openrouter_api_key="sk-test",
        request_timeout_seconds=10,
    )
    ranker = TechnicalRanker(settings)
    items = [
        DiscoveredItem(id="a", title="benchmark", url="https://a.example.com/", domain="a.example.com"),
        DiscoveredItem(id="b", title="revolutionary", url="https://b.example.com/", domain="b.example.com"),
    ]
    result = await ranker.assess_many(items, dry_run=False)

    assert set(result) == {"a", "b"}
    for item_id in ("a", "b"):
        assessment = result[item_id]
        # The fallback preserves the heuristic's *values*…
        assert assessment.technical_depth >= 0.2
        # …but marks itself distinctly so dry-run and runtime errors are
        # differentiable downstream.
        assert assessment.notes == "heuristic-fallback: ConnectError"


async def test_assess_many_empty_input() -> None:
    ranker = TechnicalRanker(_settings())
    assert await ranker.assess_many([], dry_run=False) == {}