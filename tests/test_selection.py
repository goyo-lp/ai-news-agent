from datetime import UTC, datetime
from pathlib import Path

from ai_news_agent.schemas import (
    Article,
    RouteType,
    Score,
    Source,
    SourceCategory,
    SourceTier,
    StoryCluster,
)
from ai_news_agent.selection import (
    eligibility_reason,
    publisher_family,
    render_draft,
    select_digest,
    topic_bucket,
)
from ai_news_agent.storage import SQLiteStore

NOW = datetime(2026, 9, 8, 12, 0, tzinfo=UTC)


def make_article(
    article_id: str,
    title: str = "Model release changes builder costs",
    source_id: str = "source-01",
    canonical: str | None = None,
) -> Article:
    url = f"https://example.com/ai/{article_id}"
    canonical_url = canonical or url
    return Article(
        id=article_id,
        source_id=source_id,
        url=url,
        canonical_url=canonical_url,
        title=title,
        published_at=NOW,
        first_seen_at=NOW,
    )


def make_cluster(cluster_id: str, article_id: str) -> StoryCluster:
    return StoryCluster(
        id=cluster_id,
        article_ids=(article_id,),
        representative_article_id=article_id,
        rationale="seed",
        created_at=NOW,
    )


def make_score(cluster_id: str, **overrides) -> Score:
    payload = {
        "id": f"{cluster_id}-editorial",
        "story_cluster_id": cluster_id,
        "relevance": 4,
        "impact": 4,
        "novelty": 4,
        "evidence": 4,
        "timeliness": 4,
        "evidence_ids": ("e1",),
        "rationale": "Strong evidence.",
    }
    payload.update(overrides)
    return Score(**payload)


def make_store(tmp_path: Path) -> SQLiteStore:
    store = SQLiteStore(tmp_path / "select.db")
    store.migrate()
    return store


def setup(
    store: SQLiteStore,
    specs: list[tuple[str, str, dict]],
) -> tuple[tuple[StoryCluster, ...], dict[str, Score], dict[str, Article]]:
    clusters: list[StoryCluster] = []
    scores: dict[str, Score] = {}
    articles: dict[str, Article] = {}
    for cluster_id, article_id, score_kwargs in specs:
        title = score_kwargs.pop("title", "Model release changes builder costs")
        source = score_kwargs.pop("source", "source-01")
        article = make_article(article_id, title=title, source_id=source)
        cluster = make_cluster(cluster_id, article_id)
        score = make_score(cluster_id, **score_kwargs)
        articles[article_id] = article
        clusters.append(cluster)
        scores[cluster_id] = score
    return tuple(clusters), scores, articles


def test_topic_and_family_helpers(tmp_path: Path) -> None:
    assert topic_bucket("New model release with open weights") == "model-release"
    assert topic_bucket("Court ruling sets AI policy duties") == "policy"
    assert topic_bucket("A quiet community note") == "general"
    article = make_article("a1", source_id="family-a")
    with make_store(tmp_path) as store:
        assert publisher_family(article, store) == "family-a"
        store.save(
            Source(
                id="family-a",
                name="Family A",
                publisher_family="family-a",
                category=SourceCategory.INDEPENDENT_REPORTING,
                tier=SourceTier.CORE,
                homepage_url="https://example.com/",
                route_type=RouteType.RSS,
                route_url="https://example.com/feed.xml",
                feed_scope="AI",
                enabled=True,
                disabled_reason=None,
                last_checked_at=NOW,
                last_success_at=NOW,
            )
        )
        assert publisher_family(article, store) == "family-a"


def test_quality_floor_and_minimums(tmp_path: Path) -> None:
    assert eligibility_reason(make_score("c1")) is None
    assert "below floor" in eligibility_reason(
        make_score("c1", relevance=3, impact=2, novelty=2, evidence=3, timeliness=3)
    )
    assert "relevance" in eligibility_reason(make_score("c1", relevance=2))
    assert "evidence" in eligibility_reason(make_score("c1", evidence=2))
    assert "references" in eligibility_reason(make_score("c1", evidence_ids=()))

    specs = [
        ("c-low", "a-low", {"relevance": 2}),
        ("c-ok", "a-ok", {}),
    ]
    with make_store(tmp_path) as store:
        clusters, scores, articles = setup(store, specs)
        summary = select_digest(clusters, scores, articles, store)

    assert summary.selected_ids == ("c-ok",)
    assert summary.shortfall == 9
    assert summary.shortfall_reason is not None
    by_id = {item.cluster_id: item for item in summary.decisions}
    assert by_id["c-low"].selected is False
    assert by_id["c-ok"].rank == 1


def test_missing_score_and_article_are_ineligible(tmp_path: Path) -> None:
    with make_store(tmp_path) as store:
        clusters, scores, articles = setup(store, [("c1", "a1", {})])
        missing_score = select_digest((clusters[0],), {}, articles, store)
        assert missing_score.selected == 0
        assert "missing editorial score" in missing_score.decisions[0].reason
        missing_article = select_digest((clusters[0],), scores, {}, store)
        assert "missing from store" in missing_article.decisions[0].reason


def test_duplicate_events_keep_first_by_score(tmp_path: Path) -> None:
    with make_store(tmp_path) as store:
        shared = "https://example.com/ai/shared"
        a1 = make_article("a1", canonical=shared)
        a2 = make_article("a2", canonical=shared.upper())
        c1, c2 = make_cluster("c1", "a1"), make_cluster("c2", "a2")
        scores = {"c1": make_score("c1"), "c2": make_score("c2")}
        summary = select_digest((c1, c2), scores, {"a1": a1, "a2": a2}, store)

    assert summary.selected_ids == ("c1",)
    by_id = {item.cluster_id: item for item in summary.decisions}
    assert "duplicate" in by_id["c2"].reason


def test_stable_tie_break_by_cluster_id(tmp_path: Path) -> None:
    with make_store(tmp_path) as store:
        clusters, scores, articles = setup(
            store, [("c-b", "a-b", {}), ("c-a", "a-a", {})]
        )
        summary = select_digest(clusters, scores, articles, store)

    assert summary.selected_ids == ("c-a", "c-b")


def test_quiet_day_publishes_fewer_without_padding(tmp_path: Path) -> None:
    with make_store(tmp_path) as store:
        clusters, scores, articles = setup(store, [("c1", "a1", {}), ("c2", "a2", {})])
        summary = select_digest(clusters, scores, articles, store, max_selected=10)

    assert summary.selected == 2
    assert summary.shortfall == 8
    assert "2 of 10" in (summary.shortfall_reason or "")
    assert summary.reserve == 0


def test_reserve_holds_next_eligible_in_order(tmp_path: Path) -> None:
    specs = [(f"c{i}", f"a{i}", {}) for i in range(4)]
    with make_store(tmp_path) as store:
        clusters, scores, articles = setup(store, specs)
        summary = select_digest(
            clusters, scores, articles, store, max_selected=2, max_reserve=2
        )

    assert len(summary.selected_ids) == 2
    assert len(summary.reserve_ids) == 2
    assert set(summary.selected_ids) | set(summary.reserve_ids) == {
        f"c{i}" for i in range(4)
    }
    by_id = {item.cluster_id: item for item in summary.decisions}
    assert by_id[summary.reserve_ids[0]].reserve_rank == 1


def test_one_topic_dominant_keeps_consequential_outlier(tmp_path: Path) -> None:
    specs = [
        (f"c{i}", f"a{i}", {"title": f"Model release number {i}"}) for i in range(5)
    ]
    specs.append(("c-big", "a-big", {"title": "Court ruling sets binding AI duties"}))
    with make_store(tmp_path) as store:
        clusters, scores, articles = setup(store, specs)
        # Force the outlier far below the pack: it stays reserve, pack intact.
        scores["c-big"] = make_score(
            "c-big", relevance=4, impact=3, novelty=3, evidence=3, timeliness=3
        )
        summary = select_digest(
            clusters, scores, articles, store, max_selected=4, max_reserve=5
        )

    assert "c-big" not in summary.selected_ids
    assert not summary.diversity_notes


def test_soft_family_cap_swaps_close_call_with_note(tmp_path: Path) -> None:
    specs = [
        ("c1", "a1", {"title": "Model release alpha", "source": "fam-a"}),
        ("c2", "a2", {"title": "Model release beta", "source": "fam-a"}),
        ("c3", "a3", {"title": "Model release gamma", "source": "fam-a"}),
        ("c4", "a4", {"title": "Security breach discloses misuse", "source": "fam-b"}),
    ]
    with make_store(tmp_path) as store:
        clusters, scores, articles = setup(store, specs)
        scores["c1"] = make_score(
            "c1", relevance=5, impact=5, novelty=5, evidence=5, timeliness=5
        )
        scores["c2"] = make_score(
            "c2", relevance=5, impact=4, novelty=4, evidence=4, timeliness=4
        )
        scores["c3"] = make_score(
            "c3", relevance=4, impact=4, novelty=4, evidence=4, timeliness=4
        )
        scores["c4"] = make_score(
            "c4", relevance=4, impact=4, novelty=3, evidence=4, timeliness=4
        )
        summary = select_digest(
            clusters, scores, articles, store, max_selected=3, max_reserve=2
        )

    assert summary.selected_ids == ("c1", "c2", "c4")
    assert summary.diversity_notes
    assert "c4" in summary.diversity_notes[0] and "c3" in summary.diversity_notes[0]


def test_draft_renders_links_scores_and_shortfall(tmp_path: Path) -> None:
    with make_store(tmp_path) as store:
        clusters, scores, articles = setup(store, [("c1", "a1", {})])
        summary = select_digest(clusters, scores, articles, store, max_selected=3)
        draft = render_draft(
            summary,
            {cluster.id: cluster for cluster in clusters},
            articles,
            scores,
            store,
        )

    assert draft.startswith("# Digest draft (unverified)")
    assert "https://example.com/ai/a1" in draft
    assert "Score:" in draft
    assert "Shortfall" in draft
    assert "Rationale:" in draft


def test_draft_marks_missing_context(tmp_path: Path) -> None:
    with make_store(tmp_path) as store:
        summary = select_digest((), {}, {}, store)
        draft = render_draft(summary, {"c1": make_cluster("c1", "a1")}, {}, {}, store)

    assert "Shortfall" in draft


def test_select_command_writes_draft(tmp_path: Path) -> None:
    import json

    from typer.testing import CliRunner

    from ai_news_agent.cli import app
    from ai_news_agent.schemas import StoryCluster

    runner = CliRunner()
    database = tmp_path / "select.db"
    draft_path = tmp_path / "draft.md"
    with SQLiteStore(database) as store:
        store.migrate()
        article = make_article("a1")
        store.save(article)
        store.save(
            StoryCluster(
                id="c1",
                article_ids=("a1",),
                representative_article_id="a1",
                rationale="seed",
                created_at=NOW,
            )
        )
        store.save(make_score("c1"))

    result = runner.invoke(
        app,
        ["select", "--database", str(database), "--draft-path", str(draft_path)],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    assert payload["selected"] == 1
    assert payload["selected_ids"] == ["c1"]
    assert "Digest draft" in draft_path.read_text(encoding="utf-8")
