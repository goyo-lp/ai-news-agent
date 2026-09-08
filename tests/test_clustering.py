from datetime import UTC, datetime, timedelta
from pathlib import Path

from ai_news_agent.clustering import (
    cluster_articles,
    hours_apart,
    jaccard,
    stable_cluster_id,
    title_entities,
    title_tokens,
)
from ai_news_agent.schemas import Article, ArticleText, StoryCluster
from ai_news_agent.storage import SQLiteStore

NOW = datetime(2026, 9, 8, 12, 0, tzinfo=UTC)


def make_article(
    article_id: str,
    source: str = "source-01",
    title: str = "Fixture story",
    hours_before: float = 1,
    url: str | None = None,
    authors: tuple[str, ...] = (),
) -> Article:
    published = NOW - timedelta(hours=hours_before)
    link = url or f"https://{source}.example/ai/{article_id}"
    return Article(
        id=article_id,
        source_id=source,
        url=link,
        canonical_url=link,
        title=title,
        authors=authors,
        published_at=published,
        first_seen_at=published,
    )


def make_store(tmp_path: Path) -> SQLiteStore:
    store = SQLiteStore(tmp_path / "clusters.db")
    store.migrate()
    return store


def seed_text(
    store: SQLiteStore, article_id: str, text: str = "Full body text. " * 40
) -> None:
    store.save(
        ArticleText(
            id=f"{article_id}-text",
            article_id=article_id,
            text=text,
            content_hash=(
                "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
            ),
            fetched_at=NOW,
        )
    )


# Similarity primitives


def test_title_tokens_lowercase_and_split() -> None:
    assert title_tokens("Model B is HERE!") == {"model", "b", "is", "here"}
    assert "Fixture Lab" in title_entities("Fixture Lab releases Model B")


def test_jaccard_handles_empty_sets() -> None:
    assert jaccard(frozenset(), frozenset({"a"})) == 0.0
    assert jaccard(frozenset({"a", "b"}), frozenset({"b", "c"})) == 1 / 3


def test_hours_apart_is_absolute() -> None:
    assert hours_apart(NOW, NOW - timedelta(hours=30)) == 30.0


def test_stable_cluster_id_ignores_member_order() -> None:
    assert stable_cluster_id(("b", "a")) == stable_cluster_id(("a", "b"))
    assert stable_cluster_id(("a",)) != stable_cluster_id(("a", "b"))


# Event grouping


def test_syndicated_coverage_becomes_one_cluster() -> None:
    articles = (
        make_article(
            "a1", "source-01", "Fixture Lab releases Model B with open weights"
        ),
        make_article(
            "a2", "source-02", "Fixture Lab Releases Model B With Open Weights"
        ),
        make_article(
            "a3", "source-03", "Model B is here: open weights from Fixture Lab"
        ),
    )

    summary = cluster_articles(articles, now=NOW)

    assert summary.clusters == 1
    assert summary.duplicates_collapsed == 0
    (result,) = summary.results
    assert result.cluster.article_ids == ("a1", "a2", "a3")
    assert result.cluster.representative_article_id == "a1"
    assert "grouped" in result.cluster.rationale
    assert result.recycled is False


def test_separate_launches_stay_apart() -> None:
    articles = (
        make_article(
            "a1", "source-01", "Fixture Lab releases Model B with open weights"
        ),
        make_article(
            "b1", "source-02", "Rival Corp unveils Robot Chef for home kitchens"
        ),
    )

    summary = cluster_articles(articles, now=NOW)

    assert summary.clusters == 2
    assert all("singleton" in item.cluster.rationale for item in summary.results)


def test_duplicate_canonical_urls_collapse() -> None:
    articles = (
        make_article("a1", url="https://x.example/ai/same"),
        make_article("a2", url="https://x.example/ai/same"),
    )

    summary = cluster_articles(articles, now=NOW)

    assert summary.duplicates_collapsed == 1
    assert summary.articles_clustered == 1
    assert summary.results[0].cluster.article_ids == ("a1",)


def test_identical_content_merges_despite_rewrite(tmp_path: Path) -> None:
    articles = (
        make_article("a1", title="Alpha release notes"),
        make_article("a2", title="Totally different headline"),
    )
    with make_store(tmp_path) as store:
        seed_text(store, "a1")
        seed_text(store, "a2")
        summary = cluster_articles(articles, now=NOW, store=store)

    assert summary.clusters == 1
    assert "identical article content" in summary.results[0].cluster.rationale


def test_ambiguous_near_miss_stays_separate_with_scores(tmp_path: Path) -> None:
    articles = (
        make_article("c1", "source-01", "Fixture Lab updates safety policy"),
        make_article("c2", "source-02", "Rival Corp updates safety policy"),
        make_article("d1", "source-03", "Quantum error correction milestone reached"),
    )

    summary = cluster_articles(articles, now=NOW)

    assert summary.clusters == 3
    for item in summary.results:
        if item.cluster.article_ids != ("d1",):
            assert "needs model judgment to merge" in item.cluster.rationale
        else:
            assert "needs model judgment" not in item.cluster.rationale


def test_shared_entities_boost_a_rewrite_into_one_event() -> None:
    articles = (
        make_article("a1", "source-01", "Fixture Lab ships fast inference stack today"),
        make_article("a2", "source-02", "Fixture Lab announces fast inference preview"),
    )

    summary = cluster_articles(articles, now=NOW)

    assert summary.clusters == 1
    assert "shared entities" in summary.results[0].cluster.rationale


def test_representative_prefers_byline_without_text() -> None:
    articles = (
        make_article("a1", title="Fixture Lab releases Model B with open weights"),
        make_article(
            "a2",
            title="Fixture Lab releases Model B with open weights",
            authors=("Jane Reporter",),
        ),
    )

    summary = cluster_articles(articles, now=NOW)

    assert summary.results[0].cluster.representative_article_id == "a2"
    assert "has byline" in summary.results[0].cluster.rationale


def test_stale_republication_does_not_merge(tmp_path: Path) -> None:
    articles = (
        make_article("a1", title="Fixture Lab releases Model B", hours_before=1),
        make_article("a2", title="Fixture Lab releases Model B", hours_before=80),
    )

    summary = cluster_articles(articles, now=NOW)

    assert summary.clusters == 2


def test_representative_prefers_full_text(tmp_path: Path) -> None:
    articles = (
        make_article("a1", title="Fixture Lab releases Model B with open weights"),
        make_article("a2", title="Fixture Lab releases Model B with open weights"),
    )
    with make_store(tmp_path) as store:
        seed_text(store, "a2")
        summary = cluster_articles(articles, now=NOW, store=store)

    assert summary.results[0].cluster.representative_article_id == "a2"
    assert "has full text" in summary.results[0].cluster.rationale


# Digest history


def test_recycled_coverage_is_flagged(tmp_path: Path) -> None:
    articles = (
        make_article("a1", title="Fixture Lab releases Model B"),
        make_article("a2", title="Fixture Lab releases Model B"),
    )

    summary = cluster_articles(
        articles, now=NOW, published_article_ids=frozenset({"a1", "a2"})
    )

    assert summary.recycled == 1
    assert "recycled" in summary.results[0].cluster.rationale


def test_substantive_follow_up_forms_a_fresh_cluster() -> None:
    old = make_article("old1", title="Fixture Lab previews Model B", hours_before=72)
    follow_up = make_article(
        "new1", title="Fixture Lab releases Model B with open weights", hours_before=1
    )

    summary = cluster_articles(
        (old, follow_up), now=NOW, published_article_ids=frozenset({"old1"})
    )

    assert summary.clusters == 2
    assert summary.recycled == 1
    fresh = next(
        item for item in summary.results if item.cluster.article_ids == ("new1",)
    )
    assert fresh.recycled is False


# Persistence and idempotency


def test_clusters_persist_and_rerun_is_stable(tmp_path: Path) -> None:
    articles = (
        make_article("a1", title="Fixture Lab releases Model B with open weights"),
        make_article("a2", title="Fixture Lab Releases Model B With Open Weights"),
        make_article("b1", title="Rival Corp unveils Robot Chef for home kitchens"),
    )
    with make_store(tmp_path) as store:
        first = cluster_articles(articles, now=NOW, store=store)
        second = cluster_articles(articles, now=NOW, store=store)

        first_ids = sorted(item.cluster.id for item in first.results)
        second_ids = sorted(item.cluster.id for item in second.results)
        assert first_ids == second_ids
        stored = list(store.iter_records(StoryCluster))
        assert len(stored) == 2
        assert sorted(item.id for item in stored) == first_ids


def test_empty_input_yields_empty_summary() -> None:
    summary = cluster_articles((), now=NOW)

    assert summary.attempted == 0
    assert summary.clusters == 0
