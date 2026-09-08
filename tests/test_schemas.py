from datetime import datetime, timedelta

import pytest
from pydantic import ValidationError

from ai_news_agent.fixtures import load_sample_records
from ai_news_agent.schemas import (
    Article,
    CoverageStats,
    Run,
    Score,
    Source,
    StoryCluster,
    Summary,
)


def record(model: type[object]) -> object:
    return next(item for item in load_sample_records() if isinstance(item, model))


def test_fixture_validates_every_required_record() -> None:
    records = load_sample_records()

    assert len(records) == 9
    assert {type(item).__name__ for item in records} == {
        "Article",
        "ArticleText",
        "Evidence",
        "Run",
        "Score",
        "Source",
        "StoryCluster",
        "Summary",
        "VerificationResult",
    }


def test_score_is_computed_from_accepted_weights() -> None:
    score = record(Score)

    assert isinstance(score, Score)
    assert score.weighted_total == 91


def test_article_rejects_naive_timestamp() -> None:
    article = record(Article)
    assert isinstance(article, Article)
    payload = article.model_dump()
    payload["published_at"] = datetime(2026, 9, 7, 11, 0)

    with pytest.raises(ValidationError, match="timestamps must include a timezone"):
        Article.model_validate(payload)


def test_source_state_requires_consistent_disabled_reason() -> None:
    source = record(Source)
    assert isinstance(source, Source)
    payload = source.model_dump()
    payload["enabled"] = False
    payload["disabled_reason"] = None

    with pytest.raises(ValidationError, match="disabled sources require"):
        Source.model_validate(payload)


def test_cluster_requires_representative_member_and_unique_articles() -> None:
    cluster = record(StoryCluster)
    assert isinstance(cluster, StoryCluster)
    payload = cluster.model_dump()
    payload["representative_article_id"] = "missing"

    with pytest.raises(ValidationError, match="must belong to the cluster"):
        StoryCluster.model_validate(payload)

    payload = cluster.model_dump()
    payload["article_ids"] = ["article-fixture", "article-fixture"]
    with pytest.raises(ValidationError, match="must be unique"):
        StoryCluster.model_validate(payload)


def test_summary_requires_exactly_three_sentences() -> None:
    summary = record(Summary)
    assert isinstance(summary, Summary)
    payload = summary.model_dump()
    payload["sentences"] = ["Only one sentence."]

    with pytest.raises(ValidationError):
        Summary.model_validate(payload)


def test_coverage_rejects_more_outcomes_than_attempts() -> None:
    with pytest.raises(ValidationError, match="cannot exceed attempted"):
        CoverageStats(attempted=1, successful=1, failed=1)


def test_run_requires_finished_at_for_terminal_state() -> None:
    run = record(Run)
    assert isinstance(run, Run)
    payload = run.model_dump()
    payload["finished_at"] = None

    with pytest.raises(ValidationError, match="finished run requires"):
        Run.model_validate(payload)


def test_run_accepts_aware_timestamps() -> None:
    run = record(Run)
    assert isinstance(run, Run)
    assert run.started_at.utcoffset() == timedelta(0)
