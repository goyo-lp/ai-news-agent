import httpx

from app.config import Settings
from app.schemas.article import Article
from app.services.openrouter_client import (
    OpenRouterClient,
    enforce_sentence_count,
    parse_relevance_scores,
    split_sentences,
)


def _article(article_id: str = "a1") -> Article:
    return Article(
        id=article_id,
        source_name="Test Source",
        source_rss="https://example.com/feed",
        title="OpenAI launches new model",
        url=f"https://example.com/{article_id}",
    )


def test_enforce_sentence_count_exact_three() -> None:
    text = "Sentence one. Sentence two. Sentence three. Sentence four."
    output = enforce_sentence_count(text, count=3)
    assert len(split_sentences(output)) == 3
    assert output


def test_parse_relevance_scores_extracts_and_normalizes() -> None:
    text = 'Here are the scores: {"1": 85, "2": 20, "3": 150}'
    scores = parse_relevance_scores(text, {"1", "2", "3"})
    assert scores == {"1": 0.85, "2": 0.2, "3": 1.0}


def test_parse_relevance_scores_drops_unknown_and_malformed_keys() -> None:
    text = '{"1": 50, "9": 80, "2": "not a number"}'
    scores = parse_relevance_scores(text, {"1", "2"})
    assert scores == {"1": 0.5}


def test_parse_relevance_scores_handles_non_json() -> None:
    assert parse_relevance_scores("no scores here", {"1"}) == {}
    assert parse_relevance_scores("{broken json", {"1"}) == {}


async def test_summarize_articles_reports_api_failures_as_errors(monkeypatch) -> None:
    client = OpenRouterClient(Settings(_env_file=None, openrouter_api_key="test-key"))

    async def failing_request(*_args, **_kwargs):
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(client, "_request_completion", failing_request)

    summarized, errors = await client.summarize_articles([_article()], dry_run=False)

    assert len(summarized) == 1
    assert summarized[0].summary  # fallback summary still produced
    assert len(errors) == 1
    assert "a1" in errors[0]


async def test_summarize_articles_dry_run_falls_back_without_errors() -> None:
    client = OpenRouterClient(Settings(_env_file=None, openrouter_api_key="test-key"))

    summarized, errors = await client.summarize_articles([_article()], dry_run=True)

    assert len(summarized) == 1
    assert summarized[0].summary
    assert errors == []


async def test_score_articles_relevance_reports_failure(monkeypatch) -> None:
    client = OpenRouterClient(Settings(_env_file=None, openrouter_api_key="test-key"))

    async def failing_request(*_args, **_kwargs):
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(client, "_request_completion", failing_request)

    scores, error = await client.score_articles_relevance([_article()], dry_run=False)

    assert scores == {}
    assert error is not None and "relevance scoring failed" in error
