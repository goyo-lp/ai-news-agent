from app.services.openrouter_client import enforce_sentence_count, parse_relevance_scores


def test_enforce_sentence_count_exact_three() -> None:
    text = "Sentence one. Sentence two. Sentence three. Sentence four."
    output = enforce_sentence_count(text, count=3)
    assert output.count(".") >= 3
    assert output.endswith(".")


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
