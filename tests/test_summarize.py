from app.services.openrouter_client import enforce_sentence_count


def test_enforce_sentence_count_exact_three() -> None:
    text = "Sentence one. Sentence two. Sentence three. Sentence four."
    output = enforce_sentence_count(text, count=3)
    assert output.count(".") >= 3
    assert output.endswith(".")
