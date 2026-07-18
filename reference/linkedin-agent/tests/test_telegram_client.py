from app.schemas import LinkedInPost
from app.services.telegram_client import TELEGRAM_MESSAGE_LIMIT, build_telegram_message


def test_build_telegram_message_contains_title_body_and_hashtags() -> None:
    post = LinkedInPost(
        post_id="post-1",
        angle="technical",
        headline="What changed in AI this week?",
        body="A new model release improved tool-use reliability in production workflows.",
        hashtags=["#AI", "#MachineLearning", "#AgenticAI"],
        supporting_topic_ids=["t1"],
        citation_urls=["https://example.com/a"],
    )

    message = build_telegram_message(post)
    assert "<b>What changed in AI this week?</b>" in message
    assert "A new model release improved tool-use reliability in production workflows." in message
    assert "Recommended hashtags:" in message
    assert "#AI #MachineLearning #AgenticAI" in message
    assert len(message) <= TELEGRAM_MESSAGE_LIMIT


def test_build_telegram_message_enforces_length_limit() -> None:
    long_body = "Sentence. " * 1200
    post = LinkedInPost(
        post_id="post-2",
        angle="business",
        headline="Long post",
        body=long_body,
        hashtags=["#AI", "#EnterpriseAI", "#Innovation"],
        supporting_topic_ids=["t2"],
        citation_urls=["https://example.com/b"],
    )

    message = build_telegram_message(post)
    assert len(message) <= TELEGRAM_MESSAGE_LIMIT


def test_build_telegram_message_auto_paragraphizes_single_block_body() -> None:
    post = LinkedInPost(
        post_id="post-3",
        angle="technical",
        headline="Paragraph formatting",
        body=(
            "Sentence one about implementation details. "
            "Sentence two about tradeoffs. "
            "Sentence three about evaluation. "
            "Sentence four about risks."
        ),
        hashtags=["#AI", "#Engineering", "#Agents"],
        supporting_topic_ids=["t3"],
        citation_urls=["https://example.com/c"],
    )

    message = build_telegram_message(post)
    assert "\n\nSentence three about evaluation." in message
