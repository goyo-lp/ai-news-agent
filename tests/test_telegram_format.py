from app.services.telegram_client import TELEGRAM_CAPTION_LIMIT, build_telegram_caption


def test_build_telegram_caption_has_link_and_limit() -> None:
    caption = build_telegram_caption(
        url="https://example.com/a",
        title="Interesting AI Story",
        summary="This is sentence one. This is sentence two. This is sentence three.",
    )
    assert caption.startswith('<a href="https://example.com/a">')
    assert len(caption) <= TELEGRAM_CAPTION_LIMIT


def test_build_telegram_caption_rejects_non_http_scheme() -> None:
    caption = build_telegram_caption(
        url="javascript:alert(1)",
        title="Malicious Title",
        summary="Summary text.",
    )
    assert caption.startswith('<a href="#">')
    assert "javascript:" not in caption
