from __future__ import annotations

import pytest

from app.config import Settings


def test_missing_required_runtime_fields_requires_telegram_creds_when_not_dry_run() -> None:
    settings = Settings(_env_file=None)
    missing = settings.missing_required_runtime_fields(dry_run=False)
    assert "TELEGRAM_BOT_TOKEN" in missing
    assert "TELEGRAM_CHAT_ID" in missing


def test_missing_required_runtime_fields_empty_in_dry_run() -> None:
    settings = Settings(_env_file=None)
    assert settings.missing_required_runtime_fields(dry_run=True) == []


def test_settings_defaults() -> None:
    settings = Settings(_env_file=None)
    assert settings.openrouter_model == "deepseek/deepseek-v4-flash"
    assert settings.max_articles_per_run == 50
    assert settings.history_retention_days == 14


def test_ported_linkedin_knobs_have_defaults() -> None:
    """The Stage A/B split, Tavily research-evidence knobs, and deep-agent
    bounds are ported from the LinkedIn agent (Decisions B/E/J) with sensible
    defaults so dry-run needs no new required keys."""
    settings = Settings(_env_file=None)
    assert settings.openrouter_stage_a_model == "openai/gpt-oss-120b"
    assert settings.openrouter_stage_b_research_model == "anthropic/claude-sonnet-5"
    assert settings.openrouter_stage_b_writer_model == "anthropic/claude-opus-4-8"

    assert settings.tavily_api_key is None
    assert settings.tavily_base_url == "https://api.tavily.com"
    assert settings.tavily_search_depth == "advanced"

    assert settings.deep_agent_enabled is True
    assert settings.deep_agent_timeout_seconds == 75
    assert settings.deep_agent_max_evidence_sources == 5

    assert settings.max_topics_per_run == 5
    assert settings.orchestrator_data_dir == "data/orchestrator"


def test_news_bot_profile_falls_back_to_legacy_credentials() -> None:
    """Backward compat: a repo configured before bot-profile env vars existed
    keeps delivering the digest unchanged — the `news` profile is backfilled
    from the legacy top-level fields."""
    settings = Settings(
        _env_file=None,
        telegram_bot_token="legacy-token",
        telegram_chat_id="legacy-chat",
    )
    profile = settings.bot_profile("news")
    assert profile is not None
    assert profile.token == "legacy-token"
    assert profile.chat_id == "legacy-chat"
    assert profile.is_complete() is True


def test_news_bot_profile_explicit_env_wins_over_legacy_fallback() -> None:
    """An explicit news profile splits the digest onto a dedicated bot without
    touching the legacy fields — explicit wins over fallback."""
    settings = Settings(
        _env_file=None,
        telegram_bot_token="legacy-token",
        telegram_chat_id="legacy-chat",
        telegram_news_bot_token="news-token",
        telegram_news_chat_id="news-chat",
    )
    profile = settings.bot_profile("news")
    assert profile is not None
    assert profile.token == "news-token"
    assert profile.chat_id == "news-chat"


def test_linkedin_bot_profile_optional_and_has_no_fallback() -> None:
    """The linkedin profile is optional until its bot exists (P1.2: dry-run
    needs no new required keys). It does NOT fall back to the legacy creds —
    proposals must never accidentally route to the news chat."""
    settings = Settings(
        _env_file=None,
        telegram_bot_token="legacy-token",
        telegram_chat_id="legacy-chat",
    )
    assert settings.bot_profile("linkedin") is None
    assert settings.bot_profiles() == {} or set(settings.bot_profiles()) == {"news"}

    linkedin = Settings(
        _env_file=None,
        telegram_linkedin_bot_token="li-token",
        telegram_linkedin_chat_id="li-chat",
    )
    profile = linkedin.bot_profile("linkedin")
    assert profile is not None
    assert profile.token == "li-token"
    assert profile.chat_id == "li-chat"


def test_bot_profile_rejects_unknown_name() -> None:
    """Closed registry: an unknown bot name fails loudly instead of silently
    routing to the wrong chat."""
    settings = Settings(_env_file=None)
    with pytest.raises(ValueError):
        settings.bot_profile("twitter")  # type: ignore[arg-type]


def test_bot_profile_none_when_incomplete() -> None:
    """A profile with only a token but no chat id (or vice versa) is treated as
    not-configured so delivery can report it as missing rather than partial."""
    settings = Settings(
        _env_file=None,
        telegram_linkedin_bot_token="li-token",
        # chat id unset
    )
    assert settings.bot_profile("linkedin") is None