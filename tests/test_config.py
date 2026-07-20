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


def test_orchestrator_data_dir_default() -> None:
    """The single orchestrator filesystem knob that already has a consumer
    (the fetch_curated_ai_news tool) has a stable default. Other ported knobs
    (Stage A/B models, Tavily, deep-agent bounds, orchestration limits) land
    with their consumer — declaring them now would pin defaults no reader
    validates."""
    settings = Settings(_env_file=None)
    assert settings.orchestrator_data_dir == "data/orchestrator"


def test_settings_rejects_unknown_ported_knobs() -> None:
    """A knob with no consumer is config dressed as code — the ported LinkedIn
    knobs must NOT silently exist on Settings before their consumer ships. This
    pins the contract so a future PR landing a consumer cannot accidentally let
    extra pre-declared fields slip in.

    Stage A models and `max_topics_per_run` now have a consumer (the
    technical_rank tool, P2.1); the remaining ported knobs (Stage B, Tavily,
    deep-agent bounds, orchestration limits) stay deferred until their consumer
    PRs land."""
    settings = Settings(_env_file=None)
    for removed in (
        "openrouter_stage_b_research_model",
        "openrouter_stage_b_writer_model",
        "tavily_api_key",
        "tavily_base_url",
        "deep_agent_enabled",
        "deep_agent_model",
        "deep_research_topic_concurrency",
        "adaptive_investigation_concurrency",
        "telegram_send_concurrency",
        "outputs_dir",
        "style_profile_file",
        "style_samples_dir",
    ):
        assert not hasattr(settings, removed), f"{removed} should not be on Settings yet"


def test_stage_a_knobs_have_defaults() -> None:
    """P2.1 lands the Stage A technical-ranker knobs with their consumer. They
    default to the cheap/deterministic model from Decision J; no .env needed for
    dry-run (the heuristic fallback fires when the OpenRouter key is absent)."""
    settings = Settings(_env_file=None)
    assert settings.openrouter_stage_a_model == "openai/gpt-oss-120b"
    assert settings.openrouter_stage_a_secondary_model is None
    assert settings.max_topics_per_run == 5


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


def test_news_bot_profile_whitespace_token_falls_back_to_legacy() -> None:
    """A whitespace-only explicit profile token is treated as unset so the
    legacy fallback still applies — the fallback and the completeness check
    share one stripped definition (the bug a truthy `or` would silently drop)."""
    settings = Settings(
        _env_file=None,
        telegram_bot_token="legacy-token",
        telegram_chat_id="legacy-chat",
        telegram_news_bot_token="   ",  # whitespace -> not set
        telegram_news_chat_id=None,
    )
    profile = settings.bot_profile("news")
    assert profile is not None
    assert profile.token == "legacy-token"
    assert profile.chat_id == "legacy-chat"
    assert profile.is_complete() is True


def test_news_bot_profile_whitespace_eveywhere_is_not_configured() -> None:
    """When every candidate (explicit + legacy) is whitespace-only, the profile
    is None — not a BotProfile carrying empty strings."""
    settings = Settings(
        _env_file=None,
        telegram_bot_token="   ",
        telegram_chat_id="   ",
    )
    assert settings.bot_profile("news") is None