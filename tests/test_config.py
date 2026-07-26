from __future__ import annotations

import os

import pytest

from app.config import Settings, configure_langsmith_env


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

    Stage A models + max_topics_per_run (P2.1 technical_rank), Tavily knobs
    (P2.3 tavily tools), verifier knobs (P2.4 verify_claim), and the
    style-profile paths (P3.2 load_style_profile), the Stage-B research model
    (P4.1 research-subagent), the Stage-B writer model + skills dir (P4.2
    writer-subagent), and now outputs_dir (P6.3 export_report) have consumers;
    the remaining ported knobs (deep-agent bounds, orchestration limits) stay
    deferred until their consumer PRs land."""
    settings = Settings(_env_file=None)
    for removed in (
        "deep_agent_enabled",
        "deep_agent_model",
        "deep_research_topic_concurrency",
        "adaptive_investigation_concurrency",
        "telegram_send_concurrency",
    ):
        assert not hasattr(settings, removed), f"{removed} should not be on Settings yet"


def test_style_profile_knobs_have_defaults() -> None:
    """P3.2 lands the style-profile paths with their consumer (load_style_profile).
    The seed profile and writing-samples dir default under data/."""
    settings = Settings(_env_file=None)
    assert settings.style_profile_file == "data/style_profile.json"
    assert settings.style_samples_dir == "data/style_samples"


def test_stage_b_research_model_has_default() -> None:
    """P4.1 lands the Stage-B research model with its consumer (research-subagent).
    Decision J standardizes every agent model on deepseek/deepseek-v4-flash; the
    knob stays separate so the research tier can be overridden independently."""
    settings = Settings(_env_file=None)
    assert settings.openrouter_stage_b_research_model == "deepseek/deepseek-v4-flash"


def test_stage_b_writer_knobs_have_defaults() -> None:
    """P4.2 lands the Stage-B writer model + skills dir with their consumer
    (writer-subagent). Decision J default is deepseek/deepseek-v4-flash; the
    writer knob is the one to upgrade first if voice quality warrants it."""
    settings = Settings(_env_file=None)
    assert settings.openrouter_stage_b_writer_model == "deepseek/deepseek-v4-flash"
    assert settings.skills_dir == "skills"


def test_verifier_knobs_have_defaults() -> None:
    """P2.4 lands the verifier model + bounds with their consumer. The verifier
    keeps its own knob (not a reuse of `openrouter_model`) so it can be pointed
    elsewhere without disturbing the news summarizer; the Decision J default is
    deepseek/deepseek-v4-flash. Ensemble / concurrency bounds default to
    single-model / 4-parallel."""
    settings = Settings(_env_file=None)
    assert settings.openrouter_verifier_model == "deepseek/deepseek-v4-flash"
    assert settings.openrouter_verifier_secondary_model is None
    assert settings.verification_concurrency == 4
    assert settings.verification_sources_per_topic == 3


def test_searxng_knobs_have_defaults() -> None:
    """Research search runs against a self-hosted SearXNG instance (keyless).
    An empty base URL means 'not configured' -> the client returns dry-run mock
    results, so nothing requires a running instance to iterate."""
    settings = Settings(_env_file=None)
    assert settings.searxng_base_url == ""
    assert settings.searxng_categories == "news"
    assert settings.searxng_language == "en"
    assert settings.searxng_http_concurrency == 6


def test_stage_a_knobs_have_defaults() -> None:
    """P2.1 lands the Stage A technical-ranker knobs with their consumer. The
    model defaults to deepseek/deepseek-v4-flash (Decision J standardizes every
    agent model on it); no .env needed for dry-run (the heuristic fallback fires
    when the OpenRouter key is absent)."""
    settings = Settings(_env_file=None)
    assert settings.openrouter_stage_a_model == "deepseek/deepseek-v4-flash"
    assert settings.openrouter_stage_a_secondary_model is None
    assert settings.max_topics_per_run == 5


def test_per_subagent_models_resolve_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """P7.3 acceptance: 'Models resolved from env; override test passes.' Each
    agent tier (Stage-A ranker, verifier, Stage-B research, Stage-B writer) is
    an independent env-driven knob. Overriding all four to *distinct* values and
    asserting each resolves to its own proves both that the config reads from
    env and that the knobs don't alias each other — so an operator can point one
    tier at a stronger model without touching the rest. (That an overridden knob
    then flows into the model builder is covered per tier by the subagent build
    tests.)"""
    overrides = {
        "OPENROUTER_STAGE_A_MODEL": "vendor/ranker-model",
        "OPENROUTER_VERIFIER_MODEL": "vendor/verifier-model",
        "OPENROUTER_STAGE_B_RESEARCH_MODEL": "vendor/research-model",
        "OPENROUTER_STAGE_B_WRITER_MODEL": "vendor/writer-model",
    }
    for key, value in overrides.items():
        monkeypatch.setenv(key, value)

    settings = Settings(_env_file=None)

    assert settings.openrouter_stage_a_model == "vendor/ranker-model"
    assert settings.openrouter_verifier_model == "vendor/verifier-model"
    assert settings.openrouter_stage_b_research_model == "vendor/research-model"
    assert settings.openrouter_stage_b_writer_model == "vendor/writer-model"


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


def test_configure_langsmith_env_clears_langsmiths_env_var_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``langsmith.utils.get_env_var`` — which ``tracing_is_enabled()`` and
    ``get_tracer_project()`` both call — is ``@lru_cache``d. Importing
    langgraph/deepagents (which every real process does before this function
    ever runs) can trigger that first, cached read while LANGSMITH_TRACING is
    still unset; setting the env var afterwards is then a no-op unless the
    cache is cleared too, which is what regressed here in the first place.

    This only spies on ``cache_clear`` rather than actually flipping global
    tracing state end-to-end: doing that inside the shared pytest process was
    tried and confirmed real — it left LangSmith's tracer enabled for every
    later test in the run, which then tried (and failed) to submit real
    traces to LangSmith using this test's fake key. Checking that our code
    calls the documented mitigation is the safe version of the same pin."""
    from langsmith.utils import get_env_var, get_tracer_project

    calls: list[str] = []
    monkeypatch.setattr(get_env_var, "cache_clear", lambda: calls.append("get_env_var"))
    monkeypatch.setattr(get_tracer_project, "cache_clear", lambda: calls.append("get_tracer_project"))

    # configure_langsmith_env sets these directly via os.environ[...] = ..., not
    # through monkeypatch, so register them here first to get an auto-revert.
    for key in ("LANGSMITH_TRACING", "LANGSMITH_API_KEY", "LANGSMITH_PROJECT"):
        monkeypatch.delenv(key, raising=False)

    settings = Settings(
        _env_file=None,
        langsmith_api_key="test-key",
        langsmith_project="test-project",
        langsmith_tracing=True,
    )
    configure_langsmith_env(settings)

    assert "get_env_var" in calls
    assert "get_tracer_project" in calls


def test_configure_langsmith_env_mirrors_endpoint_when_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """LANGSMITH_ENDPOINT matters for non-default (e.g. EU) workspaces — an
    API key issued for one region is rejected by the other. Left unset, the
    langsmith SDK's own default (the US endpoint) applies."""
    monkeypatch.delenv("LANGSMITH_ENDPOINT", raising=False)

    settings = Settings(_env_file=None, langsmith_endpoint="https://eu.api.smith.langchain.com")
    configure_langsmith_env(settings)

    assert os.environ["LANGSMITH_ENDPOINT"] == "https://eu.api.smith.langchain.com"


def test_configure_langsmith_env_leaves_endpoint_unset_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("LANGSMITH_ENDPOINT", raising=False)

    settings = Settings(_env_file=None)
    configure_langsmith_env(settings)

    assert "LANGSMITH_ENDPOINT" not in os.environ