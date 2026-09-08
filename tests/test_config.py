from pathlib import Path

from ai_news_agent.config import Settings


def test_settings_load_environment(monkeypatch) -> None:
    monkeypatch.setenv("AI_NEWS_DATABASE_PATH", "/tmp/ai-news-test.db")
    monkeypatch.setenv("AI_NEWS_LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "test-secret-value")

    settings = Settings(_env_file=None)

    assert settings.ai_news_database_path == Path("/tmp/ai-news-test.db")
    assert settings.ai_news_log_level == "DEBUG"
    assert settings.tracing_ready is True
    assert "test-secret-value" not in repr(settings)


def test_tracing_is_disabled_by_default() -> None:
    settings = Settings(_env_file=None)

    assert settings.langsmith_tracing is False
    assert settings.tracing_ready is False


def test_judge_uses_openrouter_by_default(monkeypatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    settings = Settings(_env_file=None)

    assert settings.judge_ready is False
    assert settings.openrouter_model == "nvidia/nemotron-3-super-120b-a12b:free"
    assert settings.openrouter_base_url == "https://openrouter.ai/api/v1"


def test_judge_ready_with_key(monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-judge-key")

    settings = Settings(_env_file=None)

    assert settings.judge_ready is True
    assert "test-judge-key" not in repr(settings)
