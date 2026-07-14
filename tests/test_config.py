from __future__ import annotations

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