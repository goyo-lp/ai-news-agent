from __future__ import annotations

import os
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    openrouter_api_key: str | None = None
    openrouter_model: str = "openai/gpt-oss-20b"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_site_url: str | None = None
    openrouter_app_name: str = "AI News Agent"

    telegram_bot_token: str | None = None
    telegram_chat_id: str | None = None
    telegram_parse_mode: str = "HTML"

    langsmith_api_key: str | None = None
    langsmith_project: str = "ai-news-agent"
    langsmith_tracing: bool = True
    langgraphics_enabled: bool = True
    langgraphics_open_browser: bool = True
    langgraphics_host: str = "localhost"
    langgraphics_port: int = 8764
    langgraphics_ws_port: int = 8765
    langgraphics_direction: str = "TB"
    langgraphics_mode: str = "auto"
    langgraphics_inspect: str = "off"
    langgraphics_theme: str = "system"

    sources_file: str = "data/news-sources.yaml"
    request_timeout_seconds: int = 20
    http_concurrency: int = 8
    max_feed_items_per_source: int = 50
    max_articles_per_run: int = 50
    user_agent: str = "AINewsAgent/0.1"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    def missing_required_runtime_fields(self, dry_run: bool) -> list[str]:
        missing: list[str] = []

        if not dry_run:
            if not (self.telegram_bot_token or "").strip():
                missing.append("TELEGRAM_BOT_TOKEN")
            if not (self.telegram_chat_id or "").strip():
                missing.append("TELEGRAM_CHAT_ID")

        return missing


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def configure_langsmith_env(settings: Settings) -> None:
    if settings.langsmith_api_key:
        os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key
    os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
    os.environ["LANGSMITH_TRACING"] = "true" if settings.langsmith_tracing else "false"
