"""Environment-backed application configuration."""

from pathlib import Path
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings loaded from environment variables or a local .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    ai_news_database_path: Path = Path(".local/ai-news-agent.db")
    ai_news_log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    ai_news_log_format: Literal["console", "json"] = "json"
    ai_news_environment: str = "local"

    langsmith_tracing: bool = False
    langsmith_project: str = "ai-news-agent-local"
    langsmith_api_key: SecretStr | None = None
    langsmith_workspace_id: str | None = None
    langsmith_endpoint: str | None = None

    @property
    def tracing_ready(self) -> bool:
        """Return whether live tracing has both opt-in and credentials."""

        return self.langsmith_tracing and self.langsmith_api_key is not None
