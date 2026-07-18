from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "ai-linked-imposting-agent"

    tavily_api_key: str | None = None
    tavily_base_url: str = "https://api.tavily.com"
    tavily_topic: str = "news"
    tavily_search_depth: str = "advanced"
    tavily_time_range: str = "day"

    openrouter_api_key: str | None = None
    openrouter_stage_a_model: str = "openai/gpt-oss-120b"
    openrouter_stage_a_secondary_model: str | None = None
    openrouter_model: str = "anthropic/claude-haiku-4.5"
    openrouter_post_secondary_model: str | None = None
    openrouter_verifier_secondary_model: str | None = None
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_site_url: str | None = None
    openrouter_app_name: str = "ai-linked-imposting-agent"
    deep_agent_enabled: bool = True
    deep_agent_model: str = "anthropic/claude-haiku-4.5"
    deep_agent_timeout_seconds: int = 75
    deep_agent_max_evidence_sources: int = 5
    deep_agent_max_evidence_chars: int = 1400

    telegram_bot_token: str | None = None
    telegram_chat_id: str | None = None
    telegram_parse_mode: str = "HTML"

    trusted_sources_file: str = "data/trusted-sources.yaml"
    sources_file: str = "data/news-sources.yaml"
    style_samples_dir: str = "style_samples"
    style_profile_file: str = "data/style_profile.json"
    outputs_dir: str = "outputs"

    discovery_hours_back: int = 48
    discovery_max_results_per_query: int = 8
    discovery_max_candidates: int = 60
    max_topics_per_run: int = 5
    seed_articles_per_run: int = 50
    max_feed_items_per_source: int = 50
    rss_http_concurrency: int = 8
    tavily_http_concurrency: int = 6
    verification_sources_per_topic: int = 3
    deep_research_topic_concurrency: int = 4
    adaptive_investigation_concurrency: int = 3
    verification_concurrency: int = 4
    telegram_send_concurrency: int = 3
    pipeline_batch_concurrency: int = 2

    request_timeout_seconds: int = 20

    langsmith_api_key: str | None = None
    langsmith_project: str = "ai-linked-imposting-agent"
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

    query_templates: list[str] = Field(
        default_factory=lambda: [
            "latest ai model launches and releases",
            "new ai agents platform announcements",
            "enterprise ai deployment announcements",
            "ai infrastructure chips inference breakthroughs",
            "ai startup funding and acquisitions",
            "multimodal ai research breakthroughs",
            "open source ai model release news",
            "ai safety and evaluation benchmark updates",
            "developer tools for ai application building",
            "new methods in llm training and reasoning",
        ]
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    def missing_required_runtime_fields(self, dry_run: bool) -> list[str]:
        missing: list[str] = []
        if not dry_run:
            if not (self.tavily_api_key or "").strip():
                missing.append("TAVILY_API_KEY")
            if not (self.openrouter_api_key or "").strip():
                missing.append("OPENROUTER_API_KEY")
            if not (self.telegram_bot_token or "").strip():
                missing.append("TELEGRAM_BOT_TOKEN")
            if not (self.telegram_chat_id or "").strip():
                missing.append("TELEGRAM_CHAT_ID")
        return missing

    @property
    def trusted_sources_path(self) -> Path:
        return Path(self.trusted_sources_file)

    @property
    def sources_path(self) -> Path:
        return Path(self.sources_file)

    @property
    def style_profile_path(self) -> Path:
        return Path(self.style_profile_file)

    @property
    def style_samples_path(self) -> Path:
        return Path(self.style_samples_dir)

    @property
    def outputs_path(self) -> Path:
        return Path(self.outputs_dir)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def configure_langsmith_env(settings: Settings) -> None:
    if settings.langsmith_api_key:
        os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key
    os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
    os.environ["LANGSMITH_TRACING"] = "true" if settings.langsmith_tracing else "false"
