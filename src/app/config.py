from __future__ import annotations

import os
from functools import lru_cache
from typing import Literal, get_args

from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

# Named Telegram destinations the process serves. Today there are exactly two
# (Decision C in architecture/ai-news.html): `news` keeps the daily digest,
# `linkedin` receives LinkedIn post proposals. The registry is closed so the
# delivery layer can fail loudly on an unknown profile instead of silently
# routing to the wrong chat (risk row in the plan).
BotName = Literal["news", "linkedin"]


def _resolve_first(*candidates: str | None) -> str | None:
    """Pick the first non-empty, stripped value from a chain of optional
    strings. Used for bot-profile fallback so the "is this configured?" check
    and the "which explicit value wins?" resolution share one definition —
    a whitespace-only token is treated as unset (consistent with is_complete)."""
    for value in candidates:
        if value is not None and value.strip():
            return value.strip()
    return None


class BotProfile(BaseModel):
    """A single Telegram destination: one bot token + one chat id. Named
    profiles live in settings so delivery routing is config-driven, not
    hard-coded by call site."""

    name: BotName
    token: str
    chat_id: str

    def is_complete(self) -> bool:
        return bool(self.token.strip()) and bool(self.chat_id.strip())


class Settings(BaseSettings):
    openrouter_api_key: str | None = None
    openrouter_model: str = "deepseek/deepseek-v4-flash"
    openrouter_reasoning_effort: str | None = "high"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_site_url: str | None = None
    openrouter_app_name: str = "AI News Agent"

    # Stage A technical-ranker model (Decision J: cheap, deterministic rank).
    # Landed with its consumer — the technical_rank tool (P2.1). The legacy
    # `openrouter_model` above stays the news pipeline's summarization model and
    # is NOT repurposed as Stage A.
    openrouter_stage_a_model: str = "openai/gpt-oss-120b"
    openrouter_stage_a_secondary_model: str | None = None

    # Cap on ranked topics the technical_rank tool writes per run. Landed with
    # its consumer (P2.1); the coordinator also reads this when delegating.
    max_topics_per_run: int = 5

    # Tavily research-evidence knobs. Landed with their consumer — the
    # tavily_search / tavily_extract tools (P2.3). Per Decision E, Tavily stays
    # *only* as per-topic research evidence; the LinkedIn agent's own Tavily/RSS
    # discovery path is dropped (discovery_* knobs are intentionally NOT ported).
    tavily_api_key: str | None = None
    tavily_base_url: str = "https://api.tavily.com"
    tavily_topic: str = "news"
    tavily_search_depth: str = "advanced"
    tavily_time_range: str = "day"
    tavily_http_concurrency: int = 6

    # Per-subagent model / deep-agent / orchestration knobs live with their
    # consumer (P3.1 linkedin-voice, P4.* subagents, P5.* coordinator). They are
    # intentionally NOT pre-declared here: a knob with no consumer is config
    # dressed as code, and landing it before the consumer exists pins defaults
    # that may turn out wrong once the real reader is built.

    # Telegram — legacy top-level credentials. Kept as the backward-compatible
    # source for the `news` bot profile so the existing digest `run` path and
    # TelegramClient (which still reads these fields directly) are unchanged
    # until P6.1 refactors delivery onto the registry.
    telegram_bot_token: str | None = None
    telegram_chat_id: str | None = None
    telegram_parse_mode: str = "HTML"

    # Named bot profiles. The `news` profile is backfilled from the legacy
    # top-level credentials above when these are unset (backward compat); the
    # `linkedin` profile is optional until that bot exists (P1.2 acceptance:
    # dry-run needs no new required keys).
    telegram_news_bot_token: str | None = None
    telegram_news_chat_id: str | None = None
    telegram_linkedin_bot_token: str | None = None
    telegram_linkedin_chat_id: str | None = None

    langsmith_api_key: str | None = None
    langsmith_project: str = "ai-news-agent"
    langsmith_tracing: bool = True

    sources_file: str = "data/news-sources.yaml"
    history_file: str = "data/delivery-history.json"
    history_retention_days: int = 14
    request_timeout_seconds: int = 20
    http_concurrency: int = 8
    max_feed_items_per_source: int = 50
    max_articles_per_run: int = 50
    max_articles_per_source: int = 3
    user_agent: str = "AINewsAgent/0.1"

    # Orchestrator filesystem convention (StateBackend lands in P5.1; this
    # locates the dir the fetch_curated_ai_news tool writes to). Other dirs
    # (`outputs_dir`, `style_profile_file`, `style_samples_dir`) live with
    # their consumer, not pre-declared.
    orchestrator_data_dir: str = "data/orchestrator"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    def missing_required_runtime_fields(self, dry_run: bool) -> list[str]:
        """Telegram credentials are only required for real runs; --dry-run never
        calls Telegram, so it's exempt from this check. For now this is keyed on
        the legacy top-level `news` fields (the digest path); the `linkedin`
        profile is validated lazily at delivery time, where a missing profile
        can be reported with the bot name that asked for it."""
        missing: list[str] = []

        if not dry_run:
            if not (self.telegram_bot_token or "").strip():
                missing.append("TELEGRAM_BOT_TOKEN")
            if not (self.telegram_chat_id or "").strip():
                missing.append("TELEGRAM_CHAT_ID")

        return missing

    def bot_profile(self, name: BotName) -> BotProfile | None:
        """Resolve a named Telegram bot profile, or None if the bot isn't
        configured yet.

        The `news` profile falls back to the legacy top-level credentials so a
        repo configured before bot-profile env vars existed keeps delivering the
        digest unchanged. An *explicit* profile env var wins over the fallback;
        that lets an operator split the news digest onto a dedicated bot later
        without touching the legacy fields. The `linkedin` profile has no
        fallback — it's optional until its bot exists (P1.2).

        Unknown names raise ValueError; the Literal type would already reject
        them statically, so this is defense-in-depth for `# type: ignore`
        callers."""
        if name not in get_args(BotName):
            raise ValueError(f"Unknown Telegram bot profile: {name!r}")

        if name == "news":
            token = _resolve_first(self.telegram_news_bot_token, self.telegram_bot_token)
            chat_id = _resolve_first(self.telegram_news_chat_id, self.telegram_chat_id)
        else:
            token = _resolve_first(self.telegram_linkedin_bot_token)
            chat_id = _resolve_first(self.telegram_linkedin_chat_id)

        if token is None or chat_id is None:
            return None
        return BotProfile(name=name, token=token, chat_id=chat_id)

    def bot_profiles(self) -> dict[BotName, BotProfile]:
        """All fully-configured bot profiles. Used by delivery tools to report
        which targets are available in a run (e.g. dry-run preview)."""
        return {
            name: profile for name in get_args(BotName) if (profile := self.bot_profile(name)) is not None
        }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def configure_langsmith_env(settings: Settings) -> None:
    """Mirror LangSmith settings into env vars: the langsmith SDK reads its config
    from the environment, not from values passed programmatically."""
    if settings.langsmith_api_key:
        os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key
    os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
    os.environ["LANGSMITH_TRACING"] = "true" if settings.langsmith_tracing else "false"