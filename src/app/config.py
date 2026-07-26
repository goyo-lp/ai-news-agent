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

    # Stage A technical-ranker model. Landed with its consumer — the
    # technical_rank tool (P2.1). Decision J now standardizes every agent model
    # on deepseek/deepseek-v4-flash (the same model the news pipeline uses); the
    # per-tier knobs are kept as separate override points, not because the
    # defaults differ. The legacy `openrouter_model` above is still the news
    # pipeline's own summarization knob and is NOT repurposed as Stage A.
    openrouter_stage_a_model: str = "deepseek/deepseek-v4-flash"
    openrouter_stage_a_secondary_model: str | None = None

    # Cap on ranked topics the technical_rank tool writes per run. Landed with
    # its consumer (P2.1); the coordinator also reads this when delegating.
    max_topics_per_run: int = 5

    # SearXNG research-evidence knobs. Search runs against a self-hosted SearXNG
    # instance (keyless, no billing) — the operator points searxng_base_url at
    # their instance (see docker-compose.searxng.yml). Per Decision E this stays
    # *only* as per-topic research evidence; discovery is the RSS pipeline's job.
    # An empty base URL means "not configured" -> the client returns dry-run mock
    # results, so nothing requires a running instance to iterate.
    searxng_base_url: str = ""
    searxng_categories: str = "news"
    searxng_language: str = "en"
    searxng_http_concurrency: int = 6

    # web_search circuit breaker. Landed with its consumer — the web_search
    # tool. When a *configured* SearXNG instance returns empty/failed results
    # this many times in a row, the tool opens the circuit and tells the
    # researcher (in the structured reason) to stop searching and pivot to
    # web_extract on the cluster's supporting_urls. The 2026-07-25 trace
    # showed 57 web_search calls in one run against an instance that was
    # silently returning zero results — 11+ futile searches per topic.
    searxng_empty_circuit_breaker: int = 3

    # Verifier model + bounds. Landed with their consumer — the verify_claim
    # tool (P2.4). Kept as its own knob (not a reuse of `openrouter_model`) so
    # the verifier can be pointed at a different model without disturbing the
    # news summarizer; the Decision J default is deepseek/deepseek-v4-flash. The
    # optional secondary drives the ensemble path (strictest-wins reconciliation).
    openrouter_verifier_model: str = "deepseek/deepseek-v4-flash"
    openrouter_verifier_secondary_model: str | None = None
    verification_concurrency: int = 4
    verification_sources_per_topic: int = 3

    # verify_claim churn guards. Landed with their consumer — the verify_claim
    # tool. The 2026-07-25 production trace showed the research subagent
    # re-verifying one brief up to 16x (with single calls taking 80-94s) when
    # corroboration was thin: the tool had no attempt budget and no wall-clock
    # bound. `verification_max_attempts` caps tool invocations per topic_id per
    # process (the soften-and-reverify loop gets one retry); the timeout bounds
    # one verification call.
    verification_max_attempts: int = 2
    verification_timeout_seconds: int = 90

    # Evidence floor — the minimum verification strength a brief must reach
    # before it may become a draft / delivery. Landed with their consumers:
    # the submit_draft tool (write-time), deliver_telegram (send-time), the
    # deterministic spine (delegation-time), export_report (reporting). A brief
    # passes when it is `verified`, or `partially_verified` with confidence >=
    # `verification_min_confidence` AND at least `verification_min_citations`
    # citations. Anything weaker is skipped — the 2026-07-25 trace shipped
    # single-source confidence-0.3 posts, which is exactly what this blocks.
    verification_min_confidence: float = 0.5
    verification_min_citations: int = 2

    # Stage-B research subagent model (the research subagent decides whether to
    # fetch the real article, what to verify, and what's technically new).
    # Landed with its consumer — the research subagent (P4.1). Its own knob so
    # the research tier can be upgraded independently; the Decision J default is
    # deepseek/deepseek-v4-flash.
    openrouter_stage_b_research_model: str = "deepseek/deepseek-v4-flash"

    # Stage-B writer subagent model. Landed with its consumer — the writer
    # subagent (P4.2). Voice quality is the product, so this is the knob to
    # upgrade first if a stronger model is warranted; the Decision J default is
    # deepseek/deepseek-v4-flash.
    openrouter_stage_b_writer_model: str = "deepseek/deepseek-v4-flash"

    # Editor relevance veto (P-spine). One batched OpenRouter call at selection
    # time that drops candidates which are not high-signal AI business/product
    # stories (e.g. pure dev-tooling release notes, OS platform tweaks) before
    # any research spend lands. The 2026-07-25 run shipped an Android-ADB story
    # and a Ruff release note from an "AI news" agent — this is the topical
    # filter that was missing. Falls back to ``openrouter_model`` when unset;
    # disabled = deterministic selection only (no LLM editorial judgment).
    openrouter_editor_model: str | None = None
    editor_veto_enabled: bool = True

    # Deterministic spine wall-clock bounds (the spine replaced the LLM
    # coordinator as the default propose path). Each research/writer subagent
    # invocation is wrapped in asyncio.wait_for with these budgets — the
    # per-topic timeout the research.py docstring always promised but the
    # LLM-coordinator path never implemented. The 2026-07-25 trace showed one
    # research task running 261s; a hung subagent used to hang the whole run.
    #
    # 240s, not the original 420s: in the 2026-07-26 trace two of five topics
    # burned the full 420s. One had already written its verified brief 86s
    # before the clock killed it and spent the remainder re-fetching URLs it
    # had already fetched; the other produced no brief at all. A researcher
    # that has not written a brief in four minutes is looping, not thinking.
    research_task_timeout_seconds: int = 240
    writer_task_timeout_seconds: int = 300

    # Draft provenance signing key (HMAC-SHA256). The submit_draft tool signs
    # every draft it writes; export_report and deliver_telegram verify the
    # signature so a draft authored outside the writer tool (e.g. an LLM
    # coordinator writing drafts/<post_id>.json directly via write_file — the
    # exact failure observed in the 2026-07-25 production trace) is refused.
    # Empty derives a key from OPENROUTER_API_KEY / TELEGRAM_BOT_TOKEN so
    # existing installs are signed without a new secret; set explicitly to
    # rotate independently.
    draft_signing_key: str = ""

    # Skills source directory (contains linkedin-voice/SKILL.md). The writer
    # subagent (P4.2) passes this to SkillsMiddleware. Landed with its consumer.
    skills_dir: str = "skills"

    # Style-profile loader knobs. Landed with their consumer — the
    # load_style_profile tool (P3.2). `style_profile_file` is the seed profile
    # the loader reads by default; `style_samples_dir` holds the author's
    # writing samples the loader rebuilds from on demand.
    style_profile_file: str = "data/style_profile.json"
    style_samples_dir: str = "data/style_samples"

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
    # Unset uses the langsmith SDK's own default (the US endpoint). Required
    # for EU-region workspaces — an EU-issued API key is rejected (403) by
    # the US endpoint and vice versa, so this has to reach a real workspace.
    langsmith_endpoint: str | None = None
    # Runs per ingest batch. The SDK's default of 100 produced 4-11MB compressed
    # multipart uploads in the 2026-07-26 run, every one of which died on a
    # socket write timeout — so the run was fully traced locally and none of it
    # reached the workspace. Smaller batches upload rather than time out; the
    # cost is more requests, which are cheap next to losing the trace.
    langsmith_batch_ingest_size_limit: int = 20

    sources_file: str = "data/news-sources.yaml"
    history_file: str = "data/delivery-history.json"
    history_retention_days: int = 14
    # Wall-clock bound on ONE OpenRouter attempt. This is enforced with
    # asyncio.wait_for, not by handing the value to httpx: httpx's read timeout
    # is per-read, and OpenRouter drip-feeds keep-alive bytes while a
    # non-streaming completion generates, so every byte resets the clock and an
    # httpx-only timeout never fires. See openrouter_client._request_completion.
    request_timeout_seconds: int = 20
    # Total budget for the optional LLM re-rank in rank_node, across all
    # attempts. Exceeding it drops the blend and keeps the deterministic
    # ranking; it must stay well under the rank node's graph-level timeout.
    llm_rerank_timeout_seconds: int = 25
    http_concurrency: int = 8
    max_feed_items_per_source: int = 50
    max_articles_per_run: int = 50
    max_articles_per_source: int = 3
    user_agent: str = "AINewsAgent/0.1"

    # Orchestrator filesystem convention — the dir on real disk where the
    # deterministic tools (fetch_curated_ai_news, technical_rank,
    # verify_claim, quality_gate, load_style_profile, fetch_article) write
    # structured artifacts via stdlib AND where deepagents' built-in
    # write_file / read_file (used by the research/writer subagents) route
    # through the coordinator-mounted backend. P5.1 mounts a
    # `FilesystemBackend` rooted here — NOT a `StateBackend`, which would
    # isolate subagent write_file payloads to LangGraph state and break the
    # cross-tool visibility contract documented in subagents/research.py and
    # subagents/writer.py. Other dirs (`style_profile_file`,
    # `style_samples_dir`) live with their consumer, not pre-declared.
    orchestrator_data_dir: str = "data/orchestrator"

    # Run-output directory — where the export_report tool (P6.3) writes the
    # per-run export bundle (posts.md + run_report.json + briefs.json) under
    # ``<outputs_dir>/<YYYY-MM-DD>/``. Landed with its consumer (P6.3).
    # Defaults to ``data/outputs`` alongside the orchestrator data dir so a
    # fresh checkout produces a tree where both are siblings under ``data/``.
    outputs_dir: str = "data/outputs"

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
    from the environment, not from values passed programmatically.

    ``langsmith.utils.get_env_var`` (which ``tracing_is_enabled()`` and
    ``get_tracer_project()`` both call) is ``@lru_cache``d. By the time this
    function runs, importing ``langgraph``/``deepagents`` earlier in the
    process has usually already called it once — caching a "no env var set"
    result from before these lines ran. Without clearing that cache, setting
    the env vars here has no effect: LangSmith silently stays disabled for
    the rest of the process even with a real API key and TRACING=true."""
    if settings.langsmith_api_key:
        os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key
    os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
    os.environ["LANGSMITH_TRACING"] = "true" if settings.langsmith_tracing else "false"
    if settings.langsmith_endpoint:
        os.environ["LANGSMITH_ENDPOINT"] = settings.langsmith_endpoint
    os.environ["LANGSMITH_BATCH_INGEST_SIZE_LIMIT"] = str(
        settings.langsmith_batch_ingest_size_limit
    )

    from langsmith.utils import get_env_var, get_tracer_project

    get_env_var.cache_clear()  # type: ignore[attr-defined]
    get_tracer_project.cache_clear()