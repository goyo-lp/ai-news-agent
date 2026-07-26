"""Shared OpenRouter chat-model construction for the coordinator and its
subagents.

Every agentic surface (the research subagent's Stage-B model, P4.1; the writer
subagent's Stage-B model, P4.2; the coordinator itself, P5) talks to the same
OpenRouter endpoint with the same attribution headers. The reference LinkedIn
agent built this ``ChatOpenAI`` inline inside each consumer, which meant the
base-url + ``HTTP-Referer`` / ``X-Title`` header protocol lived in several
places — a change to the endpoint contract would have to be made in each. This
module is the single place that knows how to turn a model id + Settings into a
live chat model, so that detail is encapsulated once.
"""
from __future__ import annotations

from typing import Any

from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from app.config import Settings
from app.orchestrator.usage import USAGE_CALLBACK_HANDLER

# ChatOpenAI refuses to construct with an empty api key (it enforces credentials
# eagerly, not at request time). Subagent specs must still build for dry-run and
# tests when no key is configured, so substitute a labeled placeholder when the
# real key is unset. It only affects construction — a real request with it would
# 401 loudly, which is the correct failure for "ran for real without a key".
_DRY_RUN_PLACEHOLDER_KEY = "dry-run-no-openrouter-key"


def build_openrouter_chat_model(settings: Settings, *, model: str) -> ChatOpenAI:
    """Construct a ``ChatOpenAI`` pointed at OpenRouter for ``model``.

    Attribution headers (``HTTP-Referer`` / ``X-Title``) are attached only when
    configured, matching OpenRouter's optional-but-recommended app-identification
    protocol. Safe to call with an unset ``OPENROUTER_API_KEY`` (dry-run /
    tests): a placeholder key lets construction succeed; only a real request
    would fail."""
    headers: dict[str, str] = {}
    if settings.openrouter_site_url:
        headers["HTTP-Referer"] = settings.openrouter_site_url
    if settings.openrouter_app_name:
        headers["X-Title"] = settings.openrouter_app_name

    kwargs: dict[str, Any] = {}
    if headers:
        kwargs["default_headers"] = headers

    return ChatOpenAI(
        model=model,
        api_key=SecretStr(settings.openrouter_api_key or _DRY_RUN_PLACEHOLDER_KEY),
        base_url=settings.openrouter_base_url,
        # Ask OpenRouter to always return the usage block (incl. cost) on every
        # agentic call. Without this the standard OpenAI-style response omits
        # cost AND token counts for some models — the 2026-07-25 production
        # trace showed token_usage=null on all 230 LLM runs, making spend
        # invisible in LangSmith and in the run-usage tracker. This matches
        # what the raw-httpx ranker/verifier already send.
        extra_body={"usage": {"include": True}},
        # Records token usage for every call into the active run (P7.2). The
        # handler is a no-op outside a track_run_usage block, so this is inert
        # in tests and dry runs.
        callbacks=[USAGE_CALLBACK_HANDLER],
        **kwargs,
    )
