"""P4.1 — research-subagent spec + the shared OpenRouter model builder.

No live LLM: these pin the spec's shape (name, tools, model, prompt contract)
so the coordinator (P5) wires a well-formed subagent, and guard the brief-field
contract in the prompt against ResearchBrief drift.
"""
from __future__ import annotations

from langchain_openai import ChatOpenAI

from app.config import Settings
from app.orchestrator.models import build_openrouter_chat_model
from app.orchestrator.schemas import ResearchBrief
from app.orchestrator.subagents.research import (
    AUTHORED_BRIEF_FIELDS,
    RESEARCH_SUBAGENT_NAME,
    build_research_subagent,
)


def _settings(**kw: object) -> Settings:
    return Settings(_env_file=None, **kw)


# --------------------------------------------------------------------------- #
# model builder
# --------------------------------------------------------------------------- #


def test_model_builder_points_at_openrouter() -> None:
    model = build_openrouter_chat_model(_settings(), model="anthropic/claude-sonnet-4.5")
    assert isinstance(model, ChatOpenAI)
    assert model.model_name == "anthropic/claude-sonnet-4.5"
    assert "openrouter.ai" in str(model.openai_api_base)


def test_model_builder_attaches_attribution_headers_when_configured() -> None:
    model = build_openrouter_chat_model(
        _settings(openrouter_site_url="https://example.com", openrouter_app_name="Test App"),
        model="x/y",
    )
    headers = model.default_headers or {}
    assert headers.get("HTTP-Referer") == "https://example.com"
    assert headers.get("X-Title") == "Test App"


def test_model_builder_omits_headers_when_unset() -> None:
    # app_name defaults to a non-empty value, so unset the two knobs explicitly.
    model = build_openrouter_chat_model(
        _settings(openrouter_site_url=None, openrouter_app_name=""),
        model="x/y",
    )
    assert not (model.default_headers or {})


def test_model_builder_constructs_without_api_key() -> None:
    # dry-run / tests must be able to build the spec with no key.
    model = build_openrouter_chat_model(_settings(openrouter_api_key=None), model="x/y")
    assert isinstance(model, ChatOpenAI)


# --------------------------------------------------------------------------- #
# subagent spec
# --------------------------------------------------------------------------- #


def test_subagent_core_shape() -> None:
    sub = build_research_subagent(_settings())
    assert sub["name"] == RESEARCH_SUBAGENT_NAME
    assert sub["description"].strip()
    assert sub["system_prompt"].strip()


def test_subagent_wires_exactly_the_research_tools() -> None:
    sub = build_research_subagent(_settings())
    tool_names = {t.name for t in sub["tools"]}
    assert tool_names == {
        "fetch_article",
        "tavily_search",
        "web_extract",
        "verify_claim",
    }


def test_subagent_uses_stage_b_research_model() -> None:
    settings = _settings(openrouter_stage_b_research_model="anthropic/claude-opus-4.1")
    sub = build_research_subagent(settings)
    model = sub["model"]
    assert isinstance(model, ChatOpenAI)
    assert model.model_name == "anthropic/claude-opus-4.1"


def test_prompt_lists_every_authored_brief_field() -> None:
    """The prompt tells the model which fields to author; if ResearchBrief gains
    a field the subagent should write, the prompt must mention it or the model
    won't emit it."""
    prompt = build_research_subagent(_settings())["system_prompt"]
    for field in AUTHORED_BRIEF_FIELDS:
        assert field in prompt, f"prompt omits authored brief field {field!r}"


def test_authored_fields_exclude_every_verification_field() -> None:
    """verify_claim owns every verification_* field; the subagent must not
    pre-set any of them. Asserts the prefix rule holds for the whole schema, so
    a future verification_* field is excluded automatically."""
    verify_owned = [f for f in ResearchBrief.model_fields if f.startswith("verification_")]
    assert verify_owned  # sanity: the schema does have verify-owned fields
    for field in verify_owned:
        assert field not in AUTHORED_BRIEF_FIELDS
    # and the authored set is exactly the non-verification_ remainder
    assert AUTHORED_BRIEF_FIELDS == [
        f for f in ResearchBrief.model_fields if not f.startswith("verification_")
    ]


def test_prompt_states_the_file_contract_and_guardrails() -> None:
    prompt = build_research_subagent(_settings())["system_prompt"].lower()
    # writes the brief file, then verifies it
    assert "briefs/<topic_id>.json" in prompt
    assert "verify_claim" in prompt
    # anti-fabrication + single-topic guardrails
    assert "never invent" in prompt or "never fabricate" in prompt
    assert "one topic" in prompt
