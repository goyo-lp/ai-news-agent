"""P5.2 — coordinator deep-agent build test.

Pins the wiring contract documented in ``agent.py``'s module docstring:
the right backend, the three coordinator-side tools, both P4 subagents, no
skills, no system_prompt (P5.3 owns it). Monkeypatching
``create_deep_agent`` (the deepagents entry point) lets the tests
introspect the kwargs ``build_coordinator_agent`` feeds through without
reaching into deepagent internals that could shift between SDK versions;
the real path is exercised by ``test_build_returns_compiled_state_graph_real_path``
+ ``test_dry_run_invoke_completes_without_error``.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.graph.state import CompiledStateGraph

from app.config import Settings
from app.orchestrator import agent as agent_module
from app.orchestrator.agent import COORDINATOR_AGENT_NAME, build_coordinator_agent
from app.orchestrator.subagents.research import RESEARCH_SUBAGENT_NAME
from app.orchestrator.subagents.writer import WRITER_SUBAGENT_NAME


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        orchestrator_data_dir=str(tmp_path),
        # empty key: the fake model is the path actually exercised, so the
        # placeholder-key behavior of build_openrouter_chat_model doesn't
        # matter — we just want a green build() without touching the lru_cache.
        openrouter_api_key="",
    )


class _FakeDoneModel(BaseChatModel):
    """Single-response fake chat model for dry-run build / invoke tests.

    Overrides ``bind_tools`` to return ``self`` — deepagents' middleware stack
    calls ``model.bind_tools(...)`` on construction; without this override the
    default ``BaseChatModel.bind_tools`` raises ``NotImplementedError`` and
    the agent never builds. The fake never emits tool calls (only an
    AIMessage with content), so the tool dispatch branch is never hit.
    """

    response: str = "Done."

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=self.response))]
        )

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        return self._generate(messages, stop, run_manager, **kwargs)

    def bind_tools(self, tools: Any, **kwargs: Any) -> "_FakeDoneModel":  # type: ignore[override]
        return self

    @property
    def _llm_type(self) -> str:
        return "fake-done"


def _capture_create_deep_agent(captured: dict[str, Any]) -> Any:
    """Produce a stand-in for ``create_deep_agent`` that records its kwargs
    and returns a sentinel object — so the wiring test can assert what
    ``build_coordinator_agent`` fed through without invoking the real SDK
    (or its middleware stack). The sentinel exposes a ``name`` attribute so
    the agent-name assertion holds."""

    class _StubGraph:
        def __init__(self, name: str | None) -> None:
            self.name = name

    def _stub(**kwargs: Any) -> Any:
        captured.update(kwargs)
        return _StubGraph(name=kwargs.get("name"))

    return _stub


# --------------------------------------------------------------------------- #
# Build contract (monkeypatches out create_deep_agent so we can introspect
# exactly what build_coordinator_agent feeds through)
# --------------------------------------------------------------------------- #


def test_build_invokes_create_deep_agent_with_expected_shape(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The factory passes the expected kwargs to ``create_deep_agent`` —
    a backend (the FilesystemBackend from P5.1), three coordinator-side tools,
    both P4 subagents, no skills, no system_prompt (P5.3 owns it), the
    coordinator name. Pinning the kwargs catches a silent rename / missing
    piece before it surfaces as a runtime failure once deepagents is updated."""
    captured: dict[str, Any] = {}
    monkeypatch.setattr(agent_module, "create_deep_agent", _capture_create_deep_agent(captured))

    build_coordinator_agent(_settings(tmp_path), model=_FakeDoneModel())

    # Tools: coordinator-side only.
    tool_names = [getattr(t, "name", None) for t in captured.get("tools", [])]
    assert "fetch_curated_ai_news" in tool_names
    assert "technical_rank" in tool_names
    assert "quality_gate" in tool_names
    # Research/writer-owned tools explicitly NOT promoted to coordinator level.
    assert "fetch_article" not in tool_names
    assert "web_search" not in tool_names
    assert "web_extract" not in tool_names
    assert "verify_claim" not in tool_names

    # Subagents: both P4 subagents under their canonical names. deepagents'
    # SubAgent spec is dict-shaped (item access), per test_research_subagent.py
    # — same convention; don't reach for `.name` attribute access.
    sub_names = [s["name"] for s in captured.get("subagents", [])]
    assert RESEARCH_SUBAGENT_NAME in sub_names
    assert WRITER_SUBAGENT_NAME in sub_names

    # Backend: the FilesystemBackend P5.1 builds.
    from deepagents.backends.filesystem import FilesystemBackend
    backend = captured.get("backend")
    assert isinstance(backend, FilesystemBackend)
    assert Path(backend.cwd).resolve() == tmp_path.resolve()

    # Other wiring choices this PR owns:
    # P5.3 wires COORDINATOR_SYSTEM_PROMPT (built from settings, not the
    # module-level singleton); pin that it landed as a non-None string that
    # carries the run-order + max_topics guardrail the model reads.
    system_prompt = captured.get("system_prompt")
    assert isinstance(system_prompt, str)
    assert "PIPELINE" in system_prompt
    assert "max_topics_per_run" in system_prompt
    assert captured.get("skills") is None  # the linkedin-voice skill is writer-subagent-owned
    assert captured.get("name") == COORDINATOR_AGENT_NAME


def test_coordinator_subagent_tool_overlap_is_intentional(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The coordinator's ``quality_gate`` and the writer subagent's own
    ``quality_gate`` share the literal name. That's intentional, not
    accidental: the writer runs ``quality_gate`` inside its own context to
    gate its draft (closed-loop retry); the coordinator reads the verdict
    file directly from disk at delivery time and *also* exposes
    ``quality_gate`` as its own tool so it can re-gate at surface time
    without delegating.

    Pinning the intentional overlap catches a future "de-duplicate" change
    that would remove either side silently; pinning the *non-overlap* of
    everything else (fetch_curated_ai_news + technical_rank exclusive to
    the coordinator; fetch_article / web_search / web_extract / verify_claim
    exclusive to research; quality_gate shared by coordinator + writer)
    documents the matrix so a future tool addition or promotion surfaces
    here before it ships a surprise collision at runtime."""
    captured: dict[str, Any] = {}
    monkeypatch.setattr(agent_module, "create_deep_agent", _capture_create_deep_agent(captured))
    build_coordinator_agent(_settings(tmp_path), model=_FakeDoneModel())

    coord_tool_names = {getattr(t, "name", None) for t in captured["tools"]}
    sub_specs = {s["name"]: s for s in captured["subagents"]}
    research_tools = {
        getattr(t, "name", None) for t in sub_specs[RESEARCH_SUBAGENT_NAME]["tools"]
    }
    writer_tools = {
        getattr(t, "name", None) for t in sub_specs[WRITER_SUBAGENT_NAME]["tools"]
    }

    # Quality gate is intentionally shared by coordinator + writer.
    assert "quality_gate" in coord_tool_names
    assert "quality_gate" in writer_tools
    # Research tools live ONLY on the research subagent — never the coordinator.
    assert {"fetch_article", "web_search", "web_extract", "verify_claim"} <= research_tools
    assert {"fetch_article", "web_search", "web_extract", "verify_claim"}.isdisjoint(
        coord_tool_names
    )
    # fetch_curated_ai_news + technical_rank live ONLY on the coordinator —
    # never the subagents.
    assert "fetch_curated_ai_news" in coord_tool_names
    assert "technical_rank" in coord_tool_names
    assert {"fetch_curated_ai_news", "technical_rank"}.isdisjoint(research_tools)
    assert {"fetch_curated_ai_news", "technical_rank"}.isdisjoint(writer_tools)
    # The writer carries submit_draft (the ONLY sanctioned draft-creation
    # path — provenance-signed, evidence-floored) + quality_gate (its
    # closed-loop gate tool). submit_draft is deliberately NOT promoted to
    # the coordinator: draft creation is a writer-owned choke point (the
    # 2026-07-25 trace showed the coordinator self-authoring drafts when it
    # had write access).
    assert writer_tools == {"quality_gate", "submit_draft"}
    assert "submit_draft" not in coord_tool_names
    assert "submit_draft" not in research_tools


def test_build_returns_compiled_state_graph_real_path(tmp_path: Path) -> None:
    """The real (un-monkeypatched) ``create_deep_agent`` returns a
    ``CompiledStateGraph``. Sanity-build so a deepagents version bump that
    changes the return type surfaces here before downstream callers rely on
    it."""
    agent = build_coordinator_agent(_settings(tmp_path), model=_FakeDoneModel())
    assert isinstance(agent, CompiledStateGraph)


def test_agent_name_carried_through(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The compiled graph carries the coordinator name — observability + tracing
    (P7) will read it. Pin so a future rename surfaces here, not as a
    surprise in tracing output."""
    captured: dict[str, Any] = {}
    monkeypatch.setattr(agent_module, "create_deep_agent", _capture_create_deep_agent(captured))
    build_coordinator_agent(_settings(tmp_path), model=_FakeDoneModel())
    assert captured.get("name") == "ai-news-coordinator"
    assert COORDINATOR_AGENT_NAME == "ai-news-coordinator"


def test_subagents_built_from_same_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The research/writer subagent factories run with the *same* Settings
    instance ``build_coordinator_agent`` resolved, so they share one config
    (backend root, OpenRouter key, data dir, model defaults). Catches a
    future ``build_*_subagent(get_settings())`` re-resolution inside the
    factory that would silently split config (a tool seeing a different
    data_dir than the subagent — principle #3 violation just waiting to
    happen)."""
    s_passed = _settings(tmp_path)
    captured_settings: list[Settings] = []

    real_research = agent_module.build_research_subagent
    real_writer = agent_module.build_writer_subagent

    def _spy_research(settings: Any = None, **_: Any) -> Any:
        captured_settings.append(settings)
        return real_research(settings)

    def _spy_writer(settings: Any = None, **_: Any) -> Any:
        captured_settings.append(settings)
        return real_writer(settings)

    monkeypatch.setattr(agent_module, "build_research_subagent", _spy_research)
    monkeypatch.setattr(agent_module, "build_writer_subagent", _spy_writer)

    captured: dict[str, Any] = {}
    monkeypatch.setattr(agent_module, "create_deep_agent", _capture_create_deep_agent(captured))
    build_coordinator_agent(s_passed, model=_FakeDoneModel())

    assert len(captured_settings) == 2
    assert all(s is s_passed for s in captured_settings), (
        "subagent builders must receive the exact Settings instance the factory got; "
        "re-resolution would split config between subagents and the coordinator."
    )


def test_tools_built_from_same_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Same shared-Settings contract for the tool factories: the three
    coordinator-side tools are built with the exact Settings instance the
    factory got, not re-resolved. ``fetch_curated_ai_news`` reads the same
    ``orchestrator_data_dir`` ``build_orchestrator_backend`` rooted at — pin
    that both sides see one config."""
    s_passed = _settings(tmp_path)
    captured_settings: list[Settings] = []

    real_news = agent_module.build_fetch_curated_ai_news_tool
    real_rank = agent_module.build_technical_rank_tool
    real_quality = agent_module.build_quality_gate_tool

    def _spy(factory: Any, name: str) -> Any:
        def _wrapped(settings: Any = None, **_: Any) -> Any:
            captured_settings.append(settings)
            return factory(settings)
        return _wrapped

    monkeypatch.setattr(agent_module, "build_fetch_curated_ai_news_tool", _spy(real_news, "news"))
    monkeypatch.setattr(agent_module, "build_technical_rank_tool", _spy(real_rank, "rank"))
    monkeypatch.setattr(agent_module, "build_quality_gate_tool", _spy(real_quality, "quality"))

    captured: dict[str, Any] = {}
    monkeypatch.setattr(agent_module, "create_deep_agent", _capture_create_deep_agent(captured))
    build_coordinator_agent(s_passed, model=_FakeDoneModel())

    assert len(captured_settings) == 3
    assert all(s is s_passed for s in captured_settings)


def test_settings_lazy_resolution_when_not_supplied(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``settings or get_settings()`` is the established lazy seam (every
    tool factory does it, state.py does it). When settings=None, the factory
    falls back to the lru_cache. Pin via env var so an accidentally-passed
    ``None`` doesn't silently use the *real* repo's data dir inside tests."""
    from app.config import get_settings

    monkeypatch.setenv("ORCHESTRATOR_DATA_DIR", str(tmp_path))
    captured: dict[str, Any] = {}
    monkeypatch.setattr(agent_module, "create_deep_agent", _capture_create_deep_agent(captured))
    get_settings.cache_clear()
    try:
        build_coordinator_agent(model=_FakeDoneModel())
        backend = captured.get("backend")
        from deepagents.backends.filesystem import FilesystemBackend
        assert isinstance(backend, FilesystemBackend)
        assert Path(backend.cwd).resolve() == tmp_path.resolve()
    finally:
        get_settings.cache_clear()
        monkeypatch.delenv("ORCHESTRATOR_DATA_DIR", raising=False)


def test_default_model_knob_landed_with_consumer(tmp_path: Path) -> None:
    """Per the repo's pinned config policy, the ``openrouter_coordinator_model``
    knob lands with its consumer (``build_coordinator_agent``). Pin its
    existence + default so a future removal / rename of the knob surfaces
    here as a consumer-without-config failure."""
    s = _settings(tmp_path)
    assert s.openrouter_coordinator_model == "deepseek/deepseek-v4-flash"


def test_production_model_path_uses_coordinator_knob(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When no ``model`` is supplied (the production path), the factory calls
    ``build_openrouter_chat_model(settings, model=settings.openrouter_coordinator_model)``.
    Pin by monkeypatching the model builder, setting a sentinel model id on
    Settings, and asserting the builder was called with that id."""
    s = _settings(tmp_path)
    # Use a sentinel model id we can scan for.
    s.openrouter_coordinator_model = "coordinator-sentinel-model"
    captured_model_args: list[str] = []

    def _spy_model(settings: Any, *, model: str) -> Any:
        captured_model_args.append(model)
        return _FakeDoneModel()  # return a real fake so create_deep_agent still builds

    monkeypatch.setattr(agent_module, "build_openrouter_chat_model", _spy_model)
    captured_create: dict[str, Any] = {}
    monkeypatch.setattr(agent_module, "create_deep_agent", _capture_create_deep_agent(captured_create))

    build_coordinator_agent(s)  # production path: no model= supplied
    assert captured_model_args == ["coordinator-sentinel-model"]
    # The chat model that ended up on create_deep_agent is the fake we returned.
    assert isinstance(captured_create.get("model"), _FakeDoneModel)


# --------------------------------------------------------------------------- #
# Dry-run invoke acceptance (P5.2's stated acceptance)
# --------------------------------------------------------------------------- #


def test_dry_run_invoke_completes_without_error(tmp_path: Path) -> None:
    """P5.2 acceptance: 'Agent builds; dry-run invoke completes without error.'
    The fake model emits a single AIMessage('Done.') with no tool calls, so
    the loop completes in one model step. This is the *minimum* end-to-end
    acceptance; the orchestration prompt (P5.3) + a multi-step fixture dry
    run land next PR."""
    s = _settings(tmp_path)
    agent = build_coordinator_agent(s, model=_FakeDoneModel())
    assert isinstance(agent, CompiledStateGraph)
    result = asyncio.run(
        agent.ainvoke({"messages": [{"role": "user", "content": "Say done."}]})
    )
    messages = result["messages"]
    assert messages, "invoke returned empty message list"
    last = messages[-1]
    content = getattr(last, "content", "")
    # The fake returns exactly 'Done.' — a non-empty final reply confirms the
    # loop terminated cleanly with no exception, no tool-call recursion, no
    # bind_tools NotImplementedError.
    assert isinstance(content, str)
    assert content.strip() == "Done."


def test_dry_run_invoke_writes_no_files_for_no_tool_call(tmp_path: Path) -> None:
    """A no-tool-call completion should not leave stray artifacts under the
    orchestrator data dir — confirms the build doesn't eagerly write state
    on a no-op invoke (which would be a leak that breaks test isolation in
    P5.3's heavier e2e fixtures)."""
    s = _settings(tmp_path)
    agent = build_coordinator_agent(s, model=_FakeDoneModel())
    asyncio.run(
        agent.ainvoke({"messages": [{"role": "user", "content": "Say done."}]})
    )
    children = sorted(p.name for p in tmp_path.iterdir())
    # The backend's mkdir already exists (build_orchestrator_backend creates
    # the root). A no-op completion leaves no artifacts beyond that.
    assert children == [], f"unexpected artifacts after no-op invoke: {children}"