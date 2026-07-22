"""P5.3 — coordinator orchestration prompt + end-to-end dry run.

Two contracts:

1. **COORDINATOR_SYSTEM_PROMPT shape** (test_coordinator_prompt_*):
   pins the prompt's section anchors (ROLE/PIPELINE/PRINCIPLE_3/
   DELEGATION_RULES/GUARDRAILS/DELIVERY), the guardrails the documented
   plan calls out (max topics, no fabricated URLs, single-topic task
   delegation, principle-#3 invariant), and that the
   ``max_topics_per_run`` interpolation carries the live settings value
   (an operator dial-up surfaces in the contract block the model reads,
   not just in the technical_rank clamp).

2. **End-to-end dry run yields N post proposals from a fixture date**
   (test_e2e_dryrun_*): drives the actual deepagents coordinator agent
   with a scripted fake coordinator model that emits the run-order the
   prompt prescribes, exercising the real ``fetch_curated_ai_news`` +
   ``technical_rank`` + ``write_file`` (deepagents built-in) +
   ``quality_gate`` tools end-to-end. Produces a passing ``drafts/
   <post_id>.json`` from a one-article fixture date.

   Subagent delegation via ``task()`` is the prompt's design, but
   scripting a faithful e2e across the subagent models' own deepagents
   loops is out of this PR's scope — the subagent specs are covered
   shape-wise in tests/test_research_subagent.py + test_writer_subagent
   (P4) and their deterministic tools have full coverage in tests/
   test_tool_*. A deeper, subagent-inclusive e2e run lands with Phase 7's
   observability work (LangSmith tracing) where the full chain can be
   driven and inspected under a real LLM. Honest about scope.

Path-semantics note (production follow-up): deepagents' built-in
``write_file`` / ``read_file`` middleware calls ``validate_path`` which
normalizes relative paths to "/<name>" (absolute-virtual). The P5.1
``build_orchestrator_backend`` mounts a ``FilesystemBackend`` with
``virtual_mode=False``, so passes-through absolute paths to the OS; a
RELATIVE path like ``briefs/topic.json`` normalizes to ``/briefs/topic.json``
and the backend treats that as an absolute path rooted at the OS root,
which is read-only on macOS. The P4 subagent prompts currently instruct
the model to "write_file at ``briefs/<topic_id>.json``" (relative). For
this test, the scripted fake model emits ABSOLUTE paths (resolved at
build time against ``tmp_path``, the orchestrator data dir). A follow-up
production-wiring PR will either (a) switch the mount to
``virtual_mode=True`` + symlink ``<data_dir>/skills`` -> the real skills
dir so the writer-subagent's absolute skills source still resolves, or
(b) interpolate ``orchestrator_data_dir`` (absolute) into the subagent
system prompts so the model emits absolute ``write_file`` paths. The
choice is deferred because P5.3's stated acceptance ("yields N post
proposals from a fixture date") doesn't require production-time LLM-driven
file writes — that integration lands under Phase 7 observability once a
real LLM can be driven end-to-end.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.graph.state import CompiledStateGraph

from app.config import Settings
from app.orchestrator import agent as agent_module
from app.orchestrator.agent import build_coordinator_agent
from app.orchestrator.prompts import (
    COORDINATOR_PROMPT_SECTIONS,
    COORDINATOR_SYSTEM_PROMPT,
    build_coordinator_system_prompt,
)
from app.orchestrator.schemas import CuratedArticle, PostProposal


def _settings(tmp_path: Path, **overrides: Any) -> Settings:
    return Settings(
        _env_file=None,
        orchestrator_data_dir=str(tmp_path),
        openrouter_api_key="",  # disables live OpenRouter; tools fall back to deterministic paths
        openrouter_coordinator_model="coordinator-sentinel",
        **overrides,
    )


def _fixture_article(topic_id: str = "topic-a") -> CuratedArticle:
    """A single CuratedArticle that survives technical_rank's hype-filter
    and clustering into one topic — deterministic across runs."""
    return CuratedArticle(
        id=topic_id,
        source_name="Test Source",
        title="Stable Diffusion 3.5 releases with new attention mechanism",
        url=f"https://example.com/{topic_id}",
        summary="A new attention mechanism improving throughput of image models.",
        score=0.9,
        duplicate_count=1,
        cluster_size=1,
    )


# --------------------------------------------------------------------------- #
# Prompt contract
# --------------------------------------------------------------------------- #


def test_prompt_carries_all_documented_sections() -> None:
    """All six section anchors (ROLE, PIPELINE, PRINCIPLE_3, DELEGATION_RULES,
    GUARDRAILS, DELIVERY) are present in the prompt. Catches a docstring reorg
    that drops a section the prompt-level acceptance criteria relies on."""
    prompt = build_coordinator_system_prompt(
        Settings(_env_file=None, max_topics_per_run=5)
    )
    for anchor in COORDINATOR_PROMPT_SECTIONS:
        # Each anchor is a markdown `## <ANCHOR>` header.
        assert f"## {anchor}" in prompt, f"prompt missing section anchor: {anchor}"


def test_prompt_interpolates_max_topics_per_run() -> None:
    """The PIPELINE section interpolates ``max_topics_per_run`` so an operator's
    dial-up surfaces in the contract block the model reads, not just in the
    downstream clamp technical_rank enforces. Pin a non-default value flows
    through and the default value is in-writing too."""
    prompt_custom = build_coordinator_system_prompt(
        Settings(_env_file=None, max_topics_per_run=9)
    )
    assert "9)" in prompt_custom  # the interpolated `(... topics.` line
    assert "9 topics" not in prompt_custom  # interpolation produces `9)`, not `N topics`

    prompt_default = build_coordinator_system_prompt(
        Settings(_env_file=None, max_topics_per_run=5)
    )
    assert "5)" in prompt_default


def test_prompt_states_no_fabricated_urls_guardrail() -> None:
    """Critical guardrail: the prompt must name 'no fabricated URLs' as a
    hard constraint, and tie it to the verification_status of the brief
    (the model skips the writer when the brief says insufficient_evidence
    or failed, instead of pushing downstream)."""
    prompt = build_coordinator_system_prompt(Settings(_env_file=None, max_topics_per_run=5))
    assert "fabricated" in prompt.lower()
    assert "no fabricated" in prompt.lower() or "never fabricate" in prompt.lower()
    # The guardrail section names the consequence explicitly so the model
    # understands the *why*, not just the rule.
    assert "insufficient_evidence" in prompt or "failed" in prompt


def test_prompt_states_single_topic_delegation_rule() -> None:
    """Per-topic delegation via ``task()`` is the principle-#3 invariant at
    the delegation seam. Pinning that the prompt prohibits bundling
    multiple topics per task call — the gate's single_topic_id check
    depends on it."""
    prompt = build_coordinator_system_prompt(Settings(_env_file=None, max_topics_per_run=5))
    assert "one topic per task" in prompt.lower()
    assert "task" in prompt.lower()


def test_prompt_states_principle_3_invariant() -> None:
    """Principle #3 (structured data lives on disk, tool/task returns are
    compressed pointers — never the bulk payload) is the load-bearing
    convention of the entire orchestrator; the prompt has to name it so the
    model doesn't helpfully paste article bodies back into its own replies.

    Slices the PRINCIPLE_3 section out of the prompt and asserts the
    compressed-pointer phrasing + an example artifact path appears INSIDE
    the slice — not just the section anchor or a stray 'filesystem' mention
    somewhere else in the prompt."""
    prompt = build_coordinator_system_prompt(Settings(_env_file=None, max_topics_per_run=5))
    start = prompt.find("## PRINCIPLE_3")
    end = prompt.find("## ", start + 1)
    assert start != -1, "prompt missing PRINCIPLE_3 section"
    slice_text = prompt[start:end] if end != -1 else prompt[start:]
    assert "compressed pointer" in slice_text.lower(), (
        "PRINCIPLE_3 section must spell out the compressed-pointer invariant"
    )
    # An example artifact path inside the section pins the prose to the
    # actual filesystem convention state.py + the tools already ship.
    assert "briefs/" in slice_text or "drafts/" in slice_text, (
        "PRINCIPLE_3 section must name an on-disk artifact path so the model "
        "knows where structured data lives"
    )


def test_prompt_states_subagent_names() -> None:
    """The DELEGATION_RULES section is the prompt's load-bearing seam for
    per-topic delegation. Pinning the prompt names the canonical subagent
    ids (``research-subagent``, ``writer-subagent``) catches a future
    rename of either `RESEARCH_SUBAGENT_NAME` / `WRITER_SUBAGENT_NAME` in
    the spec modules (P4) from silently diverging from the prompt's
    text — which is exactly the prompt-level drift this PR's prompt tests
    are designed to catch."""
    from app.orchestrator.subagents.research import RESEARCH_SUBAGENT_NAME
    from app.orchestrator.subagents.writer import WRITER_SUBAGENT_NAME

    prompt = build_coordinator_system_prompt(Settings(_env_file=None, max_topics_per_run=5))
    assert RESEARCH_SUBAGENT_NAME in prompt
    assert WRITER_SUBAGENT_NAME in prompt


def test_prompt_states_max_topics_guardrail_in_pipeline_and_guardrails() -> None:
    """The cap appears twice: once in the PIPELINE section (declarative) and
    once in the GUARDRAILS section (consequence-bearing). Pin both so a future
    edit drops one without the test catching it."""
    prompt = build_coordinator_system_prompt(Settings(_env_file=None, max_topics_per_run=7))
    pipeline_idx = prompt.find("## PIPELINE")
    guardrails_idx = prompt.find("## GUARDRAILS")
    assert pipeline_idx != -1 and guardrails_idx != -1 and pipeline_idx < guardrails_idx
    pipeline_segment = prompt[:guardrails_idx]
    guardrails_segment = prompt[guardrails_idx:]
    assert "7)" in pipeline_segment  # interpolated into the PIPELINE section
    assert "max_topics_per_run" in guardrails_segment  # named in the GUARDRAILS section


def test_module_level_singleton_matches_default_settings() -> None:
    """The module-level singleton ``COORDINATOR_SYSTEM_PROMPT`` is the
    build-from-default-settings. Pin so a future lazy-init with side effects
    doesn't ship a different string at module load vs at factory call."""
    default_prompt = build_coordinator_system_prompt(Settings(_env_file=None))
    assert COORDINATOR_SYSTEM_PROMPT == default_prompt


# --------------------------------------------------------------------------- #
# E2E dry run — deepagents coordinator drives the real tools
# --------------------------------------------------------------------------- #


class _ScriptedModel(BaseChatModel):
    """Fake chat model that emits a pre-scripted sequence of AIMessages.

    Each AIMessage may carry ``tool_calls`` (mid-sequence) or be plain text
    (final). deepagents' loop reads them in order, dispatches the tool call,
    feeds the ToolMessage back, and re-invokes the model for the next
    scripted message. When the scripted iterator is exhausted, the model
    emits a final plain AIMessage — the loop terminates.

    Overrides ``bind_tools`` to return ``self`` so deepagents' middleware
    stack can build the agent (the default ``BaseChatModel.bind_tools``
    raises ``NotImplementedError``); no tool call is ever emitted by the
    stub bound call, so no recursion through ``bound_model.ainvoke``.
    """

    script: list[AIMessage] = []

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        # The script is consumed in order; when exhausted, emit a no-tool
        # AIMessage so the loop terminates the next model step.
        idx = getattr(self, "_idx", 0)
        if idx < len(self.script):
            msg = self.script[idx]
            self._idx = idx + 1  # type: ignore[attr-defined]
        else:
            msg = AIMessage(content="Done.")
        return ChatResult(generations=[ChatGeneration(message=msg)])

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        return self._generate(messages, stop, run_manager, **kwargs)

    def bind_tools(self, tools: Any, **kwargs: Any) -> "_ScriptedModel":  # type: ignore[override]
        return self

    @property
    def _llm_type(self) -> str:
        return "scripted"


def _tool_call(name: str, args: dict[str, Any]) -> AIMessage:
    """Build an AIMessage carrying one tool call. The deepagents ToolNode
    inspects `tool_calls`, which is a list of dicts with `name`, `args`,
    `id`."""
    import uuid

    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": name,
                "args": args,
                "id": f"call-{uuid.uuid4().hex[:8]}",
                "type": "tool_call",
            }
        ],
    )


def _build_scripted_coordinator(
    tmp_path: Path,
    *,
    script: list[AIMessage],
    monkeypatch: pytest.MonkeyPatch,
) -> CompiledStateGraph[Any, Any, Any, Any]:
    """Helper: build the coordinator agent with a ``_ScriptedModel`` carrying
    the supplied script, under settings whose orchestrator data dir is the tmp
    path so the run writes nothing outside of it."""
    fake = _ScriptedModel(script=script)
    return build_coordinator_agent(_settings(tmp_path), model=fake)


def _abs(data_dir: Path, *parts: str) -> str:
    """Resolve a relative orchestrator path against the data dir's absolute
    location, returning the string form deepagents' ``write_file`` middleware
    expects — the middleware's ``validate_path`` then passes it through
    unchanged (it's already absolute) and the P5.1 mount writes it on real
    disk. See the module docstring for the path-semantics + production
    scope-out context."""
    return str(data_dir.joinpath(*parts))


def _stub_brief_json(topic_id: str) -> str:
    """A valid ResearchBrief JSON the coordinator writes (in lieu of the
    research subagent running for real). The writer-equivalent step in this
    test only inspects ``verification_status``, so we set ``verified`` and
    empty everything else deterministically."""
    return json.dumps(
        {
            "topic_id": topic_id,
            "headline": "Test headline",
            "summary": "Test summary that does not contain hype markers and is long enough.",
            "technical_significance": "A specific attention mechanism change in the model.",
            "business_impact": "Lower inference cost at parity quality for image use cases.",
            "why_now": "Released yesterday in a stable build.",
            "key_points": ["attention refactor", "throughput improvement"],
            "risks": ["real-world performance unproven"],
            "citations": [
                {"title": "Test citation", "url": f"https://example.com/{topic_id}", "domain": "example.com"}
            ],
            "verification_status": "verified",
            "verification_confidence": 0.8,
            "verification_notes": ["mock verification note"],
        }
    )


def _stub_verified_brief_json(topic_id: str) -> str:
    """A verified ResearchBrief JSON the coordinator writes (in lieu of the
    verify_claim tool running for real on a stubbed brief). The content here
    is byte-identical to the brief stub — we are NOT modelling a content
    delta between brief and verified because nothing in this e2e test paths
    through ``verify_claim``'s rewriting (the scripted model writes both
    files itself, bypassing the verifier). Testing that delta is the
    verify_claim tool's own unit tests' job (see test_tool_verify_claim.py)."""
    return _stub_brief_json(topic_id)


def _stub_draft_json(post_id: str, topic_id: str) -> str:
    """A valid PostProposal JSON that passes the quality gate. Body word count
    in [105, 182]; exactly one supporting_topic_id; >=3 hashtags; no hype
    markers. The cleaned-word-count and other gate checks all pass."""
    body_words = (
        "Stable Diffusion 3.5 ships a new attention mechanism that lowers "
        "the real cost of running image models at parity quality. The change "
        "is a small rewrite of how the model attends across patches and "
        "shows up as a throughput improvement in the release notes. "
        "Independent benchmarks are limited at this stage. Teams building on "
        "the SDK should pin their current model version and test the new path "
        "in parallel before switching production traffic. The spend math is "
        "straightforward: if inference was your bottleneck, this likely moves "
        "the per-image cost down without regressing fidelity. For research "
        "workloads the gain is minor. For production serving, this is one of "
        "the more useful steady improvements."
    )
    assert 105 <= len(body_words.split()) <= 182, "draft body must be in the gate's word window"
    return json.dumps(
        {
            "post_id": post_id,
            "angle": "steady improvement",
            "headline": "Stable Diffusion 3.5's quiet attention refactor",
            "body": body_words,
            "hashtags": ["#stablediffusion", "#attention", "#inference"],
            "supporting_topic_ids": [topic_id],
            "citation_urls": [f"https://example.com/{topic_id}"],
            "confidence": 0.8,
        }
    )


def _mock_run_curation_one_article(
    article: CuratedArticle, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Patch ``news.run_curation`` to skip the real pipeline and return a
    single CuratedArticle (the fixture date's only article)."""

    async def _fake_run_curation(
        limit: int | None = None, settings: Settings | None = None
    ) -> tuple[list[CuratedArticle], int]:
        return [article], 1

    monkeypatch.setattr(agent_module, "run_curation", _fake_run_curation, raising=False)
    # The news tool imports `run_curation` at module load — patch the
    # attribute in news.py's module namespace too.
    from app.orchestrator.tools import news as tools_news

    monkeypatch.setattr(tools_news, "run_curation", _fake_run_curation)


def test_e2e_dryrun_yields_one_post_proposal_from_fixture_date(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """P5.3 acceptance: 'End-to-end dry run yields N post proposals from a
    fixture date.'

    Drives the real deepagents coordinator with a scripted fake model that
    emits the run-order the COORDINATOR_SYSTEM_PROMPT prescribes (write_todos
    -> fetch_curated_ai_news -> technical_rank -> write the brief + verified
    brief to disk -> write the draft to disk -> quality_gate -> final
    summary). Uses the live ``fetch_curated_ai_news`` (mocked ``run_curation``
    that returns one fixture article), the live ``technical_rank`` (heuristic
    path, no OpenRouter key), the deepagents built-in ``write_file`` (to
    stage brief + draft on disk), and the live ``quality_gate`` (verifies the
    draft passes).

    Subagent delegation via ``task()`` is out of this test's scope — see
    the module docstring for the honest scope-out note. Briefs + drafts are
    staged by the coordinator model itself via ``write_file``, which
    matches the deterministic-model shape of the test without involving
    the subagent models' own deepagents loops.

    Asserts the END state: topics.json exists, briefs/<topic>.verified.json
    exists, drafts/<post_id>.json exists, drafts/<post_id>.gate.json exists
    and the verdict inside it is ``passed=True``."""
    s = _settings(tmp_path, max_topics_per_run=1)
    article = _fixture_article("topic-a")

    _mock_run_curation_one_article(article, monkeypatch)

    # Script: emit in order. (The quality_gate `post_id` slug is hardcoded
    # because the script doesn't parse read_file output in real time; the
    # brief's `topic_id` is also hardcoded so verify_claim's read matches.)
    topic_id = "topic-a"
    post_id = "post-1"

    script: list[AIMessage] = [
        _tool_call("write_todos", {"todos": [
            {"content": "Fetch curated news", "status": "in_progress"},
            {"content": "Rank", "status": "pending"},
            {"content": "Stage brief + draft", "status": "pending"},
            {"content": "Gate", "status": "pending"},
        ]}),
        _tool_call("fetch_curated_ai_news", {}),
        _tool_call("technical_rank", {}),
        # Stage the brief + verified brief (coordinator itself writes; the
        # real flow would delegate to research-subagent here).
        _tool_call("write_file", {
            "file_path": _abs(tmp_path, "briefs", f"{topic_id}.json"),
            "content": _stub_brief_json(topic_id),
        }),
        _tool_call("write_file", {
            "file_path": _abs(tmp_path, "briefs", f"{topic_id}.verified.json"),
            "content": _stub_verified_brief_json(topic_id),
        }),
        # Stage the draft (coordinator itself writes; the real flow would
        # delegate to writer-subagent here).
        _tool_call("write_file", {
            "file_path": _abs(tmp_path, "drafts", f"{post_id}.json"),
            "content": _stub_draft_json(post_id, topic_id),
        }),
        # Gate it — quality_gate is a coordinator-level tool.
        _tool_call("quality_gate", {"post_id": post_id}),
        # Final summary text (no tool calls -> loop terminates).
        AIMessage(content=f"Done. 1 post proposal produced at drafts/{post_id}.json"),
    ]

    agent = build_coordinator_agent(s, model=_ScriptedModel(script=script))
    assert isinstance(agent, CompiledStateGraph)

    # Drive the run. The "today" message is the user kick-off that
    # triggers deepagents to call the model and dispatch the script.
    kickoff = "Today is 2026-07-21. Produce a LinkedIn post per the system prompt."
    result = asyncio.run(
        agent.ainvoke({"messages": [{"role": "user", "content": kickoff}]})
    )

    # --- END-state assertions: the artifacts on disk are the acceptance ---
    articles_path = tmp_path / "articles.json"
    topics_path = tmp_path / "topics.json"
    brief_path = tmp_path / "briefs" / f"{topic_id}.json"
    verified_path = tmp_path / "briefs" / f"{topic_id}.verified.json"
    draft_path = tmp_path / "drafts" / f"{post_id}.json"
    gate_path = tmp_path / "drafts" / f"{post_id}.gate.json"

    assert articles_path.exists(), "fetch_curated_ai_news should have written articles.json"
    articles = json.loads(articles_path.read_text(encoding="utf-8"))
    assert len(articles) == 1
    assert articles[0]["id"] == "topic-a"

    assert topics_path.exists(), "technical_rank should have written topics.json"
    topics = json.loads(topics_path.read_text(encoding="utf-8"))
    assert len(topics) >= 1
    # at least one row exists; the deterministic heuristic should have produced
    # one topic from one article (cap is max_topics_per_run=1).

    assert brief_path.exists(), "scripted write_file should have written the brief"
    assert verified_path.exists()

    assert draft_path.exists(), "scripted write_file should have written the draft"
    proposal = PostProposal.model_validate_json(draft_path.read_text(encoding="utf-8"))
    assert proposal.post_id == post_id

    assert gate_path.exists(), "quality_gate should have written the verdict"
    gate_verdict = json.loads(gate_path.read_text(encoding="utf-8"))
    assert gate_verdict["passed"] is True, (
        f"quality_gate verdict should be passed=True; got: {gate_verdict}"
    )

    # The final AIMessage text carries the summary. The coordinator did NOT
    # paste the post body into its reply (principle #3); only the path.
    final = result["messages"][-1]
    final_text = getattr(final, "content", "") or ""
    assert "drafts/post-1.json" in final_text
    # The post body itself is NOT in the final reply (principle-#3 invariant).
    assert proposal.body[:40] not in final_text


def test_e2e_dryrun_no_leaked_artifacts_outside_data_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A dry run should not leak artifacts outside the orchestrator data dir.
    Walks ``tmp_path.parent`` (not ``tmp_path`` itself — that would be
    tautological since ``tmp_path`` IS the orchestrator data dir by
    construction) and asserts the only new subtree the run created under
    the parent is ``tmp_path`` itself — i.e. no sibling ``briefs/``,
    ``drafts/``, ``articles.json``, ``topics.json`` appeared next to it.

    Surfaces a future regression where a tool / the agent's deepagents
    middleware routes a write outside the orchestrator data dir — e.g.
    relative paths under the read-only OS root (the path-semantics
    limitation documented in this PR's module docstring)."""

    # Pre-snapshot the parent's children: pytest's tmp_path parent has
    # one dir per test; just record what's already on disk before the run.
    pre = {p.name for p in tmp_path.parent.iterdir()} if tmp_path.parent.exists() else set()
    pre |= {tmp_path.name}

    s = _settings(tmp_path, max_topics_per_run=1)
    _mock_run_curation_one_article(_fixture_article("topic-b"), monkeypatch)
    topic_id = "topic-b"
    post_id = "post-2"
    script: list[AIMessage] = [
        _tool_call("fetch_curated_ai_news", {}),
        _tool_call("technical_rank", {}),
        _tool_call("write_file", {
            "file_path": _abs(tmp_path, "briefs", f"{topic_id}.verified.json"),
            "content": _stub_verified_brief_json(topic_id),
        }),
        _tool_call("write_file", {
            "file_path": _abs(tmp_path, "drafts", f"{post_id}.json"),
            "content": _stub_draft_json(post_id, topic_id),
        }),
        _tool_call("quality_gate", {"post_id": post_id}),
        AIMessage(content="Done."),
    ]
    agent = build_coordinator_agent(s, model=_ScriptedModel(script=script))
    asyncio.run(agent.ainvoke({"messages": [{"role": "user", "content": "Run for today."}]}))

    # Walk the PARENT, not `tmp_path`. Anything that escaped would land as
    # a *sibling* of `tmp_path` in its parent.
    leaked_siblings: list[Path] = []
    for entry in tmp_path.parent.iterdir():
        if entry.name in pre:
            continue
        leaked_siblings.append(entry)

    # Sanity: the orchestrator data dir itself did accumulate artifacts.
    assert (tmp_path / "articles.json").exists()
    assert (tmp_path / "topics.json").exists()
    assert (tmp_path / "briefs" / f"{topic_id}.verified.json").exists()
    assert (tmp_path / "drafts" / f"{post_id}.json").exists()
    assert (tmp_path / "drafts" / f"{post_id}.gate.json").exists()

    assert not leaked_siblings, (
        f"run leaked artifacts as siblings of the orchestrator data dir "
        f"(outside its root): {leaked_siblings}"
    )