"""Shared harness for driving the coordinator deep agent deterministically.

Both the P5.3 e2e dry-run (``test_e2e_dryrun.py``) and the P8.1 snapshot
(``test_e2e_snapshot.py``) drive the *real* deepagents coordinator with a
scripted fake model that emits the exact run-order the coordinator prompt
prescribes — no OpenRouter, no LLM nondeterminism. Keeping the scripted model +
fixture builders here means the two tests share one harness instead of drifting
copies of a ~200-line stub.

The scripted model writes the brief + draft artifacts itself via the deepagents
built-in ``write_file`` (in lieu of the research/writer subagents running for
real), using ABSOLUTE paths resolved against the orchestrator data dir — see
``test_e2e_dryrun.py``'s module docstring for the path-semantics context.
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from app.config import Settings
from app.orchestrator import agent as agent_module
from app.orchestrator.schemas import CuratedArticle
from app.orchestrator.services.provenance import PROVENANCE_KEY, sign_draft


def settings_for(tmp_path: Path, **overrides: Any) -> Settings:
    """Coordinator settings rooted at ``tmp_path`` with OpenRouter disabled so
    tools take their deterministic dry-run / heuristic paths."""
    return Settings(
        _env_file=None,
        orchestrator_data_dir=str(tmp_path),
        openrouter_api_key="",  # disables live OpenRouter
        openrouter_coordinator_model="coordinator-sentinel",
        **overrides,
    )


def fixture_article(topic_id: str = "topic-a") -> CuratedArticle:
    """A single CuratedArticle that survives technical_rank's hype-filter and
    clustering into one topic — deterministic across runs."""
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


class ScriptedModel(BaseChatModel):
    """Fake chat model that emits a pre-scripted sequence of AIMessages.

    deepagents' loop reads them in order, dispatches each tool call, feeds the
    ToolMessage back, and re-invokes the model for the next scripted message.
    When the script is exhausted it emits a plain AIMessage so the loop
    terminates. ``bind_tools`` returns ``self`` so the middleware stack can
    build the agent without the default ``NotImplementedError``.
    """

    script: list[AIMessage] = []

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
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

    def bind_tools(self, tools: Any, **kwargs: Any) -> "ScriptedModel":  # type: ignore[override]
        return self

    @property
    def _llm_type(self) -> str:
        return "scripted"


def tool_call(name: str, args: dict[str, Any]) -> AIMessage:
    """Build an AIMessage carrying one tool call (deepagents' ToolNode inspects
    ``tool_calls`` — a list of dicts with ``name``, ``args``, ``id``)."""
    return AIMessage(
        content="",
        tool_calls=[
            {"name": name, "args": args, "id": f"call-{uuid.uuid4().hex[:8]}", "type": "tool_call"}
        ],
    )


def abs_path(data_dir: Path, *parts: str) -> str:
    """Resolve a relative orchestrator path against the data dir's absolute
    location — the string form deepagents' ``write_file`` middleware expects
    (already absolute, so ``validate_path`` passes it through and the P5.1 mount
    writes it on real disk)."""
    return str(data_dir.joinpath(*parts))


def stub_brief_json(topic_id: str) -> str:
    """A valid ResearchBrief JSON the scripted coordinator writes (in lieu of
    the research subagent). ``verification_status`` = ``verified``."""
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


# A quality-gate-passing post body: word count in [105, 182], no hype markers.
# Shared by the e2e draft stub and the P8.2 parity fixtures so the "canonical
# clean proposal" is tuned to the gate's word window in exactly one place.
CLEAN_GATE_BODY = (
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


def stub_draft_json(post_id: str, topic_id: str) -> str:
    """A valid PostProposal JSON that passes the quality gate: body word count in
    [105, 182]; exactly one supporting_topic_id; >=3 hashtags; no hype markers.
    Includes a valid ``_provenance`` block so the export + delivery integrity
    checks pass."""
    body_words = CLEAN_GATE_BODY
    assert 105 <= len(body_words.split()) <= 182, "draft body must be in the gate's word window"
    proposal_fields = {
        "post_id": post_id,
        "angle": "steady improvement",
        "headline": "Stable Diffusion 3.5's quiet attention refactor",
        "body": body_words,
        "hashtags": ["#stablediffusion", "#attention", "#inference"],
        "supporting_topic_ids": [topic_id],
        "citation_urls": [f"https://example.com/{topic_id}"],
        "confidence": 0.8,
    }
    # Sign with defaults (dev fallback key) so provenance verification passes.
    settings = Settings(_env_file=None)
    provenance = sign_draft(proposal_fields, settings)
    proposal_fields[PROVENANCE_KEY] = provenance
    return json.dumps(proposal_fields)


def mock_run_curation_one_article(
    article: CuratedArticle, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Patch ``run_curation`` (in both the agent module and the news tool module)
    to skip the real pipeline and return a single fixture article."""

    async def _fake_run_curation(
        limit: int | None = None, settings: Settings | None = None
    ) -> tuple[list[CuratedArticle], int]:
        return [article], 1

    monkeypatch.setattr(agent_module, "run_curation", _fake_run_curation, raising=False)
    from app.orchestrator.tools import news as tools_news

    monkeypatch.setattr(tools_news, "run_curation", _fake_run_curation)
