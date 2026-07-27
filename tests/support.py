"""Shared fixtures for driving the deterministic propose spine in tests.

The spine takes compiled research/writer agents as an explicit injection seam
(``run_propose_spine(research_agent=..., writer_agent=...)``), so a test drives
the real pipeline by supplying stub agents that do what the real subagents do —
write a brief, submit a draft, run the gate — without an LLM.

The stubs deliberately go through the *real* ``submit_draft`` and
``quality_gate`` tools rather than writing files directly: that keeps
provenance signing, the evidence floor and citation-fidelity checks live in
every spine test instead of monkeypatched away.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

from app.config import Settings
from app.orchestrator.schemas import CuratedArticle
from app.orchestrator.tools.draft import build_submit_draft_tool
from app.orchestrator.tools.quality import build_quality_gate_tool


def settings_for(tmp_path: Path, **overrides: Any) -> Settings:
    """Spine settings rooted at ``tmp_path``.

    ``openrouter_api_key`` defaults to empty so every LLM-backed tool takes its
    deterministic heuristic/dry-run path and no test touches the network. Tests
    that need the propose config-gate to pass override it and pair that with
    ``force_offline_ranking``."""
    kwargs: dict[str, Any] = {
        "orchestrator_data_dir": str(tmp_path),
        "outputs_dir": str(tmp_path / "outputs"),
        "openrouter_api_key": "",
        "max_topics_per_run": 1,
    }
    kwargs.update(overrides)
    return Settings(_env_file=None, **kwargs)


def fixture_article(
    topic_id: str = "topic-a",
    *,
    title: str = "Stable Diffusion 3.5 releases with new attention mechanism",
) -> CuratedArticle:
    """A single CuratedArticle that survives technical_rank's hype-filter and
    clustering into one topic — deterministic across runs.

    Pass a distinct ``title`` when a test needs multiple genuinely separate
    topics: technical_rank's clustering keys off title token overlap, not
    id/url, so two calls with the default title collapse into one story
    regardless of ``topic_id``."""
    return CuratedArticle(
        id=topic_id,
        source_name="Test Source",
        title=title,
        url=f"https://example.com/{topic_id}",
        summary="A new attention mechanism improving throughput of image models.",
        score=0.9,
        duplicate_count=1,
        cluster_size=1,
    )


def brief_fields(topic_id: str, **overrides: Any) -> dict[str, Any]:
    """A ResearchBrief payload at ``verified`` status — clears the evidence
    floor unconditionally (see ``meets_evidence_floor``)."""
    fields: dict[str, Any] = {
        "topic_id": topic_id,
        "headline": "Test headline",
        "summary": "Test summary that does not contain hype markers and is long enough.",
        "technical_significance": "A specific attention mechanism change in the model.",
        "business_impact": "Lower inference cost at parity quality for image use cases.",
        "why_now": "Released yesterday in a stable build.",
        "key_points": ["attention refactor", "throughput improvement"],
        "risks": ["real-world performance unproven"],
        "citations": [
            {
                "title": "Test citation",
                "url": f"https://example.com/{topic_id}",
                "domain": "example.com",
            }
        ],
        "verification_status": "verified",
        "verification_confidence": 0.8,
        "verification_notes": ["mock verification note"],
    }
    fields.update(overrides)
    return fields


# A quality-gate-passing post body: word count in [105, 182], no hype markers.
# Shared by the spine stubs and the parity fixtures so the "canonical clean
# proposal" is tuned to the gate's word window in exactly one place.
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

SHORT_BODY = "Too short to pass the quality gate."


def proposal_fields(post_id: str, topic_id: str, *, body: str = CLEAN_GATE_BODY) -> dict[str, Any]:
    """A PostProposal payload. With the default body it passes the gate: word
    count in [105, 182]; exactly one supporting_topic_id; >=3 hashtags; no hype
    markers; citation_urls drawn from the brief's citations."""
    return {
        "post_id": post_id,
        "angle": "steady improvement",
        "headline": "Stable Diffusion 3.5's quiet attention refactor",
        "body": body,
        "hashtags": ["#stablediffusion", "#attention", "#inference"],
        "supporting_topic_ids": [topic_id],
        "citation_urls": [f"https://example.com/{topic_id}"],
        "confidence": 0.8,
    }


_RESEARCH_TOPIC_RE = re.compile(r"^Research topic (?P<topic_id>[\w.-]+):")
# ``[\w.-]+`` rather than ``\S+``: the writer description ends the post_id with
# a sentence period, and a greedy \S+ would swallow it into the id.
_WRITER_IDS_RE = re.compile(
    r"topic_id=(?P<topic_id>[\w.-]+) as post_id=(?P<post_id>[\w-]+)"
)


class StubResearchAgent:
    """Stands in for the research subagent: writes the verified brief the
    evidence floor and the writer both read. ``fail_for`` names topic_ids to
    raise on (exercising the spine's per-topic error isolation); ``brief_kwargs``
    overrides brief fields (e.g. a below-floor verification_status)."""

    def __init__(
        self,
        settings: Settings,
        *,
        fail_for: set[str] | None = None,
        brief_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self._settings = settings
        self._fail_for = fail_for or set()
        self._brief_kwargs = brief_kwargs or {}
        self.calls: list[str] = []

    async def ainvoke(self, payload: dict[str, Any], config: Any = None, **_: Any) -> dict[str, Any]:
        description = payload["messages"][-1]["content"]
        match = _RESEARCH_TOPIC_RE.search(description)
        assert match, f"unexpected research description: {description!r}"
        topic_id = match.group("topic_id")
        self.calls.append(topic_id)
        if topic_id in self._fail_for:
            raise RuntimeError(f"scripted research failure for {topic_id}")

        from app.orchestrator import state

        path = state.verified_brief_path(topic_id, self._settings.orchestrator_data_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(brief_fields(topic_id, **self._brief_kwargs)), encoding="utf-8"
        )
        return {"messages": [{"role": "assistant", "content": f"researched {topic_id}"}]}


class StubWriterAgent:
    """Stands in for the writer subagent: submits the draft through the real
    ``submit_draft`` tool (provenance + floor enforced) and runs the real
    ``quality_gate`` tool. ``body_for`` maps post_id -> body so a test can make
    a specific draft fail the gate."""

    def __init__(
        self,
        settings: Settings,
        *,
        body_for: dict[str, str] | None = None,
        fail_for: set[str] | None = None,
    ) -> None:
        self._settings = settings
        self._body_for = body_for or {}
        self._fail_for = fail_for or set()
        self.calls: list[tuple[str, str]] = []

    async def ainvoke(self, payload: dict[str, Any], config: Any = None, **_: Any) -> dict[str, Any]:
        description = payload["messages"][-1]["content"]
        match = _WRITER_IDS_RE.search(description)
        assert match, f"unexpected writer description: {description!r}"
        topic_id, post_id = match.group("topic_id"), match.group("post_id")
        self.calls.append((topic_id, post_id))
        if post_id in self._fail_for:
            raise RuntimeError(f"scripted writer failure for {post_id}")

        body = self._body_for.get(post_id, CLEAN_GATE_BODY)
        submit = build_submit_draft_tool(self._settings)
        await submit.ainvoke(
            {"post_id": post_id, "proposal": proposal_fields(post_id, topic_id, body=body)}
        )
        gate = build_quality_gate_tool(self._settings)
        await gate.ainvoke({"post_id": post_id})
        return {"messages": [{"role": "assistant", "content": f"wrote {post_id}"}]}


def mock_run_curation(
    articles: list[CuratedArticle], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Patch ``run_curation`` at the news tool's import site so the spine's
    fetch stage returns fixture articles instead of running the real pipeline."""

    async def _fake_run_curation(
        limit: int | None = None, settings: Settings | None = None
    ) -> tuple[list[CuratedArticle], int]:
        return articles, len(articles)

    from app.orchestrator.tools import news as tools_news

    monkeypatch.setattr(tools_news, "run_curation", _fake_run_curation)


def force_offline_ranking(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin the technical ranker to its heuristic path regardless of the
    configured key, so a test that needs a non-empty ``openrouter_api_key``
    (the propose config-gate) still never calls OpenRouter."""
    from app.orchestrator.services.technical_ranker import TechnicalRanker

    async def _heuristic_only(self: Any, items: list[Any], dry_run: bool) -> dict[str, Any]:
        return {item.id: self._heuristic_assessment(item) for item in items}

    monkeypatch.setattr(TechnicalRanker, "assess_many", _heuristic_only)


def disable_editor_veto(monkeypatch: pytest.MonkeyPatch) -> None:
    """Neutralize the editorial veto's LLM call — the spine tests exercise
    selection/floor/gate, not the veto prompt (``test_editor_veto`` covers
    that). Keeps every topic in the pool."""
    from app.orchestrator import spine as spine_mod

    async def _keep_all(topics: list[Any], settings: Settings, *, dry_run: bool) -> Any:
        return list(topics), []

    monkeypatch.setattr(spine_mod, "veto_irrelevant_topics", _keep_all)
