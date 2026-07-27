"""The deterministic propose spine — the default `propose` path.

Covers the pure selection/id helpers directly, then drives ``run_propose_spine``
end to end with stub research/writer agents through the *real* fetch, rank,
floor, submit_draft and quality_gate machinery.
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from app.orchestrator.schemas import TopicCandidate
from app.orchestrator.spine import post_id_for, run_propose_spine, select_topics
from support import (
    SHORT_BODY,
    StubResearchAgent,
    StubWriterAgent,
    disable_editor_veto,
    fixture_article,
    mock_run_curation,
    settings_for,
)


def _topic(topic_id: str, *, score: float, domain: str) -> TopicCandidate:
    return TopicCandidate(
        topic_id=topic_id,
        title=f"Title for {topic_id}",
        summary_hint="A hint.",
        primary_url=f"https://{domain}/{topic_id}",
        primary_domain=domain,
        supporting_urls=[],
        score=score,
        cluster_size=1,
        rationale="test",
    )


# --------------------------------------------------------------------------- #
# select_topics
# --------------------------------------------------------------------------- #


def test_select_topics_prefers_one_per_domain_before_filling() -> None:
    """First pass takes the best topic per domain; only then does the second
    pass backfill remaining slots in score order."""
    topics = [
        _topic("a1", score=0.9, domain="openai.com"),
        _topic("a2", score=0.8, domain="openai.com"),
        _topic("b1", score=0.7, domain="anthropic.com"),
    ]
    assert [t.topic_id for t in select_topics(topics, 2)] == ["a1", "b1"]


def test_select_topics_backfills_from_same_domain_when_cap_not_reached() -> None:
    topics = [
        _topic("a1", score=0.9, domain="openai.com"),
        _topic("a2", score=0.8, domain="openai.com"),
        _topic("b1", score=0.7, domain="anthropic.com"),
    ]
    assert [t.topic_id for t in select_topics(topics, 3)] == ["a1", "a2", "b1"]


def test_select_topics_returns_score_ordered_result() -> None:
    """Whatever the input order, the result is descending by score."""
    topics = [
        _topic("low", score=0.1, domain="a.com"),
        _topic("high", score=0.9, domain="b.com"),
        _topic("mid", score=0.5, domain="c.com"),
    ]
    assert [t.topic_id for t in select_topics(topics, 3)] == ["high", "mid", "low"]


def test_select_topics_honours_cap_and_rejects_nonpositive() -> None:
    topics = [_topic(f"t{i}", score=i / 10, domain=f"d{i}.com") for i in range(5)]
    assert len(select_topics(topics, 2)) == 2
    assert select_topics(topics, 0) == []
    assert select_topics(topics, -1) == []


def test_select_topics_on_empty_pool() -> None:
    assert select_topics([], 5) == []


# --------------------------------------------------------------------------- #
# post_id_for
# --------------------------------------------------------------------------- #


def test_post_id_is_deterministic_and_slugged() -> None:
    taken: set[str] = set()
    assert (
        post_id_for("Stable Diffusion 3.5 Releases!", "2026-07-26", taken)
        == "post-20260726-stable-diffusion-3-5-releases"
    )


def test_post_id_suffixes_collisions() -> None:
    taken: set[str] = set()
    first = post_id_for("Same Title", "2026-07-26", taken)
    second = post_id_for("Same Title", "2026-07-26", taken)
    third = post_id_for("Same Title", "2026-07-26", taken)
    assert first == "post-20260726-same-title"
    assert second == "post-20260726-same-title-2"
    assert third == "post-20260726-same-title-3"


def test_post_id_falls_back_when_title_has_no_alphanumerics() -> None:
    assert post_id_for("!!! ???", "2026-07-26", set()) == "post-20260726-topic"


def test_post_id_truncates_long_titles() -> None:
    post_id = post_id_for("word " * 40, "2026-07-26", set())
    slug = post_id.removeprefix("post-20260726-")
    assert len(slug) <= 48
    assert not slug.endswith("-")


# --------------------------------------------------------------------------- #
# run_propose_spine — end to end with stub agents
# --------------------------------------------------------------------------- #


def _run(settings, monkeypatch, **kwargs):  # type: ignore[no-untyped-def]
    """Drive one spine run with stub agents and the veto disabled."""
    disable_editor_veto(monkeypatch)
    research = kwargs.pop("research", None) or StubResearchAgent(settings)
    writer = kwargs.pop("writer", None) or StubWriterAgent(settings)
    return asyncio.run(
        run_propose_spine(
            settings,
            run_id="test-run",
            research_agent=research,
            writer_agent=writer,
            **kwargs,
        )
    ), research, writer


def test_spine_happy_path_produces_gated_draft(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One article in -> one researched topic, one drafted post, gate passed,
    and the artifacts land where export + delivery expect them."""
    settings = settings_for(tmp_path)
    mock_run_curation([fixture_article("topic-a")], monkeypatch)

    summary, research, writer = _run(settings, monkeypatch)

    assert summary.status == "ok"
    assert summary.fetched == 1
    assert summary.topics == 1
    assert len(summary.selected) == 1
    assert summary.skipped_floor == []
    assert summary.drafts_passed == 1
    assert len(research.calls) == 1
    assert len(writer.calls) == 1

    post_id = summary.written[0].post_id
    assert (tmp_path / "drafts" / f"{post_id}.json").exists()
    gate = json.loads((tmp_path / "drafts" / f"{post_id}.gate.json").read_text())
    assert gate["passed"] is True


def test_spine_reuses_prefetched_articles_without_curating_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`both` hands the digest lane's output straight to the spine. The
    curation pipeline must not run a second time — that second pass cost ~2min
    of duplicated feed fetching and caused the 429s — and the articles must
    land on disk by the same path the fetch tool would have used, so every
    downstream stage is unaware of the difference."""
    settings = settings_for(tmp_path)

    async def _explode(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("run_curation must not run when articles are prefetched")

    from app.orchestrator.tools import news as tools_news

    monkeypatch.setattr(tools_news, "run_curation", _explode)

    summary, research, _writer = _run(
        settings, monkeypatch, prefetched=[fixture_article("topic-a")]
    )

    assert summary.fetched == 1
    assert summary.status == "ok"
    assert json.loads((tmp_path / "articles.json").read_text())[0]["id"] == "topic-a"
    assert len(research.calls) == 1


def test_spine_reports_no_articles_and_stops(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An empty fetch short-circuits before any LLM-backed stage runs."""
    settings = settings_for(tmp_path)
    mock_run_curation([], monkeypatch)

    summary, research, writer = _run(settings, monkeypatch)

    assert summary.status == "no_articles"
    assert summary.fetched == 0
    assert research.calls == []
    assert writer.calls == []


def test_spine_skips_topics_whose_brief_is_below_the_evidence_floor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A brief at ``insufficient_evidence`` never reaches the writer — the
    floor is enforced in code, not in a prompt."""
    settings = settings_for(tmp_path)
    mock_run_curation([fixture_article("topic-a")], monkeypatch)
    research = StubResearchAgent(
        settings, brief_kwargs={"verification_status": "insufficient_evidence"}
    )

    summary, _research, writer = _run(settings, monkeypatch, research=research)

    assert summary.status == "nothing_above_floor"
    assert len(summary.skipped_floor) == 1
    assert "below the evidence floor" in summary.skipped_floor[0].reason
    assert writer.calls == []  # never delegated


def test_spine_skips_topic_when_research_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A research failure kills one topic, not the run: no verified brief ->
    floored out with a 'no verified brief' reason."""
    settings = settings_for(tmp_path)
    mock_run_curation([fixture_article("topic-a")], monkeypatch)
    research = StubResearchAgent(settings, fail_for={"topic-a"})

    summary, _research, writer = _run(settings, monkeypatch, research=research)

    assert summary.research[0].status == "error"
    assert summary.status == "nothing_above_floor"
    assert summary.skipped_floor[0].reason == "no brief on disk"
    assert writer.calls == []


def test_spine_names_the_verification_status_when_a_brief_is_unverified(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unverified brief is floored out with the actual status in the reason,
    not a bare "no brief" — the spine reads through the same loader the rest of
    the pipeline uses, which prefers the verified copy but falls back to the
    unverified one so the refusal can name what it found."""
    settings = settings_for(tmp_path)
    mock_run_curation([fixture_article("topic-a")], monkeypatch)
    research = StubResearchAgent(
        settings, brief_kwargs={"verification_status": "unverified"}
    )

    summary, _research, writer = _run(settings, monkeypatch, research=research)

    assert summary.status == "nothing_above_floor"
    assert "unverified" in summary.skipped_floor[0].reason
    assert writer.calls == []


def _expected_post_id(article_title: str) -> str:
    """The post_id the spine will mint for a topic with this title today —
    ``post_id_for`` is deterministic, so a test can name it up front instead of
    running the spine once to discover it."""
    date_slug = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return post_id_for(article_title, date_slug, set())


def test_spine_records_gate_failure_without_failing_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A draft that fails the quality gate is reported as gate_passed=False and
    excluded from drafts_passed, but the run still completes."""
    settings = settings_for(tmp_path)
    article = fixture_article("topic-a")
    mock_run_curation([article], monkeypatch)
    writer = StubWriterAgent(
        settings, body_for={_expected_post_id(article.title): SHORT_BODY}
    )

    summary, _research, _writer = _run(settings, monkeypatch, writer=writer)

    assert summary.status == "ok"
    assert summary.drafts_passed == 0
    assert summary.written[0].gate_passed is False


def test_spine_writer_failure_is_isolated_to_one_post(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A writer exception is captured as that post's status; the run reports
    ok with zero gated drafts rather than raising."""
    settings = settings_for(tmp_path)
    article = fixture_article("topic-a")
    mock_run_curation([article], monkeypatch)
    writer = StubWriterAgent(settings, fail_for={_expected_post_id(article.title)})

    summary, _research, _writer = _run(settings, monkeypatch, writer=writer)

    assert summary.status == "ok"
    assert summary.written[0].writer_status == "error"
    assert summary.written[0].gate_passed is False
    assert summary.drafts_passed == 0


def test_spine_respects_the_topic_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """max_topics_per_run caps delegation in code — two rankable articles, a
    cap of 1, one research call.

    min_drafts_per_run=1 pins the target to "first success ends the run" so
    this test isolates the per-wave cap from the backfill loop below: with the
    real default of 5, topic-a's pass wouldn't meet the target and the spine
    would correctly go on to research topic-b too — that's the point of
    backfill, not a violation of this cap.
    """
    settings = settings_for(tmp_path, max_topics_per_run=1, min_drafts_per_run=1)
    mock_run_curation(
        [
            fixture_article("topic-a", title="Stable Diffusion 3.5 releases new attention mechanism"),
            fixture_article("topic-b", title="Anthropic launches Claude Opus 5 for coding agents"),
        ],
        monkeypatch,
    )

    summary, research, _writer = _run(settings, monkeypatch)

    assert len(summary.selected) == 1
    assert len(research.calls) == 1


def test_spine_backfills_from_the_vetoed_pool_to_reach_the_draft_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """min_drafts_per_run keeps drawing MORE topics from the same vetoed pool
    across waves when earlier ones don't reach the target on their own — the
    2026-07-26 production gap: a 20-topic day selected 5, lost 4 to the floor
    or a rate limit, and the other 15 vetted candidates sat unused. Forcing
    max_topics_per_run=1 with a target of 3 means all 3 topics MUST be
    attempted to satisfy the target, regardless of which one technical_rank
    happens to order first — the test doesn't depend on tie-breaking."""
    settings = settings_for(tmp_path, max_topics_per_run=1, min_drafts_per_run=3)
    mock_run_curation(
        [
            fixture_article("topic-a", title="Stable Diffusion 3.5 releases new attention mechanism"),
            fixture_article("topic-b", title="Anthropic launches Claude Opus 5 for coding agents"),
            fixture_article("topic-c", title="OpenAI ships GPT-5.6 preview with faster inference"),
        ],
        monkeypatch,
    )

    summary, research, writer = _run(settings, monkeypatch)

    assert summary.status == "ok"
    assert summary.drafts_passed == 3
    assert len(summary.selected) == 3
    assert len(research.calls) == 3
    assert len(writer.calls) == 3


def test_spine_stops_backfilling_once_the_pool_is_exhausted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A target the day's pool cannot supply must not loop forever — once
    every vetted candidate has been tried and none cleared the floor, the
    spine reports what actually happened instead of spinning."""
    settings = settings_for(tmp_path, max_topics_per_run=1, min_drafts_per_run=5)
    mock_run_curation(
        [
            fixture_article("topic-a", title="Stable Diffusion 3.5 releases new attention mechanism"),
            fixture_article("topic-b", title="Anthropic launches Claude Opus 5 for coding agents"),
        ],
        monkeypatch,
    )
    research = StubResearchAgent(
        settings, brief_kwargs={"verification_status": "insufficient_evidence"}
    )

    summary, _research, writer = _run(settings, monkeypatch, research=research)

    assert summary.status == "nothing_above_floor"
    assert len(summary.selected) == 2
    assert len(research.calls) == 2
    assert writer.calls == []


def test_spine_max_topic_attempts_per_run_bounds_a_hopeless_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unreachable target must not run up the whole day's pool — the
    attempts ceiling is the explicit worst-case cost/time bound, independent
    of both the target and how many candidates happen to be available."""
    settings = settings_for(
        tmp_path,
        max_topics_per_run=1,
        min_drafts_per_run=10,
        max_topic_attempts_per_run=2,
    )
    mock_run_curation(
        [
            fixture_article("topic-a", title="Stable Diffusion 3.5 releases new attention mechanism"),
            fixture_article("topic-b", title="Anthropic launches Claude Opus 5 for coding agents"),
            fixture_article("topic-c", title="OpenAI ships GPT-5.6 preview with faster inference"),
            fixture_article("topic-d", title="Meta releases Llama 5 open-weight model family"),
            fixture_article("topic-e", title="Mistral launches a new agentic coding model"),
        ],
        monkeypatch,
    )
    research = StubResearchAgent(
        settings, brief_kwargs={"verification_status": "insufficient_evidence"}
    )

    summary, _research, writer = _run(settings, monkeypatch, research=research)

    assert summary.status == "nothing_above_floor"
    assert len(research.calls) == 2
    assert writer.calls == []


def test_spine_times_out_a_slow_subagent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A subagent that overruns its budget is cancelled and reported as a
    timeout — one topic dies, the run survives."""
    settings = settings_for(tmp_path, research_task_timeout_seconds=1)
    mock_run_curation([fixture_article("topic-a")], monkeypatch)

    class _SlowAgent:
        calls: list[str] = []

        async def ainvoke(self, payload, config=None, **_):  # type: ignore[no-untyped-def]
            await asyncio.sleep(5)
            return {"messages": []}

    summary, _r, writer = _run(settings, monkeypatch, research=_SlowAgent())

    assert summary.research[0].status == "timeout"
    assert summary.status == "nothing_above_floor"
    assert writer.calls == []
