"""P6.3 — export_report coordinator tool tests.

Pins the wiring + behavior contract per the plan's P6.3 acceptance:

* Artifacts written: posts.md, run_report.json, briefs.json under
  ``<outputs_dir>/<YYYY-MM-DD>/``.
* Idempotency: a same-date bundle is refused by default with
  ``status="error" reason="overwrite_required"``; ``overwrite=True``
  overwrites.
* Date slug validation: the leaf must match ``YYYY-MM-DD`` exactly; any
  other shape (incl. path-fragments / relative escapes) is rejected.
* Schema-match: posts.md's briefs.json validates against
  ``ResearchBrief``; the tool's run_report carries counts + paths only
  (principle #3 — compressed pointer to the bundle, never the bundle's
  contents).
* Compressed summary back to the coordinator's LLM carries only
  ``{status, date, bundle_dir, files, counts}`` — never the bundle's
  artifacts themselves.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from app.config import Settings
from app.orchestrator.schemas import PostProposal, ResearchBrief, TopicCandidate
from app.orchestrator.tools.export import (
    ExportReportArgs,
    _DATE_SLUG_RE,
    _format_posts_md,
    _normalize_date_slug,
    build_export_report_tool,
)


def _settings(tmp_path: Path, **overrides: Any) -> Settings:
    base: dict[str, Any] = dict(
        _env_file=None,
        orchestrator_data_dir=str(tmp_path / "data"),
        outputs_dir=str(tmp_path / "outputs"),
        openrouter_api_key="",
    )
    base.update(overrides)
    return Settings(**base)


def _passing_proposal(post_id: str = "post-1", topic_id: str = "topic-a") -> PostProposal:
    body = (
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
    return PostProposal(
        post_id=post_id,
        angle="steady improvement",
        headline="Stable Diffusion 3.5's quiet attention refactor",
        body=body,
        hashtags=["#stablediffusion", "#attention", "#inference"],
        supporting_topic_ids=[topic_id],
        citation_urls=[f"https://example.com/{topic_id}"],
        confidence=0.8,
    )


def _brief(topic_id: str = "topic-a") -> ResearchBrief:
    return ResearchBrief(
        topic_id=topic_id,
        headline="Test headline",
        summary="A short brief.",
        technical_significance="A specific mechanism change in the model.",
        business_impact="Lower inference cost at parity quality.",
        why_now="Released recently.",
        key_points=["attention refactor"],
        risks=["unproven at scale"],
        citations=[
            {"title": "citation", "url": f"https://example.com/{topic_id}", "domain": "example.com"}
        ],
        verification_status="verified",
        verification_confidence=0.8,
    )


def _topic(topic_id: str = "topic-a") -> TopicCandidate:
    return TopicCandidate(
        topic_id=topic_id,
        title="Test topic",
        summary_hint="hint",
        primary_url=f"https://example.com/{topic_id}",
        primary_domain="example.com",
        score=0.9,
        cluster_size=1,
        rationale="why",
    )


def _stage_run(
    data_dir: Path,
    *,
    topics: list[TopicCandidate] | None = None,
    briefs: list[ResearchBrief] | None = None,
    drafts: list[tuple[PostProposal, dict[str, Any] | None]] | None = None,
    articles_count: int = 1,
) -> None:
    """Stage a complete run under data_dir: articles + topics + briefs +
    drafts + gate verdicts (whatever the test supplies). Each section is
    optional so a test can stage a partial run."""
    data_dir.mkdir(parents=True, exist_ok=True)

    if articles_count:
        (data_dir / "articles.json").write_text(
            json.dumps([{"id": f"a{i}"} for i in range(articles_count)])
        )

    if topics is not None:
        (data_dir / "topics.json").write_text(
            json.dumps([t.model_dump(mode="json") for t in topics], indent=2, default=str),
            encoding="utf-8",
        )

    if briefs is not None:
        briefs_dir = data_dir / "briefs"
        briefs_dir.mkdir(parents=True, exist_ok=True)
        for b in briefs:
            (briefs_dir / f"{b.topic_id}.json").write_text(
                b.model_dump_json(indent=2), encoding="utf-8"
            )
            # Stage the verified copy too — the export prefers .verified.json.
            (briefs_dir / f"{b.topic_id}.verified.json").write_text(
                b.model_dump_json(indent=2), encoding="utf-8"
            )

    if drafts is not None:
        drafts_dir = data_dir / "drafts"
        drafts_dir.mkdir(parents=True, exist_ok=True)
        for proposal, gate_verdict in drafts:
            (drafts_dir / f"{proposal.post_id}.json").write_text(
                proposal.model_dump_json(indent=2), encoding="utf-8"
            )
            if gate_verdict is not None:
                (drafts_dir / f"{proposal.post_id}.gate.json").write_text(
                    json.dumps(gate_verdict), encoding="utf-8"
                )


def _gate_verdict(post_id: str = "post-1", *, passed: bool = True) -> dict[str, Any]:
    return {
        "post_id": post_id,
        "passed": passed,
        "reasons": [] if passed else ["too short"],
        "word_count": 130,
        "hashtags_count": 3,
        "single_topic": True,
        "has_hype": False,
        "cleaned_body": "stub cleaned body",
    }


# --------------------------------------------------------------------------- #
# Date slug validation
# --------------------------------------------------------------------------- #


def test_normalize_date_slug_today_when_none() -> None:
    """date=None → today's UTC date stamp (YYYY-MM-DD shape)."""
    slug = _normalize_date_slug(None)
    assert _DATE_SLUG_RE.match(slug), f"today's slug must match the regex: {slug!r}"


def test_normalize_date_slug_passes_valid_shape() -> None:
    assert _normalize_date_slug("2026-07-21") == "2026-07-21"


@pytest.mark.parametrize(
    "bad",
    [
        "",
        "2026-7-21",
        "2026/07/21",
        "26-07-21",
        "2026-007-21",  # zero-padded month/day out of range — calendar-agnostic
        "20260-07-21",
        "2026-07-21/../..",
        "..",
        "/etc",
        "2026-07-21/",
        " 2026-07-21 ",
    ],
)
def test_normalize_date_slug_rejects_invalid_shape(bad: str) -> None:
    with pytest.raises(ValueError, match="YYYY-MM-DD"):
        _normalize_date_slug(bad)


# --------------------------------------------------------------------------- #
# Happy-path full bundle
# --------------------------------------------------------------------------- #


async def test_export_writes_three_files_with_expected_content(tmp_path: Path) -> None:
    """End-to-end: stage a run with 1 article + 1 topic + 1 verified brief +
    1 gated draft; export produces posts.md + run_report.json + briefs.json
    under <outputs>/<date>/ with schema-validated content."""
    s = _settings(tmp_path)
    data_dir = Path(s.orchestrator_data_dir)
    topic = _topic("topic-a")
    brief = _brief("topic-a")
    proposal = _passing_proposal("post-1", topic_id="topic-a")
    gate = _gate_verdict("post-1", passed=True)
    _stage_run(
        data_dir,
        topics=[topic],
        briefs=[brief],
        drafts=[(proposal, gate)],
        articles_count=1,
    )

    tool = build_export_report_tool(s)
    result_raw = await tool.ainvoke({"date": "2026-07-21"})
    result = json.loads(result_raw)

    assert result["status"] == "ok"
    assert result["date"] == "2026-07-21"
    bundle = Path(result["bundle_dir"])
    assert bundle == Path(s.outputs_dir) / "2026-07-21"

    # Three files exist.
    posts_md = bundle / "posts.md"
    run_report = bundle / "run_report.json"
    briefs_export = bundle / "briefs.json"
    assert posts_md.exists()
    assert run_report.exists()
    assert briefs_export.exists()

    # posts.md contains the post body, hashtags, citations, gate footer.
    md = posts_md.read_text(encoding="utf-8")
    assert proposal.headline in md
    assert proposal.body in md
    assert "#stablediffusion" in md
    assert "https://example.com/topic-a" in md
    assert "**Gate:** passed" in md
    assert "**Briefs:** topic-a" in md  # cross-link to the brief's topic_id

    # run_report.json carries counts + paths, validated structure.
    report = json.loads(run_report.read_text(encoding="utf-8"))
    assert report["date"] == "2026-07-21"
    assert report["counts"]["topics"] == 1
    assert report["counts"]["briefs"] == 1
    assert report["counts"]["drafts"] == 1
    assert report["counts"]["drafts_passed"] == 1
    assert report["counts"]["drafts_failed"] == 0
    assert report["counts"]["drafts_gate_absent"] == 0
    assert report["topics"][0]["topic_id"] == "topic-a"
    assert report["drafts"][0]["post_id"] == "post-1"
    assert report["drafts"][0]["gate_passed"] is True
    # paths-only — the report never copies the artifact's contents
    # (e.g. body / cleaned_body / brief summary) inside itself.
    assert "body" not in json.dumps(report)

    # briefs.json is an array of ResearchBrief-shaped objects.
    briefs_export_payload = json.loads(briefs_export.read_text(encoding="utf-8"))
    assert isinstance(briefs_export_payload, list)
    assert len(briefs_export_payload) == 1
    # Validate via the boundary model so schema drift surfaces here.
    parsed = ResearchBrief.model_validate(briefs_export_payload[0])
    assert parsed.topic_id == "topic-a"


async def test_export_summary_compressed_no_bundle_contents(tmp_path: Path) -> None:
    """Principle #3: the compressed summary the coordinator's LLM gets back
    never carries the bundle's contents. Keys are only {status, date,
    bundle_dir, files, counts}."""
    s = _settings(tmp_path)
    data_dir = Path(s.orchestrator_data_dir)
    proposal = _passing_proposal("post-1")
    gate = _gate_verdict("post-1")
    _stage_run(
        data_dir,
        topics=[_topic()],
        briefs=[_brief()],
        drafts=[(proposal, gate)],
        articles_count=1,
    )
    tool = build_export_report_tool(s)
    result_raw = await tool.ainvoke({"date": "2026-07-21"})
    result = json.loads(result_raw)
    assert set(result.keys()) <= {
        "status",
        "date",
        "bundle_dir",
        "files",
        "counts",
    }
    assert "body" not in result_raw
    assert proposal.body[:30] not in result_raw
    assert "cleaned_body" not in result_raw


async def test_export_brief_with_defaults_round_trips(tmp_path: Path) -> None:
    """A brief with the schema defaults (verification_status='unverified',
    empty citations/key_points/risks) round-trips through ResearchBrief's
    model_validate. Pins that the schema-match acceptance in the plan
    holds for the default-population path, not just the populated path.
    Addresses subagent code-review minor M3."""
    s = _settings(tmp_path)
    data_dir = Path(s.orchestrator_data_dir)
    # Construct a minimal brief: no optional fields set.
    bare_brief = ResearchBrief(
        topic_id="topic-bare",
        headline="Bare headline",
        summary="Bare summary.",
        technical_significance="Bare ts.",
        business_impact="Bare bi.",
        why_now="Bare wn.",
    )
    # Defaults present as the model_field defaults surface:
    assert bare_brief.verification_status == "unverified"
    assert bare_brief.citations == []
    assert bare_brief.key_points == []
    topic = _topic("topic-bare")
    proposal = _passing_proposal("post-bare", topic_id="topic-bare")
    gate = _gate_verdict("post-bare", passed=True)
    _stage_run(
        data_dir,
        topics=[topic],
        briefs=[bare_brief],
        drafts=[(proposal, gate)],
        articles_count=1,
    )

    tool = build_export_report_tool(s)
    result = json.loads(await tool.ainvoke({"date": "2026-07-21"}))
    assert result["status"] == "ok"

    bundle = Path(result["bundle_dir"])
    briefs_payload = json.loads(
        (bundle / "briefs.json").read_text(encoding="utf-8")
    )
    assert len(briefs_payload) == 1
    parsed = ResearchBrief.model_validate(briefs_payload[0])
    assert parsed.topic_id == "topic-bare"
    # The default surface-field values round-tripped:
    assert parsed.verification_status == "unverified"
    assert parsed.citations == []
    assert parsed.key_points == []
    assert parsed.risks == []
    assert parsed.verification_confidence == 0.0
    assert parsed.verification_notes == []


async def test_export_run_report_gate_passed_carries_strict_bool(
    tmp_path: Path,
) -> None:
    """The run_report's per-draft ``gate_passed`` field carries the same
    strict-bool-identity contract deliver_telegram (P6.2) enforces. Pins
    that a malformed gate verdict with ``passed="True"`` (string) doesn't
    leak the string into run_report — surfaces as None consistently with
    the deliver path's rejection. Addresses subagent code-review minor M2."""
    s = _settings(tmp_path)
    data_dir = Path(s.orchestrator_data_dir)
    proposal = _passing_proposal("post-1")
    # Malformed: passed="True" (string) — a model-written helpfully-overridden
    # verdict file would inject this. The export surfaces it as None, NOT
    # as the raw string — same strict contract deliver_telegram enforces.
    bad_gate = {
        "post_id": "post-1",
        "passed": "True",  # string, not bool
        "reasons": [],
    }
    _stage_run(
        data_dir,
        topics=[_topic()],
        briefs=[_brief()],
        drafts=[(proposal, bad_gate)],
        articles_count=1,
    )
    tool = build_export_report_tool(s)
    result = json.loads(await tool.ainvoke({"date": "2026-07-21"}))
    assert result["status"] == "ok"
    bundle = Path(result["bundle_dir"])
    report = json.loads((bundle / "run_report.json").read_text(encoding="utf-8"))
    assert report["counts"]["drafts_passed"] == 0  # not counted as passed
    assert report["counts"]["drafts_failed"] == 0
    # The strict identity check counts the malformed verdict as neither
    # passed, failed, NOR absent — the gate file *exists* (it's just
    # malformed). drafts_gate_absent counts draft-with-no-gate-file at
    # all; this draft DOES have a gate file. All three counts are thus 0,
    # and the per-draft gate_passed field surfaces None so the consumer
    # sees the verdict isn't a real bool True/False.
    assert report["counts"]["drafts_gate_absent"] == 0
    per_draft_gate = report["drafts"][0]["gate_passed"]
    assert per_draft_gate is None  # not the raw string "True"


# --------------------------------------------------------------------------- #
# Idempotency guard
# --------------------------------------------------------------------------- #


async def test_export_refuses_to_overwrite_existing_bundle_by_default(tmp_path: Path) -> None:
    """A same-date non-empty bundle is refused unless overwrite=True —
    the tool's 'don't be a footgun' law. Pinned so a coordinator LLM that
    calls export twice on the same date doesn't blow away the first bundle."""
    s = _settings(tmp_path)
    data_dir = Path(s.orchestrator_data_dir)
    proposal = _passing_proposal("post-1")
    gate = _gate_verdict("post-1")
    _stage_run(
        data_dir,
        topics=[_topic()],
        briefs=[_brief()],
        drafts=[(proposal, gate)],
        articles_count=1,
    )
    tool = build_export_report_tool(s)

    # First write succeeds.
    first = json.loads(await tool.ainvoke({"date": "2026-07-21"}))
    assert first["status"] == "ok"

    # Second write — same date, no overwrite — is refused.
    second = json.loads(await tool.ainvoke({"date": "2026-07-21"}))
    assert second["status"] == "error"
    assert second["reason"] == "overwrite_required"
    assert "bundle_dir" in second
    # The prior bundle is intact.
    assert (Path(second["bundle_dir"]) / "posts.md").exists()


async def test_export_overwrite_true_replaces_prior_bundle(tmp_path: Path) -> None:
    """overwrite=True lets a same-date export replace the prior bundle —
    the explicit opt-in. Pins the tool's no-accidental-overwrite law
    doesn't lock operators out of re-running for any reason."""
    s = _settings(tmp_path)
    data_dir = Path(s.orchestrator_data_dir)
    proposal = _passing_proposal("post-1")
    gate = _gate_verdict("post-1")
    _stage_run(
        data_dir,
        topics=[_topic()],
        briefs=[_brief()],
        drafts=[(proposal, gate)],
        articles_count=1,
    )
    tool = build_export_report_tool(s)

    first = json.loads(await tool.ainvoke({"date": "2026-07-21"}))
    assert first["status"] == "ok"

    # Drop the artifacts + re-stage with different content so the second
    # export would produce different posts.md content. Without overwrite, it
    # would still be refused.
    second = json.loads(await tool.ainvoke({"date": "2026-07-21", "overwrite": True}))
    assert second["status"] == "ok"
    assert Path(second["bundle_dir"]).exists()


async def test_export_same_date_empty_bundle_directory_is_not_refused(tmp_path: Path) -> None:
    """An *empty* same-date bundle directory (e.g. a prior failed run left
    behind a created-but-empty dir) is allowed without overwrite —
    the footgun guard fires on *non-empty existing*, not *existing*."""
    s = _settings(tmp_path)
    data_dir = Path(s.orchestrator_data_dir)
    proposal = _passing_proposal("post-1")
    gate = _gate_verdict("post-1")
    _stage_run(
        data_dir,
        topics=[_topic()],
        briefs=[_brief()],
        drafts=[(proposal, gate)],
        articles_count=1,
    )
    # Pre-create an empty bundle dir.
    bundle = Path(s.outputs_dir) / "2026-07-21"
    bundle.mkdir(parents=True, exist_ok=True)
    assert bundle.exists() and not any(bundle.iterdir())

    tool = build_export_report_tool(s)
    result = json.loads(await tool.ainvoke({"date": "2026-07-21"}))
    assert result["status"] == "ok"


# --------------------------------------------------------------------------- #
# Invalid date slug
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "bad_date",
    ["", "2026-7-21", "2026/07/21", "26-07-21", "..", "/etc", "2026-07-21/../.."],
)
async def test_export_invalid_date_returns_error(tmp_path: Path, bad_date: str) -> None:
    """The date argument goes directly into the export leaf path; even when
    the slug validator is bypassed accidentally, the LLM can't accidentally
    route the export outside <outputs_dir>."""
    s = _settings(tmp_path)
    tool = build_export_report_tool(s)
    result = json.loads(await tool.ainvoke({"date": bad_date}))
    assert result["status"] == "error"
    assert result["reason"] == "invalid_date"
    assert "YYYY-MM-DD" in result["error"]


# --------------------------------------------------------------------------- #
# Partial runs (missing artifacts)
# --------------------------------------------------------------------------- #


async def test_export_partial_run_no_artifacts(tmp_path: Path) -> None:
    """A data dir with nothing in it still produces a valid (empty) export
    bundle — the tool's stance is 'export what's there, surface absence
    honestly' (counts=0 for everything)."""
    s = _settings(tmp_path)
    Path(s.orchestrator_data_dir).mkdir(parents=True, exist_ok=True)
    tool = build_export_report_tool(s)
    result = json.loads(await tool.ainvoke({"date": "2026-07-21"}))
    assert result["status"] == "ok"
    assert result["counts"]["topics"] == 0
    assert result["counts"]["briefs"] == 0
    assert result["counts"]["drafts"] == 0

    bundle = Path(result["bundle_dir"])
    md = (bundle / "posts.md").read_text(encoding="utf-8")
    assert "(no drafts produced)" in md


async def test_export_partial_run_only_drafts_no_topics(tmp_path: Path) -> None:
    """When the run produced drafts but no topics (e.g. manual backfill),
    the drafts are still exported — but **no briefs** are loaded (brief
    iteration is keyed off the topics list)."""
    s = _settings(tmp_path)
    data_dir = Path(s.orchestrator_data_dir)
    proposal = _passing_proposal("post-1")
    gate = _gate_verdict("post-1")
    _stage_run(data_dir, topics=None, briefs=None, drafts=[(proposal, gate)])
    tool = build_export_report_tool(s)
    result = json.loads(await tool.ainvoke({"date": "2026-07-21"}))
    assert result["status"] == "ok"
    assert result["counts"]["drafts"] == 1
    assert result["counts"]["briefs"] == 0
    # The posts.md's briefs-cross-link footer isn't rendered (no briefs).
    md = (Path(result["bundle_dir"]) / "posts.md").read_text(encoding="utf-8")
    assert "Briefs:" not in md


async def test_export_partial_run_draft_with_no_gate_verdict(tmp_path: Path) -> None:
    """A draft whose gate verdict is absent counts as drafts_gate_absent;
    its Gate footer in posts.md says "absent". The export doesn't fail —
    partial artifacts surface honestly in counts."""
    s = _settings(tmp_path)
    data_dir = Path(s.orchestrator_data_dir)
    proposal = _passing_proposal("post-1")
    _stage_run(
        data_dir,
        topics=[_topic()],
        briefs=[_brief()],
        drafts=[(proposal, None)],
    )
    tool = build_export_report_tool(s)
    result = json.loads(await tool.ainvoke({"date": "2026-07-21"}))
    assert result["status"] == "ok"
    assert result["counts"]["drafts_gate_absent"] == 1
    md = (Path(result["bundle_dir"]) / "posts.md").read_text(encoding="utf-8")
    assert "**Gate:** absent" in md


async def test_export_partial_run_failed_gate(tmp_path: Path) -> None:
    """A draft the gate failed counts as drafts_failed; its Gate footer
    says "failed". The export doesn't gate the failing draft — exporting it
    is informational, not delivery."""
    s = _settings(tmp_path)
    data_dir = Path(s.orchestrator_data_dir)
    proposal = _passing_proposal("post-1")
    failed_gate = _gate_verdict("post-1", passed=False)
    _stage_run(
        data_dir,
        topics=[_topic()],
        briefs=[_brief()],
        drafts=[(proposal, failed_gate)],
    )
    tool = build_export_report_tool(s)
    result = json.loads(await tool.ainvoke({"date": "2026-07-21"}))
    assert result["status"] == "ok"
    assert result["counts"]["drafts_failed"] == 1
    md = (Path(result["bundle_dir"]) / "posts.md").read_text(encoding="utf-8")
    assert "**Gate:** failed" in md


# --------------------------------------------------------------------------- #
# formatter as a pure unit (no I/O)
# --------------------------------------------------------------------------- #


def test_format_posts_md_empty_drafts_section() -> None:
    """Empty drafts list renders the (no drafts produced) sentinel so an
    empty bundle's posts.md isn't an empty file."""
    md = _format_posts_md([], [], "2026-07-21", {})
    assert "no drafts produced" in md
    assert "2026-07-21" in md


def test_format_posts_md_skips_empty_sections() -> None:
    """If a proposal has no hashtags and no citations, those sections don't
    render — no stray empty bullet."""
    proposal = _passing_proposal("post-1")
    proposal.hashtags = []
    proposal.citation_urls = []
    draft_triple = (proposal, {"passed": True}, proposal.model_dump(mode="json"))
    md = _format_posts_md(
        [draft_triple], [], "2026-07-21", {"post-1": {"provenance_ok": True, "floor_ok": True, "floor_reason": ""}}
    )
    assert "**Citations:**" not in md
    assert "**Gate:** passed" in md
    assert proposal.body in md


def test_format_posts_md_gate_footer_states() -> None:
    """The Gate footer renders as `passed` / `failed` / `absent` per the
    verdict state — symmetry with the counts in run_report."""
    proposal = _passing_proposal("post-1")
    raw = proposal.model_dump(mode="json")
    integrity = {"post-1": {"provenance_ok": True, "floor_ok": True, "floor_reason": ""}}
    # passed
    assert "**Gate:** passed" in _format_posts_md(
        [(proposal, {"passed": True}, raw)], [], "2026-07-21", integrity
    )
    # failed
    assert "**Gate:** failed" in _format_posts_md(
        [(proposal, {"passed": False}, raw)], [], "2026-07-21", integrity
    )
    # absent (gate_verdict None or missing the key)
    assert "**Gate:** absent" in _format_posts_md(
        [(proposal, None, raw)], [], "2026-07-21", integrity
    )
    assert "**Gate:** absent" in _format_posts_md(
        [(proposal, {"post_id": "post-1"}, raw)], [], "2026-07-21", integrity
    )


# --------------------------------------------------------------------------- #
# Factory + singleton
# --------------------------------------------------------------------------- #


def test_singleton_build_does_not_require_settings() -> None:
    """The module-level singleton builds at import time without settings —
    same pattern as every sibling tool."""
    from app.orchestrator.tools.export import export_report_tool

    assert export_report_tool.name == "export_report"
    assert export_report_tool.args_schema is ExportReportArgs


async def test_factory_lazy_settings_resolution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """settings=None falls back to the lru_cache via get_settings() (parity
    with every sibling tool). Pin via env var so a None doesn't silently
    write into the real repo's data tree."""
    from app.config import get_settings

    monkeypatch.setenv("ORCHESTRATOR_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("OUTPUTS_DIR", str(tmp_path / "outputs"))
    get_settings.cache_clear()
    try:
        tool = build_export_report_tool()  # no settings supplied
        result = json.loads(await tool.ainvoke({"date": "2026-07-21"}))
        assert result["status"] == "ok"
        bundle = Path(result["bundle_dir"])
        assert bundle == tmp_path / "outputs" / "2026-07-21"
    finally:
        get_settings.cache_clear()
        monkeypatch.delenv("ORCHESTRATOR_DATA_DIR", raising=False)
        monkeypatch.delenv("OUTPUTS_DIR", raising=False)