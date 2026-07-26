"""P6.2 — deliver_telegram coordinator tool tests.

Pins the wiring + behavior contract:

* The tool hard-codes ``bot="linkedin"`` — a LinkedIn post proposal can't
  be routed anywhere else; the plan's risk row names silent misroute to
  the news chat as the failure mode the tool exists to prevent.
* Refuses to ship a draft whose ``drafts/<post_id>.gate.json`` verdict
  doesn't report ``passed=True`` — surfaces ``status="error" reason=
  "gate_not_passed"`` (or ``"gate_verdict_missing"`` if no verdict file
  exists) rather than sending a draft the writer hasn't certified.
* Reads ``drafts/<post_id>.json``, validates the JSON against
  ``PostProposal``, formats the body + hashtags + citations as a single
  Telegram text message, sends via ``TelegramClient.send_message``.
* Per guiding principle #3 the summary the tool returns to the
  coordinator's LLM never contains the post body — only ``post_id``,
  routing info, and a delivery status.
* Path-traversal guarded at the post_id (inline rule, parity with
  quality.py's own — centralization onto state._validate_slug is the
  documented P5.1 follow-up that pre-dates this PR and stays out of
  scope).
* Telegram send retry + token redaction inherited from P6.1's
  ``_post_with_retry``; covered in ``tests/test_telegram_send.py`` —
  not re-tested here.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from app.config import Settings
from app.orchestrator.schemas import PostProposal
from app.orchestrator.tools.telegram import (
    DeliverTelegramArgs,
    _format_post_message,
    build_deliver_telegram_tool,
)


def _settings(tmp_path: Path, **overrides: Any) -> Settings:
    base: dict[str, Any] = dict(
        _env_file=None,
        orchestrator_data_dir=str(tmp_path),
        openrouter_api_key="",
        telegram_linkedin_bot_token="li-tok",
        telegram_linkedin_chat_id="li-chat",
        request_timeout_seconds=10,
    )
    base.update(overrides)
    return Settings(**base)


def _passing_proposal(post_id: str = "post-1", topic_id: str = "topic-a") -> PostProposal:
    """A proposal that passes the quality gate (word count in [105, 182],
    single topic_id, 3 hashtags, no hype markers). Reused from
    tests/test_e2e_dryrun.py's stub to keep cross-test fixtures aligned."""
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


def _draft(tmp_path: Path, post_id: str = "post-1", *, proposal: PostProposal | None = None) -> Path:
    """Stage a SIGNED draft at drafts/<post_id>.json — the shape the writer's
    submit_draft tool produces (provenance block included). Delivery refuses
    unsigned drafts, so the fixture must sign like production does. Returns
    the written path."""
    from app.orchestrator.tools.draft import write_signed_draft

    p = proposal or _passing_proposal(post_id)
    return write_signed_draft(p, _settings(tmp_path))


def _unsigned_draft(tmp_path: Path, post_id: str = "post-1") -> Path:
    """Stage a draft the way an LLM would via write_file — valid PostProposal
    JSON, NO provenance block. Used to pin the provenance refusal."""
    p = _passing_proposal(post_id)
    draft_dir = tmp_path / "drafts"
    draft_dir.mkdir(parents=True, exist_ok=True)
    path = draft_dir / f"{post_id}.json"
    path.write_text(p.model_dump_json(indent=2), encoding="utf-8")
    return path


def _brief(tmp_path: Path, topic_id: str = "topic-a", *, status: str = "verified") -> Path:
    """Stage a verified brief whose citations cover the passing proposal's
    citation_urls, so the evidence floor + citation tracing pass. Returns
    the written path."""
    brief_dir = tmp_path / "briefs"
    brief_dir.mkdir(parents=True, exist_ok=True)
    path = brief_dir / f"{topic_id}.verified.json"
    payload = {
        "topic_id": topic_id,
        "headline": "Stable Diffusion 3.5 attention refactor",
        "summary": "A new attention mechanism lowers serving cost.",
        "technical_significance": "Throughput improvement at parity quality.",
        "business_impact": "Lower per-image cost for production serving.",
        "why_now": "Released this week.",
        "key_points": ["new attention path", "throughput up"],
        "risks": [],
        "citations": [
            {
                "title": "SD3.5 release notes",
                "url": f"https://example.com/{topic_id}",
                "domain": "example.com",
            }
        ],
        "verification_status": status,
        "verification_confidence": 0.9,
        "verification_notes": [],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _gate(
    tmp_path: Path,
    post_id: str = "post-1",
    *,
    passed: bool = True,
    reasons: list[str] | None = None,
) -> Path:
    """Stage a gate verdict at drafts/<post_id>.gate.json. Returns the path."""
    draft_dir = tmp_path / "drafts"
    draft_dir.mkdir(parents=True, exist_ok=True)
    path = draft_dir / f"{post_id}.gate.json"
    payload = {
        "post_id": post_id,
        "passed": passed,
        "reasons": reasons or [],
        "word_count": 130,
        "hashtags_count": 3,
        "single_topic": True,
        "has_hype": False,
        "cleaned_body": "stub cleaned body",
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# _format_post_message — pure formatter, no network
# --------------------------------------------------------------------------- #


def test_format_includes_headline_body_hashtags_citations() -> None:
    proposal = _passing_proposal()
    msg = _format_post_message(proposal)
    # All four sections land in the assembled message.
    assert proposal.headline in msg
    assert proposal.body in msg
    assert "#stablediffusion" in msg
    assert "#attention" in msg
    assert "#inference" in msg
    assert "https://example.com/topic-a" in msg
    # Citations are numbered.
    assert "1. <a href=" in msg


def test_format_skips_empty_sections() -> None:
    """If the proposal has no hashtags or no citations, those sections don't
    appear at all — not as empty lines or empty list markers."""
    proposal = _passing_proposal()
    proposal.hashtags = []
    proposal.citation_urls = []
    msg = _format_post_message(proposal)
    assert "1. <a href=" not in msg  # no citations block
    # No stray empty newline blocks.
    assert "\n\n\n" not in msg


def test_format_html_escapes_title_and_body() -> None:
    """The headline + body are user-typed prose; HTML-escape them so an em
    dash or a literal < doesn't break the Telegram HTML parser. Citations
    URLs are escaped via quote=True so they survive the href attribute."""
    proposal = _passing_proposal()
    proposal.headline = "A <b>bold</b> title & a 'quote'"
    proposal.body = "Body with <script> & ampersand."
    msg = _format_post_message(proposal)
    assert "<b>bold</b>" not in msg  # the literal <b> was escaped
    assert "&lt;b&gt;bold&lt;/b&gt;" in msg
    assert "&amp;" in msg  # the & was escaped to &amp;
    # The citation URL still appears inside a clean href (quote-escaped):
    assert '<a href="https://example.com/topic-a">' in msg


def test_format_truncates_overlong_assembled_message() -> None:
    """Defense-in-depth: an overlong assembled message (head + body + tags +
    citations) is truncated at ``_MAX_MESSAGE_LEN`` — never 400s the
    sendMessage call. Practically never happens at the gate's 182-word cap,
    but a future gate relaxation or a citations blowup would trip this."""
    from app.orchestrator.tools.telegram import _MAX_MESSAGE_LEN

    proposal = _passing_proposal()
    # Pump the citations list way past the point where it'd exceed the cap
    # even at the gate's 182-word body — 4096 chars / ~30 chars per citation
    # line -> ~140 citations.
    proposal.citation_urls = [f"https://example.com/long/url/{i:05d}" for i in range(200)]
    msg = _format_post_message(proposal)
    assert len(msg) <= _MAX_MESSAGE_LEN
    assert msg.endswith("...")


# --------------------------------------------------------------------------- #
# Tool: error branches (no network)
# --------------------------------------------------------------------------- #


async def test_invalid_post_id_rejected_before_path_resolution(tmp_path: Path) -> None:
    """Path-traversal guard at the post_id — surface as status=error with
    reason=invalid_post_id; no read attempt."""
    tool = build_deliver_telegram_tool(_settings(tmp_path))
    for bad in ("a/b", "..", ".", "", "/leading", "endslash/"):
        result_raw = await tool.ainvoke({"post_id": bad})
        result = json.loads(result_raw)
        assert result["status"] == "error"
        assert result["reason"] == "invalid_post_id"
        assert "post_id" in result


async def test_draft_not_found(tmp_path: Path) -> None:
    """A post_id with no draft on disk surfaces status=error reason=
    draft_not_found — never falls through to a guess."""
    tool = build_deliver_telegram_tool(_settings(tmp_path))
    result_raw = await tool.ainvoke({"post_id": "ghost"})
    result = json.loads(result_raw)
    assert result["status"] == "error"
    assert result["reason"] == "draft_not_found"


async def test_draft_invalid_json(tmp_path: Path) -> None:
    """Malformed draft JSON surfaces status=error reason=draft_invalid_json;
    the model doesn't guess a partial parse."""
    tool = build_deliver_telegram_tool(_settings(tmp_path))
    draft_dir = tmp_path / "drafts"
    draft_dir.mkdir(parents=True, exist_ok=True)
    (draft_dir / "post-bad.json").write_text("{not json}", encoding="utf-8")
    result_raw = await tool.ainvoke({"post_id": "post-bad"})
    result = json.loads(result_raw)
    assert result["status"] == "error"
    assert result["reason"] == "draft_invalid_json"


async def test_gate_verdict_missing(tmp_path: Path) -> None:
    """Draft on disk but no .gate.json verdict → refuse delivery. The gate is
    the last deterministic check; a missing verdict means the writer skipped
    a step — surface honestly, don't ship."""
    tool = build_deliver_telegram_tool(_settings(tmp_path))
    _draft(tmp_path, "post-1")
    result_raw = await tool.ainvoke({"post_id": "post-1"})
    result = json.loads(result_raw)
    assert result["status"] == "error"
    assert result["reason"] == "gate_verdict_missing"


async def test_unsigned_draft_refuses_delivery(tmp_path: Path) -> None:
    """A draft written outside the writer's submit_draft tool (valid
    PostProposal JSON, no provenance signature — exactly what an LLM's
    write_file produces) is refused BEFORE the gate is even consulted. This
    is the deterministic stop for the 2026-07-25 failure mode (coordinator
    self-authored + self-gated all five drafts)."""
    tool = build_deliver_telegram_tool(_settings(tmp_path))
    _unsigned_draft(tmp_path, "post-1")
    _gate(tmp_path, "post-1")
    _brief(tmp_path)
    result_raw = await tool.ainvoke({"post_id": "post-1"})
    result = json.loads(result_raw)
    assert result["status"] == "error"
    assert result["reason"] == "provenance_invalid"


async def test_tampered_draft_refuses_delivery(tmp_path: Path) -> None:
    """A signed draft whose body was edited after signing (signature no
    longer matches the payload) is refused — provenance covers content,
    not just origin."""
    tool = build_deliver_telegram_tool(_settings(tmp_path))
    draft_path = _draft(tmp_path, "post-1")
    payload = json.loads(draft_path.read_text(encoding="utf-8"))
    payload["body"] = payload["body"] + " Tampered sentence."
    draft_path.write_text(json.dumps(payload), encoding="utf-8")
    _gate(tmp_path, "post-1")
    _brief(tmp_path)
    result_raw = await tool.ainvoke({"post_id": "post-1"})
    result = json.loads(result_raw)
    assert result["status"] == "error"
    assert result["reason"] == "provenance_invalid"


async def test_below_floor_brief_refuses_delivery(tmp_path: Path) -> None:
    """A signed, gate-passed draft whose brief is below the evidence floor
    (single citation, low confidence) is refused at send time — the floor
    is enforced at every choke point, not just upstream."""
    tool = build_deliver_telegram_tool(_settings(tmp_path))
    _draft(tmp_path, "post-1")
    _gate(tmp_path, "post-1")
    _brief(tmp_path, status="partially_verified")
    # The fixture brief has 1 citation + we need low confidence: rewrite it.
    brief_path = tmp_path / "briefs" / "topic-a.verified.json"
    payload = json.loads(brief_path.read_text(encoding="utf-8"))
    payload["verification_confidence"] = 0.3
    brief_path.write_text(json.dumps(payload), encoding="utf-8")
    result_raw = await tool.ainvoke({"post_id": "post-1"})
    result = json.loads(result_raw)
    assert result["status"] == "error"
    assert result["reason"] == "verification_floor"


async def test_gate_verdict_not_passed_refuses_delivery(tmp_path: Path) -> None:
    """A draft whose gate verdict reports passed=False is refused — surfaces
    status=error reason=gate_not_passed with gate_passed=False. Catches the
    "model helpfully overrode the gate" failure mode at the coordinator seam."""
    tool = build_deliver_telegram_tool(_settings(tmp_path))
    _draft(tmp_path, "post-1")
    _gate(tmp_path, "post-1", passed=False, reasons=["too short"])
    result_raw = await tool.ainvoke({"post_id": "post-1"})
    result = json.loads(result_raw)
    assert result["status"] == "error"
    assert result["reason"] == "gate_not_passed"
    assert result["gate_passed"] is False
    # The summary does NOT inline the gate's reasons list (that lives on disk).
    assert "reasons" not in result


async def test_gate_verdict_invalid_json(tmp_path: Path) -> None:
    """A malformed gate verdict is treated as an error — never silently
    interpreted as a pass."""
    tool = build_deliver_telegram_tool(_settings(tmp_path))
    _draft(tmp_path, "post-1")
    gate_path = tmp_path / "drafts" / "post-1.gate.json"
    gate_path.write_text("{not json}", encoding="utf-8")
    result_raw = await tool.ainvoke({"post_id": "post-1"})
    result = json.loads(result_raw)
    assert result["status"] == "error"
    assert result["reason"] == "gate_verdict_invalid_json"


# --------------------------------------------------------------------------- #
# Tool: dry-run + send (mocked TelegramClient.send_message)
# --------------------------------------------------------------------------- #


async def test_delivery_dry_run_when_linkedin_profile_unconfigured(tmp_path: Path) -> None:
    """When the linkedin bot isn't configured (token OR chat_id empty), the
    tool falls back to dry-run rather than sending through the network —
    surfaces status=dry_run bot=linkedin + a preview_chars count. This is
    the P1.2 "dry-run needs no new required keys" invariant enforced here
    on the consumer side."""
    s = _settings(
        tmp_path,
        telegram_linkedin_bot_token=None,
        telegram_linkedin_chat_id=None,
    )
    tool = build_deliver_telegram_tool(s)
    _draft(tmp_path, "post-1")
    _gate(tmp_path, "post-1")
    _brief(tmp_path)
    result_raw = await tool.ainvoke({"post_id": "post-1"})
    result = json.loads(result_raw)
    assert result["status"] == "dry_run"
    assert result["bot"] == "linkedin"
    assert isinstance(result["preview_chars"], int)
    assert result["preview_chars"] > 0
    # The post body is never in the coordinator-level summary (principle-#3).
    body = _passing_proposal().body
    assert body[:40] not in json.dumps(result)
    assert "message_id" not in result or result["message_id"] is None


async def test_delivery_send_success(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Happy path: a gated + verified draft is formatted + sent to the
    linkedin bot, the result carries status=sent + bot=linkedin +
    message_id. Asserts via a monkeypatched TelegramClient.send_message
    that records the bot kwarg (so a future refactor that drops the
    hard-coded bot=linkedin surfaces here)."""
    sent_calls: list[dict[str, Any]] = []

    async def _stub_send_message(self, text, *, bot="news", dry_run=False, disable_web_page_preview=True):
        sent_calls.append({"text": text, "bot": bot, "dry_run": dry_run, "disable_web_page_preview": disable_web_page_preview})
        return {"status": "sent", "bot": bot, "message_id": 99}

    from app.services import telegram_client as tc

    monkeypatch.setattr(tc.TelegramClient, "send_message", _stub_send_message)

    tool = build_deliver_telegram_tool(_settings(tmp_path))
    _draft(tmp_path, "post-1")
    _gate(tmp_path, "post-1")
    _brief(tmp_path)
    result_raw = await tool.ainvoke({"post_id": "post-1"})
    result = json.loads(result_raw)

    assert result["status"] == "sent"
    assert result["bot"] == "linkedin"
    assert result["message_id"] == 99
    assert sent_calls, "send_message should have been called once"
    assert sent_calls[0]["bot"] == "linkedin"
    # disable_web_page_preview is True (LinkedIn post — no auto-previews).
    assert sent_calls[0]["disable_web_page_preview"] is True
    # The sent text carries the post's body + hashtags + citations.
    sent_text = sent_calls[0]["text"]
    assert "Stable Diffusion 3.5" in sent_text
    assert "#stablediffusion" in sent_text
    assert "https://example.com/topic-a" in sent_text


async def test_delivery_routes_only_to_linkedin_never_news(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Critical parity: the tool hard-codes ``bot="linkedin"``. Even when
    the news profile is configured and the linkedin profile is configured,
    a LinkedIn post proposal never lands at the news chat. Catches a future
    "convenience drop the bot arg" refactor that silently misroutes."""
    sent_bots: list[str] = []

    async def _stub_send_message(self, text, *, bot="news", dry_run=False, disable_web_page_preview=True):
        sent_bots.append(bot)
        return {"status": "sent", "bot": bot, "message_id": 1}

    from app.services import telegram_client as tc

    monkeypatch.setattr(tc.TelegramClient, "send_message", _stub_send_message)

    tool = build_deliver_telegram_tool(_settings(tmp_path))
    _draft(tmp_path, "post-1")
    _gate(tmp_path, "post-1")
    _brief(tmp_path)
    await tool.ainvoke({"post_id": "post-1"})

    assert sent_bots == ["linkedin"]


async def test_summary_never_contains_post_body(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Principle #3: the tool's compressed summary doesn't carry the post
    body (or the full message text). The coordinator branches on status
    alone; the body lives on disk at drafts/<post_id>.json."""
    from app.services import telegram_client as tc

    async def _stub_send_message(self, text, *, bot="news", dry_run=False, disable_web_page_preview=True):
        return {"status": "sent", "bot": bot, "message_id": 7}

    monkeypatch.setattr(tc.TelegramClient, "send_message", _stub_send_message)
    tool = build_deliver_telegram_tool(_settings(tmp_path))
    _draft(tmp_path, "post-1")
    _gate(tmp_path, "post-1")
    _brief(tmp_path)
    result_raw = await tool.ainvoke({"post_id": "post-1"})
    body = _passing_proposal().body
    assert body[:40] not in result_raw
    assert "Stable Diffusion 3.5" not in result_raw
    result = json.loads(result_raw)
    assert result["status"] == "sent"
    # The summary keys are exactly the routing set — no body, no headline.
    expected_keys = {"post_id", "status", "bot", "message_id", "preview_chars", "error"}
    assert set(result.keys()) <= expected_keys


def test_singleton_build_does_not_require_settings(tmp_path: Path) -> None:
    """The module-level convenience singleton builds at import time without
    requiring settings — same pattern as fetch_curated_ai_news_tool /
    technical_rank_tool / quality_gate_tool. Pins the seam so a future
    lazy-init refactor doesn't break production imports."""
    from app.orchestrator.tools.telegram import deliver_telegram_tool

    assert deliver_telegram_tool.name == "deliver_telegram"
    # The args schema is the documented one.
    assert deliver_telegram_tool.args_schema is DeliverTelegramArgs


@pytest.mark.parametrize(
    "bad_passed",
    ["True", "true", "yes", 1, 1.0, ["no"], {"__bool__": True}, "passed"],
)
async def test_gate_verdict_malformed_passed_value_refuses_delivery(
    tmp_path: Path, bad_passed: Any
) -> None:
    """A gate verdict whose ``passed`` field is truthy but NOT exactly the
    Python bool ``True`` is refused. Pins the strict-identity contract
    against a model-written verdict file helpfully stuffing 'True' / 1 /
    ['yes'] / etc. — exactly the "model helpfully overrode the gate"
    failure mode the tool exists to prevent. (Subagent code-review B1.)"""
    tool = build_deliver_telegram_tool(_settings(tmp_path))
    _draft(tmp_path, "post-1")
    gate_path = tmp_path / "drafts" / "post-1.gate.json"
    gate_path.write_text(
        json.dumps({"post_id": "post-1", "passed": bad_passed}),
        encoding="utf-8",
    )
    result_raw = await tool.ainvoke({"post_id": "post-1"})
    result = json.loads(result_raw)
    assert result["status"] == "error"
    assert result["reason"] == "gate_not_passed"
    # The summary reports the leaked value (helps the coordinator surface
    # the verdict's actual shape, not the literal True/False).
    assert result["gate_passed"] == bad_passed


async def test_factory_lazy_settings_resolution(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """When settings=None is supplied (the production path), the factory
    falls back to the lru_cache via ``get_settings()``. Pin via env var
    so an accidentally-passed None doesn't silently write into the real
    repo's data tree."""
    from app.config import get_settings

    monkeypatch.setenv("ORCHESTRATOR_DATA_DIR", str(tmp_path))
    get_settings.cache_clear()
    try:
        tool = build_deliver_telegram_tool()  # no settings supplied
        # Drive a path-traversal-rejection to ensure settings resolved to
        # the tmp_path orchestrator_data_dir (a failed read would fail
        # if the data dir were the real repo's data/orchestrator).
        result_raw = await tool.ainvoke({"post_id": "ghost"})
        result = json.loads(result_raw)
        assert result["reason"] == "draft_not_found"
        # No file should have been written under tmp_path.
        assert not (tmp_path / "drafts").exists()
    finally:
        get_settings.cache_clear()
        monkeypatch.delenv("ORCHESTRATOR_DATA_DIR", raising=False)