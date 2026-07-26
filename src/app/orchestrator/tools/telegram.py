"""deliver_telegram — the coordinator's delivery tool for LinkedIn post
proposals.

Reads a ``PostProposal`` from ``drafts/<post_id>.json``, asserts the
deterministic quality gate has passed it (``drafts/<post_id>.gate.json``
reports ``passed=True``), formats the post body + hashtags + citations
as a single Telegram text message, and sends it to the configured
``linkedin`` bot profile via :meth:`TelegramClient.send_message`.

Per guiding principle #3 the proposal is the artifact on disk: the tool
reads the draft file, sends the formatted message, and returns a
compressed JSON summary — it does NOT paste the post body back into the
coordinator's chat reply. The coordinator's run-summary just lists the
``post_id`` + the delivery outcome per proposal.

Refuses to ship a draft whose gate verdict is anything other than
``passed=True`` — surfaces ``status="error" reason="gate_not_passed"``
or ``"gate_verdict_missing"`` rather than sending a draft the writer
hasn't certified. The gate is the last deterministic check before the
post reaches the audience; bypassing it here is exactly the "model
helpfully overrode the gate" failure mode the gate exists to prevent.

Target environment: deepagents ``create_deep_agent`` — async-only tool,
same sibling pattern as news / technical_rank / fetch_article / web /
verify_claim / quality_gate. Multi-bot routing via P6.1's named-bot
profile seam; this tool hard-codes ``bot="linkedin"`` because the
coordinator can't choose to route a *LinkedIn post proposal* anywhere
else — sending it to the news chat would be a coordinated misroute.

Paths, the path-traversal guard, and the brief lookup all come from
:mod:`app.orchestrator.state`, which owns the filesystem convention.
"""
from __future__ import annotations

import html
import json
import logging
from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from app.config import Settings, get_settings
from app.orchestrator import state
from app.orchestrator.schemas import PostProposal
from app.orchestrator.services.evidence_floor import meets_evidence_floor
from app.orchestrator.services.provenance import verify_draft
from app.services.telegram_client import TELEGRAM_TEXT_LIMIT, TelegramClient

logger = logging.getLogger(__name__)

# The send-side cap is 4096 (Telegram's sendMessage text limit). The post
# body is capped to 182 words by the quality gate, well under that limit;
# the formatter caps the assembled message (body + hashtags + citations)
# to 4096 as defense-in-depth so a future gate relaxation (or a citations
# blowup) doesn't 400 the sendMessage call.
_MAX_MESSAGE_LEN = TELEGRAM_TEXT_LIMIT


class DeliverTelegramArgs(BaseModel):
    """Tool input. ``post_id`` identifies the draft on disk
    (``drafts/<post_id>.json``); the tool also reads
    ``drafts/<post_id>.gate.json`` to verify the gate passed."""

    post_id: str = Field(
        ...,
        description="The proposal's post_id; the draft is read from drafts/<post_id>.json.",
    )


def _format_post_message(proposal: PostProposal) -> str:
    """Render a PostProposal as a single Telegram text message.

    Layout (HTML, parse_mode=HTML by default in Settings):
      <headline>

      <body>

      <hashtags joined by space>

      <citation URLs as numbered list>

    Empty sections are skipped. Body is used verbatim — the writer's voice
    is the product, so no reformatting/normalization here. Hashtags are
    space-joined (LinkedIn convention). Citations are HTML-escaped and
    rendered as a numbered list of clickable links so the audience can
    verify the post's sources. The whole message is capped to
    ``_MAX_MESSAGE_LEN`` (Telegram's sendMessage text limit, defense-in-
    depth on top of the gate's 182-word body cap)."""
    parts: list[str] = []
    if proposal.headline.strip():
        parts.append(html.escape(proposal.headline.strip(), quote=False))
    if proposal.body.strip():
        parts.append(html.escape(proposal.body.strip(), quote=False))
    hashtag_line = " ".join(h.strip() for h in proposal.hashtags if h.strip())
    if hashtag_line:
        parts.append(html.escape(hashtag_line, quote=False))
    if proposal.citation_urls:
        citation_lines: list[str] = []
        for idx, url in enumerate(proposal.citation_urls, start=1):
            if not url.strip():
                continue
            safe_url = html.escape(url.strip(), quote=True)
            citation_lines.append(f'{idx}. <a href="{safe_url}">{safe_url}</a>')
        if citation_lines:
            parts.append("\n".join(citation_lines))

    message = "\n\n".join(parts)
    if len(message) > _MAX_MESSAGE_LEN:
        # Truncate at the limit, preserving the headline + as much of the
        # body as fits. Drop citations first (they're the lowest-priority
        # text), then hashtags, then body-tail. Practically never happens
        # at the gate's 182-word body cap; this is harden-against-future-
        # gate-relaxation. The cut must not land inside an HTML tag or
        # entity — a mid-tag slice (e.g. `<a hr`) 400s the sendMessage call.
        message = _safe_truncate_html(message, _MAX_MESSAGE_LEN - 3).rstrip() + "..."
    return message


def _safe_truncate_html(text: str, limit: int) -> str:
    """Truncate to ``limit`` chars without cutting inside an HTML tag
    (``<...>``) or entity (``&...;``). If the naive cut lands inside either,
    back off to just before it. The message is otherwise valid HTML, so
    truncating at a safe boundary keeps it valid."""
    if len(text) <= limit:
        return text
    cut = text[:limit]
    last_lt, last_gt = cut.rfind("<"), cut.rfind(">")
    if last_lt > last_gt:  # cut lands inside a tag
        cut = cut[:last_lt]
    last_amp, last_semi = cut.rfind("&"), cut.rfind(";")
    if last_amp > last_semi:  # cut lands inside an entity
        cut = cut[:last_amp]
    return cut


def _check_evidence_floor(settings: Settings, proposal: PostProposal) -> tuple[bool, str]:
    """Load the brief behind a draft and apply the evidence floor. The
    verified copy wins; the pre-verification copy is the fallback (its
    ``unverified`` status will fail the floor with an honest reason). A
    missing/unreadable brief fails closed."""
    if not proposal.supporting_topic_ids:
        return False, "draft has no supporting_topic_ids; cannot check the evidence floor"
    topic_id = proposal.supporting_topic_ids[0]
    brief = state.read_brief(topic_id, settings.orchestrator_data_dir)
    if brief is None:
        return False, f"no readable brief for topic {topic_id!r}; cannot verify the evidence floor"
    passes, why = meets_evidence_floor(brief, settings)
    return passes, ("" if passes else f"brief {topic_id!r} below evidence floor: {why}")


async def _deliver_one(post_id: str, settings: Settings) -> dict[str, Any]:
    """Read + provenance-check + gate-check + floor-check + format + send one
    draft. Returns the compressed summary that rides back to the coordinator's
    LLM — never the post body or the gate verdict's reasons list (those live
    on disk)."""
    # 1. Resolve paths + path-traversal guard.
    try:
        draft_file = state.draft_path(post_id, settings.orchestrator_data_dir)
        gate_file = state.gate_path(post_id, settings.orchestrator_data_dir)
    except ValueError as exc:
        return {
            "post_id": post_id,
            "status": "error",
            "reason": "invalid_post_id",
            "error": str(exc),
        }

    # 2. Read the draft.
    if not draft_file.exists():
        return {
            "post_id": post_id,
            "status": "error",
            "reason": "draft_not_found",
            "error": f"No draft at {draft_file}",
        }

    try:
        raw_draft = json.loads(draft_file.read_text(encoding="utf-8"))
        proposal = PostProposal.model_validate(raw_draft)
    except Exception as exc:
        return {
            "post_id": post_id,
            "status": "error",
            "reason": "draft_invalid_json",
            "error": str(exc),
        }

    # 2b. Provenance: only drafts signed by the writer's submit_draft tool may
    # ship. An LLM (coordinator or writer) can always write_file a draft —
    # what it cannot do is forge the HMAC key it never sees. This is the
    # deterministic stop for the 2026-07-25 failure mode (coordinator
    # self-authored + self-gated all five drafts).
    if not isinstance(raw_draft, dict) or not verify_draft(raw_draft, settings):
        return {
            "post_id": post_id,
            "status": "error",
            "reason": "provenance_invalid",
            "error": (
                "Draft is not signed by the writer's submit_draft tool; "
                "re-author it via the writer-subagent before delivery."
            ),
        }

    # 3. Read the gate verdict and verify passed=True. Refuse to ship a
    # draft the gate didn't certify — that's the whole point of the gate.
    if not gate_file.exists():
        return {
            "post_id": post_id,
            "status": "error",
            "reason": "gate_verdict_missing",
            "error": (
                f"No gate verdict at {gate_file}; run quality_gate before "
                "deliver_telegram."
            ),
        }
    try:
        gate_verdict = json.loads(gate_file.read_text(encoding="utf-8"))
    except Exception as exc:
        return {
            "post_id": post_id,
            "status": "error",
            "reason": "gate_verdict_invalid_json",
            "error": str(exc),
        }

    if gate_verdict.get("passed") is not True:
        # Note the strict ``is not True`` — the gate's canonical producer
        # (quality_gate via dataclasses.asdict + json.dumps) writes a real
        # Python bool that round-trips to ``True`` / ``False``. But the
        # file on disk is *untrusted*: the writer subagent (an LLM) can
        # author ``drafts/<post_id>.gate.json`` via deepagents' built-in
        # ``write_file`` with a "helpful" ``{"passed": "True"}`` /
        # ``{"passed": 1}`` / ``{"passed": ["no"]}`` etc. A truthy-but-
        # not-bool value would silently pass a ``not gate_verdict.get(
        # "passed")`` check — exactly the "model helpfully overrode the
        # gate" failure mode this delivery path exists to prevent. The
        # ``is not True`` identity check rejects every truthy-not-bool
        # variant loudly. The pinned test
        # ``test_gate_verdict_malformed_passed_value_refuses_delivery``
        # covers several flavors of the leak shape.
        return {
            "post_id": post_id,
            "status": "error",
            "reason": "gate_not_passed",
            "gate_passed": gate_verdict.get("passed"),
            "error": (
                f"Gate verdict at {gate_file} does not report "
                f"passed=True (got {gate_verdict.get('passed')!r}); "
                "re-delegate the writer to fix the draft before delivery."
            ),
        }

    # 3b. Evidence floor: the brief behind the draft must still clear the
    # floor at send time (defense-in-depth for hand-written/legacy drafts —
    # the spine + submit_draft already enforce this upstream).
    floor_ok, floor_reason = _check_evidence_floor(settings, proposal)
    if not floor_ok:
        return {
            "post_id": post_id,
            "status": "error",
            "reason": "verification_floor",
            "error": floor_reason,
        }

    # 4. Format the message.
    message_body = _format_post_message(proposal)

    # 5. Send to the linkedin bot. Hard-coded bot="linkedin" — a LinkedIn
    # post proposal can't be routed anywhere else; sending it to the news
    # chat would be a coordinated misroute.
    client = TelegramClient(settings)
    send_result = await client.send_message(
        message_body,
        bot="linkedin",
        dry_run=not (settings.telegram_linkedin_bot_token or "").strip()
        or not (settings.telegram_linkedin_chat_id or "").strip(),
    )

    # 6. Compressed summary back to the coordinator's LLM.
    return {
        "post_id": post_id,
        "status": send_result.get("status", "error"),
        "bot": "linkedin",
        "message_id": send_result.get("message_id"),
        # The dry_run preview never includes the full message body in the
        # summary — that's principle #3. Surface just enough to let the
        # coordinator log the routing decision (bot + status + chars).
        "preview_chars": len(send_result.get("preview", "")) if send_result.get("status") == "dry_run" else None,
        "error": send_result.get("error"),
    }


def build_deliver_telegram_tool(settings: Settings | None = None) -> StructuredTool:
    """Construct the deliver_telegram LangChain tool.

    Settings resolve lazily on first call when not supplied, mirroring the
    news / technical_rank / quality_gate / style tool factories."""
    bound_settings = settings

    async def _async(post_id: str) -> str:
        s = bound_settings or get_settings()
        result = await _deliver_one(post_id, s)
        return json.dumps(result, default=str)

    return StructuredTool.from_function(
        func=None,
        coroutine=_async,
        name="deliver_telegram",
        description=(
            "Read the post proposal at drafts/<post_id>.json, verify its "
            "quality_gate verdict at drafts/<post_id>.gate.json reports "
            "passed=True, format the body + hashtags + citations as a single "
            "Telegram message, and send it to the linkedin bot profile. "
            "Returns a JSON summary with {post_id, status, bot, message_id, "
            "error} — never the post body. status='sent' on success, 'error' "
            "with a reason (invalid_post_id / draft_not_found / "
            "draft_invalid_json / provenance_invalid / verification_floor / "
            "gate_verdict_missing / gate_not_passed / send_failed) on failure. "
            "Refuses to ship a draft the gate hasn't passed, a draft not "
            "signed by the writer's submit_draft tool, or a draft whose brief "
            "is below the evidence floor. Hard-coded bot=linkedin — a "
            "LinkedIn post proposal can't be routed to the news chat."
        ),
        args_schema=DeliverTelegramArgs,
    )


# Convenience singleton — same pattern as fetch_curated_ai_news_tool /
# technical_rank_tool / quality_gate_tool. Production callers use this;
# tests inject settings via the factory.
deliver_telegram_tool = build_deliver_telegram_tool()