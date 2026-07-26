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
from app.orchestrator.schemas import PostProposal
from app.orchestrator.services.drafts import (
    Certification,
    DraftLoadError,
    LoadedDraft,
    certify,
    load_draft,
)
from app.services.telegram_client import TELEGRAM_TEXT_LIMIT, TelegramClient

logger = logging.getLogger(__name__)


def _error(reason: str, message: str, **extra: Any) -> dict[str, Any]:
    """The tool's structured failure shape. Every refusal is a value the LLM
    can branch on, never a raise into the agent loop."""
    return {"status": "error", "reason": reason, "error": message, **extra}

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


def _refusal(draft: LoadedDraft, cert: Certification) -> dict[str, Any] | None:
    """Why this draft must not ship, or None if it may.

    Order is the contract: provenance before the gate (an unsigned draft is
    refused without the gate even being consulted — its verdict is as
    untrustworthy as the draft), and the gate before the evidence floor (a
    draft the gate rejected is refused whether or not a brief exists).
    """
    if not cert.provenance_ok:
        return _error(
            "provenance_invalid",
            "Draft is not signed by the writer's submit_draft tool; "
            "re-author it via the writer-subagent before delivery.",
        )
    if cert.gate_status == "missing":
        return _error(
            "gate_verdict_missing",
            f"No gate verdict for {draft.post_id}; run quality_gate before deliver_telegram.",
        )
    if cert.gate_status == "malformed":
        return _error(
            "gate_verdict_invalid_json",
            f"Gate verdict for {draft.post_id} is not readable JSON.",
        )
    if cert.gate_passed is not True:
        return _error(
            "gate_not_passed",
            f"Gate verdict for {draft.post_id} does not report passed=True "
            f"(got {(draft.gate_verdict or {}).get('passed')!r}); "
            "re-delegate the writer to fix the draft before delivery.",
            gate_passed=(draft.gate_verdict or {}).get("passed"),
        )
    if not cert.floor_ok:
        return _error("verification_floor", cert.floor_reason)
    return None


async def _deliver_one(post_id: str, settings: Settings) -> dict[str, Any]:
    """Read + certify + format + send one draft. Returns the compressed summary
    that rides back to the caller — never the post body or the gate verdict's
    reasons list (those live on disk)."""
    try:
        draft = load_draft(post_id, settings.orchestrator_data_dir)
    except DraftLoadError as exc:
        return {"post_id": post_id, **_error(exc.reason, str(exc))}

    refusal = _refusal(draft, certify(draft, settings))
    if refusal is not None:
        return {"post_id": post_id, **refusal}

    # Hard-coded bot="linkedin" — a LinkedIn post proposal can't be routed
    # anywhere else; sending it to the news chat would be a coordinated
    # misroute. Auto-dry-runs when the linkedin profile is unconfigured.
    client = TelegramClient(settings)
    send_result = await client.send_message(
        _format_post_message(draft.proposal),
        bot="linkedin",
        dry_run=not (settings.telegram_linkedin_bot_token or "").strip()
        or not (settings.telegram_linkedin_chat_id or "").strip(),
    )

    return {
        "post_id": post_id,
        "status": send_result.get("status", "error"),
        "bot": "linkedin",
        "message_id": send_result.get("message_id"),
        # The dry_run preview never includes the full message body in the
        # summary — that's principle #3. Surface just enough to let the caller
        # log the routing decision (bot + status + chars).
        "preview_chars": (
            len(send_result.get("preview", ""))
            if send_result.get("status") == "dry_run"
            else None
        ),
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