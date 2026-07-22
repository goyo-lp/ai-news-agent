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

Path-traversal guard: ``post_id`` is sanitized via the inline rule
already enforced at quality.py / verify_claim.py (no `/`, no ``..`` /
``.`` / empty) — same parity-bound guard the state.py module (P5.1)
centralizes. The guard is duplicated inline here for the same reason
quality.py duplicates it: the custom tools' inline guards predate
state.py's centralized helper, and adopting the helper per-tool is
incremental (P5.1's docstring admits this scope-out). A future refactor
swaps this inline guard for ``state._validate_slug(post_id, kind=
'post_id')``; the parity test pins the agreement. A greppable
``# TODO(swap-inline-guard-onto-state._validate_slug):`` marker stands
next to the inline guard so the deferral is source-level greppable,
not prose-only.

Similarly, ``quality.py`` exposes a shared ``read_proposal_from_state``
helper for the ``PostProposal.model_validate_json`` read; this tool
reads the draft inline (same pre-helper-race shape). Adopting the helper
is out of this PR's scope and lives with the same per-tool refactor.
"""
from __future__ import annotations

import html
import json
import logging
from pathlib import Path
from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from app.config import Settings, get_settings
from app.orchestrator.schemas import PostProposal
from app.services.telegram_client import TELEGRAM_TEXT_LIMIT, TelegramClient

logger = logging.getLogger(__name__)

_DRAFTS_SUBDIR = "drafts"
_DRAFT_SUFFIX = ".json"
_GATE_SUFFIX = ".gate.json"

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


def _draft_path(data_dir: str, post_id: str, *, gate: bool = False) -> Path:
    """Resolve the on-disk path for a draft or its gate verdict. Same inline
    guard as quality.py / verify_claim.py — see module docstring for the
    parity-bound rationale and the future swap onto state._validate_slug."""
    # TODO(swap-inline-guard-onto-state._validate_slug): parity pinned by
    # tests/test_state.py::test_draft_and_gate_paths_match_quality_tool_convention;
    # swap onto state._validate_slug when adopting state helpers per-tool.
    if "/" in post_id or post_id in {"", ".", ".."}:
        raise ValueError(f"Invalid post_id: {post_id!r}")
    suffix = _GATE_SUFFIX if gate else _DRAFT_SUFFIX
    return Path(data_dir) / _DRAFTS_SUBDIR / f"{post_id}{suffix}"


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
        # gate-relaxation.
        message = message[: _MAX_MESSAGE_LEN - 3].rstrip() + "..."
    return message


async def _deliver_one(post_id: str, settings: Settings) -> dict[str, Any]:
    """Read + gate-check + format + send one draft. Returns the compressed
    summary that rides back to the coordinator's LLM — never the post body
    or the gate verdict's reasons list (those live on disk)."""
    # 1. Resolve paths + path-traversal guard.
    try:
        draft_file = _draft_path(settings.orchestrator_data_dir, post_id)
        gate_file = _draft_path(settings.orchestrator_data_dir, post_id, gate=True)
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
        proposal = PostProposal.model_validate_json(
            draft_file.read_text(encoding="utf-8")
        )
    except Exception as exc:
        return {
            "post_id": post_id,
            "status": "error",
            "reason": "draft_invalid_json",
            "error": str(exc),
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
            "draft_invalid_json / gate_verdict_missing / gate_not_passed / "
            "send_failed) on failure. Refuses to ship a draft the gate "
            "hasn't passed. Hard-coded bot=linkedin — a LinkedIn post "
            "proposal can't be routed to the news chat."
        ),
        args_schema=DeliverTelegramArgs,
    )


# Convenience singleton — same pattern as fetch_curated_ai_news_tool /
# technical_rank_tool / quality_gate_tool. Production callers use this;
# tests inject settings via the factory.
deliver_telegram_tool = build_deliver_telegram_tool()