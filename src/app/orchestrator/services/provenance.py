"""Draft provenance — cryptographic proof a draft was written by the writer
subagent's ``submit_draft`` tool, not by an LLM freelancing with
``write_file``.

Motivation (2026-07-25 production trace): the LLM coordinator skipped the
writer-subagent entirely, authored all five drafts itself via deepagents'
built-in ``write_file``, and self-gated them — bypassing the linkedin-voice
skill, the writer model tier, and the writer's write-until-pass loop, none of
which any prompt guardrail stopped. A prompt can ask a model not to author
prose; it cannot *prevent* it. This module is the prevention: every draft
that reaches export/delivery must carry a valid HMAC-SHA256 signature over
its canonical content, computed with a key the models never see.

The signature covers the ``PostProposal`` fields only (not the provenance
block itself), serialized as canonical JSON (sorted keys, compact separators)
so verification is stable across writers/readers. The provenance block rides
*inside* ``drafts/<post_id>.json`` under the ``_provenance`` key — pydantic's
default ``model_validate`` ignores unknown extra fields, so existing readers
(``read_proposal_from_state``, export, telegram) keep working unchanged.

Key resolution: ``settings.draft_signing_key`` when set; otherwise derived
from an already-secret env credential (OPENROUTER_API_KEY first, then
TELEGRAM_BOT_TOKEN) so existing installs get distinct keys with no new
secret to manage; otherwise a labeled dev key (tests / dry-run environments
— an attacker with repo access in that context gains nothing, there are no
secrets to forge against).
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
from typing import Any, Mapping

from app.config import Settings

logger = logging.getLogger(__name__)

# The key inside the draft JSON that carries the provenance block. Leading
# underscore keeps it visually separate from the PostProposal contract and
# collides with no schema field.
PROVENANCE_KEY = "_provenance"

_PROVENANCE_VERSION = 1
_DEV_FALLBACK_KEY = "ai-news-agent-dev-signing-key"


def _resolve_key(settings: Settings) -> bytes:
    """Resolve the HMAC key for this install. Never logged, never returned
    to a tool summary — the whole point is that no model context ever sees
    it."""
    explicit = (settings.draft_signing_key or "").strip()
    if explicit:
        return explicit.encode("utf-8")
    for candidate in (settings.openrouter_api_key, settings.telegram_bot_token):
        if candidate and candidate.strip():
            return hashlib.sha256(
                f"draft-signing:{candidate.strip()}".encode("utf-8")
            ).digest()
    return _DEV_FALLBACK_KEY.encode("utf-8")


def _canonical_payload(proposal_fields: Mapping[str, Any]) -> bytes:
    """Canonical JSON encoding of the signed fields: sorted keys, compact
    separators, UTF-8. Two writers producing the same PostProposal produce
    byte-identical signable payloads."""
    return json.dumps(
        proposal_fields,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")


def sign_draft(proposal_fields: Mapping[str, Any], settings: Settings) -> dict[str, Any]:
    """Compute the provenance block for a draft. ``proposal_fields`` is the
    PostProposal dump (no ``_provenance`` key). Returns the block to embed
    under ``_provenance``."""
    sig = hmac.new(
        _resolve_key(settings), _canonical_payload(proposal_fields), hashlib.sha256
    ).hexdigest()
    return {"version": _PROVENANCE_VERSION, "signed_by": "submit_draft", "sig": sig}


def verify_draft(raw_draft: Mapping[str, Any], settings: Settings) -> bool:
    """Verify a raw draft dict (as read from disk) carries a valid signature
    over its PostProposal fields. Returns False for a missing/malformed
    provenance block, a version mismatch, or a signature mismatch — every
    failure mode is "not provably writer-authored", so they all refuse
    identically."""
    provenance = raw_draft.get(PROVENANCE_KEY)
    if not isinstance(provenance, Mapping):
        return False
    if provenance.get("version") != _PROVENANCE_VERSION:
        return False
    sig = provenance.get("sig")
    if not isinstance(sig, str) or not sig:
        return False
    payload = {k: v for k, v in raw_draft.items() if k != PROVENANCE_KEY}
    expected = hmac.new(
        _resolve_key(settings), _canonical_payload(payload), hashlib.sha256
    ).hexdigest()
    # compare_digest: constant-time — this is an HMAC check, treat it like one.
    return hmac.compare_digest(sig, expected)


def verify_draft_file_payload(raw: Any) -> Mapping[str, Any] | None:
    """Type-guard helper for callers that json.loads a draft file: returns the
    mapping when it is one, else None."""
    return raw if isinstance(raw, Mapping) else None


__all__ = [
    "PROVENANCE_KEY",
    "sign_draft",
    "verify_draft",
    "verify_draft_file_payload",
]
