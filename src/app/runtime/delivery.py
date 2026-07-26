"""Proposal delivery to the LinkedIn Telegram bot.

Delivery is deliberately a separate layer from production: the spine writes
drafts, ``export_report`` commits the run, and only then does this module ship
what passed. Keeping the ordering outside the spine is what makes the
"export is the commit record" rule (Decision C) visible at the call site.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.config import Settings
from app.orchestrator import state
from app.orchestrator.tools.telegram import build_deliver_telegram_tool


async def deliver_proposals(settings: Settings) -> list[dict[str, Any]]:
    """Send every draft under ``drafts/`` to the LinkedIn Telegram bot via the
    ``deliver_telegram`` tool. The tool re-verifies each draft's gate verdict and
    refuses any it didn't certify, hard-codes ``bot="linkedin"``, and auto-dry-runs
    when the linkedin profile is unconfigured — so this loop can stay dumb:
    deliver every draft, let the tool gate + route. Returns one compressed result
    dict per draft."""
    drafts_dir = Path(settings.orchestrator_data_dir) / state.DRAFTS_DIRNAME
    if not drafts_dir.exists():
        return []
    tool = build_deliver_telegram_tool(settings)
    results: list[dict[str, Any]] = []
    for draft in sorted(drafts_dir.glob("*.json")):
        if draft.name.endswith(".gate.json"):
            continue
        post_id = draft.name[: -len(".json")]
        results.append(json.loads(await tool.ainvoke({"post_id": post_id})))
    return results


__all__ = ["deliver_proposals"]
