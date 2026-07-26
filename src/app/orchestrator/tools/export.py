"""export_report — the per-run export tool.

Thin adapter over :mod:`app.orchestrator.services.export_bundle`, which owns
reading a run's artifacts, rendering them, and writing the dated bundle under
``<outputs_dir>/<YYYY-MM-DD>/``. This module owns only the tool's argument
schema and its JSON return contract.

Target environment: deepagents ``create_deep_agent`` — async-only tool, same
sibling pattern as every other orchestrator tool.
"""
from __future__ import annotations

import json

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from app.config import Settings, get_settings
from app.orchestrator.services.export_bundle import export_run


class ExportReportArgs(BaseModel):
    """Tool input. ``date`` is an optional ``YYYY-MM-DD`` override for the
    export's directory leaf; defaults to today (UTC). ``overwrite`` allows the
    export to replace a prior same-date bundle; default False (the tool refuses
    with status=error reason=overwrite_required when one exists)."""

    date: str | None = Field(
        default=None,
        description=(
            "Optional YYYY-MM-DD datestamp for the export directory leaf "
            "(<outputs_dir>/<date>/). Defaults to today (UTC). Must match "
            "YYYY-MM-DD exactly; any other shape (incl. path-fragments) is "
            "rejected."
        ),
    )
    overwrite: bool = Field(
        default=False,
        description=(
            "Allow the export to overwrite a prior same-date bundle. Default "
            "False — the tool refuses with status=error reason="
            "overwrite_required when a same-date bundle directory already "
            "exists."
        ),
    )


def build_export_report_tool(settings: Settings | None = None) -> StructuredTool:
    """Construct the export_report LangChain tool.

    Settings resolve lazily on first call when not supplied, mirroring every
    other orchestrator tool factory."""
    bound_settings = settings

    async def _async(date: str | None = None, overwrite: bool = False) -> str:
        s = bound_settings or get_settings()
        return json.dumps(
            export_run(s, date_arg=date, overwrite=overwrite), default=str
        )

    return StructuredTool.from_function(
        func=None,
        coroutine=_async,
        name="export_report",
        description=(
            "Read the artifacts a run has produced under orchestrator_data_dir "
            "(articles + topics + briefs + drafts + gate verdicts) and write a "
            "compact export bundle under <outputs_dir>/<YYYY-MM-DD>/ with "
            "posts.md (rendered posts), run_report.json (structured summary "
            "with counts + paths), and briefs.json (every brief as a JSON "
            "array). Returns a JSON summary with {status, date, bundle_dir, "
            "files, counts} — never the bundle contents. Defaults to today "
            "(UTC); an optional `date` override must match YYYY-MM-DD exactly. "
            "Refuses to overwrite a prior same-date bundle unless "
            "`overwrite=True` — the first law of run-export tools is 'don't be "
            "a footgun'."
        ),
        args_schema=ExportReportArgs,
    )


# Convenience singleton — same pattern as every sibling tool.
export_report_tool = build_export_report_tool()
