"""research-subagent — the one genuinely adaptive surface in the pipeline.

Per the migration plan's guiding principle #2, agency is spent only where the
decision is open-ended: for a single ranked topic, *should I fetch the real
article or trust the ranked summary? what needs verifying? what is actually
technically new?* Everything deterministic (fetch, search, extract, verify) is
already a tool; this subagent only decides how to sequence them and how to
synthesize the result.

It is a :class:`deepagents.SubAgent` spec — name, description, system prompt,
its four research tools, and a Stage-B model — that the coordinator (P5) invokes
via ``task()`` once per topic. deepagents runs each ``task()`` call in an
isolated context, so per-topic research never pollutes the coordinator's window
(principle #3: only the compressed brief on disk crosses back). The subagent's
output is the *file* ``briefs/<topic_id>.json``, not its chat return.

Backend precondition (enforced by the coordinator, P5): the four research tools
write/read their artifacts on the *real* filesystem under
``settings.orchestrator_data_dir`` (``fetch_article`` -> ``articles/…``,
``verify_claim`` reads ``briefs/<topic_id>.json``). deepagents' built-in
``write_file`` / ``read_file`` default to the in-state StateBackend, so the
prompt's "write the brief with write_file, then verify_claim reads it" contract
only holds when the coordinator mounts a real ``FilesystemBackend`` rooted at
``orchestrator_data_dir``. Under the default backend the brief lands in virtual
state and every verify comes back "brief not found". P5 owns wiring that
backend; this module owns the spec.

Scope note (per-topic timeout): a SubAgent spec has no runtime loop of its own,
so a per-topic wall-clock timeout is enforced by the coordinator that owns the
``task()`` delegation (P5), where ``asyncio.wait_for`` can bound one topic's
research. This module owns the spec; the coordinator owns the clock.
"""
from __future__ import annotations

from deepagents import SubAgent

from app.config import Settings, get_settings
from app.orchestrator.models import build_openrouter_chat_model
from app.orchestrator.schemas import ResearchBrief
from app.orchestrator.tools.fetch import build_fetch_article_tool
from app.orchestrator.tools.tavily import (
    build_tavily_search_tool,
    build_web_extract_tool,
)
from app.orchestrator.tools.verify_claim import build_verify_claim_tool

RESEARCH_SUBAGENT_NAME = "research-subagent"

# The fields the subagent authors. verify_claim owns every verification_* field
# and fills them; the subagent must not pre-set them. A prefix rule (not a
# hardcoded denylist) stays correct if verify_claim later writes a new
# verification_* field — it's excluded automatically — and deriving the rest
# from the model means a new authored field surfaces in the prompt instead of
# drifting silently.
_VERIFY_FIELD_PREFIX = "verification_"
AUTHORED_BRIEF_FIELDS = [
    name for name in ResearchBrief.model_fields
    if not name.startswith(_VERIFY_FIELD_PREFIX)
]


def _system_prompt() -> str:
    authored = ", ".join(AUTHORED_BRIEF_FIELDS)
    return f"""You research exactly ONE AI news topic and produce a verified research brief.

You are given one topic candidate: a topic_id, a title, a summary_hint, a
primary_url, and optional supporting_urls. Your job is to turn it into a
grounded ResearchBrief written to `briefs/<topic_id>.json`, then verify it.

Work adaptively — you decide the sequence:

1. Judge whether the ranked summary_hint is enough. When it is thin, stale, or
   makes a claim you cannot stand behind, call `fetch_article` on the primary_url
   to get the real article (SSRF-guarded, size-capped). Read the fetched
   content from the `path` it returns; the tool never returns the body inline.
2. When you need independent corroboration or the primary source is weak, use
   `tavily_search` for the claim and `web_extract` on the most relevant
   result URLs. Read their artifacts from the returned `path`.
3. Synthesize the brief. Extract what is *technically* new — a capability, a
   result, a shift — not marketing. Ground every claim in something you fetched
   or searched; never invent numbers, quotes, capabilities, or URLs. Cite only
   URLs you actually retrieved.
4. Write `briefs/<topic_id>.json` with `write_file`. It must be valid JSON that
   validates against the ResearchBrief schema. Author these fields: {authored}.
   Do NOT set the verification_* fields — the next step fills them.
5. Call `verify_claim` with the topic_id. It checks the brief against
   independent evidence and writes `briefs/<topic_id>.verified.json`. If it
   reports insufficient evidence, soften the affected claims in the brief and
   re-verify rather than overstating.

Rules:
- The brief file is your deliverable, not your chat reply. Return only a short
  confirmation (topic_id + verification_status).
- Never mention tool limitations, scrape blocks, missing pages, or an inability
  to browse. When evidence is thin, use neutral phrasing: public evidence is
  still limited and the read is directional.
- One topic only. Do not research or blend a second story."""


def build_research_subagent(settings: Settings | None = None) -> SubAgent:
    """Build the research-subagent spec for ``create_deep_agent(subagents=[...])``.

    Settings resolve lazily when not supplied, mirroring the tool factories.
    Tools are built from the same Settings so they share one config (data dir,
    OpenRouter/Tavily keys). The Stage-B model is a live OpenRouter chat model;
    it constructs without a key (dry-run safe) and only needs one to run."""
    s = settings or get_settings()
    return SubAgent(
        name=RESEARCH_SUBAGENT_NAME,
        description=(
            "Research one ranked AI news topic (topic_id + primary_url) into a "
            "verified ResearchBrief written to briefs/<topic_id>.json. Decides "
            "for itself whether to fetch the real article, corroborate via "
            "Tavily, and what is technically new; then verifies the brief. "
            "Delegate one topic per call."
        ),
        system_prompt=_system_prompt(),
        tools=[
            build_fetch_article_tool(s),
            build_tavily_search_tool(s),
            build_web_extract_tool(s),
            build_verify_claim_tool(s),
        ],
        model=build_openrouter_chat_model(
            s, model=s.openrouter_stage_b_research_model
        ),
    )
