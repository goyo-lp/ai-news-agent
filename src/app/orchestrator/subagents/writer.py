"""writer-subagent — turns a verified brief into a post proposal in the author's
voice.

Unlike the research subagent, the writer's decisions are narrow: it isn't
gathering evidence, it's rendering an already-verified brief into one LinkedIn
post that sounds like the author and clears the deterministic gate. The voice
is the genuinely-instructional part (guiding principle #1), so it lives in the
``linkedin-voice`` skill (P3.1) the writer loads, not in this prompt.

It is a :class:`deepagents.SubAgent` spec the coordinator (P5) invokes via
``task()`` once per topic. Its deliverable is the file ``drafts/<post_id>.json``
(a :class:`PostProposal`), gated by the one tool it carries, ``quality_gate``.

Backend precondition (enforced by the coordinator, P5): the writer reads the
verified brief (``briefs/<topic_id>.verified.json``) and the style profile
(``style_profile.json``) with ``read_file`` and writes ``drafts/<post_id>.json``
with ``write_file``; ``quality_gate`` then reads that draft from the *real*
filesystem under ``settings.orchestrator_data_dir``. As with the research
subagent, this only works when the coordinator mounts a real ``FilesystemBackend``
rooted there (deepagents' file tools otherwise default to the in-state
StateBackend).

The skills source is different: SkillsMiddleware shares that same backend, so a
*relative* skills path would resolve under the ``orchestrator_data_dir`` root —
but the linkedin-voice skill lives at the repo-root ``skills/`` dir, not under
the data dir. So the factory resolves ``settings.skills_dir`` to an *absolute*
path, which a non-virtual ``FilesystemBackend`` reads as-is (independent of its
root). The skill is loaded from the real repo; the brief/draft I/O stays under
the data dir.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from deepagents import SubAgent, create_deep_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph.state import CompiledStateGraph

from app.config import Settings, get_settings
from app.orchestrator.models import build_openrouter_chat_model
from app.orchestrator.schemas import PostProposal
from app.orchestrator.state import build_orchestrator_backend
from app.orchestrator.tools.draft import build_submit_draft_tool
from app.orchestrator.tools.quality import build_quality_gate_tool

WRITER_SUBAGENT_NAME = "writer-subagent"

# Every PostProposal field is author-owned. Derived from the model so a new
# field surfaces in the prompt instead of being silently omitted from drafts.
AUTHORED_POST_FIELDS = list(PostProposal.model_fields)


def _system_prompt(data_dir: str) -> str:
    authored = ", ".join(AUTHORED_POST_FIELDS)
    return f"""You write exactly ONE LinkedIn post from a verified research brief, in the author's voice.

First load the `linkedin-voice` skill (see the Skills system) and read its
SKILL.md in full — it defines the voice, structure, and the hard constraints
your post must satisfy. Follow it; it is the source of truth for how the post
should sound.

FILE PATHS: your data directory is `{data_dir}` (an absolute path). Every
`read_file` call MUST use an absolute path under it — e.g.
`{data_dir}/briefs/<topic_id>.verified.json`. Never pass a bare relative path
like `briefs/<topic_id>.json`; it will not resolve.

You are given a topic_id and a post_id. Then:

1. Read the verified brief `{data_dir}/briefs/<topic_id>.verified.json`. If
   that file is absent or its verification is weak, STOP — do not draft from
   the unverified brief; report the topic as not writable. Every factual
   claim in your post must trace to this brief — never invent numbers,
   quotes, capabilities, or URLs, and cite only URLs the brief's citations
   provide. If the brief's verification_status is not "verified", soften or
   drop the affected claims.
2. Read the author's `{data_dir}/style_profile.json` for voice signals (common
   openers, vocabulary, tone) and let them shape the phrasing. If it is
   absent, proceed with the skill's defaults — do not invent a profile.
3. Submit the post with the `submit_draft` tool: pass the post_id and the
   full proposal object with these fields: {authored}. Set
   supporting_topic_ids to exactly [topic_id] and citation_urls from the
   brief's citations. `submit_draft` enforces the evidence floor
   deterministically — if it returns status=error with
   reason=verification_floor, the topic must not be drafted: stop and report
   it as skipped. NEVER write `drafts/<post_id>.json` with `write_file`;
   unsigned drafts are refused downstream.
4. Call `quality_gate` with the post_id. It enforces, deterministically:
   body 105–182 words (measured after a light jargon/CTA clean), exactly one
   supporting_topic_id, at least 3 hashtags, no hype markers, and citation
   fidelity against the brief. If it returns status="failed", read the
   reasons from the verdict file at the returned `path`, fix the proposal,
   resubmit with `submit_draft`, and re-gate. Repeat until it passes; do not
   ship a failing draft.

Rules:
- The draft file is your deliverable, not your chat reply. Return only a short
  confirmation (post_id + whether the gate passed).
- Write for the gate up front: aim a few words inside the 105–182 window, use
  plain language over jargon, and never use hype phrasing — a first-pass gate
  pass beats a retry loop.
- One topic, one post."""


def build_writer_subagent(settings: Settings | None = None) -> SubAgent:
    """Build the writer-subagent spec for ``create_deep_agent(subagents=[...])``.

    Settings resolve lazily when not supplied, mirroring the tool/subagent
    factories. The quality_gate tool is built from the same Settings so it reads
    the same data dir. The Stage-B writer model is a live OpenRouter chat model
    (constructs without a key for dry-run/tests). The linkedin-voice skill source
    is resolved to an absolute path so it loads from the repo regardless of the
    coordinator's backend root (see module docstring)."""
    s = settings or get_settings()
    skills_source = str(Path(s.skills_dir).resolve())
    data_dir = str(Path(s.orchestrator_data_dir).resolve())
    return SubAgent(
        name=WRITER_SUBAGENT_NAME,
        description=(
            "Write one LinkedIn post proposal from a verified ResearchBrief in "
            "the author's voice, gated by quality_gate. Reads "
            "briefs/<topic_id>.verified.json + style_profile.json, submits "
            "drafts/<post_id>.json (a PostProposal) via submit_draft. "
            "Delegate one topic per call."
        ),
        system_prompt=_system_prompt(data_dir),
        tools=[build_submit_draft_tool(s), build_quality_gate_tool(s)],
        model=build_openrouter_chat_model(
            s, model=s.openrouter_stage_b_writer_model
        ),
        skills=[skills_source],
    )


def build_writer_agent(
    settings: Settings | None = None,
    *,
    model: BaseChatModel | None = None,
) -> CompiledStateGraph[Any, Any, Any, Any]:
    """Build the writer as a standalone runnable deep agent (no coordinator).

    The deterministic spine (``app.orchestrator.spine``) invokes this directly
    once per topic — same prompt, same tools, same model tier and skills as
    the ``task()``-delegated spec, but the delegation boundary (and its
    per-topic wall-clock timeout) lives in code, not in a coordinator model's
    prompt-following. ``model`` is the seam tests use to inject a fake."""
    s = settings or get_settings()
    skills_source = str(Path(s.skills_dir).resolve())
    data_dir = str(Path(s.orchestrator_data_dir).resolve())
    return create_deep_agent(
        model=model
        or build_openrouter_chat_model(s, model=s.openrouter_stage_b_writer_model),
        tools=[build_submit_draft_tool(s), build_quality_gate_tool(s)],
        backend=build_orchestrator_backend(s),
        system_prompt=_system_prompt(data_dir),
        skills=[skills_source],
        name=WRITER_SUBAGENT_NAME,
    )
