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

from deepagents import SubAgent

from app.config import Settings, get_settings
from app.orchestrator.models import build_openrouter_chat_model
from app.orchestrator.schemas import PostProposal
from app.orchestrator.tools.quality import build_quality_gate_tool

WRITER_SUBAGENT_NAME = "writer-subagent"

# Every PostProposal field is author-owned. Derived from the model so a new
# field surfaces in the prompt instead of being silently omitted from drafts.
AUTHORED_POST_FIELDS = list(PostProposal.model_fields)


def _system_prompt() -> str:
    authored = ", ".join(AUTHORED_POST_FIELDS)
    return f"""You write exactly ONE LinkedIn post from a verified research brief, in the author's voice.

First load the `linkedin-voice` skill (see the Skills system) and read its
SKILL.md in full — it defines the voice, structure, and the hard constraints
your post must satisfy. Follow it; it is the source of truth for how the post
should sound.

You are given a topic_id and a post_id. Then:

1. Read the verified brief `briefs/<topic_id>.verified.json` (fall back to
   `briefs/<topic_id>.json` if the verified file is absent). Every factual
   claim in your post must trace to this brief — never invent numbers, quotes,
   capabilities, or URLs, and cite only URLs the brief provides. If the brief's
   verification_status is not "verified", soften or drop the affected claims.
2. Read the author's `style_profile.json` for voice signals (common openers,
   vocabulary, tone) and let them shape the phrasing.
3. Compose the post and write it to `drafts/<post_id>.json` with `write_file`,
   as valid JSON validating against the PostProposal schema. Author these
   fields: {authored}. Set supporting_topic_ids to exactly [topic_id] and
   citation_urls from the brief's citations.
4. Call `quality_gate` with the post_id. It enforces, deterministically:
   body 105–182 words (measured after a light jargon/CTA clean), exactly one
   supporting_topic_id, at least 3 hashtags, and no hype markers. If it returns
   status="failed", read the reasons from the verdict file at the returned
   `path`, fix the draft (rewrite drafts/<post_id>.json), and re-gate. Repeat
   until it passes; do not ship a failing draft.

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
    return SubAgent(
        name=WRITER_SUBAGENT_NAME,
        description=(
            "Write one LinkedIn post proposal from a verified ResearchBrief in "
            "the author's voice, gated by quality_gate. Reads "
            "briefs/<topic_id>.verified.json + style_profile.json, writes "
            "drafts/<post_id>.json (a PostProposal). Delegate one topic per call."
        ),
        system_prompt=_system_prompt(),
        tools=[build_quality_gate_tool(s)],
        model=build_openrouter_chat_model(
            s, model=s.openrouter_stage_b_writer_model
        ),
        skills=[skills_source],
    )
