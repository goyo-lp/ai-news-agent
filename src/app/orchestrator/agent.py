"""Coordinator deep-agent wiring — Phase 5.2.

The single place that turns settings + already-built pieces (the FilesystemBackend
from P5.1's ``state.py``, the deterministic coordinator-side tools, the research
+ writer subagents from P4) into a ``CompiledStateGraph`` ready for the
orchestration prompt (P5.3) and end-to-end invoke.

What it wires and what it deliberately does *not*:

- **backend** = ``build_orchestrator_backend(settings)`` (P5.1). This is the
  load-bearing mount the P4 subagents document as a precondition: deepagents'
  built-in ``write_file`` / ``read_file`` route through it, and so does the
  research/writer subagent ``task()`` path — every ``write_file`` they emit
  lands on real disk at ``orchestrator_data_dir`` where the deterministic
  custom tools read via stdlib.

- **tools** = the *coordinator-side* deterministic tools:
  ``fetch_curated_ai_news`` (P1.3, producer), ``technical_rank`` (P2.1,
  filter), ``quality_gate`` (P2.5, post-write verdict the coordinator also
  reads directly at delivery time). The research tools (``fetch_article``,
  ``web_search``, ``web_extract``, ``verify_claim``) and the writer tool
  (``quality_gate`` as a write-side gate) are *subagent-owned*: handing
  them to the coordinator too would invite the model to perform Stage-B
  research inline in the coordinator's window — spending planner tokens on
  researcher work and bypassing the per-topic isolated context that
  ``task()`` provides. The deterministic custom tools already return
  compressed summaries (never the bulk payload), so principle #3 isn't the
  binding risk at this seam — agency / cost budget is.

- **subagents** = ``[build_research_subagent(s), build_writer_subagent(s)]``
  (P4.1 + P4.2). The coordinator delegates to these via ``task()`` *per
  topic*; each task call runs in an isolated context, so the ranked
  article list never fits inside the coordinator's window — only the
  brief / draft artifacts on disk cross back as compressed summaries.

- **skills** = ``None``. The one skill (``linkedin-voice``, P3.1) belongs
  to the *writer subagent* (P4.2 wires its absolute source path into
  ``build_writer_subagent``). Mounting it on the coordinator drifts the
  skill into a context the writer can't see (SkillsMiddleware reads from
  the *coordinator*'s mount when the coordinator invokes tools directly),
  and invites the coordinator model to author posts in the author's voice
  — exactly the work that was supposed to be delegated.

- **system_prompt** = ``None`` for now. P5.3 owns the orchestration
  prompt. Acceptance for *this* PR is "Agent builds; dry-run invoke
  completes without error" — deepagents' default prompt is sufficient for
  that; the orchestration plan lands next PR.

- **context_schema** = ``None``. No run-scoped context the tools need to
  share yet (the LiveRunContext — run id, profile, dry-run bit — would
  land with delivery / observability phases, not with the build stub).

- **model** = ``build_openrouter_chat_model(s, model=s.openrouter_coordinator_model)``.
  New knob ``openrouter_coordinator_model`` lands here with its consumer
  (per the repo's pinned config policy). Default Decision J's
  ``deepseek/deepseek-v4-flash``. Tests pass an explicit ``model`` to
  avoid hitting OpenRouter — same seam ``build_orchestrator_backend`` uses
  for settings, applied to the model.

The factory takes ``model=None`` *by us* — the type stays ``BaseChatModel`` so
deepagents' ``create_deep_agent`` type-checks; we never call it without one
in production, and tests inject a fake. This pattern mirrors how the
research/writer subagents accept settings lazily.
"""
from __future__ import annotations

from typing import Any

from deepagents import create_deep_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph.state import CompiledStateGraph

from app.config import Settings, get_settings
from app.orchestrator.models import build_openrouter_chat_model
from app.orchestrator.state import build_orchestrator_backend
from app.orchestrator.subagents.research import build_research_subagent
from app.orchestrator.subagents.writer import build_writer_subagent
from app.orchestrator.tools.news import build_fetch_curated_ai_news_tool
from app.orchestrator.tools.quality import build_quality_gate_tool
from app.orchestrator.tools.technical_rank import build_technical_rank_tool

COORDINATOR_AGENT_NAME = "ai-news-coordinator"


def build_coordinator_agent(
    settings: Settings | None = None,
    *,
    model: BaseChatModel | None = None,
) -> CompiledStateGraph[Any, Any, Any, Any]:
    """Construct the coordinator ``CompiledStateGraph``.

    Args:
        settings: lazily resolved via ``get_settings()`` when not supplied
            (mirrors every tool factory + the state backend factory).
        model: optional pre-initialized chat model — the seam tests use to
            inject a fake chat model that completes one loop without hitting
            OpenRouter. When ``None`` (the production path), the model is
            built from ``settings.openrouter_coordinator_model`` via the
            shared OpenRouter model builder (P4.1's ``models.py``), so the
            same attribution headers / endpoint / placeholder-key behavior
            the subagents already rely on applies here too.

    Returns:
        A ``CompiledStateGraph`` ready for ``await agent.ainvoke(...)``.

    Notes on what is deliberately *not* set here (kept honest in the module
    docstring too):
      * ``system_prompt`` — owned by P5.3, not this build stub.
      * ``context_schema`` — no run-scoped context the tools share yet.
      * ``middleware`` — left at default; cost / observability middleware
        lands with their consumers (Phase 7).
      * ``checkpointer`` — defaults to ``None`` (in-memory); persisted runs
        and retries across crashes land with Phase 7's observability work.
    """
    s = settings or get_settings()
    chat_model = model or build_openrouter_chat_model(
        s, model=s.openrouter_coordinator_model
    )
    backend = build_orchestrator_backend(s)

    # Coordinator-level tools only — see module docstring for why the research
    # / writer / verify tools are NOT here.
    tools = [
        build_fetch_curated_ai_news_tool(s),
        build_technical_rank_tool(s),
        build_quality_gate_tool(s),
    ]

    # Subagents built from the same Settings so they share one config
    # (backend root, OpenRouter key, data dir, model defaults).
    subagents = [
        build_research_subagent(s),
        build_writer_subagent(s),
    ]

    return create_deep_agent(
        model=chat_model,
        tools=tools,
        subagents=subagents,
        backend=backend,
        # No coordinator-level system prompt yet — P5.3 owns the
        # orchestration prompt (write_todos -> fetch -> rank -> per-topic
        # delegate -> quality-gate -> deliver). deepagents' default prompt
        # is sufficient for the "build + invoke completes" acceptance here.
        system_prompt=None,
        skills=None,
        name=COORDINATOR_AGENT_NAME,
    )