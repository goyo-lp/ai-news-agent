# Migration decisions — locked in

Every fork below is settled (2026-07-18). The LinkedIn agent's source is vendored in-repo at
`reference/linkedin-agent/` for the port, and it already ships on
`deepagents.create_deep_agent` — so the target framework is proven in our own code, not just
assumed.

This file is the tracked record of the migration decisions; the visual version lived in the
local-only `architecture/ai-news.html` (gitignored) during planning. Downstream Phase 0+ PRs
build on the choices below.

## A — Where it lives
Evolve the AI News Agent repo into the coordinator. Its 5-node news pipeline is reused
in-process as the `fetch_curated_ai_news` tool — no cross-repo packaging. The LinkedIn logic is
ported in (Tavily, verifier, technical ranker, style profile, post generation, deep-agent
layer), each module vendored alongside the tool/subagent that wraps it.

Source copied to `reference/linkedin-agent/`. Trade-off: more porting, but the news producer
needs zero packaging and the plan lives here.

## B — Framework
`deepagents.create_deep_agent` (coordinator + subagents), not a single ReAct agent.
Context isolation, per-subagent model override (keeps the Stage A/B split), and reuse of
existing logic as tools. Already a dependency and in use inside the LinkedIn agent's
investigation node.

## C — Delivery
Telegram only — two bots, one process. The `news` bot keeps the daily digest; a new `linkedin`
bot receives post proposals. Proposals auto-send as drafts once they pass the quality gate —
manually post the good ones to LinkedIn. No LinkedIn API, no review hold.

Quality gate is the only filter; failures are dropped, not retried endlessly.

## D — News digest survives
Keep both outputs. The existing daily news digest keeps flowing to the `news` bot unchanged;
LinkedIn proposals are a second product, not a replacement.

## E — Topic source
The News Agent RSS pipeline is the sole producer. `ingest → enrich → rank → dedupe` feeds
`technical_rank`. The LinkedIn agent's own Tavily/RSS discovery is dropped; Tavily stays only as
per-topic research evidence.

## F — Migration style
Full rebuild per this plan. Build the coordinator + research/writer subagents fresh, prove
parity, cut over, then delete the legacy 13-node graph — not an incremental wrap of the old
graph.

## G — Cross-run dedup
Reuse the news pipeline's `delivery-history.json` + novelty. Topics are already deduped
upstream; no separate LinkedIn-post history file.

## H — Trigger
Manual CLI for now (`python -m app.main propose`). The news `run` keeps its existing trigger.
Scheduling added later once quality is dialed in.

## I — Observability
LangSmith tracing only. The LinkedIn agent's `langgraphics` (git+https) node-graph visualizer
is dropped — the coordinator isn't a fixed LangGraph, so it no longer maps cleanly. Removes a
git dependency.

## J — Models (via OpenRouter, env-tunable)
Every agent model uses **`deepseek/deepseek-v4-flash`** — the same model the news pipeline
already summarizes with. One model across all four tiers:
- **Stage A rank** (`openrouter_stage_a_model`)
- **Verifier** (`openrouter_verifier_model`)
- **Research subagent** (`openrouter_stage_b_research_model`)
- **Writer subagent** (`openrouter_stage_b_writer_model`)

The per-tier knobs stay separate (not collapsed into one) so any tier can be pointed at a
different model via env without touching the others — the writer is the natural first upgrade
if post voice quality warrants it. The Stage A / Stage B split (principle #4) is preserved
structurally; only the defaults are unified.

## Guiding principles

1. **Deterministic work stays code, exposed as tools.** Never agentify a fixed recipe — you'd
   pay tokens to re-derive tested logic.
2. **Spend agency only where decisions are open-ended.** The one genuinely adaptive surface is
   per-topic research: "fetch the real article or trust the text? verify what? what's
   technically new?"
3. **Structured data lives in the filesystem/StateBackend; subagents return only compressed
   findings.** A subagent's single text report is lossy — never route the full ranked-article
   list through it.
4. **Preserve the Stage A / Stage B model split** via per-subagent model overrides — each tier
   keeps its own knob so it can diverge later, even though Decision J now defaults them all to
   one model.
5. **Reuse, don't rewrite.** The news pipeline, SSRF-guarded fetch, Tavily client, Telegram
   client, and style profile are assets — wrap them.
6. **Every change ships as a ≤500-LOC PR with tests.** Dry-run parity against the old pipeline
   before any cutover.

## Reuse inventory

`run_curation` (News Agent), `http_utils.py` (SSRF + size cap), `tavily_client.py`,
`brief_verifier.py`, `technical_ranker.py` / `scoring.py`, `quality_gate`, `style_profile.py`,
`telegram_client.py`, `api_usage_tracker.py`. The plan wraps every one of these rather than
reimplementing it — all now vendored in-repo at `reference/linkedin-agent/` for the port.