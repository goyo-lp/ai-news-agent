# Editorial agent

PR07 investigates shortlisted story clusters and ranks them with per-criterion
scores. Deterministic evidence tools run first; a structured judge supplies
relevance, impact, novelty, evidence, and timeliness (0–5) plus evidence
references, uncertainty, and a rationale. The weighted total is computed in code
and persisted as a `Score`.

## Run ranking

```bash
uv sync --locked --all-groups
uv run ai-news-agent rank --database .local/ai-news-agent.db --digest-run-id manual-001
```

The command reads stored clusters (up to `--limit`, default 100), investigates
each one, saves one `Score` per ranked cluster (`<cluster-id>-editorial`), and
emits `attempted`, `ranked`, `incomplete`, `revisions`, and `ranking` (best
first, stable tie-break by cluster ID). Pass `--published <article-id>`
(repeatable) for digest history.

The judge needs `OPENROUTER_API_KEY` in local `.env` (see `.env.example`);
without it the command fails fast instead of guessing.

## Behavior

- LangChain tools: `read_article`, `inspect_coverage`, `fetch_evidence`, and
  `consult_history` operate on the SQLite store. `make_editorial_tools()` binds
  them to one run; `build_editorial_agent()` wires them into
  `langchain.agents.create_agent` for live investigation.
- Primary article text stays separate from supporting investigation context.
  Revisions enrich the supporting context only; the representative article is
  never replaced by a supporting source.
- Uncertain verdicts (`needs_more_evidence=true`) get one targeted
  reinvestigation when tool-call, judge-call, and elapsed-time budgets allow.
- Budgets default to 40 tool calls, 20 judge calls, and 120 seconds per run.
  Exhaustion marks remaining clusters `incomplete` with
  `termination_reason=budget-exhausted` instead of looping.
- Malformed judge output becomes `incomplete` (`malformed-output`); transport
  failures become `incomplete` (`judge-unavailable`); missing credentials fail
  the run. No Score is saved for incomplete clusters.
- Each run is traced (`editorial-ranking`) when LangSmith tracing is enabled,
  with model identity, prompt version, calls, and termination reason in the
  summary and shared digest run ID on the trace metadata.

## Verification

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

Editorial tests use an injected fake judge for scoring, revision, budgets, and
tool separation, plus transport-level tests of the OpenRouter client.
