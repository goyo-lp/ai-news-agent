# Verification and bounded repair

PR10 keeps unverified drafts out of delivery. Deterministic code checks run
first (evidence resolution, unsupported numbers, stale stories, duplicate
events); a separate model verifier then judges grounding, relevance,
freshness, duplication, and missed stronger candidates with claim-level
findings and an explicit pass, revise, replace, or reject verdict. A LangGraph
workflow with typed state, durable checkpoints, and bounded edges reverifies
repairs and reserve replacements, terminating retries and recording
shortfalls.

## Run verification

```bash
uv sync --locked --all-groups
uv run ai-news-agent verify --database .local/ai-news-agent.db \
  --digest-run-id manual-001
```

The command verifies stored draft summaries (pass `--cluster <id>`
repeatably to limit the set, `--reserve <id>` for replacement candidates),
persists one `VerificationResult` per attempt (`<summary-id>-v<attempt>`),
marks summaries verified or rejected, and emits `attempted`, `verified`,
`rejected`, `replaced`, `shortfall`, and `verified_ids`. Only verified
summaries reach delivery via `verified_summaries_for_delivery()`; the digest
meets its minimum only with at least three verified stories.

The verifier needs `OPENROUTER_API_KEY` in local `.env` (see `.env.example`);
without it the command fails fast instead of guessing.

## Behavior

- Deterministic ERROR findings short-circuit the model call: missing
  articles, unresolvable evidence, numbers absent from evidence, stories older
  than 36 hours, and duplicate canonical URLs become reject or replace with
  recorded findings.
- Revise regenerates the draft once through the summary writer and
  reverifies; replace swaps in the next reserve candidate; both terminate
  after `--max-attempts` (default 2) and excluded items record shortfalls.
- The LangGraph state carries pending IDs, attempts, verified IDs, outcomes,
  and replacements with a `MemorySaver` checkpoint keyed by digest run ID, so
  resumed executions upsert by stable IDs instead of duplicating artifacts.
- Each verification is traced (`verify-summary`, `verification-workflow`)
  when LangSmith tracing is enabled, with structured outcomes attached.

## Verification

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

Verification tests use injected fake verifiers and writers for wrong numbers,
stale and duplicate stories, repair, reserve replacement, retry termination,
and resume idempotency, plus transport-level tests of the OpenRouter client.
