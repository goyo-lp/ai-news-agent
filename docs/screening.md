# Candidate screening

PR06 reduces clustered stories to a shortlist of roughly 30–50 candidates.
Deterministic date and history rules run first; survivors go to a low-cost
structured relevance judge (OpenRouter, configured but swappable). Rejected
candidates keep their reasons, borderline items are preserved for review, and
verdicts are cached by content, policy, prompt, and model version.

## Run screening

```bash
uv sync --locked --all-groups
uv run ai-news-agent screen --database .local/ai-news-agent.db --max-candidates 50
```

The command reads stored clusters (up to `--limit`, default 100), applies the
rules below, retrieves full text for kept stories through PR04's tool, merges
clusters with identical retrieved text, and emits a JSON summary with
`attempted`, `shortlisted`, `borderline`, `rejected`, and `merged` counts.
Pass `--published <article-id>` (repeatable) for already-published articles.

The judge needs `OPENROUTER_API_KEY` in local `.env` (see `.env.example`);
without it the command fails fast instead of guessing.

## Behavior

- Clusters are rejected before any model call when the representative article
  is missing, already published, or older than 36 hours. Publication 24–36
  hours ago stays eligible as a recorded late arrival.
- The judge answers one strict JSON object per story
  (`relevant`, `borderline`, `reason`) against the editorial exclusions:
  non-AI stories, AI-as-marketing, explainers, rumors without evidence, and
  transaction-only funding news are out; genuinely uncertain cases come back
  borderline.
- Malformed judge output becomes a cached borderline item; transport failures
  become uncached borderline items with a visible error count; rejected
  credentials fail the run instead of degrading silently.
- Relevant stories fill the cap first (newest first), then borderline; the
  overflow is rejected with its reason, so every exclusion is auditable.
- Kept stories get full-text retrieval, and clusters whose representatives
  share identical retrieved text merge with their source cluster IDs recorded
  in the rationale.
- Each run is traced (`screen-clusters`) when LangSmith tracing is enabled.

## Verification

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

Screening tests use an injected fake judge for date rules, caching, caps,
and re-merging, plus transport-level tests of the OpenRouter client. A live
two-call probe against the configured free model verified end-to-end
classification during development.
