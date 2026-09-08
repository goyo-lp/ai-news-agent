# Digest selection

PR08 turns scored clusters into one coherent digest of at most ten stories plus
a ranked reserve list. Eligibility enforces distinct events and the editorial
quality floor in code; soft publisher/topic preferences reorder close calls
with retained rationales. Quiet days publish fewer than ten with a recorded
shortfall instead of padding.

## Run selection

```bash
uv sync --locked --all-groups
uv run ai-news-agent select --database .local/ai-news-agent.db --draft-path .local/draft.md
```

The command reads stored clusters, scores, and articles (up to `--limit`,
default 100), selects up to `--max-selected` (default 10) with `--max-reserve`
reserves (default 5), and emits `attempted`, `eligible`, `selected`,
`reserve`, `shortfall`, `selected_ids`, and `reserve_ids`. `--draft-path`
writes a markdown preview for review before summarization.

## Behavior

- A story is eligible only with a saved `Score`, an accessible representative
  article, a distinct canonical URL, evidence references, relevance and
  evidence at least 3, and a weighted total of at least 65.
- Candidates sort by weighted total descending with a stable cluster-ID
  tie-break. Duplicates beyond the first canonical URL stay out with reasons.
- Soft diversity caps (default 2 per publisher family, 4 per topic bucket)
  defer violators, fill open slots with deferred items, and allow one swap when
  a deferred candidate trails by at most 15 points. Larger gaps keep the more
  consequential story; every swap records both scores and a rationale.
- Reserves are the next eligible clusters in score order. Shortfalls report
  selected versus eligible versus attempted counts.
- Each run is traced (`select-digest`) when LangSmith tracing is enabled.

## Verification

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

Selection tests cover the quality floor, duplicate collapse, quiet-day
shortfalls, one-topic dominance without rigid quotas, stable tie-breaks, and
draft rendering.
