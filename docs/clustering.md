# Story clustering

PR05 groups collected `Article` records into distinct news events persisted
as `StoryCluster` records. Matching is fully deterministic: no model calls,
no API keys, no network access. Ambiguous near-misses stay separate with
their scores recorded in the rationale, which is the seam where a future
model-judged pass can revisit them.

## Run clustering

```bash
uv sync --locked --all-groups
uv run ai-news-agent cluster --database .local/ai-news-agent.db --limit 200
```

The command reads stored articles (up to `--limit`), writes one `StoryCluster`
per distinct event, and emits a JSON summary with `attempted`, `clusters`,
`duplicates_collapsed`, and `recycled` counts. Cluster IDs are stable hashes
of the sorted member IDs, so re-running the same articles changes nothing.
Pass `--published <article-id>` (repeatable) for already-published articles;
until digests persist their own history, this is operator-supplied.

## Behavior

- Exact canonical-URL duplicates collapse to the earliest-seen article before
  matching; identical full-text content hashes always merge, regardless of
  headline or date.
- Otherwise two articles merge when they share at least half their headline
  tokens within a 48-hour publication window, or shared capitalized entities
  plus moderate title overlap. Anything older than 48 hours apart never
  merges on text signals alone (republication is not a new event).
- Pairs in the ambiguous band stay separate, and the rationale records both
  overlap scores with a note that merging needs model judgment.
- The representative is the member with retrieved full text, then a byline,
  then earliest publication; supporting articles are retained as member IDs.
  Every rationale states why the group formed (or why a singleton stands
  alone).
- A cluster whose every member was already published is flagged `recycled`.
  A new article on an old topic forms a fresh, non-recycled cluster, so
  substantive follow-ups qualify while repeats do not.
- Each run is traced (`cluster-articles`) when LangSmith tracing is enabled.

## Verification

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

Clustering tests cover syndicated rewrites merging, distinct launches staying
apart, URL and content-hash dedupe, entity-boosted merges, ambiguous
near-misses, stale republication, representative preference, recycled
flagging, follow-up freshness, and rerun stability.
