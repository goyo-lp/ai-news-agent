# Feed ingestion

PR03 collects the 50-source roster in `config/sources.yaml` into validated
`Article` records with raw-feed `Evidence`. Every route starts as a candidate;
no source is active until a live run verifies reachability, freshness,
canonical links, dates, and one readable article.

## Roster

`config/sources.yaml` holds one entry per source with its publisher family,
category, tier, homepage, route (`rss` | `atom` | `official_page_adapter`),
discovery provenance, feed scope, and editorial notes. All 50 entries ship
disabled with `pending-pr03-validation` until ingestion validates them.

The 12 `official_page_adapter` routes have an interface but no validated
scraper yet; ingestion records them as `unavailable` with an explicit reason
rather than silent success. The VentureBeat (HTTP 429) and BAIR (timeouts)
rechecks, large TechNode (~11.6 MB) and METR (~8.9 MB) responses, the
newsletter-only Epoch AI feed, and the Google family grouping are noted in
the roster and handled below.

## Run ingestion

```bash
uv sync --locked --all-groups
uv run ai-news-agent ingest --config config/sources.yaml \
  --database .local/ai-news-agent.db --digest-run-id manual-001
```

The command saves `Source` health (`last_checked_at`, `last_success_at`),
`Article` and `feed_entry` `Evidence` records, conditional-request `feed_state`,
and a `Run` with `attempted`, `successful`, `unchanged`, `failed`, and
`unavailable` counts. It emits a JSON summary to stdout and structured logs
to stderr. Re-running the same feed creates no duplicate articles: article IDs
are stable hashes of the canonical URL and existing records keep their
original `first_seen_at`.

## Behavior

- Bounded concurrent fetching (`--max-workers`, default 8) with per-feed
  timeouts (`--timeout`, default 15 s), retry backoff for timeouts, 429s, and
  5xx, and fail-fast behavior for other 4xx responses.
- Conditional requests (`If-None-Match`, `If-Modified-Since`) with persisted
  `feed_state`; HTTP 304, matching content hashes, and fully duplicate feeds
  all report as `unchanged`.
- Bounded streaming (`--max-bytes`, default 15 MB) so TechNode and METR cannot
  exhaust memory; oversized responses fail the source without failing the run.
- URL normalization strips fragments and marketing parameters, resolves
  relative links, and rejects private, local, non-HTTP, and credential-bearing
  destinations.
- Date normalization accepts RFC 822 and ISO 8601 timestamps and requires
  timezone-aware storage. Missing dates fall back to `first_seen_at` for
  collection; malformed entries are quarantined with a warning instead of
  rejecting the feed, while malformed XML fails only that source.
- Raw feed evidence is preserved as `feed_entry` excerpts linked to each
  article; adapter sources never invent articles.
- Each run is traced under the shared `digest_run_id` (`ingest-sources` with
  per-source `ingest-source` children) when LangSmith tracing is enabled.

## Verification

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

Feed fixtures in `tests/fixtures/feeds/` cover valid RSS, valid Atom,
malformed XML, missing dates, and repeated entries. Ingestion tests cover
timeouts, 304 handling, retry budgets, and two identical runs producing no
duplicates.
