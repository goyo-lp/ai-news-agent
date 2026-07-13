# Testing Guide

## Running Tests

```bash
uv run pytest -q
```

All tests are synchronous (pytest-asyncio is configured with `asyncio_mode = "auto"` in `pyproject.toml` but current tests don't use async fixtures).

## Type and Lint Checks

```bash
uv run mypy src
uv run ruff check .
```

Configuration: `ruff` with `line-length=100`, `target-version=py311`. `mypy` with `python_version=3.11`, `disallow_untyped_defs=true`, `warn_return_any=true`.

## Test Files

### `tests/test_rank.py` (11 tests)

The most comprehensive test file. Tests ranking, clustering, date filtering, relevance prioritization, history-aware novelty, and per-source capping.

| Test | What It Validates |
|------|------------------|
| `test_rank_articles_limits_output` | Output respects `limit` parameter (30 articles, limit=20 → 20 returned) |
| `test_rank_articles_orders_descending_score` | Fresh article with duplicates scores higher than old article from unknown source |
| `test_rank_articles_clusters_same_story_across_sources` | Two articles with similar titles from different sources cluster into 1 representative |
| `test_rank_articles_prioritizes_high_relevance_over_event_roundups` | "Startup raises Series B" outranks "Top AI events and webinar roundup" even when roundup is from OpenAI Blog |
| `test_rank_articles_uses_description_for_relevance_signals` | Relevance scoring uses description text, not just title — startup/deal keywords in description boost score |
| `test_rank_articles_caps_items_per_source` | With `max_per_source=3`, arXiv flood is capped to 3 items while other sources pass through |
| `test_rank_articles_boosts_frontier_company_mentions` | Article mentioning Anthropic/Claude outranks a generic version of the same headline |
| `test_rank_articles_penalizes_recently_delivered_titles` | When `recent_titles` is provided, articles matching delivered titles get lower novelty scores than baseline |
| `test_filter_articles_published_today_keeps_same_day` | Date filter keeps only articles published on the reference date |
| `test_filter_articles_published_today_respects_reference_timezone` | Date filter converts published_at to local timezone before comparing dates |
| `test_filter_articles_published_today_excludes_missing_dates` | Articles with `published_at=None` are excluded by date filter |

### `tests/test_ingest.py` (2 tests)

| Test | What It Validates |
|------|------------------|
| `test_normalize_url_removes_tracking_params` | `utm_source`, `fbclid` are stripped; non-tracking params (`id`) are kept |
| `test_dedupe_articles_keeps_newest_and_counts_duplicates` | Dedup keeps newest article by `published_at` and sets `duplicate_count` correctly |

### `tests/test_enrich.py` (1 test)

| Test | What It Validates |
|------|------------------|
| `test_extract_open_graph_fields` | OpenGraph extraction correctly parses `og:title`, `og:description`, `og:image` from HTML |

### `tests/test_summarize.py` (4 tests)

| Test | What It Validates |
|------|------------------|
| `test_enforce_sentence_count_exact_three` | `enforce_sentence_count` truncates to exactly 3 sentences when input has more |
| `test_parse_relevance_scores_extracts_and_normalizes` | Parses JSON object from LLM text, normalizes 0–100 to 0.0–1.0, caps at 1.0 |
| `test_parse_relevance_scores_drops_unknown_and_malformed_keys` | Drops keys not in `valid_keys` set and non-numeric values |
| `test_parse_relevance_scores_handles_non_json` | Returns `{}` when text has no JSON or broken JSON |

### `tests/test_telegram_format.py` (1 test)

| Test | What It Validates |
|------|------------------|
| `test_build_telegram_caption_has_link_and_limit` | Caption starts with `<a href="...">` and respects `TELEGRAM_CAPTION_LIMIT` (1024 chars) |

### `tests/test_history.py` (5 tests)

Tests the delivery history service (`src/app/services/history.py`).

| Test | What It Validates |
|------|------------------|
| `test_record_and_load_round_trip` | `record_deliveries` writes entries that `load_history` reads back; `delivered_titles` extracts titles |
| `test_record_skips_failed_deliveries` | Only `status: "sent"` results are recorded; failed deliveries are skipped |
| `test_load_history_prunes_expired_entries` | Entries older than `retention_days` are pruned on load |
| `test_load_history_missing_or_corrupt_file` | Missing file returns `[]`; corrupt JSON returns `[]` |
| `test_filter_previously_delivered_drops_known_urls` | Articles whose URL is in history are filtered out |

### `tests/test_workflow.py` (2 tests)

Tests the conditional routing function `route_after_rank` in `src/app/graph/workflow.py`.

| Test | What It Validates |
|------|------------------|
| `test_route_after_rank_continues_with_articles` | When `articles_selected` is non-empty, routes to `"summarize"` |
| `test_route_after_rank_ends_when_empty` | When `articles_selected` is empty or missing, routes to `END` |

## Test Helpers

`tests/test_rank.py` includes two factory functions:
- `_article(id, source, hours_old, ...)` — creates an Article with `published_at` set to N hours ago
- `_dated_article(id, published_at)` — creates an Article with an explicit datetime (used for date filter tests)

## Coverage Gaps

The current test suite focuses on unit-level behavior of individual services and routing logic. Notable areas without tests:
- End-to-end pipeline execution (dry-run integration test)
- Telegram send retry logic and 429 handling
- OpenRouter API call flow (summarization and relevance scoring) with live HTTP
- RSS feed parsing with real feed data
- Enrichment with blocked domains and image fallback

When adding tests, follow the existing pattern: construct `Article` objects directly, call service functions, assert on return values. No mocking framework is used — tests are purely deterministic.
