# AI News Agent Build Plan (Codex Execution Playbook)

## Goal

Build and maintain a Python + LangGraph AI news agent that:
- Reads RSS sources from `data/news-sources.yaml`
- Deduplicates exact URLs and cross-source same-story coverage
- Ranks and selects top 50 stories per run (or fewer)
- Sends exactly 1 Telegram message per selected story
- Uses Telegram-friendly formatting:
  1. Clickable title linking to source URL
  2. Photo
  3. Exactly 3-sentence summary
- Uses OpenRouter model `openai/gpt-oss-20b`
- Uses LangSmith as the only observability tool
- Uses LangGraphics for live local graph visualization during runs
- Runs via manual CLI command only

## Non-Negotiable Constraints

- Language: Python 3.11+
- Framework: LangGraph
- LLM provider: OpenRouter
- Observability: LangSmith only
- Visualization: LangGraphics
- Secrets: `.env` only
- Delivery: Telegram Bot API
- Run mode: manual CLI trigger only
- Output volume: max 50 articles per run

## Current Architecture

1. Ingestion Node
- Parses RSS feeds and normalizes records.
- Removes tracking query params and deduplicates exact URL matches.

2. Enrichment Node
- Resolves canonical URL and extracts `og:title`, `og:description`, `og:image`.
- Applies source-specific image fallback rules.

3. Ranking Node
- Clusters same-story coverage across sources.
- Uses relevance-first scoring to prioritize:
  - new tech/features/releases
  - startups/funding
  - technical breakthroughs
  - enterprise adoption/deployments
  - AI deals/partnerships/acquisitions
- Demotes lower-priority event/roundup style items unless they also contain high-signal relevance.
- Selects cluster representatives and keeps top 50.

4. Summarization Node
- Generates exactly 3-sentence summaries via OpenRouter.

5. Delivery Node
- Sends one Telegram message per article in HTML mode.

## Shared Data Contract

Article fields:
- `id`
- `source_name`
- `source_rss`
- `title`
- `url`
- `published_at`
- `description` (optional)
- `og_title` (optional)
- `og_description` (optional)
- `image_url` (optional)
- `summary` (optional)
- `score` (optional)
- `duplicate_count`
- `cluster_id` (optional)
- `cluster_size` (optional)

Run state fields:
- `run_id`
- `started_at`
- `articles_raw`
- `articles_enriched`
- `articles_ranked`
- `articles_top20` (legacy key name; currently stores selected output list)
- `delivery_results`
- `errors`

## Telegram Message Contract

Use Telegram `HTML` parse mode.

Per article send in one message:
1. `<a href="ARTICLE_URL">TITLE</a>`
2. Photo (if available)
3. Exactly 3-sentence summary

Formatting rules:
- Escape HTML entities in title and summary.
- Respect Telegram caption/message length limits.
- If no valid image exists, send text-only message.

## Codex Execution Sequence (Detailed)

### Phase 1: Bootstrap and Environment
Deliverables:
- `.venv`
- `requirements.txt`
- `pyproject.toml`
- base package layout
- `.env.example`

Validation:
- `uv venv .venv`
- `uv run ruff check .`
- `uv run mypy src`

### Phase 2: Config, Schemas, and State
Deliverables:
- env-driven config loading
- article/schema models
- graph state model

Validation:
- unit tests for config/schema parsing

### Phase 3: RSS Ingestion
Deliverables:
- RSS fetch client
- ingest node

Validation:
- ingest tests with duplicate URL scenarios

### Phase 4: OpenGraph Enrichment
Deliverables:
- extractor service
- enrich node

Validation:
- tests for OG extraction, fallback image behavior, blocked domains

### Phase 5: Relevance-First Ranking
Deliverables:
- scoring service
- rank node

Validation:
- tests that high-signal relevance outranks generic event/roundup content
- tests that same-story clustering returns one representative
- limit behavior remains correct (<= 50)

### Phase 6: Summarization
Deliverables:
- OpenRouter client
- summarize node

Validation:
- tests enforce exactly 3-sentence output contract

### Phase 7: Telegram Delivery
Deliverables:
- Telegram client
- delivery node
- formatting utility

Validation:
- tests for HTML-safe formatting and one-message-per-article behavior

### Phase 8: Graph Wiring + CLI
Deliverables:
- workflow wiring in LangGraph
- CLI `run` command with `--dry-run`, `--limit`, `--verbose`

Validation:
- end-to-end dry-run execution

### Phase 9: Observability + Visualization
Deliverables:
- LangSmith traces for node and LLM spans
- LangGraphics live visualization on local host/port

Validation:
- verify trace visibility and graph rendering during run

### Phase 10: Hardening and Docs
Deliverables:
- runbook-level README
- CLI command docs
- troubleshooting notes

Validation:
- lint + type checks + tests all pass
- manual integration run against real Telegram bot

## Definition of Done

Project is done when all are true:
1. Manual CLI run completes reliably.
2. Pipeline selects top 50 or fewer articles.
3. Each selected article sends one Telegram message.
4. Each message includes clickable title, image when available, and exactly 3-sentence summary.
5. OpenRouter calls use `openai/gpt-oss-20b`.
6. LangSmith traces include full run + node spans.
7. Tests, lint, and type checks pass.
8. Exact duplicates and same-story cross-source duplicates are both handled.
9. Relevance-first ranking prioritizes product/market-moving AI news over lower-signal event roundup content.
