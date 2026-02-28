# AI News Agent Build Plan (Codex Execution Playbook)

## Goal

Build a Python + LangGraph AI news agent that:
- Reads RSS sources from `news-sources.yaml`
- Deduplicates both exact URLs and cross-source “same story” coverage
- Ranks and selects top 50 stories per run
- Sends exactly 1 Telegram message per selected story
- Message format (Telegram-friendly HTML):
  1. Clickable title linking to source URL
  2. Photo
  3. Exactly 3-sentence summary
- Uses OpenRouter model `openai/gpt-oss-20b`
- Uses LangSmith as the only observability system
- Uses LangGraphics for live local graph visualization during each run
- Runs by manual CLI command only (no scheduler)

## Non-Negotiable Constraints

- Language: Python 3.11+
- Framework: LangGraph
- LLM provider: OpenRouter
- Observability: LangSmith only
- Visualization: LangGraphics (auto-start on CLI run)
- Secrets: `.env` only
- Delivery: Telegram Bot API
- Run mode: manual CLI trigger only
- Output volume: max 20 articles per run

## Target Architecture

1. Orchestrator Node
- Drives the LangGraph flow and shared state.

2. Ingestion Node
- Parses RSS feeds and normalizes article records.

3. Enrichment Node
- Resolves canonical URL, extracts OG metadata, applies image fallbacks.

4. Ranking Node
- Clusters same-story articles across different sources, selects cluster representatives, scores and keeps top 50.

5. Summarization Node
- Produces exactly 3-sentence summaries via OpenRouter.

6. Delivery Node
- Sends one Telegram message per article with retry/backoff.

## Codex Operating Instructions

Codex should execute in small, verifiable phases and commit after each phase.

For each phase:
1. Implement only the scope listed.
2. Run the listed validation commands.
3. Fix failures before moving on.
4. Update docs/examples when interfaces change.

If blocked by missing credentials, Codex should:
1. Add clear placeholders to `.env.example`.
2. Continue with mocked/local tests.
3. Mark external integration tests as pending.

## Repository Structure To Create

```text
ai-news-agent/
  src/
    app/
      graph/
        state.py
        workflow.py
      nodes/
        ingest.py
        enrich.py
        rank.py
        summarize.py
        deliver.py
      services/
        rss_client.py
        extractor.py
        openrouter_client.py
        telegram_client.py
        scoring.py
      schemas/
        article.py
      config.py
      logging.py
      main.py
  tests/
    test_ingest.py
    test_enrich.py
    test_rank.py
    test_summarize.py
    test_telegram_format.py
  data/
    news-sources.yaml
  .venv/
  .env.example
  requirements.txt
  pyproject.toml
  README.md
```

## Shared Data Contract

Article fields (required unless noted):
- `id` (stable hash)
- `source_name`
- `source_rss`
- `title`
- `url`
- `published_at`
- `og_title` (optional)
- `og_description` (optional)
- `image_url` (optional)
- `summary` (optional)
- `score` (optional)
- `cluster_id` (optional)
- `cluster_size` (optional)

Run state fields:
- `run_id`
- `started_at`
- `articles_raw`
- `articles_enriched`
- `articles_ranked`
- `articles_top20`
- `delivery_results`
- `errors`

## Telegram Message Contract

Use Telegram `HTML` parse mode.

Per article send in one message:
1. `<a href="ARTICLE_URL">TITLE</a>`
2. Photo
: Preferred: send as photo caption message.
: Fallback: send photo then text message only if Telegram API constraints require split.
3. 3-sentence summary text.

Formatting rules:
- Escape HTML entities in title and summary.
- Do not exceed Telegram limits.
- If no valid image exists, send text-only message with title + summary.

## Phase-by-Phase Build Plan For Codex

### Phase 1: Bootstrap project

Deliverables:
- `.venv` virtual environment
- `requirements.txt`
- `pyproject.toml`
- base package layout
- `.env.example`
- `README.md` quickstart

Dependencies:
- `langgraph`, `langchain`, `httpx`, `feedparser`, `beautifulsoup4`, `lxml`, `pydantic`, `pydantic-settings`, `python-dotenv`, `langsmith`, `pytest`, `pytest-asyncio`, `mypy`, `ruff`

Validation:
- `uv venv .venv`
- `uv run ruff check .`
- `uv run mypy src`

### Phase 2: Config + schemas + state

Deliverables:
- `config.py` with env loading
- `schemas/article.py`
- `graph/state.py`

Validation:
- Unit tests for config parsing and schema validation.

### Phase 3: Ingestion

Deliverables:
- `rss_client.py`
- `nodes/ingest.py`

Behavior:
- Load sources from `data/news-sources.yaml`
- Parse RSS entries
- Normalize and dedupe (canonical URL + hash)

Validation:
- `tests/test_ingest.py` with fixture feeds
- Assert dedupe correctness

### Phase 4: Enrichment

Deliverables:
- `extractor.py`
- `nodes/enrich.py`

Behavior:
- Resolve final URLs
- Extract `og:title`, `og:description`, `og:image`
- Apply fallback rules from source config

Validation:
- `tests/test_enrich.py`
- Cases: valid OG image, missing OG image, blocked domain

### Phase 5: Ranking

Deliverables:
- `scoring.py`
- `nodes/rank.py`

Behavior:
- Build same-story clusters across sources using title/content similarity and time window checks
- Select one representative article per cluster
- Apply deterministic score from recency, source weight, URL-duplication signal, cluster-support signal, novelty
- Keep top 50 representative stories

Validation:
- `tests/test_rank.py`
- Assert sort order and exactly 20 max
- Assert same-story articles from different sources collapse into one representative

### Phase 6: Summarization

Deliverables:
- `openrouter_client.py`
- `nodes/summarize.py`

Behavior:
- Prompt model `openai/gpt-oss-20b`
- Enforce exactly 3 sentences per article
- Retry once on malformed output

Validation:
- `tests/test_summarize.py` using mocked OpenRouter responses

### Phase 7: Telegram delivery

Deliverables:
- `telegram_client.py`
- `nodes/deliver.py`
- message formatter utility

Behavior:
- One message per article
- Clickable title + image + 3-sentence summary
- Retry/backoff on Telegram errors

Validation:
- `tests/test_telegram_format.py`
- Mock Telegram API responses

### Phase 8: LangGraph wiring + CLI

Deliverables:
- `graph/workflow.py`
- `main.py` with manual run command

CLI contract:
- `uv run python -m app.main run`
- optional flags: `--dry-run`, `--limit 50`, `--verbose`

Validation:
- End-to-end dry run with sample data

### Phase 9: LangSmith instrumentation

Deliverables:
- LangSmith tracing around each node and LLM call

Validation:
- Confirm one trace per run with node-level spans

### Phase 10: Final hardening

Deliverables:
- Error taxonomy and graceful failure handling
- Idempotency guard to avoid duplicate sends in same run
- README operational runbook

Validation:
- Full test suite + lint + type checks
- Manual integration run with real Telegram chat

## Definition of Done

Project is done when all are true:
1. Manual CLI run completes without crashing.
2. Pipeline selects top 50 or fewer articles.
3. Each selected article produces one Telegram message.
4. Each message has clickable title, image when available, and exactly 3-sentence summary.
5. OpenRouter calls use `openai/gpt-oss-20b`.
6. LangSmith traces show complete run and node timings.
7. Lint, types, and tests pass.
8. Cross-source same-story duplicates are clustered and only one representative is sent.

## Acceptance Test Checklist

1. Run with `--dry-run` and inspect generated payloads.
2. Run against real Telegram bot and verify formatting on mobile.
3. Force missing image and verify fallback behavior.
4. Force malformed model output and verify re-generation.
5. Re-run same inputs and confirm duplicate suppression behavior.
