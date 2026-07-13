# AI News Agent — Quickstart

A LangGraph-based AI news pipeline that fetches RSS feeds, enriches articles with OpenGraph metadata, ranks by relevance, summarizes via OpenRouter, and delivers digest messages to Telegram.

## What This Repo Does

1. **Ingests** RSS feeds from 20+ AI news sources (`data/news-sources.yaml`)
2. **Enriches** each article with OpenGraph metadata (title, description, image)
3. **Ranks** articles using deterministic relevance scoring + LLM re-ranking, with same-story clustering and delivery-history dedup
4. **Summarizes** top articles with exactly 3-sentence LLM summaries (OpenRouter `openai/gpt-oss-20b`)
5. **Delivers** one Telegram message per article with a clickable title, photo, and summary; records deliveries to `data/delivery-history.json` for future dedup

## Quick Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # Fill in OPENROUTER_API_KEY, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
```

## Run Commands

```bash
# Dry run (no Telegram sends)
PYTHONPATH=src python -m app.main run --dry-run

# Real run (sends to Telegram)
PYTHONPATH=src python -m app.main run

# Limit and verbose
PYTHONPATH=src python -m app.main run --limit 50 --verbose
```

## Tests

```bash
PYTHONPATH=src pytest -q
```

## Documentation Sections

| Section | Page | What It Covers |
|---------|------|----------------|
| [Architecture](architecture/overview.md) | `/openwiki/architecture/overview.md` | LangGraph workflow, state model, config, observability |
| [Pipeline](workflows/pipeline.md) | `/openwiki/workflows/pipeline.md` | End-to-end flow through all 5 nodes |
| [Scoring & Ranking](domain/scoring.md) | `/openwiki/domain/scoring.md` | Relevance scoring, clustering, date filtering, source weights |
| [Integrations](integrations/overview.md) | `/openwiki/integrations/overview.md` | OpenRouter, Telegram, LangSmith, LangGraphics, RSS |
| [Operations](operations/runbook.md) | `/openwiki/operations/runbook.md` | Env vars, CLI, troubleshooting, CI/CD |
| [Testing](testing/guide.md) | `/openwiki/testing/guide.md` | Test structure, coverage, how to run |

## Tech Stack

- **Language**: Python 3.11+
- **Workflow**: LangGraph
- **LLM**: OpenRouter (`openai/gpt-oss-20b`)
- **Observability**: LangSmith
- **Visualization**: LangGraphics (live graph UI at `localhost:8764`)
- **Delivery**: Telegram Bot API
- **Config**: `pydantic-settings` + `.env`

## Key Source Files

| File | Purpose |
|------|---------|
| `src/app/main.py` | CLI entrypoint (`run` subcommand) |
| `src/app/config.py` | Settings model, env loading, LangSmith env setup |
| `src/app/graph/workflow.py` | LangGraph wiring — 5 nodes, linear pipeline |
| `src/app/graph/state.py` | `AgentState` TypedDict (shared run state, `articles_selected`) |
| `src/app/schemas/article.py` | `Article`, `SourceConfig`, `FetchRules` Pydantic models |
| `src/app/services/scoring.py` | Relevance scoring, clustering, ranking |
| `src/app/services/rss_client.py` | RSS fetching, URL normalization, dedup |
| `src/app/services/extractor.py` | OpenGraph extraction, image fallback |
| `src/app/services/openrouter_client.py` | LLM summarization, sentence enforcement, batched relevance scoring |
| `src/app/services/telegram_client.py` | Telegram send with retry, caption formatting |
| `src/app/services/history.py` | Delivery history load/record, URL dedup, retention pruning |
| `data/news-sources.yaml` | RSS source definitions and fetch rules |

## Git History Summary

- **52cb4ae** — Initial commit: full pipeline (ingest → enrich → rank → summarize → deliver)
- **56ccbc2** — Relevance-first ranking refactor: keyword-based scoring, cluster dedup, source weights
- **f5384e1** — Date filter: only rank articles published today (local timezone)
