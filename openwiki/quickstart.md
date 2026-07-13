# AI News Agent — Quickstart

A LangGraph-based AI news pipeline that fetches RSS feeds, enriches articles with OpenGraph metadata, ranks by relevance, summarizes via OpenRouter, and delivers digest messages to Telegram.

## What This Repo Does

1. **Ingests** RSS feeds from 20+ AI news sources (`data/news-sources.yaml`)
2. **Enriches** each article with OpenGraph metadata (title, description, image)
3. **Ranks** articles using deterministic relevance scoring + LLM re-ranking, with same-story clustering, delivery-history dedup, and a per-source cap to prevent any single feed from flooding the digest
4. **Summarizes** top articles with exactly 3-sentence LLM summaries (OpenRouter `openai/gpt-oss-20b`)
5. **Delivers** one Telegram message per article with a clickable title, photo, and summary; records deliveries to `data/delivery-history.json` for future dedup

## Quick Setup

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync
cp .env.example .env  # Fill in OPENROUTER_API_KEY, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
```

## Run Commands

```bash
# Dry run (no Telegram sends)
uv run ai-news-agent run --dry-run

# Real run (sends to Telegram)
uv run ai-news-agent run

# Limit and verbose
uv run ai-news-agent run --limit 50 --verbose
```

## Tests

```bash
uv run pytest -q
```

## Documentation Sections

| Section | Page | What It Covers |
|---------|------|----------------|
| [Architecture](architecture/overview.md) | `/openwiki/architecture/overview.md` | LangGraph workflow, state model, config, observability |
| [Pipeline](workflows/pipeline.md) | `/openwiki/workflows/pipeline.md` | End-to-end flow through all 5 nodes |
| [Scoring & Ranking](domain/scoring.md) | `/openwiki/domain/scoring.md` | Relevance scoring, clustering, date filtering, source weights |
| [Integrations](integrations/overview.md) | `/openwiki/integrations/overview.md` | OpenRouter, Telegram, LangSmith, RSS |
| [Operations](operations/runbook.md) | `/openwiki/operations/runbook.md` | Env vars, CLI, troubleshooting, CI/CD |
| [Testing](testing/guide.md) | `/openwiki/testing/guide.md` | Test structure, coverage, how to run |

## Tech Stack

- **Language**: Python 3.11+
- **Workflow**: LangGraph
- **LLM**: OpenRouter (`openai/gpt-oss-20b`)
- **Observability**: LangSmith
- **Delivery**: Telegram Bot API
- **Config**: `pydantic-settings` + `.env`
- **Dependency management**: [uv](https://docs.astral.sh/uv/) with `uv.lock`

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
- **33199b5** — Delivery history dedup, LLM re-rank, early-exit routing
- **bee0984** — Public repo prep: license, CI, gitignore, docs
- **5f01487** — Switch to uv for dependency management and lock file
- **784cf0d** — Fix arXiv flooding: per-source cap, frontier keywords, refocus sources on AI news
