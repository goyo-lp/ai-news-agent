# AI News Agent

LangGraph-based AI news pipeline that pulls RSS sources, enriches articles with OpenGraph metadata, deduplicates and clusters similar stories, summarizes with OpenRouter (`openai/gpt-oss-20b`), and sends Telegram digest messages.

## What This App Does

- Ingests RSS feeds from `data/news-sources.yaml`
- Normalizes URLs and removes tracking params
- Enriches each article with OpenGraph fields (`og:title`, `og:description`, `og:image`)
- Applies image fallback rules from source config
- Deduplicates exact URL duplicates
- Clusters cross-source same-story coverage and keeps one representative
- Ranks and selects up to 50 stories per run (or fewer if less are available)
- Generates exactly 3-sentence summaries
- Sends one Telegram message per selected story

## Ranking Logic (Relevance-First)

Ranking now prioritizes relevance to high-signal AI business and product developments.

Primary relevance signals:
- New tech/model/features/product launches
- New startups and funding activity
- Technical developments and breakthroughs
- Enterprise adoption/deployments
- AI deals, partnerships, and acquisitions

Lower-priority signals (demoted unless also high-signal):
- Event roundups
- Webinar/podcast recap content
- Generic newsletter-style coverage

Final ranking score combines:
- Relevance score (highest weight)
- Recency
- Source quality weight
- Duplicate/coverage signal
- Cluster support signal
- Title novelty

## Delivery Format (Per Article)

Each Telegram message is:
1. Clickable title linked to the source article
2. Photo (`og:image` or configured fallback)
3. 3-sentence summary

## Tech Stack

- Language: Python 3.11+
- Agent framework: LangGraph
- LLM: OpenRouter (`openai/gpt-oss-20b`)
- Observability: LangSmith
- Live graph visualization: LangGraphics

## Project Structure

```text
src/app/
  graph/        # LangGraph workflow/state
  nodes/        # ingest/enrich/rank/summarize/deliver nodes
  services/     # RSS, extraction, ranking, OpenRouter, Telegram, langgraphics assets
  schemas/      # Pydantic models
  config.py     # environment settings
  main.py       # CLI entrypoint

data/
  news-sources.yaml

tests/
  unit tests
```

## Setup

1. Create and activate virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create `.env` from `.env.example` and fill values.

## Required `.env` Values

Required for real runs (`--dry-run` off):
- `OPENROUTER_API_KEY`
- `TELEGRAM_BOT_TOKEN`
- `TELEGRAM_CHAT_ID`

Recommended:
- `LANGSMITH_API_KEY`
- `LANGSMITH_PROJECT`
- `LANGSMITH_TRACING=true`

Other runtime controls are listed in `.env.example`.

## Get Telegram Chat ID

1. Send any message to your bot from the target chat.
2. Run:
```bash
curl "https://api.telegram.org/bot$TELEGRAM_BOT_TOKEN/getUpdates"
```
3. Copy `chat.id` into `TELEGRAM_CHAT_ID`.

## Run

Dry run (no Telegram sends):
```bash
PYTHONPATH=src python -m app.main run --dry-run
```

Real run:
```bash
PYTHONPATH=src python -m app.main run
```

Useful flags:
- `--limit 50`
- `--verbose`

Equivalent one-liner:
```bash
source .venv/bin/activate && PYTHONPATH=src python -m app.main run
```

## Live Visualization

LangGraphics is automatically wired into the run path.

When you run the app:
- HTTP UI: `http://localhost:8764`
- WS stream: `ws://localhost:8765`

The repo includes built LangGraphics web assets and syncs them into the installed `langgraphics/static` path before `watch(...)` starts.

Config flags:
- `LANGGRAPHICS_ENABLED=true|false`
- `LANGGRAPHICS_OPEN_BROWSER=true|false`
- `LANGGRAPHICS_HOST`, `LANGGRAPHICS_PORT`, `LANGGRAPHICS_WS_PORT`

## Troubleshooting

`Configuration error: missing required .env values: TELEGRAM_CHAT_ID`
- Set `TELEGRAM_CHAT_ID` to a real numeric chat id.

`Delivery complete: 0 sent, N failed`
- Most common cause is missing/invalid Telegram credentials.
- Check bot token and chat id first.

`Source fetch failed ... 403/429`
- Some feeds block bots or rate-limit aggressively.
- This is non-fatal; the run continues with remaining sources.

## Development

Run tests:
```bash
PYTHONPATH=src pytest -q
```

Type/lint checks:
```bash
PYTHONPATH=src mypy src
ruff check .
```
