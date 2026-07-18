# ai-linked-imposting-agent

LangGraph-based agent for technical AI-news curation and reflective LinkedIn writing.

Pipeline now runs as a parallelized graph:
1. Seed from curated RSS sources (copied/adapted from your `AI news agent`) and save top 50 daily candidates.
2. Re-rank for technical implementation depth.
3. Deep research and then fan out into:
   - adaptive Deep Agent investigation
   - deterministic verifier branch
   - style-profile branch
4. Merge verification branches, generate posts, quality-gate posts.
5. Fan out delivery and artifact export, then join to finalize run report.

## What This App Does

- Ingests RSS feeds from `data/news-sources.yaml`
- Deduplicates and filters to items published today (local timezone)
- Selects top 50 seed articles and saves them to output JSON
- Re-ranks to top 5 topics emphasizing technical depth (not hype)
- Uses OpenRouter `openai/gpt-oss-120b` by default for Stage A technical ranking
- Keeps funding/deal items only when paired with technical implementation detail
- Expands top topics with Tavily evidence retrieval
- Runs a Deep Agents adaptive investigation layer with three skills:
  - technical novelty extraction
  - claim verification
  - reflective anti-hype rewriting
- Verifies research briefs with model-based fact-checking + Tavily evidence
- Merges adaptive and deterministic verification paths conservatively
- Uses OpenRouter `anthropic/claude-haiku-4.5` by default for Stage B (deep research synthesis, verification, deep-agent layer, and post generation)
- Builds a writing-style profile from `style_samples/`
- Generates 5 LinkedIn posts (one topic per post) in a reflective/non-sales tone
- Keeps post language simple for a general business audience (minimal jargon)
- Targets post length between 105 and 182 words
- Sends each post as an independent Telegram message
- Formats Telegram post bodies into readable paragraphs with blank lines
- Logs end-of-run API usage/cost summary (OpenRouter + Tavily call stats)
- Exports full run artifacts

## Telegram Delivery Contract

Each suggested post is sent as its own Telegram message with:
- A catchy title/question/intro phrase (`headline`)
- Full post text (`body`) formatted into readable paragraphs
- Bottom section: recommended hashtags

## LangGraphics Visualization

LangGraphics is enabled by default during `run`.

When running:
- HTTP UI: `http://localhost:8764`
- WS stream: `ws://localhost:8765`

Disable with `--no-graphics`.

## LangSmith Observability

All nodes are trace-wrapped and runtime enables LangSmith from env.

Set in `.env`:
- `LANGSMITH_API_KEY`
- `LANGSMITH_PROJECT=ai-linked-imposting-agent`
- `LANGSMITH_TRACING=true`

## Project Structure

```text
src/app/
  graph/        # LangGraph workflow/state
  nodes/        # discover -> normalize -> rank -> research -> (adaptive + verify + style in parallel) -> merge -> posts -> quality -> (deliver + export artifacts) -> report
  services/     # RSS seed ingestion, Tavily, ranking, Deep Agent investigator, verifier, style, generation, Telegram, export
  schemas/      # Pydantic models
  config.py     # environment settings
  main.py       # CLI entrypoint

data/
  news-sources.yaml
  trusted-sources.yaml

style_samples/
  # your writing examples (.txt, .md, .jsonl)

outputs/
  YYYY-MM-DD/
    top_50_articles.json
    technical_candidates.json
    adaptive_briefs.json
    research_briefs.json
    linkedin_posts.md
    run_report.json
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

Required for non-dry runs:
- `TAVILY_API_KEY`
- `OPENROUTER_API_KEY`
- `TELEGRAM_BOT_TOKEN`
- `TELEGRAM_CHAT_ID`

Optional:
- `OPENROUTER_STAGE_A_MODEL` (default: `openai/gpt-oss-120b`, used in Stage A technical ranking)
- `OPENROUTER_STAGE_A_SECONDARY_MODEL` (optional ensemble model for Stage A ranking)
- `OPENROUTER_MODEL` (default: `anthropic/claude-haiku-4.5`)
- `OPENROUTER_POST_SECONDARY_MODEL` (optional second model for post generation)
- `OPENROUTER_VERIFIER_SECONDARY_MODEL` (optional second model for deterministic verification)
- `DEEP_AGENT_ENABLED` (default: `true`)
- `DEEP_AGENT_MODEL` (default: `anthropic/claude-haiku-4.5`)
- `DEEP_AGENT_TIMEOUT_SECONDS` (default: `75`)
- `DEEP_AGENT_MAX_EVIDENCE_SOURCES` (default: `5`)
- `DEEP_AGENT_MAX_EVIDENCE_CHARS` (default: `1400`)
- `DEEP_RESEARCH_TOPIC_CONCURRENCY` (default: `4`)
- `ADAPTIVE_INVESTIGATION_CONCURRENCY` (default: `3`)
- `VERIFICATION_CONCURRENCY` (default: `4`)
- `TELEGRAM_SEND_CONCURRENCY` (default: `3`)
- `PIPELINE_BATCH_CONCURRENCY` (default: `2`)
- `LANGSMITH_API_KEY`

## Run

Dry run (no external API calls):

```bash
PYTHONPATH=src python -m app.main run --dry-run
```

Real run:

```bash
PYTHONPATH=src python -m app.main run
```

Batch run across multiple dates (parallelized jobs):

```bash
PYTHONPATH=src python -m app.main run-batch --dates 2026-03-01,2026-03-02 --max-concurrency 2 --no-graphics
```

Get Telegram chat id:

```bash
curl "https://api.telegram.org/bot$TELEGRAM_BOT_TOKEN/getUpdates"
```

Then set `TELEGRAM_CHAT_ID` from the response.

Useful flags:
- `--date 2026-03-02`
- `--hours-back 48`
- `--samples-dir style_samples`
- `--no-graphics`
- `--verbose`

At the end of each run, logs include API usage/cost telemetry such as:
- `openrouter_calls`
- `tavily_search_calls`
- `tavily_extract_calls`
- `openrouter_total_tokens`
- `estimated_cost_usd`
- `cost_source` (`response_usage`, `unavailable_from_provider`, or `no_openrouter_calls`)

Build style profile only:

```bash
PYTHONPATH=src python -m app.main build-style-profile --samples-dir style_samples
```

Preview latest generated posts:

```bash
PYTHONPATH=src python -m app.main preview
```

## Notes on Style Adaptation

Place writing samples in `style_samples/` as:
- `.txt`
- `.md`
- `.jsonl` (fields: `text`/`content`/`post`)

Profile is persisted at `data/style_profile.json` and reused in each run.

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
