# Proposed Tech Stack

## Core Framework

- LangGraph (agent workflow orchestration)
- LangGraphics (live local visualization of graph execution)

## Language and Runtime

- Python 3.11+
- uv for environment/package management

## LLM and AI APIs

- OpenRouter API
- Model: `openai/gpt-oss-20b`

## Data Ingestion and Enrichment

- `feedparser` for RSS parsing
- `httpx` for async HTTP requests
- `beautifulsoup4` + `lxml` for OpenGraph extraction

## Ranking and Selection

- In-code deterministic scoring service (`src/app/services/scoring.py`)
- Relevance-first ranking with same-story clustering and dedup signals

## Delivery

- Telegram Bot API via async `httpx`
- Telegram HTML parse mode

## Observability

- LangSmith only

## Execution Model

- Manual CLI-triggered runs only
- No scheduler
- No Docker requirement

## Storage and State

- Source config in YAML (`data/news-sources.yaml`)
- In-memory run state (LangGraph state object)
- No external database required in current version

## Configuration and Secrets

- `pydantic-settings`
- `.env` file only

## Quality and Testing

- `pytest`
- `pytest-asyncio`
- `ruff`
- `mypy`
