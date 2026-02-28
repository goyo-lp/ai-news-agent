# Proposed Tech Stack

## Core Framework

- LangGraph
- LangChain (supporting utilities)
- LangGraphics (live local visualization for LangGraph runs)

## Language and Runtime

- Chosen language: Python
- Python 3.11+
- uv (package/env management)
- TypeScript is not the primary implementation language for this build

## LLM and AI APIs

- OpenRouter API
- Model: `openai/gpt-oss-20b`

## Data Ingestion and Parsing

- feedparser (RSS parsing)
- httpx (HTTP client, async)
- trafilatura or readability-lxml (article text extraction fallback)
- beautifulsoup4 + lxml (HTML/OpenGraph parsing)

## Scheduling and Workflow Execution

- Manual CLI-triggered runs only (no scheduler)

## Storage

- PostgreSQL (primary store: articles, runs, delivery logs)
- Redis (cache, dedupe keys, short-lived state)

## Telegram Delivery

- Telegram Bot API
- python-telegram-bot or direct HTTP via httpx
- Telegram HTML parse mode for message formatting

## Observability

- LangSmith only

## Quality and Testing

- pytest
- pytest-asyncio
- ruff (lint)
- mypy (type checks)

## Deployment

- Local/manual execution via CLI command
- No Docker
- No scheduler
- No deployment orchestration layer

## Configuration and Secrets

- pydantic-settings
- `.env` only
