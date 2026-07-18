# Proposed Tech Stack

- Python 3.11+
- LangGraph (orchestration)
- LangGraphics (live graph visualization)
- LangSmith (tracing)
- RSS seed ingestion (feedparser + curated source YAML)
- Tavily API (evidence expansion + extraction)
- OpenRouter (Stage A ranking default `openai/gpt-oss-120b`; downstream default `anthropic/claude-haiku-4.5`)
- Deep Agents (`deepagents` + `langchain-openai`) for adaptive technical investigation layer
- Telegram Bot API (one message per suggested post)
- Pydantic + pydantic-settings
- httpx (async HTTP)
- PyYAML
- Asyncio semaphore-based bounded concurrency in services and batch runner
- Run-scoped API usage tracker (OpenRouter/Tavily call and cost telemetry)
- pytest / ruff / mypy
