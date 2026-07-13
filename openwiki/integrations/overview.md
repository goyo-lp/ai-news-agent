# Integrations

The AI News Agent integrates with five external services: OpenRouter (LLM), Telegram (delivery), LangSmith (observability), LangGraphics (visualization), and RSS feeds (data source).

## OpenRouter (LLM Summarization)

**Source**: `src/app/services/openrouter_client.py`

- **API**: `POST {OPENROUTER_BASE_URL}/chat/completions` (OpenAI-compatible)
- **Model**: `openai/gpt-oss-20b` (configurable via `OPENROUTER_MODEL`)
- **Auth**: Bearer token via `OPENROUTER_API_KEY`
- **Headers**: `HTTP-Referer` (site URL) and `X-Title` (app name) for OpenRouter attribution
- **Parameters**: `temperature=0.2`, `max_tokens=220`
- **Prompt**: System message enforces 3-sentence format; user message provides title, source, date, URL, and context
- **Retry**: If first response has <3 sentences, sends a follow-up message asking to "Rewrite your answer as exactly 3 sentences"
- **Fallback**: No API key or dry-run → template summary without LLM call. API errors → fallback summary.
- **Concurrency**: Semaphore capped at `min(HTTP_CONCURRENCY, 4)`
- **Header builder**: `_build_headers()` is a shared helper used by both `summarize_articles()` and `score_articles_relevance()`

## OpenRouter (LLM Re-Rank)

**Source**: `src/app/services/openrouter_client.py`

In addition to summarization, the OpenRouter client provides `score_articles_relevance()` — a single batched call that asks the LLM to rate each candidate article 0–100 for relevance to a digest focused on product launches, model releases, startup funding, enterprise adoption, and major deals.

- **Endpoint**: Same `POST {OPENROUTER_BASE_URL}/chat/completions`
- **Model**: `openai/gpt-oss-20b` (same as summarization)
- **Parameters**: `temperature=0.0`, `max_tokens=100 + 10 × num_articles`
- **Prompt**: Lists each article's title and truncated context (≤200 chars), asks for a JSON object mapping item numbers to scores
- **Response parsing**: `parse_relevance_scores(text, valid_keys)` extracts a JSON object from the LLM response, normalizes 0–100 scores to 0.0–1.0, and drops unknown/malformed keys
- **Blend weight**: LLM scores are blended at 30% with deterministic scores in the rank node
- **Skipped**: Returns `{}` when dry-run, no API key, no articles, or on any API failure — callers treat this as "keep deterministic ranking"

## Telegram (Delivery)

**Source**: `src/app/services/telegram_client.py`

- **API**: `https://api.telegram.org/bot{TOKEN}/{method}`
- **Methods**: `sendPhoto` (with image), `sendMessage` (text-only fallback)
- **Parse mode**: HTML (configurable via `TELEGRAM_PARSE_MODE`)
- **Caption limit**: 1024 chars (photo caption)
- **Text limit**: 4096 chars (text message)
- **HTML escaping**: All user content (URL, title, summary) is escaped with `html.escape()`
- **Caption format**: `<a href="URL">TITLE</a>\n\nSUMMARY`
- **Retry logic**: 3 attempts with linear backoff (1s, 2s, 3s). HTTP 429 respects `retry_after` from Telegram response.
- **Photo fallback**: If `sendPhoto` fails, automatically retries with `sendMessage` (text-only)
- **Credentials**: `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` required for non-dry-run

## LangSmith (Observability)

**Sources**: `src/app/services/tracing.py`, `src/app/config.py`

- **Decorator**: `@traceable(name="<node_name>")` wraps all 5 node functions
- **Env setup**: `configure_langsmith_env()` sets `LANGSMITH_API_KEY`, `LANGSMITH_PROJECT`, `LANGSMITH_TRACING` before workflow execution
- **Graceful degradation**: If LangSmith is not installed, `tracing.py` provides a no-op `traceable` decorator
- **Project**: Defaults to `ai-news-agent`
- **Tracing**: Enabled by default (`LANGSMITH_TRACING=true`)

## LangGraphics (Live Visualization)

**Sources**: `src/app/graph/workflow.py`, `src/app/services/langgraphics_assets.py`

- **Wrapping**: `langgraphics.watch(compiled_graph)` wraps the compiled graph and starts a live UI
- **HTTP UI**: `http://localhost:{LANGGRAPHICS_PORT}` (default 8764)
- **WebSocket**: `ws://localhost:{LANGGRAPHICS_WS_PORT}` (default 8765)
- **Asset management**: `ensure_langgraphics_static_assets()` copies vendored web assets from `src/app/assets/langgraphics_static/` into the installed `langgraphics/static/` directory (only if not already present)
- **Config flags**: `LANGGRAPHICS_ENABLED`, `LANGGRAPHICS_OPEN_BROWSER`, `LANGGRAPHICS_HOST`, `LANGGRAPHICS_PORT`, `LANGGRAPHICS_WS_PORT`, `LANGGRAPHICS_DIRECTION` (TB), `LANGGRAPHICS_MODE` (auto), `LANGGRAPHICS_INSPECT` (off), `LANGGRAPHICS_THEME` (system)
- **Disabled mode**: When `LANGGRAPHICS_ENABLED=false`, `build_workflow()` returns the raw compiled graph without wrapping

## RSS Sources

**Source**: `data/news-sources.yaml`, `src/app/services/rss_client.py`

### Source Configuration
The YAML file has two top-level keys:
- `fetch_defaults`: Default `FetchRules` for all sources (image fallback, user-agent requirement, blocked domains)
- `sources`: List of source configs, each with `name`, `url`, `rss`, and optional `fetch_overrides`

### Current Sources (20+)
Major outlets include: TLDR AI, TechCrunch (AI), The Verge (AI), Wired (AI), MIT Technology Review, VentureBeat (AI), ZDNet (AI), The Guardian (AI), OpenAI Blog, Google DeepMind Blog, BAIR Blog, NVIDIA Blog, AWS ML Blog, Apple ML Journal, AI News, AI Business, AI Magazine, KDNuggets, Towards Data Science, Machine Learning Mastery, Unite.AI, MarkTechPost, Analytics Vidhya, InfoQ (AI/ML).

### Fetch Behavior
- Uses `feedparser` to parse RSS/Atom feeds
- Respects `MAX_FEED_ITEMS_PER_SOURCE` (default 50 items per feed)
- HTTP concurrency bounded by `HTTP_CONCURRENCY` semaphore (default 8)
- Custom `User-Agent` header (`AINewsAgent/0.1`)
- `REQUEST_TIMEOUT_SECONDS` (default 20s) per request
