# Architecture Overview

## LangGraph Workflow

The pipeline is a linear 5-node graph compiled from `StateGraph(AgentState)`:

```
ingest → enrich → rank → [articles_selected?]
                        ├─ yes → summarize → deliver → END
                        └─ no  → END
```

**Source**: `src/app/graph/workflow.py`

The graph is built by `build_workflow()` which:
1. Creates a `StateGraph` with `AgentState` as the state type
2. Registers each node function
3. Wires a linear chain up to `rank`, then a **conditional edge** after `rank`
4. Compiles the graph
5. Optionally wraps it with LangGraphics `watch()` for live visualization

```python
graph.set_entry_point("ingest")
graph.add_edge("ingest", "enrich")
graph.add_edge("enrich", "rank")
graph.add_conditional_edges("rank", route_after_rank, {"summarize": "summarize", END: END})
graph.add_edge("summarize", "deliver")
graph.add_edge("deliver", END)
```

`route_after_rank(state)` returns `"summarize"` when `articles_selected` is non-empty, otherwise skips directly to `END`. This prevents unnecessary LLM and Telegram calls when the date/history filters remove all articles.

When `LANGGRAPHICS_ENABLED=true` (default), the compiled graph is wrapped with `langgraphics.watch()` which starts an HTTP UI at `localhost:8764` and a WebSocket stream at `localhost:8765`. Before starting, `ensure_langgraphics_static_assets()` copies vendored web assets from `src/app/assets/langgraphics_static/` into the installed `langgraphics/static/` path.

## Agent State

**Source**: `src/app/graph/state.py`

`AgentState` is a `TypedDict(total=False)` — all fields are optional. State passes through nodes as a dict; each node receives the full state and returns a new dict with updates.

| Field | Type | Set By | Purpose |
|-------|------|--------|---------|
| `run_id` | `str` | main.py | UUID per run |
| `started_at` | `str` | main.py | ISO timestamp |
| `dry_run` | `bool` | main.py | Skip Telegram sends |
| `limit` | `int` | main.py | Max articles to select (≤50) |
| `fetch_defaults` | `dict` | ingest | Default FetchRules |
| `sources` | `list[dict]` | ingest | Parsed source configs |
| `articles_raw` | `list[dict]` | ingest | Deduplicated RSS articles |
| `articles_enriched` | `list[dict]` | enrich | Articles with OpenGraph data |
| `articles_selected` | `list[dict]` | rank/summarize | Ranked, scored, and selected articles (written by rank, read by summarize and deliver) |
| `delivery_results` | `list[dict]` | deliver | Per-article send status |
| `errors` | `list[str]` | all | Non-fatal error accumulation |

## Configuration

**Source**: `src/app/config.py`

Settings are loaded via `pydantic-settings` `BaseSettings` from `.env`. The `get_settings()` function is memoized with `@lru_cache(maxsize=1)`.

### Key Settings Groups

| Group | Variables | Defaults |
|-------|-----------|----------|
| OpenRouter | `OPENROUTER_API_KEY`, `OPENROUTER_MODEL`, `OPENROUTER_BASE_URL`, `OPENROUTER_SITE_URL`, `OPENROUTER_APP_NAME` | model=`openai/gpt-oss-20b` |
| Telegram | `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`, `TELEGRAM_PARSE_MODE` | parse_mode=`HTML` |
| LangSmith | `LANGSMITH_API_KEY`, `LANGSMITH_PROJECT`, `LANGSMITH_TRACING` | project=`ai-news-agent`, tracing=true |
| LangGraphics | `LANGGRAPHICS_ENABLED`, `LANGGRAPHICS_OPEN_BROWSER`, `LANGGRAPHICS_HOST/PORT/WS_PORT/DIRECTION/MODE/INSPECT/THEME` | port=8764, ws_port=8765 |
| Runtime | `SOURCES_FILE`, `HISTORY_FILE`, `HISTORY_RETENTION_DAYS`, `REQUEST_TIMEOUT_SECONDS`, `HTTP_CONCURRENCY`, `MAX_FEED_ITEMS_PER_SOURCE`, `MAX_ARTICLES_PER_RUN`, `USER_AGENT` | concurrency=8, max_articles=50, history_file=`data/delivery-history.json`, retention=14d |

### Required Fields Validation

`missing_required_runtime_fields(dry_run)` only enforces `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` when not in dry-run mode. OpenRouter API key is optional (falls back to template summaries).

### LangSmith Environment

`configure_langsmith_env()` sets `LANGSMITH_API_KEY`, `LANGSMITH_PROJECT`, and `LANGSMITH_TRACING` as OS environment variables before the workflow runs.

## Observability

**Source**: `src/app/services/tracing.py`

All node functions are decorated with `@traceable(name="<node_name>")` from LangSmith. If LangSmith is not installed, the decorator is a no-op passthrough, so the pipeline runs without observability when needed.

## Entrypoint

**Source**: `src/app/main.py`

The CLI uses `argparse` with a single `run` subcommand:

```bash
python -m app.main run [--dry-run] [--limit N] [--verbose]
```

The `run_pipeline()` function:
1. Loads settings and configures LangSmith env
2. Validates required fields (skips Telegram check in dry-run)
3. Clamps `limit` to `[1, MAX_ARTICLES_PER_RUN]`
4. Creates `AgentState` with `run_id`, `started_at`, `dry_run`, `limit`, empty `errors`
5. Builds the workflow and invokes it asynchronously
6. Reports summary: selected count, attempted, sent, failed, dry_run
7. Returns exit code 0 (success) or 1 (delivery failures)
