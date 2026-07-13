# Operations Runbook

## Environment Setup

### Prerequisites
- Python 3.11+
- A Telegram bot token and chat ID (for real runs)
- An OpenRouter API key (for real summaries)

### Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Environment Variables

Copy `.env.example` to `.env` and fill in values:

| Variable | Required | Default | Purpose |
|----------|----------|---------|---------|
| `OPENROUTER_API_KEY` | For LLM summaries | — | OpenRouter API auth |
| `OPENROUTER_MODEL` | No | `openai/gpt-oss-20b` | LLM model |
| `OPENROUTER_BASE_URL` | No | `https://openrouter.ai/api/v1` | API endpoint |
| `OPENROUTER_SITE_URL` | No | — | HTTP-Referer header |
| `OPENROUTER_APP_NAME` | No | `AI News Agent` | X-Title header |
| `TELEGRAM_BOT_TOKEN` | For real runs | — | Bot auth token |
| `TELEGRAM_CHAT_ID` | For real runs | — | Target chat ID |
| `TELEGRAM_PARSE_MODE` | No | `HTML` | Telegram parse mode |
| `LANGSMITH_API_KEY` | Recommended | — | Observability |
| `LANGSMITH_PROJECT` | No | `ai-news-agent` | Trace project |
| `LANGSMITH_TRACING` | No | `true` | Enable tracing |
| `LANGGRAPHICS_ENABLED` | No | `true` | Live graph UI |
| `LANGGRAPHICS_OPEN_BROWSER` | No | `true` | Auto-open browser |
| `SOURCES_FILE` | No | `data/news-sources.yaml` | RSS source config |
| `HISTORY_FILE` | No | `data/delivery-history.json` | Delivery history for dedup |
| `HISTORY_RETENTION_DAYS` | No | `14` | How long delivered URLs are remembered |
| `REQUEST_TIMEOUT_SECONDS` | No | `20` | HTTP timeout |
| `HTTP_CONCURRENCY` | No | `8` | Parallel HTTP requests |
| `MAX_FEED_ITEMS_PER_SOURCE` | No | `50` | RSS items per feed |
| `MAX_ARTICLES_PER_RUN` | No | `50` | Max articles delivered |
| `USER_AGENT` | No | `AINewsAgent/0.1` | HTTP User-Agent |

### Getting Telegram Chat ID

1. Send any message to your bot from the target chat
2. Run:
   ```bash
   curl "https://api.telegram.org/bot$TELEGRAM_BOT_TOKEN/getUpdates"
   ```
3. Copy `chat.id` into `TELEGRAM_CHAT_ID`

## CLI Commands

All commands run from the project root.

```bash
# Dry run — no Telegram sends, no LLM calls (template summaries)
PYTHONPATH=src python -m app.main run --dry-run

# Real run — sends Telegram messages with LLM summaries
PYTHONPATH=src python -m app.main run

# Limit output
PYTHONPATH=src python -m app.main run --limit 25

# Verbose/debug logging
PYTHONPATH=src python -m app.main run --verbose

# Equivalent one-liner
source .venv/bin/activate && PYTHONPATH=src python -m app.main run
```

### Exit Codes
- `0` — Success (all deliveries sent or dry-run)
- `1` — One or more delivery failures
- `2` — Configuration error (missing required env vars)

## LangGraphics

LangGraphics starts automatically when `LANGGRAPHICS_ENABLED=true`:
- HTTP UI: `http://localhost:8764`
- WS stream: `ws://localhost:8765`

To disable: set `LANGGRAPHICS_ENABLED=false`.

## Troubleshooting

### `Configuration error: missing required .env values: TELEGRAM_CHAT_ID`
Set `TELEGRAM_CHAT_ID` to a real numeric chat ID. Use `--dry-run` to skip this check.

### `Delivery complete: 0 sent, N failed`
Most common cause is missing or invalid Telegram credentials. Check:
1. `TELEGRAM_BOT_TOKEN` is valid
2. `TELEGRAM_CHAT_ID` is a numeric string
3. The bot is a member of the target chat

### `Source fetch failed ... 403/429`
Some RSS feeds block bots or rate-limit aggressively. This is non-fatal — the run continues with remaining sources. If persistent, check:
1. `USER_AGENT` is set (some feeds require a real user-agent)
2. The feed URL is correct and accessible from your network
3. Reduce `HTTP_CONCURRENCY` if rate-limited

### `Run complete. selected=0 ...`
Likely causes:
1. No articles published today (date filter excludes older articles)
2. RSS feeds returned no items
3. `published_at` parsing failed for all entries

### LangGraphics not appearing
1. Check `LANGGRAPHICS_ENABLED=true`
2. Verify `langgraphics` is installed (`pip show langgraphics`)
3. Check that port 8764 is not already in use
4. Set `LANGGRAPHICS_OPEN_BROWSER=false` and open `http://localhost:8764` manually

## CI/CD

**Source**: `.github/workflows/openwiki-update.yml`

A GitHub Actions workflow runs daily at 08:00 UTC (`cron: "0 8 * * *"`) or on manual dispatch. It:
1. Installs OpenWiki (`npm install --global openwiki`)
2. Runs `openwiki code --update --print` to regenerate documentation
3. Creates a pull request with the updated `openwiki/` directory

This workflow refreshes the OpenWiki documentation pages. It does not run the news pipeline itself.
