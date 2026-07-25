# AI News Agent

LangGraph-based AI news pipeline that pulls RSS sources, enriches articles with OpenGraph metadata, deduplicates and clusters similar stories, summarizes with OpenRouter (`deepseek/deepseek-v4-flash`), and sends Telegram digest messages.

## What This App Does

- Ingests RSS feeds from `data/news-sources.yaml`
- Normalizes URLs and removes tracking params
- Enriches each article with OpenGraph fields (`og:title`, `og:description`, `og:image`)
- Applies image fallback rules from source config
- Deduplicates exact URL duplicates
- Filters to articles published today (local timezone) before ranking
- Skips stories already delivered in a previous run (tracked in `data/delivery-history.json`)
- Clusters cross-source same-story coverage and keeps one representative
- Ranks candidates deterministically, then optionally re-ranks the top pool with one batched OpenRouter call
- Caps each source at `MAX_ARTICLES_PER_SOURCE` (default 3) so no single feed floods the digest
- Selects up to 50 stories per run (default 20 as shipped in `.env.example`; the cap is `MAX_ARTICLES_PER_RUN=50`). Configure via the `MAX_ARTICLES_PER_RUN` env var or `--limit`.
- Generates exactly 3-sentence summaries
- Sends one Telegram message per selected story
- Records successfully delivered stories to history so they aren't repeated in later runs

Hardening built into the fetch/LLM/delivery paths (see [Security](#security)):
- Blocks outbound fetches to private/loopback/link-local/cloud-metadata addresses
- Enforces byte caps on RSS and article-page downloads against actual bytes received, not just the advisory `Content-Length` header
- Frames scraped article content as untrusted data in LLM prompts (prompt-injection mitigation)
- Redacts the Telegram bot token from any error message before logging

## Ranking Logic (Relevance-First)

Ranking now prioritizes relevance to high-signal AI business and product developments.

Primary relevance signals:
- New tech/model/features/product launches
- Frontier company mentions (OpenAI, Anthropic, Gemini/DeepMind, xAI, Cursor, Mistral, DeepSeek, open-source models)
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
- Novelty (dissimilarity to titles delivered in the last `HISTORY_RETENTION_DAYS` days)

The top-ranked pool (up to 40 candidates) is optionally re-scored by a single batched OpenRouter
call and blended 70/30 with the deterministic score. This step is skipped automatically on
`--dry-run`, when `OPENROUTER_API_KEY` is unset, or if the call fails — the deterministic ranking
always stands on its own.

## Delivery Format (Per Article)

Each Telegram message is:
1. Clickable title linked to the source article
2. Photo (`og:image` or configured fallback)
3. 3-sentence summary

## Tech Stack

- Language: Python 3.11+
- Agent framework: LangGraph
- LLM: OpenRouter (`deepseek/deepseek-v4-flash`)
- Observability: LangSmith

## Security

- **SSRF protection**: article-page and RSS fetches run through an `httpx` request hook (`services/http_utils.py`) that blocks requests to private/loopback/link-local/reserved IPs and known cloud-metadata hostnames, on every request including redirect hops. This is a literal-address check, not DNS resolution — a domain that *resolves* to an internal address (DNS rebinding) isn't caught.
- **Download size caps**: RSS and article-page fetches stream the response body and abort once actual bytes received exceed the cap, rather than trusting the (spoofable/absent) `Content-Length` header after the whole body is already buffered.
- **Prompt-injection framing**: scraped article title/description text is explicitly framed as untrusted data in both the summarization and relevance-scoring prompts sent to OpenRouter, since article content can be attacker-influenced (e.g. a Hacker News or Reddit submission). This mitigates, not eliminates, injection risk.
- **URL scheme allowlist**: only `http`/`https` article URLs become clickable Telegram links; anything else (e.g. `javascript:`) is replaced with `#`.
- **Secret redaction**: the Telegram bot token (embedded in the request URL per Telegram's API design) is redacted from any exception message before it's logged.
- Dependencies have been checked with `pip-audit` against the PyPI/OSV advisory database (no known vulnerabilities as of the last manual check) — this isn't yet wired into CI, so re-run it periodically: `uvx pip-audit --requirement <(uv export --no-hashes)`.

## Research Evidence (no paid dependencies)

The orchestrator's research tools use only keyless/self-hosted evidence sources — no Tavily, no API keys, LLM cost aside:

- **`web_extract`** — fetches article text locally through the SSRF-guarded, size-capped HTTP path and extracts it with [trafilatura](https://trafilatura.readthedocs.io/). Always available; no service required.
- **`web_search`** — queries a self-hosted [SearXNG](https://docs.searxng.org/) instance (keyless). Start one with:
  ```bash
  docker compose -f docker-compose.searxng.yml up -d
  ```
  then set `SEARXNG_BASE_URL=http://localhost:8080` in `.env`. The JSON API the tool uses is enabled in `searxng/settings.yml` (SearXNG ships with JSON off by default). Leave `SEARXNG_BASE_URL` empty to run without an instance — `web_search` then returns deterministic dry-run mock results.

Independent corroboration leans first on the RSS pipeline's clustered `supporting_urls` (sources it already found), with SearXNG as the fallback when the cluster is thin.

## Project Structure

```text
src/app/
├── graph/                   # LangGraph workflow/state
├── nodes/                   # ingest/enrich/rank/summarize/deliver nodes
├── services/                # RSS, extraction, ranking, OpenRouter, Telegram
│   ├── http_utils.py        # shared SSRF guard + size-capped GET
│   ├── middleware.py        # reasoning-effort injection + think-block strip
│   └── scoring_keywords.py  # keyword/phrase sets for scoring (pure data)
├── schemas/                 # Pydantic models
├── config.py                # environment settings
├── utils.py                 # small shared helpers (e.g. reference-time)
└── main.py                  # CLI entrypoint

data/
├── news-sources.yaml
└── delivery-history.json    # generated at runtime, gitignored

tests/
└── unit tests

reference/linkedin-agent/   # port kit only (not a second app); see PORT_MAP.md
architecture/               # local diagrams only (gitignored)
```

## Setup

Requires [uv](https://docs.astral.sh/uv/).

1. Install dependencies (creates `.venv` and installs from `uv.lock`):
```bash
uv sync
```

2. Create `.env` from `.env.example` and fill values.

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
uv run ai-news-agent run --dry-run
```

Real run:
```bash
uv run ai-news-agent run
```

Useful flags:
- `--limit 50`
- `--verbose`

## Propose (LinkedIn post proposals)

Drive the coordinator deep agent to turn today's ranked topics into LinkedIn
post proposals, then write a per-run export bundle:

```bash
uv run ai-news-agent propose
```

The coordinator planner and its research/writer subagents call OpenRouter, so
`OPENROUTER_API_KEY` is required (the command exits with a config error
otherwise). The run is traced (LangSmith, if configured) and its OpenRouter
usage/cost is logged at the end.

Each proposal that passes the quality gate is then **sent to the LinkedIn
Telegram bot** (`TELEGRAM_LINKEDIN_BOT_TOKEN` + `TELEGRAM_LINKEDIN_CHAT_ID`);
if those are unset the send is skipped (dry-run). The rendered `posts.md` and a
`run_report.json` also land under `<OUTPUTS_DIR>/<YYYY-MM-DD>/`. Flags:

- `--dry-run` — produce + export proposals but do **not** send them to Telegram.
- `--force` — overwrite today's existing export bundle instead of refusing.

Two delivery lanes, two commands: `run` sends the daily **news digest** to the
news bot; `propose` sends **LinkedIn post proposals** to the linkedin bot. The
news digest `run` path is unchanged.

## Troubleshooting

`Configuration error: missing required .env values: TELEGRAM_CHAT_ID`
- Set `TELEGRAM_CHAT_ID` to a real numeric chat id.

`Delivery complete: 0 sent, N failed`
- Most common cause is missing/invalid Telegram credentials.
- Check bot token and chat id first.

`Source fetch failed ... 403/429`
- Some feeds block bots or rate-limit aggressively.
- This is non-fatal; the run continues with remaining sources.

`Ranking complete: selected 0 items` / run ends right after ranking
- Either nothing was published today, or every fresh story was already delivered in a prior run.
- Delete or edit `data/delivery-history.json` to reset the dedup window if needed.

## Development

Run tests:
```bash
uv run pytest -q
```

Type/lint checks:
```bash
uv run mypy src
uv run ruff check .
```

Add a dependency:
```bash
uv add <package>          # runtime dependency
uv add --dev <package>    # dev-only dependency
```
