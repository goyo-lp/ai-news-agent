# Pipeline Workflow

The pipeline executes as a linear LangGraph chain up to the rank node, then branches: **ingest → enrich → rank → (articles? → summarize → deliver → END | END)**. Each node is an async function that receives `AgentState` and returns updated state.

## Node 1: Ingest

**Source**: `src/app/nodes/ingest.py`, `src/app/services/rss_client.py`

### What It Does
1. Loads RSS source definitions from `data/news-sources.yaml` (via `RSSClient.load_sources()`)
2. Fetches all RSS feeds concurrently with `asyncio.gather()` (bounded by `HTTP_CONCURRENCY` semaphore)
3. Parses each feed with `feedparser`, extracts title, URL, description, published date, and RSS-embedded images
4. Normalizes URLs (strips tracking params like `utm_*`, `fbclid`, `mc_cid`, etc.)
5. Deduplicates articles by normalized URL — keeps the newest version, increments `duplicate_count`

### Key Functions in `rss_client.py`
- `normalize_url(url)` — strips tracking params, removes fragments
- `parse_entry_datetime(entry)` — parses `published_parsed`/`updated_parsed`, falls back to RFC 2822 date strings
- `extract_entry_image(entry)` — checks `media_content`, `media_thumbnail`, `links`, and `image` fields
- `build_article_id(source, url, title)` — SHA-256 hash, truncated to 24 chars
- `dedupe_articles(articles)` — URL-keyed dedup, keeps newest by `published_at`

### Error Handling
Source fetch failures (403, 429, timeouts) are non-fatal. Errors are captured as strings and appended to `state["errors"]`. The pipeline continues with remaining sources.

## Node 2: Enrich

**Source**: `src/app/nodes/enrich.py`, `src/app/services/extractor.py`

### What It Does
1. Reconstructs `FetchRules` per source from state (merges defaults with per-source overrides)
2. For each article, fetches the article URL via `httpx` and extracts OpenGraph metadata:
   - `og:title` (falls back to `twitter:title`)
   - `og:description` (falls back to `description`, `twitter:description`)
   - `og:image` (falls back to `twitter:image`, `twitter:image:src`)
3. Applies domain blocking — if the URL's domain is in `blocked_domains`, skips HTTP fetch and uses RSS image fallback
4. Applies image fallback — if no `og:image` found and `image_fallback_rss_enclosure=true`, uses `rss_image_url`

### Key Details
- Uses `BeautifulSoup` with `lxml` parser for HTML parsing
- Follows redirects (`follow_redirects=True`) and re-normalizes the final URL
- Only processes HTML responses (`text/html` content type)
- HTTP errors fall back gracefully — article still passes through with whatever metadata is available

## Node 3: Rank

**Source**: `src/app/nodes/rank.py`, `src/app/services/scoring.py`

### What It Does
1. **Date filter**: Keeps only articles published today (local timezone). Articles with no `published_at` are excluded. This was added in commit `f5384e1`.
2. **History filter**: Loads delivery history from `data/delivery-history.json` (retention: `HISTORY_RETENTION_DAYS`, default 14) and drops articles whose URL was previously delivered. New file: `src/app/services/history.py`.
3. **Deterministic ranking**: Clusters same-story articles and scores them with `rank_articles()`, passing `recent_titles` from delivery history so the novelty component penalizes recently-delivered stories. Retrieves up to `_LLM_RERANK_POOL` (40) candidates — wider than the run `limit` to give the LLM re-rank a larger pool.
4. **LLM re-rank**: Calls `OpenRouterClient.score_articles_relevance()` — a single batched OpenRouter request that returns 0–100 relevance scores for all candidates. Scores are blended at `_LLM_BLEND_WEIGHT = 0.3` (30% LLM, 70% deterministic). If dry-run or no API key, this step is skipped and deterministic scores stand.
5. **Selection**: Cuts to the run `limit` after final sort.

### State Output
Sets `articles_selected` to the serialized list of selected articles.

### Conditional Routing
After rank, the workflow uses a conditional edge (`route_after_rank`) that sends articles to `summarize` only when `articles_selected` is non-empty. If the date/history filters remove all articles, the pipeline skips directly to `END`, avoiding unnecessary LLM and Telegram calls.

## Node 4: Summarize

**Source**: `src/app/nodes/summarize.py`, `src/app/services/openrouter_client.py`

### What It Does
1. Reads articles from `articles_selected` (set by the rank node)
2. For each selected article, sends a chat completion request to OpenRouter
3. System prompt: "You summarize AI news for a Telegram digest. Return exactly 3 concise sentences."
4. User prompt includes title, source, published date, URL, and context (og_description or description)
5. Validates the response has ≥3 sentences; if not, retries once with an explicit "Rewrite your answer as exactly 3 sentences" follow-up
6. Enforces exactly 3 sentences via `enforce_sentence_count()` — truncates extras or pads with fallback sentences

### State Output
Updates `articles_selected` with LLM summaries attached to each article.

### Dry Run Mode
When `dry_run=true` or no `OPENROUTER_API_KEY` is set, generates a template summary without calling the LLM:
- Title sentence: "{title} is a notable AI update from {source}."
- Context sentence: "Key context: {first 180 chars of description}."
- Action sentence: "Open the link to review full details, claims, and technical context."

### Concurrency
Uses `asyncio.Semaphore(min(HTTP_CONCURRENCY, 4))` to limit parallel LLM calls to 4.

## Node 5: Deliver

**Source**: `src/app/nodes/deliver.py`, `src/app/services/telegram_client.py`

### What It Does
1. Reads articles from `articles_selected` (set by the summarize node)
2. For each article, formats a Telegram message:
   - If image available: `sendPhoto` with caption (1024 char limit)
   - If no image: `sendMessage` with text (4096 char limit)
3. Caption format: `<a href="URL">TITLE</a>\n\nSUMMARY` (HTML-escaped)
4. Retries up to 3 attempts with exponential backoff
5. Handles HTTP 429 rate limits by sleeping `retry_after` seconds from Telegram response
6. After delivery, if not dry-run, calls `record_deliveries()` to persist successfully-sent articles (URL, title, timestamp) to `data/delivery-history.json`, pruning entries older than `HISTORY_RETENTION_DAYS`

### Dry Run Mode
Returns `{status: "dry_run", mode: "photo"|"text", preview: <formatted_message>}` without calling Telegram API.

### Error Handling
Photo send failures fall back to text-only messages. All failures are recorded in `delivery_results` and appended to `state["errors"]`.

## Data Flow Diagram

```
data/news-sources.yaml
        │
        ▼
   [ingest] ──→ articles_raw (deduped)
        │
        ▼
   [enrich] ──→ articles_enriched (with og:* fields)
        │
        ▼
   [rank]  ──→ articles_selected (date filter → history filter → cluster/score → LLM re-rank → cut to limit)
        │
        ├─ articles_selected? ──→ [summarize] ──→ [deliver] ──→ delivery_results
        │                                                  └─ record_deliveries → data/delivery-history.json
        └─ empty ─────────────→ END
```
