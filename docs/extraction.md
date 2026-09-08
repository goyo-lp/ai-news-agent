# Article retrieval

PR04 fetches the full text of collected `Article` records into validated
`ArticleText` records with `article_text` `Evidence`. Blocked, paywalled, and
thin pages are marked `insufficient` and persist nothing, so they can never
qualify for full-article summaries. Nothing is ever invented for a missing
article.

## Run retrieval

```bash
uv sync --locked --all-groups
uv run ai-news-agent retrieve --database .local/ai-news-agent.db --limit 50
```

The command reads stored articles (or `--article <id>` repeated for specific
ones), retrieves each page, and emits a JSON summary with `attempted`,
`retrieved`, `cached`, `insufficient`, and `failed` counts. Re-running an
unchanged article keeps the original `fetched_at`: text IDs are stable
(`<article-id>-text`) and identical content hashes are not rewritten.

## Behavior

- Redirects are followed manually (default 5 hops) so every hop is resolved
  and validated; private, local, non-HTTP, and credential-bearing destinations
  are rejected, including redirect targets.
- HTTP 401/403/451 responses are access blocks (`insufficient`), not
  transport failures. Timeouts, 429s, and 5xx retry with backoff; other 4xx,
  oversized bodies (default 5 MB cap), and non-HTML content fail fast.
- Extraction uses only the standard library: Open Graph / meta / `<time>`
  metadata for title, authors, and timestamps; `script`, `style`, `nav`,
  `header`, `footer`, `aside`, `form`, and `noscript` content is dropped and
  body paragraphs are collected in document order (restricted to `<article>`
  when present).
- Paywall signals (`isAccessibleForFree: false`, subscribe/sign-in markers),
  bot-challenge markers (captcha, human-verification, forced-JavaScript
  pages), extracts under 400 characters, and pages without a title are all
  `insufficient`. The final URL, content hash, and fetch timestamp are stored
  with each record; over-long text is kept with a `truncated` locator.
- Requests run sequentially with a per-host politeness interval (default
  1 second) so one run cannot hammer a single publisher.
- Retrieved text is untrusted data: it is stored and quoted as inert strings,
  full bodies are never written to logs (lengths and hashes only), and nothing
  in the pipeline interprets page content as instructions.
- Each run is traced (`retrieve-articles` with per-article
  `retrieve-article` children) when LangSmith tracing is enabled.

## Verification

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

HTML fixtures in `tests/fixtures/articles/` cover a full article, a
paywalled page, a bot-challenge page, and a thin page. Retrieval tests cover
unsafe destinations and redirects, retry budgets, rate limiting, caching, and
mixed-outcome summaries.
