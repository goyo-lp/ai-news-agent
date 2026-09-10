# Grounded summaries

PR09 writes exactly three grounded sentences per selected story and renders
email plus archive previews from one structured digest. Summaries describe the
linked representative article only, map every sentence to evidence, and stay in
`draft` status until verification. No email is sent; previews are local files.

## Run summaries

```bash
uv sync --locked --all-groups
uv run ai-news-agent summarize --database .local/ai-news-agent.db \
  --html-path .local/email.html --text-path .local/email.txt \
  --archive-path .local/archive.html
```

The command summarizes stored clusters (pass `--cluster <id>` repeatably to
limit the set), saves one `Summary` per generated cluster
(`<cluster-id>-summary`), and emits `attempted`, `generated`, and
`insufficient` counts. Clusters without retrievable full text become explicit
insufficient items instead of headline-only summaries.

The writer needs `OPENROUTER_API_KEY` in local `.env` (see `.env.example`);
without it the command fails fast instead of guessing.

## Behavior

- Sentence 1 states what happened and the actor; sentence 2 gives the most
  decision-relevant detail, number, scope, availability, or limitation;
  sentence 3 explains why it matters or states a material uncertainty.
- Reported claims are attributed and inferred significance is framed as
  analysis following directly from cited facts.
- Validation requires exactly three single-sentence fields. Abbreviations
  (`U.S.`, `e.g.`) and decimals (`3.14`) do not split sentences; line breaks,
  bullets, and semicolon chains are rejected.
- `build_digest_items()` assembles one ordered digest; `render_email_html()`,
  `render_email_text()`, and `render_archive_html()` render the same stories
  from it. Article content is HTML-escaped, drafts are marked unverified, and
  the email layout uses a 600px container for mobile widths.
- Each run is traced (`generate-summaries`) when LangSmith tracing is enabled.

## Verification

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

Summary tests use an injected fake writer for company claims, numbers, dates,
abbreviations, insufficient evidence, and escaping, plus transport-level tests
of the OpenRouter client and renderer-consistency checks.
