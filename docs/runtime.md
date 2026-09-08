# Runtime foundation

PR02 establishes a provider-neutral Python 3.12+ runtime. It includes validated
records, forward-only SQLite migrations, structured logs, a key-free fixture
command, and opt-in LangSmith tracing. Model-provider dependencies and keys are
deliberately deferred until the editorial agent is implemented.

## Install

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then sync
the exact dependency versions recorded in `uv.lock`:

```bash
uv sync --locked --all-groups
```

Copy `.env.example` to `.env` only when local configuration is needed. `.env`
and local SQLite files are ignored by Git; never commit credentials.

## Offline fixture command

The fixture command validates and stores one example of every required record.
It makes no network calls, needs no API keys, and disables tracing even if the
shell has tracing enabled:

```bash
uv run ai-news-agent fixture
```

To inspect the migrated database:

```bash
uv run ai-news-agent fixture --database .local/fixture.db
sqlite3 .local/fixture.db 'select record_type, count(*) from records group by 1;'
```

The command emits a structured log to stderr and a stable JSON result to stdout.

## Data contracts

`src/ai_news_agent/schemas.py` defines strict Pydantic records for `Source`,
`Article`, `StoryCluster`, `Score`, `Evidence`, `Summary`,
`VerificationResult`, and `Run`. `ArticleText` is a separate record so retrieved
source text cannot be confused with generated summaries. Unknown fields,
timezone-naive dates, invalid score ranges, inconsistent run states, and broken
record relationships fail validation with explicit errors.

The score total follows the accepted editorial weights and is computed by code.
SQLite stores the validated payloads as JSON envelopes keyed by record type and
ID. Re-saving an ID updates it rather than creating a duplicate. Later PRs may
add query-optimized tables through new migrations without changing these public
contracts.

## LangSmith tracing

Tracing is off by default. The runtime uses the current LangSmith environment
names and attaches a shared digest run ID, component, environment, and package
version to nested spans. Inputs, outputs, and metadata pass through recursive
credential and email redaction before export.

Set these values only for an opt-in live smoke run:

```bash
export LANGSMITH_TRACING=true
export LANGSMITH_API_KEY=your-key
export LANGSMITH_PROJECT=ai-news-agent
uv run ai-news-agent trace-smoke --digest-run-id manual-smoke-001
```

The command emits `trace-smoke-parent` and `trace-smoke-child` spans, flushes the
client, and returns nonzero if flushing fails. Background export failures are
also written to local structured logs as `langsmith_trace_export_failed`.

Official references used to validate the setup:

- [LangSmith observability quickstart](https://docs.langchain.com/langsmith/observability-quickstart)
- [Trace LangChain applications](https://docs.langchain.com/langsmith/trace-with-langchain)
- [Prevent logging sensitive data](https://docs.langchain.com/langsmith/mask-inputs-outputs)
- [Conditional tracing](https://docs.langchain.com/langsmith/conditional-tracing)

## Verification

Run the same checks used by CI:

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest
uv build
```
