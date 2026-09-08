# ai-news-agent

Open-source LangChain AI agent for AI news.

> Early scaffolding — no agent code yet. This repo currently sets up the open-source foundation: branching, protections, and PR workflow.

## Quickstart

The project requires Python 3.12+ and uses `uv` for reproducible environments:

```bash
uv sync --locked --all-groups
uv run ai-news-agent fixture
```

The fixture command exercises the schemas and SQLite storage without network
access, API keys, or tracing. See the [runtime guide](docs/runtime.md) for
configuration, verification, and the opt-in LangSmith smoke test. Feed
collection is documented in the [ingestion guide](docs/ingestion.md). Full-text
retrieval is documented in the [extraction guide](docs/extraction.md). Event
grouping is documented in the [clustering guide](docs/clustering.md). Candidate
screening is documented in the [screening guide](docs/screening.md). Agent behavior
will land under `src/` through the remaining planned pull requests.

## Editorial contract

The initial [editorial policy](docs/editorial-policy.md) defines the audience,
quality bar, evidence rules, scoring rubric, summary format, and failure
behavior. The companion [source registry contract](docs/source-registry.md)
records the proposed 50-source input that ingestion must validate before use.

## Workflow

- Default branch: `main` (protected)
- Workflow: GitHub Flow — short-lived `feat/*`, `fix/*`, `docs/*`, `chore/*` branches → PR to `main`
- PRs require: green `ci`, resolved conversations (no required approvals — solo maintainer)
- See [CONTRIBUTING.md](CONTRIBUTING.md) for branch naming, commits, and PR checklist.

## Contributing

Please read [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md), [CONTRIBUTING.md](CONTRIBUTING.md), and [SECURITY.md](SECURITY.md) before opening an issue or PR.

## License

[MIT](LICENSE)
