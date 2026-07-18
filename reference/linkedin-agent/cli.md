# CLI Commands

Run from project root.

Full pipeline (LangGraphics on by default):
```bash
source .venv/bin/activate && PYTHONPATH=src python -m app.main run
```
This runs the full graph (including adaptive Deep Agent investigation), then sends 5 suggested posts as individual Telegram messages.
Generated posts are written in simple audience-friendly language and rendered in paragraph format on Telegram.
Run completion logs include API usage/cost telemetry (`openrouter_calls`, Tavily call counts, token totals, and `estimated_cost_usd` when available).

Dry run (no external API calls):
```bash
source .venv/bin/activate && PYTHONPATH=src python -m app.main run --dry-run
```

Run with adaptive Deep Agent disabled (keeps deterministic pipeline):
```bash
source .venv/bin/activate && DEEP_AGENT_ENABLED=false PYTHONPATH=src python -m app.main run
```

Run a parallel batch for multiple dates:
```bash
source .venv/bin/activate && PYTHONPATH=src python -m app.main run-batch --dates 2026-03-01,2026-03-02 --max-concurrency 2 --no-graphics
```

Disable LangGraphics for a run:
```bash
source .venv/bin/activate && PYTHONPATH=src python -m app.main run --no-graphics
```

Specify freshness window and style folder:
```bash
source .venv/bin/activate && PYTHONPATH=src python -m app.main run --hours-back 48 --samples-dir style_samples
```

Build style profile only:
```bash
source .venv/bin/activate && PYTHONPATH=src python -m app.main build-style-profile --samples-dir style_samples
```

Preview latest generated posts:
```bash
source .venv/bin/activate && PYTHONPATH=src python -m app.main preview
```
