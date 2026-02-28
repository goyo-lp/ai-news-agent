# CLI Commands

Run from project root.

Main run (sends to Telegram):
```bash
source .venv/bin/activate && PYTHONPATH=src python -m app.main run
```

Dry run (no Telegram send):
```bash
source .venv/bin/activate && PYTHONPATH=src python -m app.main run --dry-run
```

Limit output (max 50):
```bash
source .venv/bin/activate && PYTHONPATH=src python -m app.main run --limit 50
```

Verbose logs:
```bash
source .venv/bin/activate && PYTHONPATH=src python -m app.main run --verbose
```

LangGraphics starts automatically during runs.
