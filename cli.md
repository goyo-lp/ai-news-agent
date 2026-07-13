# CLI Commands

Run from project root.

Main run (sends to Telegram):
```bash
uv run ai-news-agent run
```

Dry run (no Telegram send):
```bash
uv run ai-news-agent run --dry-run
```

Limit output (max 50):
```bash
uv run ai-news-agent run --limit 50
```

Verbose logs:
```bash
uv run ai-news-agent run --verbose
```

LangGraphics starts automatically during runs.
