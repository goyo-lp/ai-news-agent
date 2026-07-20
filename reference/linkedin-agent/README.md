# LinkedIn agent — port kit

Vendored **reference only** for porting LinkedIn research + post generation into this repo.

This is **not** a second runnable app. Graph shell, discovery/RSS, LangGraphics, nested packaging, and bulk run outputs were intentionally omitted.

## Layout

```text
PORT_MAP.md              # source → target phase / do-not-port
src/app/
  schemas/models.py      # Pydantic shapes worth adapting
  services/              # generation, verify, research, rank, style, tavily, …
  config.py              # env inventory (do not dual-Settings)
data/
  style_profile.json
  trusted-sources.yaml
style_samples/           # author voice samples
tests/                   # anchors for the modules above
```

## Rules

1. Read `PORT_MAP.md` before copying anything.
2. Do not `import` this tree from host `src/app`.
3. News pipeline remains the sole producer of candidate stories (no RSS discovery here).
4. Rebuild orchestration with the host deep-agent / coordinator design — do not revive the old graph.

Host CI excludes `reference/` from ruff and pytest (`pyproject.toml`).
