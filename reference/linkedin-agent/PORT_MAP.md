# LinkedIn agent → port map

This tree is a **port kit**, not a runnable second app. Do not import it from host `src/app`.
Copy or reimplement logic into the orchestrator packages in later PRs.

## Port (source → intended home)

| Source | Target phase / module | Notes |
|--------|----------------------|--------|
| `src/app/schemas/models.py` | P0.2 / orchestrator contracts | Shapes for briefs, posts, style; adapt to host models |
| `src/app/services/post_generator.py` | P3.x LinkedIn generation | Core generation prompts + length gates |
| `src/app/services/brief_verifier.py` | P2.x / P3.x verification | Claim checks + evidence |
| `src/app/services/deep_agent_investigator.py` | P2.x deep research | Skills / investigation loop; rebuild under `create_deep_agent` |
| `src/app/services/technical_ranker.py` | P2.x ranking | Technical-depth re-rank (news pipeline remains producer) |
| `src/app/services/style_profile.py` | P3.x style | Profile build from samples |
| `src/app/services/tavily_client.py` | P2.x research tools | Evidence search client patterns |
| `src/app/services/api_usage_tracker.py` | P4.x / shared ops | Cost/usage accounting ideas |
| `src/app/services/url_utils.py` | Prefer host `http_utils` / existing URL helpers | Port only if host lacks equivalent |
| `src/app/services/source_policy.py` | Optional P2.x | Trusted-source policy; may fold into news sources |
| `src/app/services/scoring.py` | Optional / reference only | Seed scoring; news pipeline owns discovery ranking |
| `src/app/config.py` | Env inventory only | Do not dual-Settings; map knobs into host config |
| `data/style_profile.json` | P3.x data | Seed style profile |
| `data/trusted-sources.yaml` | Optional P2.x | Policy input |
| `style_samples/*` | P3.x | Author voice samples |
| `tests/test_post_generator.py` | Port with generator | Behavior anchors |
| `tests/test_style_profile.py` | Port with style | |
| `tests/test_url_utils.py` | Only if url_utils ported | Prefer host tests |
| `tests/test_scoring.py` | Reference only | |
| `tests/test_deep_agent_investigator.py` | Rebuild with new agent | Patterns, not drop-in |

## Do not port

| Path / concept | Why |
|----------------|-----|
| Graph shell (`graph/`, `nodes/`, `main.py`) | Decision F: rebuild under deep agent / orchestrator |
| RSS discovery / `rss_seed_client` / `news-sources.yaml` | Decision E: news pipeline is sole producer |
| LangGraphics / static assets | Decision I: dropped |
| Telegram client / delivery nodes | Host already has Telegram; LinkedIn uses separate bot (decision on delivery) |
| `output_writer`, run `outputs/` dumps | Not needed to port logic; add small fixtures later if parity needs them |
| Nested `pyproject.toml` / `requirements.txt` / CLI docs | Not a second installable project |
| Importing `reference.*` or dual `app` package on PYTHONPATH | Boundary leak; copy into host modules |

## How to use

1. Open the source file listed above.
2. Reimplement behind host orchestrator boundaries (P0.2 contracts).
3. Do not wrap the old LangGraph workflow.
