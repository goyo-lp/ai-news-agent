# ai-linked-imposting-agent Build Plan

Implemented architecture:
- `discover_news` (RSS seed from `data/news-sources.yaml`, daily filter)
- `normalize_and_dedupe`
- `rank_and_cluster` (technical-depth emphasis + anti-hype weighting)
- `deep_research_top5` (Tavily expansion)
- parallel fan-out:
  - `adaptive_investigation` (Deep Agents skill-based technical review with per-brief fallback)
  - `verify_briefs` (model + Tavily verification)
  - `build_style_profile`
- `merge_briefs` (reconcile adaptive + deterministic verification)
- `generate_posts` (5 reflective one-topic posts in simple audience-friendly language; speculative generation path)
- `quality_gate`
- parallel fan-out:
  - `deliver_telegram`
  - `export_artifacts`
- `export_report`

Outputs per run:
- `outputs/YYYY-MM-DD/top_50_articles.json`
- `outputs/YYYY-MM-DD/technical_candidates.json`
- `outputs/YYYY-MM-DD/adaptive_briefs.json`
- `outputs/YYYY-MM-DD/research_briefs.json`
- `outputs/YYYY-MM-DD/linkedin_posts.md`
- `outputs/YYYY-MM-DD/run_report.json`

Telegram delivery:
- Sends each suggested post as an independent Telegram message.
- Message format includes:
  - catchy intro/title
  - full post body (paragraph-formatted for readability)
  - recommended hashtags section

Visualization:
- LangGraphics enabled by default (`http://localhost:8764`).

Model split:
- Stage A ranking defaults to `openai/gpt-oss-120b` (`OPENROUTER_STAGE_A_MODEL`).
- Downstream reasoning defaults to `anthropic/claude-haiku-4.5` (`OPENROUTER_MODEL`).
- Optional model ensembles:
  - Stage A secondary model (`OPENROUTER_STAGE_A_SECONDARY_MODEL`)
  - Verifier secondary model (`OPENROUTER_VERIFIER_SECONDARY_MODEL`)
  - Post generator secondary model (`OPENROUTER_POST_SECONDARY_MODEL`)

Parallelism coverage:
- LangGraph node-level fan-out/fan-in joins
- In-node topic/brief concurrency with semaphores
- Call-level overlap of evidence search/extraction
- Delivery concurrency for Telegram sends
- Multi-run batch concurrency via `run-batch`

Run telemetry:
- End-of-run logs now include API usage/cost summary:
  - OpenRouter calls and token totals
  - Tavily search/extract call counts
  - `estimated_cost_usd` when provider usage includes cost metadata
