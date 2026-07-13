# Scoring & Ranking

**Sources**: `src/app/services/scoring.py`, `src/app/nodes/rank.py`

The ranking system determines which articles make it into the final Telegram digest. It combines relevance-first keyword scoring, same-story clustering, recency, source quality, and novelty signals.

## Date Filtering

Before ranking, `filter_articles_published_today()` (in `src/app/nodes/rank.py`) keeps only articles published on the current local date:

- Uses `datetime.now().astimezone()` to get local timezone
- Articles with `published_at=None` are excluded
- Articles are compared by converting to local timezone and checking date equality
- This filter was added in commit `f5384e1` to ensure only fresh news reaches ranking

## Clustering

**Function**: `cluster_articles(articles)` in `scoring.py`

Groups articles covering the same story across different sources. Uses title similarity:

### `_same_story(left, right)` Logic
1. Tokenizes both titles (lowercase, stopword-filtered)
2. Requires ≥2 overlapping tokens
3. Computes `_title_similarity` = 0.65 × Jaccard + 0.35 × SequenceMatcher ratio
4. **Strong match**: title_similarity ≥ 0.78 AND overlap_ratio ≥ 0.5
5. **Moderate match**: title_similarity ≥ 0.62 AND overlap_ratio ≥ 0.7 AND `_is_time_aligned` (published within 120 hours)

### Cluster Assignment
Articles are sorted by `published_at` descending (newest first), then greedily assigned to the first matching cluster. Each cluster's first member becomes the representative for similarity comparison.

## Scoring Formula

**Function**: `score_article(article, cluster_size=1, recent_titles=None)` in `scoring.py`

```
score = 0.38 × relevance
      + 0.24 × recency
      + 0.14 × source_weight
      + 0.10 × duplication_signal
      + 0.09 × cluster_signal
      + 0.05 × novelty
```

### Relevance Score (0.0–1.0, weight 0.38)

The highest-weighted component. Uses keyword and phrase matching against the article's effective title + effective summary source.

**Primary signal categories** (each contributes a boost):
| Category | Base boost | Per extra hit | Example keywords |
|----------|-----------|---------------|------------------|
| Product launch | 0.28 | +0.03 | launch, release, rollout, ship, debut, unveils, model, agent, platform, API, SDK |
| Tech development | 0.22 | +0.03 | breakthrough, benchmark, inference, training, multimodal, reasoning, architecture, chip, accelerator |
| Startup/funding | 0.24 | +0.03 | startup, funding, seed, series, valuation, investor, venture, raise |
| Enterprise adoption | 0.22 | +0.03 | enterprise, adoption, deploy, deployment, production, customer, workflow, integration |
| Deals | 0.20 | +0.03 | deal, partnership, partner, contract, acquire, acquisition, merger, agreement |
| Frontier company | 0.24 | +0.04 | openai, chatgpt, gpt, anthropic, claude, gemini, deepmind, xai, grok, cursor, copilot, llama, mistral, deepseek, qwen, nvidia, huggingface, perplexity, windsurf, midjourney |

Extra hits are capped at 2 additional beyond the first.

**High-relevance phrases** (up to 2 matched, +0.07 each):
`new model`, `new feature`, `model release`, `product launch`, `series a/b/c`, `funding round`, `enterprise adoption`, `enterprise deployment`, `production deployment`, `strategic partnership`, `signed a deal`, `open source model`, `open weights`

**Low-priority penalty**: Keywords like `event`, `conference`, `webinar`, `podcast`, `newsletter`, `roundup`, `recap`, `tutorial`, `how to` reduce the score. Penalty is 0.08 if priority hits exist, 0.22 if none. Additional 0.03 per low-priority signal (max 3).

### Recency Score (0.0–1.0, weight 0.24)

| Age | Score |
|-----|-------|
| ≤ 6 hours | 1.0 |
| ≤ 24 hours | 0.8 |
| ≤ 48 hours | 0.6 |
| ≤ 96 hours | 0.4 |
| > 96 hours or unknown | 0.2–0.3 |

### Source Weight (0.7–1.0, weight 0.14)

Hardcoded reputation weights in `_SOURCE_WEIGHTS`:

| Source | Weight |
|--------|--------|
| OpenAI Blog, Google DeepMind Blog, Anthropic Blog, Meta AI Blog | 1.0 |
| Google AI Blog | 0.95 |
| MIT Technology Review, Simon Willison | 0.90–0.92 |
| TechCrunch (AI) | 0.90 |
| The Verge (AI), Wired (AI), Ars Technica (AI) | 0.88 |
| VentureBeat (AI) | 0.87 |
| Hacker News (Top), The Guardian (AI) | 0.85 |
| GitHub Trending, ZDNet (AI) | 0.80 |
| Reddit r/MachineLearning | 0.62 |
| Reddit r/artificial | 0.60 |
| arXiv.org (cs.AI) | 0.50 |
| Any other source | 0.70 (default) |

### Duplication Signal (0.0–1.0, weight 0.10)

`min(duplicate_count / 5.0, 1.0)` — articles seen at the same URL across multiple fetches get a small boost, capped at 5 duplicates.

### Cluster Signal (0.0–1.0, weight 0.09)

`min(cluster_size / 5.0, 1.0)` — articles that are part of a larger cluster (covered by more sources) get a small boost.

### Novelty Score (0.0–1.0, weight 0.05)

`_novelty_score(article, recent_titles)` — when delivery history is available, novelty is computed as `1.0 - max_title_similarity` against recently delivered titles. An article whose title closely matches a previously-delivered story gets a low novelty score. When no `recent_titles` are provided (e.g. no history file), novelty defaults to `1.0`.

## Final Selection

1. All articles are scored within their clusters (novelty penalizes recently-delivered titles)
2. Each cluster's highest-scoring member (ties broken by newest `published_at`) becomes the representative
3. Representatives are sorted by score descending (ties broken by newest)
4. If `max_per_source` is set, each source is capped at that count — no backfill past the cap, so a quiet day yields a shorter digest (prevents high-volume feeds like arXiv from flooding the digest)
5. Top `limit` articles are returned (default 50, clamped to `[1, MAX_ARTICLES_PER_RUN]`, per-source cap `MAX_ARTICLES_PER_SOURCE` default 3)

**Note**: In the rank node, `rank_articles()` is called with `limit=max(limit, _LLM_RERANK_POOL)` (40) to produce a wider candidate pool. The LLM re-rank then blends scores (30% LLM, 70% deterministic) and the final cut to `limit` happens after that blend. See [Pipeline Workflow](../workflows/pipeline.md#node-3-rank) for details.

## LLM Re-Rank

After deterministic scoring, the rank node calls `OpenRouterClient.score_articles_relevance()` — a single batched OpenRouter request that asks the LLM to rate each article 0–100 for relevance to a product/funding/deals-focused digest. The returned scores (normalized to 0.0–1.0) are blended into each article's deterministic score at `_LLM_BLEND_WEIGHT = 0.3`:

```
final_score = 0.7 × deterministic_score + 0.3 × llm_score
```

If dry-run, no API key, or the call fails, the deterministic scores stand unchanged. See [Integrations](../integrations/overview.md#openrouter-llm-summarization) for the API details.

## Evolution (Git History)

- **52cb4ae** (initial): Basic scoring without relevance keywords
- **56ccbc2**: Major refactor — added keyword-based relevance scoring with 5 priority categories, phrase matching, low-priority penalties, source weights, and clustering
- **f5384e1**: Added `filter_articles_published_today()` before ranking to exclude non-today articles
- **784cf0d**: Added frontier company keyword category, per-source cap (`max_per_source`), demoted arXiv/Reddit source weights, added new high-relevance phrases (`open source model`, `open weights`)
- **uncommitted**: Novelty score changed from title-length heuristic to title-similarity against delivery history. `rank_articles()` and `score_article()` now accept `recent_titles` parameter. LLM re-rank blends 30% LLM relevance scores into final ranking.
