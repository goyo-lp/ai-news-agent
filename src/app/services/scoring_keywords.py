"""Keyword and phrase sets that drive the deterministic relevance score.

Pure data, no logic: each set marks a story category the digest boosts or
penalizes. Keywords are matched against individual title/description tokens
(lowercased, stopwords removed); phrases are matched as substrings of the
normalized text. See scoring._relevance_score for how they combine.
"""

from __future__ import annotations

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in",
    "into", "is", "it", "its", "of", "on", "or", "that", "the", "their",
    "to", "story", "stories", "news", "update", "updates", "with",
}

PRODUCT_LAUNCH_KEYWORDS = {
    "launch", "launches", "launched", "release", "releases", "released",
    "rollout", "rollouts", "ship", "ships", "shipped", "debut", "debuts",
    "debuted", "introduces", "introduced", "unveils", "unveiled", "feature",
    "features", "api", "apis", "model", "models", "agent", "agents",
    "platform", "copilot", "copilots", "sdk", "tool", "tools",
}

TECH_DEVELOPMENT_KEYWORDS = {
    "breakthrough", "breakthroughs", "benchmark", "benchmarks", "inference",
    "training", "multimodal", "reasoning", "architecture", "architectures",
    "chip", "chips", "accelerator", "accelerators", "evaluation",
    "efficiency", "latency", "throughput", "sota",
}

STARTUP_FUNDING_KEYWORDS = {
    "startup", "startups", "funding", "fundraise", "fundraising", "seed",
    "preseed", "series", "valuation", "investor", "investors", "venture",
    "vc", "raise", "raises", "raised",
}

ENTERPRISE_ADOPTION_KEYWORDS = {
    "enterprise", "enterprises", "adoption", "adopt", "adopts", "adopted",
    "deploy", "deploys", "deployed", "deployment", "production", "customer",
    "customers", "workflow", "workflows", "integration", "integrates",
    "integrated", "operations", "productivity",
}

FRONTIER_KEYWORDS = {
    "openai", "chatgpt", "gpt", "anthropic", "claude", "gemini", "deepmind",
    "xai", "grok", "cursor", "copilot", "llama", "mistral", "deepseek",
    "qwen", "nvidia", "huggingface", "perplexity", "windsurf", "midjourney",
}

DEAL_KEYWORDS = {
    "deal", "deals", "partnership", "partnerships", "partner", "partners",
    "contract", "contracts", "acquire", "acquires", "acquired",
    "acquisition", "merger", "mergers", "agreement", "agreements",
}

LOW_PRIORITY_KEYWORDS = {
    "event", "events", "conference", "conferences", "summit", "webinar",
    "meetup", "podcast", "newsletter", "interview", "opinion", "roundup",
    "recap", "guide", "tutorial", "course",
}

HIGH_RELEVANCE_PHRASES = {
    "new model",
    "new feature",
    "model release",
    "product launch",
    "series a",
    "series b",
    "series c",
    "funding round",
    "enterprise adoption",
    "enterprise deployment",
    "production deployment",
    "strategic partnership",
    "signed a deal",
    "open source model",
    "open weights",
}

LOW_PRIORITY_PHRASES = {
    "weekly roundup",
    "daily roundup",
    "event recap",
    "conference recap",
    "podcast episode",
    "panel discussion",
    "how to",
    "step by step",
}
