"""Bundled, deterministic records for key-free local verification."""

import json
from importlib.resources import files

from ai_news_agent.schemas import (
    Article,
    ArticleText,
    Evidence,
    Run,
    Score,
    Source,
    StoredRecord,
    StoryCluster,
    Summary,
    VerificationResult,
)

_MODELS = {
    "source": Source,
    "article": Article,
    "article_text": ArticleText,
    "story_cluster": StoryCluster,
    "score": Score,
    "evidence": Evidence,
    "summary": Summary,
    "verification_result": VerificationResult,
    "run": Run,
}


def load_sample_records() -> tuple[StoredRecord, ...]:
    """Load and validate the package's offline fixture records."""

    fixture_path = files("ai_news_agent").joinpath("fixtures/sample_records.json")
    payload = json.loads(fixture_path.read_text(encoding="utf-8"))
    return tuple(
        _MODELS[item["type"]].model_validate(item["record"])
        for item in payload["records"]
    )
