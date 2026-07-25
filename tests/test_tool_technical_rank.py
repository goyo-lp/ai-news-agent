from __future__ import annotations

import json
from pathlib import Path

from app.config import Settings
from app.orchestrator.schemas import CuratedArticle, TopicCandidate
from app.orchestrator.tools.news import write_articles_to_state
from app.orchestrator.tools.technical_rank import (
    build_technical_rank_tool,
    technical_rank_tool,
    write_topics_to_state,
)


def _curated(article_id: str, title: str, *, summary: str = "benchmark inference", url: str = "https://x.example.com/") -> CuratedArticle:
    return CuratedArticle(
        id=article_id,
        source_name="Test Source",
        title=title,
        url=url,
        summary=summary,
    )


# Ten titles with essentially no shared vocabulary, so `_same_story`
# clustering keeps them as ten separate story clusters (a shared numeric
# suffix like "... inference {i}" clusters them into one story instead).
_DISTINCT_TITLES = [
    "Benchmark architecture inference training",
    "Kernel scheduler memory allocator patch",
    "Distributed training pipeline throughput",
    "Vector database retrieval index optimization",
    "Reinforcement learning policy gradient update",
    "Transformer attention mechanism optimization work",
    "Compiler backend code generation improvements",
    "Networking protocol latency reduction study",
    "Storage engine compaction algorithm redesign",
    "Container orchestration scheduler enhancement release",
]


def _settings(tmp_path: Path) -> Settings:
    # No OPENROUTER_API_KEY -> heuristic path -> deterministic.
    return Settings(
        _env_file=None,
        orchestrator_data_dir=str(tmp_path),
        openrouter_api_key=None,
        max_topics_per_run=5,
    )


def test_write_topics_to_state_serializes_boundary_models(tmp_path: Path) -> None:
    topics = [
        TopicCandidate(
            topic_id="a",
            title="T",
            summary_hint="hint",
            primary_url="https://x.example.com/",
            primary_domain="x.example.com",
            score=0.8,
            cluster_size=1,
            rationale="high-signal, single-source, tier-1",
        )
    ]
    path = write_topics_to_state(topics, str(tmp_path))
    assert path.name == "topics.json"
    on_disk = json.loads(path.read_text(encoding="utf-8"))
    assert on_disk[0]["topic_id"] == "a"
    # Boundary projection shape, NOT DiscoveredItem internals.
    assert "query" not in on_disk[0]
    assert "source_tier" not in on_disk[0]


def test_write_topics_to_state_creates_missing_dir(tmp_path: Path) -> None:
    nested = tmp_path / "deep" / "state"
    write_topics_to_state([], str(nested))
    assert (nested / "topics.json").exists()


async def test_tool_reads_articles_writes_topics_and_returns_summary(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    articles = [_curated("a", "Architecture benchmark inference training",
                       url="https://a.example.com/", summary="benchmark inference training"),
                _curated("b", "Revolutionary game changer massive announcement",
                       url="https://b.example.com/", summary="massive unbelievable hype announcement")]
    write_articles_to_state(articles, settings.orchestrator_data_dir)

    tool = build_technical_rank_tool(settings)
    result = json.loads(await tool.ainvoke({}))

    assert set(result) == {"count", "limit_used", "path"}
    assert result["count"] == 2
    # No limit was requested, so the cap is however many articles were
    # available (2) — NOT settings.max_topics_per_run (5). technical_rank
    # writes every viable topic; the coordinator does the down-selection.
    assert result["limit_used"] == 2
    assert result["path"].endswith("topics.json")
    topics_path = Path(result["path"])
    assert topics_path.parent == Path(settings.orchestrator_data_dir)

    on_disk = json.loads(topics_path.read_text(encoding="utf-8"))
    assert len(on_disk) == 2
    # The technical story ranks first; the hype-only story second.
    assert on_disk[0]["topic_id"] == "a"
    assert on_disk[0]["score"] > on_disk[1]["score"]


async def test_tool_does_not_clamp_to_settings_max_topics_per_run(tmp_path: Path) -> None:
    """technical_rank writes every viable topic regardless of
    max_topics_per_run — that cap is enforced by the coordinator's own
    selection when it reads topics.json (P: source-diversity reasoning), not
    by this tool. Ten distinct-domain articles with a max_topics_per_run=3
    settings value should still all make it to topics.json."""
    settings = Settings(_env_file=None, orchestrator_data_dir=str(tmp_path), max_topics_per_run=3)
    articles = [_curated(f"a{i}", title, url=f"https://x{i}.example.com/") for i, title in enumerate(_DISTINCT_TITLES)]
    write_articles_to_state(articles, settings.orchestrator_data_dir)

    tool = build_technical_rank_tool(settings)
    result = json.loads(await tool.ainvoke({}))

    assert result["count"] == 10
    assert result["limit_used"] == 10


async def test_tool_honors_explicit_smaller_limit(tmp_path: Path) -> None:
    """An explicit `limit` is still honored as an escape hatch, independent of
    max_topics_per_run."""
    settings = Settings(_env_file=None, orchestrator_data_dir=str(tmp_path), max_topics_per_run=3)
    articles = [_curated(f"a{i}", title, url=f"https://x{i}.example.com/") for i, title in enumerate(_DISTINCT_TITLES)]
    write_articles_to_state(articles, settings.orchestrator_data_dir)

    tool = build_technical_rank_tool(settings)
    result = json.loads(await tool.ainvoke({"limit": 2}))

    assert result["count"] == 2
    assert result["limit_used"] == 2


async def test_tool_summary_omits_topic_payload(tmp_path: Path) -> None:
    """Per guiding principle #3, the tool's return is a compressed summary —
    never the topics themselves."""
    settings = _settings(tmp_path)
    write_articles_to_state([_curated("a", "Architecture benchmark inference")], settings.orchestrator_data_dir)
    tool = build_technical_rank_tool(settings)
    result_raw = await tool.ainvoke({})
    assert "primary_url" not in result_raw
    assert "rationale" not in result_raw


async def test_tool_raises_when_articles_file_missing(tmp_path: Path) -> None:
    """Fetch_curated_ai_news must run first; a missing articles file is a real
    precondition error, not silently-empty output."""
    import pytest

    settings = _settings(tmp_path)
    tool = build_technical_rank_tool(settings)
    with pytest.raises(FileNotFoundError):
        await tool.ainvoke({})


def test_default_singleton_is_a_structured_tool() -> None:
    assert technical_rank_tool.name == "technical_rank"
    assert technical_rank_tool.args_schema is not None