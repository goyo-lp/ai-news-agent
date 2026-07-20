from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from app.orchestrator.schemas import CuratedArticle
from app.orchestrator.tools.news import (
    build_fetch_curated_ai_news_tool,
    fetch_curated_ai_news_tool,
    write_articles_to_state,
)


def _curated(article_id: str = "a1") -> CuratedArticle:
    return CuratedArticle(
        id=article_id,
        source_name="Test Source",
        title="OpenAI launches new model",
        url=f"https://example.com/{article_id}",
        summary="A concise summary.",
        score=0.9,
    )


def _settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    """Build a Settings whose orchestrator data dir points at the tmp dir, so
    tests don't write into the repo's real data/ tree."""
    from app.config import Settings

    return Settings(_env_file=None, orchestrator_data_dir=str(tmp_path))


def test_write_articles_to_state_serializes_boundary_models(tmp_path: Path) -> None:
    articles = [_curated("a1"), _curated("a2")]

    path = write_articles_to_state(articles, str(tmp_path))

    assert path.name == "articles.json"
    assert path.parent == tmp_path
    on_disk = json.loads(path.read_text(encoding="utf-8"))
    assert len(on_disk) == 2
    assert on_disk[0]["id"] == "a1"
    assert on_disk[0]["title"] == "OpenAI launches new model"
    # Boundary projection, not the pipeline-internal Article: no source_rss /
    # og_title leakage across the seam.
    assert "source_rss" not in on_disk[0]


def test_write_articles_to_state_creates_missing_dir(tmp_path: Path) -> None:
    nested = tmp_path / "deep" / "state"

    write_articles_to_state([], str(nested))

    assert (nested / "articles.json").exists()
    assert json.loads((nested / "articles.json").read_text()) == []


def test_write_articles_to_state_round_trips_through_curated_article(tmp_path: Path) -> None:
    """The file on disk rehydrates into the exact boundary model — that's the
    parity contract the coordinator's downstream readers depend on."""
    original = [_curated("a1")]
    path = write_articles_to_state(original, str(tmp_path))

    payload = json.loads(path.read_text(encoding="utf-8"))
    restored = [CuratedArticle.model_validate(item) for item in payload]
    assert restored == original


async def test_tool_factory_injects_settings_and_uses_its_data_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.config import Settings
    from app.orchestrator import tools as tools_mod

    settings = Settings(_env_file=None, orchestrator_data_dir=str(tmp_path), max_articles_per_run=5)

    async def fake_run_curation(
        limit: int | None = None, settings: Settings | None = None
    ) -> tuple[list[CuratedArticle], int]:
        assert limit == 3
        assert settings is not None and settings.max_articles_per_run == 5
        return [_curated("a1"), _curated("a2")], 3

    monkeypatch.setattr(tools_mod.news, "run_curation", fake_run_curation)

    tool = build_fetch_curated_ai_news_tool(settings)
    result_raw = await tool.ainvoke({"limit": 3})
    result = json.loads(result_raw)

    assert result["count"] == 2
    assert result["limit_used"] == 3
    assert result["path"].endswith("articles.json")
    assert Path(result["path"]).parent == tmp_path
    on_disk = json.loads(Path(result["path"]).read_text(encoding="utf-8"))
    assert len(on_disk) == 2


async def test_tool_clamps_limit_to_settings_max(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A model-supplied limit can't bypass the configured ceiling — the same
    cap run_curation enforces. The reported `limit_used` is the effective
    clamped value returned by run_curation, not a reconstruction."""
    from app.config import Settings
    from app.orchestrator import tools as tools_mod

    settings = Settings(_env_file=None, orchestrator_data_dir=str(tmp_path), max_articles_per_run=3)

    async def fake_run_curation(
        limit: int | None = None, settings: Settings | None = None
    ) -> tuple[list[CuratedArticle], int]:
        assert settings is not None
        resolved = limit if limit is not None else settings.max_articles_per_run
        effective = max(1, min(resolved, settings.max_articles_per_run))
        return [], effective

    monkeypatch.setattr(tools_mod.news, "run_curation", fake_run_curation)

    tool = build_fetch_curated_ai_news_tool(settings)
    result = json.loads(await tool.ainvoke({"limit": 10_000}))

    assert result["count"] == 0
    assert result["limit_used"] == 3


async def test_tool_summary_omits_article_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Per guiding principle #3, the tool's return is a compressed summary —
    never the articles themselves. Structured data lives on the filesystem."""
    from app.config import Settings
    from app.orchestrator import tools as tools_mod

    settings = Settings(_env_file=None, orchestrator_data_dir=str(tmp_path), max_articles_per_run=5)

    async def fake_run_curation(
        limit: int | None = None, settings: Settings | None = None
    ) -> tuple[list[CuratedArticle], int]:
        return [_curated("a1"), _curated("a2"), _curated("a3")], 5

    monkeypatch.setattr(tools_mod.news, "run_curation", fake_run_curation)

    tool = build_fetch_curated_ai_news_tool(settings)
    result_raw = await tool.ainvoke({})

    result = json.loads(result_raw)
    assert set(result) == {"count", "limit_used", "path"}
    assert result["count"] == 3
    assert "title" not in result_raw  # no article content in the summary


async def test_tool_threads_injected_settings_into_run_curation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The tool passes its injected Settings into run_curation — the seam that
    stops the `limit_used` reconstruction bug. A mismatched get_settings() must
    never be the source of the reported limit."""
    from app.config import Settings, get_settings
    from app.orchestrator import tools as tools_mod

    injected = Settings(_env_file=None, orchestrator_data_dir=str(tmp_path), max_articles_per_run=2)
    assert get_settings().max_articles_per_run != 2  # sanity: differs from cache

    received: dict[str, Any] = {}

    async def fake_run_curation(
        limit: int | None = None, settings: Settings | None = None
    ) -> tuple[list[CuratedArticle], int]:
        received["settings"] = settings
        assert settings is not None
        effective = max(1, min(limit or settings.max_articles_per_run, settings.max_articles_per_run))
        return [], effective

    monkeypatch.setattr(tools_mod.news, "run_curation", fake_run_curation)

    tool = build_fetch_curated_ai_news_tool(injected)
    result = json.loads(await tool.ainvoke({"limit": 100}))

    assert received["settings"] is injected
    assert result["limit_used"] == 2


def test_tool_sync_invoke_raises_not_implemented(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The tool is async-only — sync invoke must fail loudly rather than spawn
    a hidden event loop on a worker thread."""
    from app.config import Settings

    settings = Settings(_env_file=None, orchestrator_data_dir=str(tmp_path))
    tool = build_fetch_curated_ai_news_tool(settings)
    with pytest.raises(NotImplementedError):
        tool.invoke({"limit": 1})


def test_default_singleton_is_a_structured_tool() -> None:
    """The convenience module-level singleton is built and named so the
    coordinator's create_deep_agent(tools=[...]) list can import it directly."""
    assert fetch_curated_ai_news_tool.name == "fetch_curated_ai_news"
    assert fetch_curated_ai_news_tool.args_schema is not None