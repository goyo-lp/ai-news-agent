from __future__ import annotations

from typing import Any

from app import main as main_module
from app.orchestrator.schemas import CuratedArticle
from app.schemas.article import Article, serialize_articles


def _article(article_id: str = "a1") -> Article:
    return Article(
        id=article_id,
        source_name="Test Source",
        source_rss="https://example.com/feed",
        title="OpenAI launches new model",
        url=f"https://example.com/{article_id}",
    )


class _FakeCurationWorkflow:
    """Stands in for the compiled LangGraph curation workflow: records the
    initial state it was invoked with and returns a canned final state."""

    def __init__(self, final_state: dict[str, Any]) -> None:
        self._final_state = final_state
        self.received_state: dict[str, Any] | None = None

    async def ainvoke(self, initial_state: dict[str, Any]) -> dict[str, Any]:
        self.received_state = initial_state
        return self._final_state


async def test_run_curation_returns_curated_articles_from_final_state(monkeypatch) -> None:
    article = _article()
    fake_workflow = _FakeCurationWorkflow(
        {"articles_selected": serialize_articles([article]), "errors": []}
    )
    monkeypatch.setattr(main_module, "build_curation_workflow", lambda: fake_workflow)

    result = await main_module.run_curation(limit=5)

    assert len(result) == 1
    assert isinstance(result[0], CuratedArticle)
    assert result[0].id == "a1"
    assert result[0].title == "OpenAI launches new model"
    assert fake_workflow.received_state["limit"] == 5
    assert fake_workflow.received_state["dry_run"] is False


async def test_run_curation_defaults_limit_to_settings_max(monkeypatch) -> None:
    fake_workflow = _FakeCurationWorkflow({"articles_selected": []})
    monkeypatch.setattr(main_module, "build_curation_workflow", lambda: fake_workflow)

    result = await main_module.run_curation()

    assert result == []
    assert fake_workflow.received_state["limit"] == main_module.get_settings().max_articles_per_run


async def test_run_curation_clamps_limit_to_settings_max(monkeypatch) -> None:
    """A programmatic caller can't bypass the cap the CLI enforces."""
    fake_workflow = _FakeCurationWorkflow({"articles_selected": []})
    monkeypatch.setattr(main_module, "build_curation_workflow", lambda: fake_workflow)

    cap = main_module.get_settings().max_articles_per_run
    await main_module.run_curation(limit=cap + 10_000)

    assert fake_workflow.received_state["limit"] == cap