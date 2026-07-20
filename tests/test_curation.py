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

    articles, effective_limit = await main_module.run_curation(limit=5)

    assert len(articles) == 1
    assert isinstance(articles[0], CuratedArticle)
    assert articles[0].id == "a1"
    assert articles[0].title == "OpenAI launches new model"
    assert fake_workflow.received_state["limit"] == 5
    assert fake_workflow.received_state["dry_run"] is False
    assert effective_limit == 5


async def test_run_curation_defaults_limit_to_settings_max(monkeypatch) -> None:
    fake_workflow = _FakeCurationWorkflow({"articles_selected": []})
    monkeypatch.setattr(main_module, "build_curation_workflow", lambda: fake_workflow)

    articles, effective_limit = await main_module.run_curation()

    assert articles == []
    assert fake_workflow.received_state["limit"] == main_module.get_settings().max_articles_per_run
    assert effective_limit == main_module.get_settings().max_articles_per_run


async def test_run_curation_clamps_limit_to_settings_max(monkeypatch) -> None:
    """A programmatic caller can't bypass the cap the CLI enforces."""
    fake_workflow = _FakeCurationWorkflow({"articles_selected": []})
    monkeypatch.setattr(main_module, "build_curation_workflow", lambda: fake_workflow)

    cap = main_module.get_settings().max_articles_per_run
    articles, effective_limit = await main_module.run_curation(limit=cap + 10_000)

    assert fake_workflow.received_state["limit"] == cap
    assert effective_limit == cap
    assert articles == []


async def test_run_curation_uses_injected_settings_not_lru_cache(monkeypatch) -> None:
    """The injected Settings is the source of truth for the clamp — not
    get_settings(). This is the bug class the tool used to have: it reported a
    limit reconstructed from one Settings instance while run_curation clamped
    using another."""
    fake_workflow = _FakeCurationWorkflow({"articles_selected": []})
    monkeypatch.setattr(main_module, "build_curation_workflow", lambda: fake_workflow)

    from app.config import Settings

    injected = Settings(_env_file=None, max_articles_per_run=3)
    articles, effective_limit = await main_module.run_curation(
        limit=10_000, settings=injected
    )

    assert fake_workflow.received_state["limit"] == 3
    assert effective_limit == 3
    assert articles == []
