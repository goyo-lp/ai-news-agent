from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from app.config import Settings
from app.orchestrator.services import fetch_article as svc_mod
from app.orchestrator.services.fetch_article import FetchedArticle
from app.orchestrator.tools.fetch import (
    build_fetch_article_tool,
    fetch_article_tool,
    write_fetched_to_state,
)


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        orchestrator_data_dir=str(tmp_path),
        request_timeout_seconds=10,
        user_agent="test-agent",
    )


def _patch_httpx(monkeypatch: pytest.MonkeyPatch, handler) -> None:
    real_async_client = httpx.AsyncClient

    def _factory(*args: object, **kwargs: object) -> httpx.AsyncClient:
        kwargs["transport"] = httpx.MockTransport(handler)
        return real_async_client(**kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(svc_mod.httpx, "AsyncClient", _factory)


def test_write_fetched_to_state_round_trips(tmp_path: Path) -> None:
    article = FetchedArticle(
        url="https://example.com/post",
        final_url="https://example.com/post",
        title="Title",
        description="Desc",
        image_url="https://img.example.com/x.png",
        text="Body text.",
        content_type="text/html",
        bytes=42,
    )
    path = write_fetched_to_state(article, str(tmp_path))

    assert path.parent == tmp_path / "articles"
    assert path.name.startswith("example.com-")
    on_disk = json.loads(path.read_text(encoding="utf-8"))
    assert on_disk["title"] == "Title"
    assert FetchedArticle.model_validate(on_disk) == article


def test_write_fetched_to_state_creates_missing_subdir(tmp_path: Path) -> None:
    nested = tmp_path / "deep"
    write_fetched_to_state(
        FetchedArticle(url="https://x.example.com/", final_url="https://x.example.com/"),
        str(nested),
    )
    assert (nested / "articles").exists()


async def test_tool_writes_artifact_and_returns_compressed_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    html = '<html><head><meta property="og:title" content="Real Title" /></head><body><p>Real body.</p></body></html>'

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "text/html"}, content=html.encode("utf-8"))

    _patch_httpx(monkeypatch, handler)
    settings = _settings(tmp_path)
    tool = build_fetch_article_tool(settings)

    result = json.loads(await tool.ainvoke({"url": "https://example.com/post"}))

    assert set(result) == {"url", "final_url", "title", "status", "reason", "path"}
    assert result["status"] == "ok"
    assert result["reason"] is None
    assert result["title"] == "Real Title"
    assert Path(result["path"]).parent == Path(settings.orchestrator_data_dir) / "articles"
    # The artifact is on disk; the compressed summary does NOT carry the body.
    assert "text" not in result
    artifact = json.loads(Path(result["path"]).read_text(encoding="utf-8"))
    assert artifact["title"] == "Real Title"


async def test_tool_summary_reports_blocked_without_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # No httpx patch -> Redirect-no-op; instead use a literal private IP URL so
    # the SSRF guard fires before any network IO.
    settings = _settings(tmp_path)
    tool = build_fetch_article_tool(settings)

    result = json.loads(await tool.ainvoke({"url": "http://10.0.0.5/internal"}))

    assert result["status"] == "blocked"
    assert set(result) == {"url", "final_url", "title", "status", "reason", "path"}
    assert result["final_url"] is None
    assert result["title"] is None
    assert result["path"] is None
    assert result["reason"]  # populated
    # The echoed URL is normalized (no fragments/whitespace) across all branches.
    assert result["url"] == "http://10.0.0.5/internal"
    # Nothing written to the articles dir.
    assert not (Path(settings.orchestrator_data_dir) / "articles").exists()


async def test_tool_summary_reports_not_html_without_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "image/png"}, content=b"\x89PNG\r\n")

    _patch_httpx(monkeypatch, handler)
    settings = _settings(tmp_path)
    tool = build_fetch_article_tool(settings)

    result = json.loads(await tool.ainvoke({"url": "https://example.com/img.png"}))

    assert result["status"] == "not_html"
    assert "reason" in result
    assert "text" not in result  # no artifact in the summary


def test_default_singleton_is_a_structured_tool() -> None:
    assert fetch_article_tool.name == "fetch_article"
    assert fetch_article_tool.args_schema is not None