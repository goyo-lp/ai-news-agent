from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from app.config import Settings
from app.orchestrator.services.ranking import DiscoveredItem
from app.orchestrator.tools import web as tools_mod
from app.orchestrator.tools.web import (
    build_web_extract_tool,
    build_web_search_tool,
    web_extract_tool,
    web_search_tool,
    write_extract_to_state,
    write_search_to_state,
)


def _settings(tmp_path: Path, *, base_url: str = "") -> Settings:
    return Settings(
        _env_file=None,
        orchestrator_data_dir=str(tmp_path),
        searxng_base_url=base_url,
        request_timeout_seconds=10,
    )


def _patch_httpx(monkeypatch: pytest.MonkeyPatch, handler) -> None:
    """The search tool constructs its own httpx.AsyncClient inside _run_search,
    so patch the AsyncClient symbol on the web tools module. Capture the real
    AsyncClient first to avoid a recursion bug."""
    real = httpx.AsyncClient

    def _factory(*args: object, **kwargs: object) -> httpx.AsyncClient:
        kwargs["transport"] = httpx.MockTransport(handler)
        return real(**kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(tools_mod.httpx, "AsyncClient", _factory)


def test_write_search_to_state_round_trips_discovered_items(tmp_path: Path) -> None:
    items = [
        DiscoveredItem(id="a", title="T1", url="https://a.example/x", domain="a.example.com", query="q"),
        DiscoveredItem(id="b", title="T2", url="https://b.example/y", domain="b.example.com", query="q"),
    ]
    path = write_search_to_state(items, "q", str(tmp_path))

    assert path.parent == tmp_path / "web" / "search"
    assert path.name.endswith(".json")
    on_disk = json.loads(path.read_text(encoding="utf-8"))
    assert len(on_disk) == 2
    assert DiscoveredItem.model_validate(on_disk[0]) == items[0]


def test_write_extract_to_state_round_trips_url_text_map(tmp_path: Path) -> None:
    extracted = {"https://a.example/x": "text A", "https://b.example/y": "text B"}
    urls = list(extracted.keys())
    path = write_extract_to_state(extracted, urls, str(tmp_path))

    assert path.parent == tmp_path / "web" / "extracted"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["urls"] == urls
    assert payload["results"] == extracted


async def test_search_tool_dry_run_writes_mock_results_and_compressed_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # No SEARXNG_BASE_URL -> dry-run mock; no network.
    settings = _settings(tmp_path, base_url="")
    tool = build_web_search_tool(settings)

    result = json.loads(await tool.ainvoke({"query": "ai agents", "max_results": 2}))

    assert set(result) == {"query", "result_count", "status", "reason", "path"}
    assert result["status"] == "ok"
    assert result["reason"] is None
    assert result["query"] == "ai agents"
    assert result["result_count"] == 2
    assert Path(result["path"]).parent == Path(settings.orchestrator_data_dir) / "web" / "search"
    # The summary does NOT carry the result list — that's on disk only.
    assert "results" not in result
    on_disk = json.loads(Path(result["path"]).read_text(encoding="utf-8"))
    assert len(on_disk) == 2


async def test_search_tool_empty_query_returns_error_without_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings(tmp_path, base_url="")
    tool = build_web_search_tool(settings)

    result = json.loads(await tool.ainvoke({"query": "   "}))
    assert result["status"] == "error"
    assert result["query"] == ""
    assert "required" in result["reason"]
    # Nothing written.
    assert not (Path(settings.orchestrator_data_dir) / "web").exists()


async def test_search_tool_real_call_parses_and_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    response = {
        "results": [
            {
                "title": "OpenAI launches new model",
                "url": "https://openai.com/news/x",
                "content": "snippet",
                "publishedDate": "2025-01-15T12:30:45Z",
            }
        ]
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=response)

    _patch_httpx(monkeypatch, handler)
    settings = _settings(tmp_path, base_url="https://searxng.example")
    tool = build_web_search_tool(settings)

    result = json.loads(await tool.ainvoke({"query": "openai", "hours_back": 24, "max_results": 1}))

    assert result["status"] == "ok"
    assert result["result_count"] == 1
    artifact = json.loads(Path(result["path"]).read_text(encoding="utf-8"))
    assert artifact[0]["title"] == "OpenAI launches new model"


async def test_search_tool_translates_http_failure_to_error_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "boom"})

    _patch_httpx(monkeypatch, handler)
    settings = _settings(tmp_path, base_url="https://searxng.example")
    tool = build_web_search_tool(settings)

    result = json.loads(await tool.ainvoke({"query": "openai", "max_results": 1}))

    assert result["status"] == "error"
    assert result["path"] is None
    assert result["result_count"] == 0
    assert "reason" in result and result["reason"]


async def test_search_tool_respects_explicit_zero_args(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An explicit `max_results=0` is a real value, not 'unset'. The None-vs-0
    check respects the explicit zero: zero mock templates returned."""
    settings = _settings(tmp_path, base_url="")
    tool = build_web_search_tool(settings)

    result = json.loads(await tool.ainvoke({"query": "q", "max_results": 0}))
    assert result["status"] == "ok"
    assert result["result_count"] == 0  # zero template slices, not the default 8


# --- extract tool -----------------------------------------------------------
#
# The extract tool now wraps the local extract_url_texts service (SSRF-guarded
# fetch + trafilatura); the service itself is covered end-to-end in
# test_web_extract.py. Here we stub it to test the *tool's* orchestration:
# input validation, dedupe, honest success_count, persistence, and summary
# shape.


def _stub_extract(monkeypatch: pytest.MonkeyPatch, mapping: dict[str, str]) -> None:
    async def _fake(urls: list[str], settings: Settings, *, dry_run: bool = False) -> dict[str, str]:
        return {u: mapping[u] for u in urls if u in mapping}

    monkeypatch.setattr(tools_mod, "extract_url_texts", _fake)


async def test_extract_tool_writes_extracted_text_and_compressed_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_extract(
        monkeypatch,
        {"https://a.example/x": "Real text A.", "https://b.example/y": "Real text B."},
    )
    settings = _settings(tmp_path)
    tool = build_web_extract_tool(settings)

    result = json.loads(
        await tool.ainvoke({"urls": ["https://a.example/x", "https://b.example/y", "https://a.example/x"]})
    )

    assert set(result) == {"url_count", "success_count", "status", "reason", "path"}
    assert result["status"] == "ok"
    assert result["reason"] is None
    # Three input URLs, two distinct after normalization.
    assert result["url_count"] == 2
    assert result["success_count"] == 2
    # The summary does NOT carry the extracted text — that's on disk only.
    assert "results" not in result
    artifact = json.loads(Path(result["path"]).read_text(encoding="utf-8"))
    assert artifact["results"]["https://a.example/x"] == "Real text A."
    assert Path(result["path"]).parent == Path(settings.orchestrator_data_dir) / "web" / "extracted"


async def test_extract_tool_empty_urls_returns_error_without_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings(tmp_path)
    tool = build_web_extract_tool(settings)

    result = json.loads(await tool.ainvoke({"urls": []}))

    assert result["status"] == "error"
    assert result["path"] is None
    assert "required" in result["reason"]
    assert not (Path(settings.orchestrator_data_dir) / "web").exists()


async def test_extract_tool_rejects_only_whitespace_urls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings(tmp_path)
    tool = build_web_extract_tool(settings)

    result = json.loads(await tool.ainvoke({"urls": ["   ", "   "]}))

    assert result["status"] == "error"
    assert "no valid URLs" in result["reason"]


async def test_extract_tool_partial_success_reports_success_count_honestly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only one URL yielded extractable text (the other was unreachable/blocked
    and skipped inside the service) — that's `status=ok`, `success_count=1`, NOT
    an error. A skipped URL is not a failure."""
    _stub_extract(monkeypatch, {"https://a.example/x": "Real text A."})
    settings = _settings(tmp_path)
    tool = build_web_extract_tool(settings)

    result = json.loads(
        await tool.ainvoke({"urls": ["https://a.example/x", "https://b.example/y"]})
    )

    assert result["status"] == "ok"
    assert result["success_count"] == 1
    assert result["url_count"] == 2


async def test_extract_tool_service_exception_reports_error_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unexpected service-level exception surfaces as `status=error` (no
    artifact) rather than propagating into the agent loop. Individual URL
    failures are skipped inside the service; only a whole-call fault errors."""

    async def _boom(urls: list[str], settings: Settings, *, dry_run: bool = False) -> dict[str, str]:
        raise RuntimeError("extractor blew up")

    monkeypatch.setattr(tools_mod, "extract_url_texts", _boom)
    settings = _settings(tmp_path)
    tool = build_web_extract_tool(settings)

    result = json.loads(await tool.ainvoke({"urls": ["https://a.example/x"]}))

    assert result["status"] == "error"
    assert result["success_count"] == 0
    assert result["path"] is None
    assert result["reason"]


def test_default_singletons_are_structured_tools() -> None:
    assert web_search_tool.name == "web_search"
    assert web_extract_tool.name == "web_extract"
    assert web_search_tool.args_schema is not None
    assert web_extract_tool.args_schema is not None