from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from app.config import Settings
from app.orchestrator.services import tavily_client as svc_mod
from app.orchestrator.services.ranking import DiscoveredItem
from app.orchestrator.tools.tavily import (
    build_tavily_extract_tool,
    build_tavily_search_tool,
    tavily_extract_tool,
    tavily_search_tool,
    write_extract_to_state,
    write_search_to_state,
)


def _settings(tmp_path: Path, *, api_key: str | None = None) -> Settings:
    return Settings(
        _env_file=None,
        orchestrator_data_dir=str(tmp_path),
        tavily_api_key=api_key,
        tavily_base_url="https://api.tavily.example",
        request_timeout_seconds=10,
    )


def _patch_httpx(monkeypatch: pytest.MonkeyPatch, handler) -> None:
    """The tools construct their own httpx.AsyncClient internally (just like
    search_many), so patch the AsyncClient symbol on the tavily_client module
    that they route through. Capture the real AsyncClient first to avoid the
    recursion bug seen earlier in P2.2."""
    real = httpx.AsyncClient

    def _factory(*args: object, **kwargs: object) -> httpx.AsyncClient:
        kwargs["transport"] = httpx.MockTransport(handler)
        return real(**kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(svc_mod.httpx, "AsyncClient", _factory)


def test_write_search_to_state_round_trips_discovered_items(tmp_path: Path) -> None:
    items = [
        DiscoveredItem(id="a", title="T1", url="https://a.example/x", domain="a.example.com", query="q"),
        DiscoveredItem(id="b", title="T2", url="https://b.example/y", domain="b.example.com", query="q"),
    ]
    path = write_search_to_state(items, "q", str(tmp_path))

    assert path.parent == tmp_path / "tavily" / "search"
    assert path.name.endswith(".json")
    on_disk = json.loads(path.read_text(encoding="utf-8"))
    assert len(on_disk) == 2
    assert DiscoveredItem.model_validate(on_disk[0]) == items[0]


def test_write_extract_to_state_round_trips_url_text_map(tmp_path: Path) -> None:
    extracted = {"https://a.example/x": "text A", "https://b.example/y": "text B"}
    urls = list(extracted.keys())
    path = write_extract_to_state(extracted, urls, str(tmp_path))

    assert path.parent == tmp_path / "tavily" / "extracted"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["urls"] == urls
    assert payload["results"] == extracted


async def test_search_tool_dry_run_writes_mock_results_and_compressed_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # No httpx patch: dry-run path doesn't touch the network.
    settings = _settings(tmp_path, api_key=None)
    tool = build_tavily_search_tool(settings)

    result = json.loads(await tool.ainvoke({"query": "ai agents", "max_results": 2}))

    assert set(result) == {"query", "result_count", "status", "reason", "path"}
    assert result["status"] == "ok"
    assert result["reason"] is None
    assert result["query"] == "ai agents"
    assert result["result_count"] == 2
    assert Path(result["path"]).parent == Path(settings.orchestrator_data_dir) / "tavily" / "search"
    # The summary does NOT carry the result list — that's on disk only.
    assert "results" not in result
    on_disk = json.loads(Path(result["path"]).read_text(encoding="utf-8"))
    assert len(on_disk) == 2


async def test_search_tool_empty_query_returns_error_without_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings(tmp_path, api_key=None)
    tool = build_tavily_search_tool(settings)

    result = json.loads(await tool.ainvoke({"query": "   "}))
    assert result["status"] == "error"
    assert result["query"] == ""
    assert "required" in result["reason"]
    # Nothing written.
    assert not (Path(settings.orchestrator_data_dir) / "tavily").exists()


async def test_search_tool_real_call_parses_and_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    response = {
        "results": [
            {
                "title": "OpenAI launches new model",
                "url": "https://openai.com/news/x",
                "content": "snippet",
                "raw_content": "body text",
                "published_date": "2025-01-15T12:30:45Z",
            }
        ]
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=response)

    _patch_httpx(monkeypatch, handler)
    settings = _settings(tmp_path, api_key="sk-test")
    tool = build_tavily_search_tool(settings)

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
    settings = _settings(tmp_path, api_key="sk-test")
    tool = build_tavily_search_tool(settings)

    result = json.loads(await tool.ainvoke({"query": "openai", "max_results": 1}))

    assert result["status"] == "error"
    assert result["path"] is None
    assert result["result_count"] == 0
    assert "reason" in result and result["reason"]


async def test_search_tool_respects_explicit_zero_args(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """P2.3 review BLOCKER #1: an explicit `hours_back=0` or `max_results=0` is
    a real value, not 'unset'. The truthy-`or`-fallback pattern silently
    coerced both to defaults (24/8); the None-vs-0 check respects the explicit
    zero. ``max_results=0`` => zero mock templates returned (deterministic)."""
    settings = _settings(tmp_path, api_key=None)
    tool = build_tavily_search_tool(settings)

    result = json.loads(await tool.ainvoke({"query": "q", "max_results": 0}))
    assert result["status"] == "ok"
    assert result["result_count"] == 0  # zero template slices, not the default 8


# --- extract tool -----------------------------------------------------------


async def test_extract_tool_dry_run_writes_mock_text_map(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings(tmp_path, api_key=None)
    tool = build_tavily_extract_tool(settings)

    result = json.loads(
        await tool.ainvoke({"urls": ["https://a.example/x", "https://b.example/y", "https://a.example/x"]})
    )

    assert set(result) == {"url_count", "success_count", "status", "reason", "path"}
    assert result["status"] == "ok"
    assert result["reason"] is None
    # Three input URLs, two distinct after normalization.
    assert result["url_count"] == 2
    assert result["success_count"] == 2
    artifact = json.loads(Path(result["path"]).read_text(encoding="utf-8"))
    assert set(artifact["results"]) == {"https://a.example/x", "https://b.example/y"}


async def test_extract_tool_empty_urls_returns_error_without_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings(tmp_path, api_key=None)
    tool = build_tavily_extract_tool(settings)

    result = json.loads(await tool.ainvoke({"urls": []}))

    assert result["status"] == "error"
    assert result["path"] is None
    assert "required" in result["reason"]
    assert not (Path(settings.orchestrator_data_dir) / "tavily").exists()


async def test_extract_tool_rejects_only_whitespace_urls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings(tmp_path, api_key=None)
    tool = build_tavily_extract_tool(settings)

    result = json.loads(await tool.ainvoke({"urls": ["   ", "   "]}))

    assert result["status"] == "error"
    assert "no valid URLs" in result["reason"]


async def test_extract_tool_real_call_writes_extracted_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    response = {
        "results": [
            {"url": "https://a.example/x", "raw_content": "Real text A."},
            {"url": "https://b.example/y", "content": "Real text B."},
        ]
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=response)

    _patch_httpx(monkeypatch, handler)
    settings = _settings(tmp_path, api_key="sk-test")
    tool = build_tavily_extract_tool(settings)

    result = json.loads(
        await tool.ainvoke({"urls": ["https://a.example/x", "https://b.example/y"]})
    )

    assert result["status"] == "ok"
    assert result["success_count"] == 2
    artifact = json.loads(Path(result["path"]).read_text(encoding="utf-8"))
    assert artifact["results"]["https://a.example/x"] == "Real text A."


async def test_extract_tool_partial_success_reports_success_count_honestly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Endpoint answered, but only one URL produced extractable content —
    that's `status=ok`, `success_count=1` (NOT an error)."""
    response = {
        "results": [
            {"url": "https://a.example/x", "raw_content": "Real text A."},
            # b.example/y is absent from results: not extractable, or filtered.
        ]
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=response)

    _patch_httpx(monkeypatch, handler)
    settings = _settings(tmp_path, api_key="sk-test")
    tool = build_tavily_extract_tool(settings)

    result = json.loads(
        await tool.ainvoke({"urls": ["https://a.example/x", "https://b.example/y"]})
    )

    assert result["status"] == "ok"
    assert result["success_count"] == 1
    assert result["url_count"] == 2


async def test_extract_tool_endpoint_failure_reports_error_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """P2.3 review MAJOR #2: an HTTP 500 from the extract endpoint surfaces as
    `status=error`, distinct from a successful-but-empty extract."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "boom"})

    _patch_httpx(monkeypatch, handler)
    settings = _settings(tmp_path, api_key="sk-test")
    tool = build_tavily_extract_tool(settings)

    result = json.loads(await tool.ainvoke({"urls": ["https://a.example/x"]}))

    assert result["status"] == "error"
    assert result["success_count"] == 0
    assert result["path"] is None
    assert result["reason"]


def test_default_singletons_are_structured_tools() -> None:
    assert tavily_search_tool.name == "tavily_search"
    assert tavily_extract_tool.name == "tavily_extract"
    assert tavily_search_tool.args_schema is not None
    assert tavily_extract_tool.args_schema is not None