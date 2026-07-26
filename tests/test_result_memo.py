"""The research tools' per-run memo.

From the 2026-07-26 trace: inside one research fan-out, one article was fetched
8 times, another 6, and whole 4-query search sets were re-issued verbatim ~90s
apart. Two topics burned their full wall-clock budget partly on that churn. The
memo answers a repeat from memory and — the part that actually breaks the loop —
tells the model it is repeating itself.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from app.orchestrator.budgets import ResultMemo
from app.orchestrator.tools.fetch import build_fetch_article_tool
from app.orchestrator.tools.web import build_web_extract_tool, build_web_search_tool
from support import settings_for


# --------------------------------------------------------------------------- #
# The primitive
# --------------------------------------------------------------------------- #


def test_memo_returns_none_before_the_first_result() -> None:
    assert ResultMemo().replay("k") is None


def test_memo_marks_a_replay_without_disturbing_the_payload() -> None:
    memo = ResultMemo()
    memo.put("k", {"status": "ok", "path": "/tmp/a.json"})

    replayed = memo.replay("k")

    assert replayed is not None
    assert replayed["status"] == "ok"
    assert replayed["path"] == "/tmp/a.json"
    assert replayed["repeated"] is True
    assert "already ran this exact call" in replayed["note"]


def test_memo_keys_are_independent() -> None:
    memo = ResultMemo()
    memo.put("a", {"n": 1})
    memo.put("b", {"n": 2})

    assert memo.replay("a")["n"] == 1  # type: ignore[index]
    assert memo.replay("b")["n"] == 2  # type: ignore[index]


# --------------------------------------------------------------------------- #
# Wired into the tools
# --------------------------------------------------------------------------- #


def test_fetch_article_hits_the_network_once_per_url(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []

    async def counting_fetch(url, settings):  # type: ignore[no-untyped-def]
        calls.append(url)
        raise RuntimeError("network reached")

    monkeypatch.setattr("app.orchestrator.tools.fetch.fetch_article", counting_fetch)
    tool = build_fetch_article_tool(settings_for(tmp_path))

    first = json.loads(asyncio.run(tool.ainvoke({"url": "https://example.com/a"})))
    second = json.loads(asyncio.run(tool.ainvoke({"url": "https://example.com/a"})))

    assert len(calls) == 1
    assert "repeated" not in first
    assert second["repeated"] is True


def test_fetch_article_memo_normalizes_url_variants(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The researcher reaches the same page by several spellings; they must
    collapse onto one entry or the memo never fires."""
    calls: list[str] = []

    async def counting_fetch(url, settings):  # type: ignore[no-untyped-def]
        calls.append(url)
        raise RuntimeError("network reached")

    monkeypatch.setattr("app.orchestrator.tools.fetch.fetch_article", counting_fetch)
    tool = build_fetch_article_tool(settings_for(tmp_path))

    asyncio.run(tool.ainvoke({"url": "https://example.com/a?utm_source=x"}))
    asyncio.run(tool.ainvoke({"url": "https://example.com/a"}))

    assert len(calls) == 1


def test_a_different_url_is_not_a_memo_hit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []

    async def counting_fetch(url, settings):  # type: ignore[no-untyped-def]
        calls.append(url)
        raise RuntimeError("network reached")

    monkeypatch.setattr("app.orchestrator.tools.fetch.fetch_article", counting_fetch)
    tool = build_fetch_article_tool(settings_for(tmp_path))

    asyncio.run(tool.ainvoke({"url": "https://example.com/a"}))
    result = json.loads(asyncio.run(tool.ainvoke({"url": "https://example.com/b"})))

    assert len(calls) == 2
    assert "repeated" not in result


def test_memos_are_scoped_to_one_tool_instance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One run builds one tool; a second run builds a fresh one and must not
    inherit the first run's answers."""
    calls: list[str] = []

    async def counting_fetch(url, settings):  # type: ignore[no-untyped-def]
        calls.append(url)
        raise RuntimeError("network reached")

    monkeypatch.setattr("app.orchestrator.tools.fetch.fetch_article", counting_fetch)
    settings = settings_for(tmp_path)

    asyncio.run(build_fetch_article_tool(settings).ainvoke({"url": "https://e.com/a"}))
    asyncio.run(build_fetch_article_tool(settings).ainvoke({"url": "https://e.com/a"}))

    assert len(calls) == 2


def test_failed_fetches_are_remembered_too(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Deliberate: a URL that blocked will block again, and the researcher
    retries in a tight loop against a wall clock. Ending the loop is worth more
    than the chance the failure was transient."""
    calls: list[str] = []

    async def always_blocked(url, settings):  # type: ignore[no-untyped-def]
        from app.orchestrator.services.fetch_article import BlockedArticleError

        calls.append(url)
        raise BlockedArticleError("blocked host")

    monkeypatch.setattr("app.orchestrator.tools.fetch.fetch_article", always_blocked)
    tool = build_fetch_article_tool(settings_for(tmp_path))

    first = json.loads(asyncio.run(tool.ainvoke({"url": "https://example.com/x"})))
    second = json.loads(asyncio.run(tool.ainvoke({"url": "https://example.com/x"})))

    assert len(calls) == 1
    assert first["status"] == "blocked"
    assert second["status"] == "blocked"
    assert second["repeated"] is True


def test_web_search_memoizes_the_full_argument_triple(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same words with a different window is a different search, and must not
    be served from the memo."""
    calls: list[dict] = []

    async def counting_search(args, settings, circuit):  # type: ignore[no-untyped-def]
        calls.append(dict(args))
        return {"query": args["query"], "result_count": 0, "status": "ok", "path": "p"}

    monkeypatch.setattr("app.orchestrator.tools.web._run_search", counting_search)
    tool = build_web_search_tool(settings_for(tmp_path))

    asyncio.run(tool.ainvoke({"query": "kimi k3", "hours_back": 24}))
    repeat = json.loads(asyncio.run(tool.ainvoke({"query": "kimi k3", "hours_back": 24})))
    asyncio.run(tool.ainvoke({"query": "kimi k3", "hours_back": 168}))

    assert len(calls) == 2
    assert repeat["repeated"] is True


def test_web_extract_memo_ignores_url_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[list[str]] = []

    async def counting_extract(urls, settings):  # type: ignore[no-untyped-def]
        calls.append(list(urls))
        return {}

    monkeypatch.setattr(
        "app.orchestrator.tools.web.extract_url_texts", counting_extract
    )
    tool = build_web_extract_tool(settings_for(tmp_path))

    urls = ["https://a.com/1", "https://b.com/2"]
    asyncio.run(tool.ainvoke({"urls": urls}))
    repeat = json.loads(asyncio.run(tool.ainvoke({"urls": list(reversed(urls))})))

    assert len(calls) == 1
    assert repeat["repeated"] is True
