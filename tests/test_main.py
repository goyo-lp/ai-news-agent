"""The `both` CLI command: runs the news digest then the LinkedIn proposal
flow, combining their exit codes. Stubs both lanes so this stays a pure
dispatch/exit-code test, not a re-test of either lane's own behavior."""
from __future__ import annotations

import argparse
import asyncio

import httpx
import pytest

from app import main
from app.config import Settings


def _args(**overrides: object) -> argparse.Namespace:
    defaults = {"dry_run": False, "limit": None, "force": False, "verbose": False}
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


@pytest.mark.parametrize(
    ("run_exit", "propose_exit", "expected"),
    [(0, 0, 0), (1, 0, 1), (0, 2, 2), (1, 1, 1)],
)
def test_run_both_combines_exit_codes(
    monkeypatch: pytest.MonkeyPatch, run_exit: int, propose_exit: int, expected: int
) -> None:
    calls: list[str] = []

    async def fake_run_pipeline(args: argparse.Namespace) -> int:
        calls.append("run")
        return run_exit

    async def fake_run_propose(args: argparse.Namespace) -> int:
        calls.append("propose")
        return propose_exit

    monkeypatch.setattr(main, "run_pipeline", fake_run_pipeline)
    monkeypatch.setattr(main, "run_propose", fake_run_propose)

    result = asyncio.run(main.run_both(_args()))

    assert result == expected
    assert calls == ["run", "propose"]


class _FakeResponse:
    def __init__(self, status_code: int) -> None:
        self.status_code = status_code


class _FakeHealthCheckClient:
    """Stand-in for httpx.Client used by _ensure_searxng's readiness poll."""

    def __init__(self, *, status_code: int = 200, raises: bool = False) -> None:
        self._status_code = status_code
        self._raises = raises

    def __enter__(self) -> "_FakeHealthCheckClient":
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def get(self, url: str) -> _FakeResponse:
        if self._raises:
            raise httpx.ConnectError("connection refused")
        return _FakeResponse(self._status_code)


def _searxng_settings(base_url: str = "") -> Settings:
    return Settings(_env_file=None, searxng_base_url=base_url)  # type: ignore[arg-type]


def test_ensure_searxng_skips_docker_when_already_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _searxng_settings("http://example.com:9000")
    called = False

    def fake_run(*args: object, **kwargs: object) -> None:
        nonlocal called
        called = True

    monkeypatch.setattr(main.subprocess, "run", fake_run)

    main._ensure_searxng(settings)

    assert not called
    assert settings.searxng_base_url == "http://example.com:9000"


def test_ensure_searxng_sets_url_once_container_is_healthy(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _searxng_settings()
    monkeypatch.setattr(main.subprocess, "run", lambda *a, **k: None)
    monkeypatch.setattr(main.time, "sleep", lambda _: None)
    monkeypatch.setattr(main.httpx, "Client", lambda **kwargs: _FakeHealthCheckClient(status_code=200))

    main._ensure_searxng(settings)

    assert settings.searxng_base_url == main._SEARXNG_DEFAULT_URL


def test_ensure_searxng_leaves_url_empty_when_docker_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _searxng_settings()

    def fake_run(*args: object, **kwargs: object) -> None:
        raise FileNotFoundError("docker not found")

    monkeypatch.setattr(main.subprocess, "run", fake_run)

    main._ensure_searxng(settings)

    assert settings.searxng_base_url == ""


def test_ensure_searxng_leaves_url_empty_when_never_healthy(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _searxng_settings()
    monkeypatch.setattr(main.subprocess, "run", lambda *a, **k: None)
    monkeypatch.setattr(main.time, "sleep", lambda _: None)
    monkeypatch.setattr(main.httpx, "Client", lambda **kwargs: _FakeHealthCheckClient(raises=True))

    main._ensure_searxng(settings)

    assert settings.searxng_base_url == ""
