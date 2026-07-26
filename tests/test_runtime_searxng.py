"""SearXNG availability for a propose run: auto-start, readiness poll, and the
"leave a configured instance alone" rule. The health probe is stubbed
throughout — these tests pin the lifecycle, not the search client."""
from __future__ import annotations

import httpx
import pytest

from app.config import Settings
from app.runtime import searxng


class _FakeResponse:
    def __init__(self, status_code: int) -> None:
        self.status_code = status_code


class _FakeHealthCheckClient:
    """Stand-in for httpx.AsyncClient used by ensure_available's readiness poll."""

    def __init__(self, *, status_code: int = 200, raises: bool = False) -> None:
        self._status_code = status_code
        self._raises = raises

    async def __aenter__(self) -> "_FakeHealthCheckClient":
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    async def get(self, url: str) -> _FakeResponse:
        if self._raises:
            raise httpx.ConnectError("connection refused")
        return _FakeResponse(self._status_code)


def _settings(base_url: str = "") -> Settings:
    return Settings(_env_file=None, searxng_base_url=base_url)  # type: ignore[arg-type]


async def _noop_sleep(_seconds: float) -> None:
    return None


@pytest.fixture(autouse=True)
def _stub_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    """The probe issues a real search; stub it so these tests stay offline."""

    async def _fake_probe(_settings: Settings) -> int:
        return 1

    monkeypatch.setattr(searxng, "probe", _fake_probe)


async def test_skips_docker_when_already_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    """An operator pointing at their own instance is left alone."""
    settings = _settings("http://example.com:9000")
    called = False

    def fake_run(*args: object, **kwargs: object) -> None:
        nonlocal called
        called = True

    monkeypatch.setattr(searxng.subprocess, "run", fake_run)

    await searxng.ensure_available(settings)

    assert not called
    assert settings.searxng_base_url == "http://example.com:9000"


async def test_sets_url_once_container_is_healthy(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings()
    monkeypatch.setattr(searxng.subprocess, "run", lambda *a, **k: None)
    monkeypatch.setattr(searxng.asyncio, "sleep", _noop_sleep)
    monkeypatch.setattr(
        searxng.httpx, "AsyncClient", lambda **kwargs: _FakeHealthCheckClient(status_code=200)
    )

    await searxng.ensure_available(settings)

    assert settings.searxng_base_url == searxng._DEFAULT_URL


async def test_leaves_url_empty_when_docker_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """No docker -> a warning, not a crash; web_search's mock mode is the
    fallback."""
    settings = _settings()

    def fake_run(*args: object, **kwargs: object) -> None:
        raise FileNotFoundError("docker not found")

    monkeypatch.setattr(searxng.subprocess, "run", fake_run)

    await searxng.ensure_available(settings)

    assert settings.searxng_base_url == ""


async def test_leaves_url_empty_when_never_healthy(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings()
    monkeypatch.setattr(searxng.subprocess, "run", lambda *a, **k: None)
    monkeypatch.setattr(searxng.asyncio, "sleep", _noop_sleep)
    monkeypatch.setattr(
        searxng.httpx, "AsyncClient", lambda **kwargs: _FakeHealthCheckClient(raises=True)
    )

    await searxng.ensure_available(settings)

    assert settings.searxng_base_url == ""
