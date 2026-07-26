"""SearXNG availability for a propose run.

``propose``'s research stage corroborates claims through a self-hosted SearXNG
instance. Two things can go wrong quietly: the instance isn't running at all, or
it answers but returns nothing (disabled engines, rate limiting, wrong
categories). Both degrade research into mock results without any obvious
failure, so this module starts the instance when it can and always probes it
loudly.
"""
from __future__ import annotations

import asyncio
import logging
import subprocess
from pathlib import Path

import httpx

from app.config import Settings

logger = logging.getLogger(__name__)

_COMPOSE_FILE = Path(__file__).resolve().parents[3] / "docker-compose.searxng.yml"
_DEFAULT_URL = "http://localhost:8080"
_PROBE_QUERY = "OpenAI"


async def probe(settings: Settings) -> int:
    """Run one real search against the configured instance and log the result
    count loudly. The 2026-07-25 run wasted 57 web_search calls on an instance
    that was silently returning zero results — nobody noticed until trace
    review. A 0-result probe now surfaces at run start; per-query empties during
    the run are handled by web_search's circuit breaker. Returns the probe's
    result count (0 also when the probe itself fails)."""
    from app.orchestrator.services.searxng_client import SearxngClient

    client = SearxngClient(settings)
    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(settings.request_timeout_seconds)
        ) as http:
            items = await client.search_news(
                client=http,
                query=_PROBE_QUERY,
                # Month bucket: the probe checks engine health, not recency.
                hours_back=24 * 31,
                max_results=5,
                dry_run=False,
            )
    except Exception as exc:
        logger.warning(
            "SearXNG probe failed (%s) — search evidence this run may be degraded.", exc
        )
        return 0
    if items:
        logger.info("SearXNG probe ok: %d result(s) for %r.", len(items), _PROBE_QUERY)
    else:
        logger.warning(
            "SearXNG probe returned 0 results for %r — the instance answers but "
            "produces no results (engines disabled? rate-limited? wrong "
            "categories?). Research corroboration will be degraded this run.",
            _PROBE_QUERY,
        )
    return len(items)


async def ensure_available(settings: Settings) -> None:
    """Best-effort auto-start of the self-hosted SearXNG instance
    (docker-compose.searxng.yml) so propose's web_search fallback does real
    corroborating searches instead of returning mock results. Only kicks in
    when SEARXNG_BASE_URL is unset — an operator already pointing it at their
    own instance is left alone (but still probed). Any failure here (no docker,
    container never becomes healthy) just logs a warning; web_search's existing
    mock-result mode is the fallback, not a hard error.

    Async-native: the compose call runs in a thread and the readiness poll
    awaits instead of sleeping, so the event loop is never blocked."""
    if (settings.searxng_base_url or "").strip():
        await probe(settings)
        return

    try:
        await asyncio.to_thread(
            subprocess.run,
            ["docker", "compose", "-f", str(_COMPOSE_FILE), "up", "-d"],
            check=True,
            capture_output=True,
            timeout=60,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        logger.warning(
            "Could not start SearXNG (%s) — web_search will use mock results.", exc
        )
        return

    async with httpx.AsyncClient(timeout=2) as client:
        for _ in range(15):
            try:
                if (await client.get(_DEFAULT_URL)).status_code == 200:
                    settings.searxng_base_url = _DEFAULT_URL
                    logger.info("SearXNG ready at %s", _DEFAULT_URL)
                    await probe(settings)
                    return
            except httpx.HTTPError:
                pass
            await asyncio.sleep(1)

    logger.warning(
        "SearXNG container started but didn't become healthy in time — "
        "web_search will use mock results."
    )


__all__ = ["ensure_available", "probe"]
