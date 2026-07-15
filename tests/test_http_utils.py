from __future__ import annotations

import httpx

from app.services.http_utils import (
    BlockedHostError,
    ResponseTooLargeError,
    get_capped,
    is_blocked_host,
    reject_private_network_requests,
)


def test_is_blocked_host_flags_loopback_link_local_and_metadata() -> None:
    assert is_blocked_host("127.0.0.1")
    assert is_blocked_host("169.254.169.254")
    assert is_blocked_host("10.0.0.5")
    assert is_blocked_host("192.168.1.1")
    assert is_blocked_host("::1")
    assert is_blocked_host("metadata.google.internal")
    assert is_blocked_host(None)


def test_is_blocked_host_allows_public_hosts() -> None:
    assert not is_blocked_host("example.com")
    assert not is_blocked_host("8.8.8.8")


async def test_reject_private_network_requests_blocks_redirect_to_private_ip() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "public.example.com":
            return httpx.Response(302, headers={"location": "http://169.254.169.254/secret"})
        return httpx.Response(200, text="unreachable")

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(
        transport=transport,
        follow_redirects=True,
        event_hooks={"request": [reject_private_network_requests]},
    ) as client:
        try:
            await client.get("http://public.example.com/start")
            assert False, "expected BlockedHostError"
        except BlockedHostError:
            pass


async def test_get_capped_returns_response_within_limit() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="hello world")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        response = await get_capped(client, "http://example.com", {}, max_bytes=1000)

    assert response.text == "hello world"


async def test_get_capped_raises_when_body_exceeds_max_bytes() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="x" * 1000)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        try:
            await get_capped(client, "http://example.com", {}, max_bytes=100)
            assert False, "expected ResponseTooLargeError"
        except ResponseTooLargeError:
            pass
