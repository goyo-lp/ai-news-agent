from __future__ import annotations

import ipaddress

import httpx

_BLOCKED_HOSTNAMES = {
    "metadata.google.internal",
    "metadata.internal",
    "metadata.azure.com",
}


class BlockedHostError(httpx.HTTPError):
    """Raised when a request targets a private/loopback/link-local/reserved address."""


class ResponseTooLargeError(httpx.HTTPError):
    """Raised when a streamed response body exceeds the configured byte cap."""


def _is_blocked_ip(addr: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    return (
        addr.is_private
        or addr.is_loopback
        or addr.is_link_local
        or addr.is_reserved
        or addr.is_multicast
        or addr.is_unspecified
    )


def is_blocked_host(hostname: str | None) -> bool:
    """True if hostname is a literal private/loopback/link-local/reserved IP, or a
    known cloud metadata hostname.

    This is a literal-address check only — it does not resolve DNS, so a domain name
    that *resolves* to an internal address (DNS-based SSRF / DNS rebinding) is not
    caught here. It blocks the common case of a fetched URL embedding a raw internal
    or metadata-service address directly.
    """
    if not hostname:
        return True
    lowered = hostname.strip("[]").lower()
    if lowered in _BLOCKED_HOSTNAMES:
        return True
    try:
        addr = ipaddress.ip_address(lowered)
    except ValueError:
        return False
    return _is_blocked_ip(addr)


async def reject_private_network_requests(request: httpx.Request) -> None:
    """httpx request event hook: block requests to private/internal hosts.

    Runs on every request the client sends, including redirect hops (httpx invokes
    request hooks per-hop), so a public URL that redirects to an internal address is
    also caught. Intended for httpx.AsyncClient(event_hooks={"request": [...]})  on
    clients that fetch attacker-influenced URLs (e.g. article links sourced from RSS
    feeds/aggregators).
    """
    host = request.url.host
    if is_blocked_host(host):
        raise BlockedHostError(f"Refusing to fetch private/internal host: {host}")


async def get_capped(
    client: httpx.AsyncClient,
    url: str,
    headers: dict[str, str],
    max_bytes: int,
) -> httpx.Response:
    """GET a URL, aborting once the body exceeds max_bytes.

    Enforces the cap against bytes actually received, rather than trusting the
    (advisory, spoofable-or-absent) Content-Length header after a non-streaming
    get() has already buffered the whole body in memory.
    """
    async with client.stream("GET", url, headers=headers) as response:
        response.raise_for_status()
        downloaded = 0
        async for chunk in response.aiter_bytes():
            downloaded += len(chunk)
            if downloaded > max_bytes:
                raise ResponseTooLargeError(f"Response from {url} exceeded {max_bytes} bytes")
    return response
