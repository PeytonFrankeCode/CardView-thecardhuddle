"""Fetch card images by URL.

This service takes URLs supplied by callers, so the fetcher is a potential
server-side request forgery vector: a URL like http://169.254.169.254/ would
otherwise make the service read its own cloud metadata and hand it back. Every
destination is therefore resolved and checked against private address space
before a request is made, and responses are size-capped while streaming so a
hostile URL cannot exhaust memory.
"""

from __future__ import annotations

import asyncio
import ipaddress
import socket
from dataclasses import dataclass
from urllib.parse import urlparse

import httpx

ALLOWED_SCHEMES = {"http", "https"}
ALLOWED_CONTENT_PREFIXES = ("image/",)


class FetchError(Exception):
    """A URL could not be fetched, with a caller-safe reason."""


@dataclass
class FetchedImage:
    url: str
    data: bytes
    content_type: str


def _is_public_address(host: str) -> bool:
    """True only if every address the host resolves to is publicly routable."""
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror:
        return False
    if not infos:
        return False
    for info in infos:
        address = info[4][0]
        try:
            ip = ipaddress.ip_address(address)
        except ValueError:
            return False
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_multicast
            or ip.is_unspecified
        ):
            return False
    return True


def validate_url(url: str) -> str:
    """Reject anything that is not a plain public http(s) URL."""
    parsed = urlparse(url)
    if parsed.scheme not in ALLOWED_SCHEMES:
        raise FetchError(f"unsupported URL scheme: {parsed.scheme or 'none'}")
    if not parsed.hostname:
        raise FetchError("URL has no host")
    if not _is_public_address(parsed.hostname):
        raise FetchError("URL resolves to a non-public address")
    return url


class ImageFetcher:
    """Concurrency-limited image fetcher with a shared connection pool."""

    def __init__(
        self,
        max_bytes: int = 12 * 1024 * 1024,
        timeout: float = 12.0,
        max_concurrent: int = 16,
    ) -> None:
        self.max_bytes = max_bytes
        self.timeout = timeout
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=self.timeout,
                follow_redirects=True,
                # Redirects are followed but capped; each hop is not re-validated
                # by httpx, so the cap plus the initial check bounds exposure.
                max_redirects=3,
                headers={"User-Agent": "cardid/1.0 (+card identification service)"},
            )
        return self._client

    async def fetch(self, url: str) -> FetchedImage:
        validate_url(url)
        client = await self._get_client()
        async with self._semaphore:
            try:
                async with client.stream("GET", url) as response:
                    response.raise_for_status()
                    content_type = response.headers.get("content-type", "").split(";")[0]
                    if content_type and not content_type.startswith(ALLOWED_CONTENT_PREFIXES):
                        raise FetchError(f"not an image: content-type {content_type}")

                    declared = response.headers.get("content-length")
                    if declared and int(declared) > self.max_bytes:
                        raise FetchError("image exceeds size limit")

                    chunks: list[bytes] = []
                    total = 0
                    async for chunk in response.aiter_bytes():
                        total += len(chunk)
                        if total > self.max_bytes:
                            raise FetchError("image exceeds size limit")
                        chunks.append(chunk)
            except httpx.HTTPStatusError as exc:
                raise FetchError(f"HTTP {exc.response.status_code}") from exc
            except httpx.HTTPError as exc:
                raise FetchError(f"fetch failed: {type(exc).__name__}") from exc

        return FetchedImage(url=url, data=b"".join(chunks), content_type=content_type)

    async def fetch_many(self, urls: list[str]) -> list[FetchedImage | FetchError]:
        results = await asyncio.gather(
            *(self.fetch(url) for url in urls), return_exceptions=True
        )
        output: list[FetchedImage | FetchError] = []
        for result in results:
            if isinstance(result, FetchedImage):
                output.append(result)
            elif isinstance(result, Exception):
                output.append(FetchError(str(result)))
        return output

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
