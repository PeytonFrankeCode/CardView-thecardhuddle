"""URL validation. The fetcher takes caller-supplied URLs, so this is security."""

from __future__ import annotations

import pytest

from cardid.fetch import FetchError, validate_url


@pytest.mark.parametrize(
    "url",
    [
        "http://169.254.169.254/latest/meta-data/",  # cloud metadata
        "http://127.0.0.1/x.png",
        "http://localhost:8000/x.png",
        "http://10.0.0.5/internal.png",
        "http://192.168.1.1/router.png",
    ],
)
def test_private_and_loopback_addresses_are_blocked(url):
    with pytest.raises(FetchError):
        validate_url(url)


@pytest.mark.parametrize("url", ["file:///etc/passwd", "ftp://example.com/x.png",
                                 "gopher://example.com/", "data:image/png;base64,AAA"])
def test_non_http_schemes_are_blocked(url):
    with pytest.raises(FetchError):
        validate_url(url)


def test_url_without_a_host_is_blocked():
    with pytest.raises(FetchError):
        validate_url("http:///nohost.png")


def test_unresolvable_host_is_blocked():
    with pytest.raises(FetchError):
        validate_url("http://this-host-does-not-exist.invalid/x.png")


def test_public_url_is_allowed():
    assert validate_url("https://i.ebayimg.com/images/g/abc/s-l1600.jpg")
