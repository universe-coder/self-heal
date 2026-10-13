"""Validate webhook URLs to reduce SSRF risk."""

from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlparse

_IPAddress = ipaddress.IPv4Address | ipaddress.IPv6Address


def _is_private_ip(ip: _IPAddress) -> bool:
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    return bool(
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_unspecified
        or ip.is_reserved
    )


def validate_webhook_url(url: str, *, allow_insecure: bool) -> None:
    """Reject non-http(s) schemes, plain http (unless allowed) and private/local targets."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"unsupported URL scheme: {parsed.scheme!r}")
    if parsed.scheme == "http" and not allow_insecure:
        raise ValueError("only https URLs are allowed (set allow_insecure=true for http)")
    host = parsed.hostname
    if not host:
        raise ValueError("URL must include a hostname")
    if parsed.username or parsed.password:
        raise ValueError("credentials in webhook URLs are not allowed; use headers or env")

    if allow_insecure:
        return

    lowered = host.lower().rstrip(".")
    if lowered in ("localhost", "0.0.0.0") or lowered.endswith(".localhost"):
        raise ValueError("localhost targets are not allowed")
    try:
        literal: _IPAddress | None = ipaddress.ip_address(lowered)
    except ValueError:
        literal = None
    if literal is not None:
        if _is_private_ip(literal):
            raise ValueError("private/link-local IP targets are not allowed")
        return
    try:
        infos = socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
    except socket.gaierror as e:
        raise ValueError(f"could not resolve host: {host}") from e
    for info in infos:
        try:
            ip = ipaddress.ip_address(info[4][0])
        except ValueError:
            continue
        if _is_private_ip(ip):
            raise ValueError("private/link-local IP targets are not allowed")
