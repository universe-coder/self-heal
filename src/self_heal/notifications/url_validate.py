"""Validate webhook URLs to reduce SSRF risk."""

from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlparse


def _is_private_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    return bool(
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or getattr(ip, "is_reserved", False)  # py311+
    )


def validate_webhook_url(url: str, *, allow_insecure: bool) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"unsupported URL scheme: {parsed.scheme!r}")
    if parsed.scheme == "http" and not allow_insecure:
        raise ValueError("only https URLs are allowed (set allow_insecure=true for http)")
    host = parsed.hostname
    if not host:
        raise ValueError("URL must include a hostname")

    # Reject obvious SSRF targets when not explicitly allowed
    if not allow_insecure:
        lowered = host.lower()
        if lowered in ("localhost", "0.0.0.0"):
            raise ValueError("localhost targets are not allowed")
        try:
            infos = socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
        except socket.gaierror as e:
            raise ValueError(f"could not resolve host: {host}") from e
        for info in infos:
            sockaddr = info[4]
            addr = sockaddr[0]
            try:
                ip = ipaddress.ip_address(addr)
            except ValueError:
                continue
            if _is_private_ip(ip):
                raise ValueError("private/link-local IP targets are not allowed")
