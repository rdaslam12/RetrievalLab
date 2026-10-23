"""Auditable, bounded HTTP transport for user-selected adapter endpoints.

No proxy environment, redirects, retries, cookies, or response-body logging.
DNS is checked on every request and the socket uses the checked numeric address.
"""
from __future__ import annotations

from dataclasses import dataclass
import http.client
import ipaddress
import os
import re
import socket
import ssl
import time
from urllib.parse import urlsplit


class AdapterError(ValueError):
    """A safe, locally authored connection or contract error (never remote text)."""


MAX_RESPONSE_BYTES = 16 * 1024 * 1024
MAX_REQUEST_BYTES = 2 * 1024 * 1024
CONNECT_TIMEOUT = 5.0
READ_TIMEOUT = 30.0


@dataclass(frozen=True)
class Endpoint:
    scheme: str
    host: str
    port: int
    path: str


def validate_endpoint(url: str) -> Endpoint:
    """Validate syntax and literal addresses without resolving DNS."""
    try:
        if not isinstance(url, str) or len(url) > 2048 or re.search(r'[\s\\\x00-\x1f\x7f]', url):
            raise ValueError
        parsed = urlsplit(url)
        host = parsed.hostname
        if (parsed.scheme not in {'https', 'http'} or not host or parsed.username is not None
                or parsed.password is not None or parsed.query or parsed.fragment or '%' in host):
            raise ValueError
        host = host.encode('idna').decode('ascii').lower().rstrip('.')
        port = parsed.port or (443 if parsed.scheme == 'https' else 80)
        if not 1 <= port <= 65535 or not re.fullmatch(r'[a-z0-9.:-]+', host):
            raise ValueError
        endpoint = Endpoint(parsed.scheme, host, port, parsed.path.rstrip('/'))
    except (ValueError, UnicodeError):
        raise AdapterError('Adapter endpoint must be an HTTP(S) URL without credentials, query parameters, or fragments.') from None
    local_enabled = os.getenv('RETRIEVALLAB_ALLOW_LOCAL_ADAPTERS') == '1'
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        address = None
    local = host == 'localhost' or (address is not None and address.is_loopback)
    if endpoint.scheme != 'https' and not (local and local_enabled):
        raise AdapterError('Remote adapters require HTTPS; localhost HTTP requires RETRIEVALLAB_ALLOW_LOCAL_ADAPTERS=1.')
    if host == 'localhost' and not local_enabled:
        raise AdapterError('Loopback adapter endpoints are disabled in public mode.')
    if address is not None:
        _check_address(str(address), local_enabled=local_enabled)
    return endpoint


def _check_address(value: str, *, local_enabled: bool) -> None:
    address = ipaddress.ip_address(value)
    if str(address) == '168.63.129.16':  # Azure platform virtual IP
        raise AdapterError('Adapter endpoint resolves to a prohibited network address.')
    # Reject transition mechanisms that can conceal an IPv4 destination.
    if isinstance(address, ipaddress.IPv6Address) and (
        address.ipv4_mapped or address.sixtofour or address.teredo
        or address in ipaddress.ip_network('64:ff9b::/96')
        or address in ipaddress.ip_network('64:ff9b:1::/48')
    ):
        raise AdapterError('Adapter endpoint resolves to a prohibited network address.')
    if local_enabled and address.is_loopback:
        return
    if not address.is_global or address.is_multicast or address.is_unspecified or address.is_reserved:
        raise AdapterError('Adapter endpoint resolves to a prohibited network address.')


def resolve_addresses(endpoint: Endpoint) -> list[str]:
    try:
        records = socket.getaddrinfo(endpoint.host, endpoint.port, type=socket.SOCK_STREAM)
        addresses = list(dict.fromkeys(record[4][0] for record in records))
    except OSError:
        raise AdapterError('Adapter hostname could not be resolved.') from None
    if not addresses:
        raise AdapterError('Adapter hostname has no usable addresses.')
    for address in addresses:
        _check_address(address, local_enabled=os.getenv('RETRIEVALLAB_ALLOW_LOCAL_ADAPTERS') == '1')
    return addresses


class HTTPTransport:
    """One request per connection, TLS verified against the original hostname."""

    def request(self, endpoint: Endpoint, method: str, route: str, body: bytes | None,
                token: str) -> bytes:
        addresses = resolve_addresses(endpoint)
        connection = http.client.HTTPConnection(endpoint.host, endpoint.port, timeout=CONNECT_TIMEOUT)
        try:
            # Numeric IP is pinned, so DNS cannot rebind between check and connect.
            sock = socket.create_connection((addresses[0], endpoint.port), timeout=CONNECT_TIMEOUT)
            connection.sock = sock
            if endpoint.scheme == 'https':
                connection.sock = ssl.create_default_context().wrap_socket(sock, server_hostname=endpoint.host)
            connection.sock.settimeout(READ_TIMEOUT)
            headers = {'Accept': 'application/json', 'Accept-Encoding': 'identity', 'Content-Type': 'application/json'}
            if token:
                headers['Authorization'] = 'Bearer ' + token
            connection.request(method, endpoint.path + route, body=body, headers=headers)
            response = connection.getresponse()
            if 300 <= response.status < 400:
                raise AdapterError('Adapter redirects are not allowed. Use the final endpoint URL.')
            if response.status != 200:
                raise AdapterError(f'Adapter returned HTTP {response.status}; check endpoint and authentication.')
            if response.getheader('Content-Encoding', 'identity').lower() != 'identity':
                raise AdapterError('Adapter response must not be compressed.')
            if response.getheader('Content-Type', '').split(';')[0].strip().lower() != 'application/json':
                raise AdapterError('Adapter response must have application/json content type.')
            length = response.getheader('Content-Length')
            if length is not None and (not length.isdigit() or int(length) > MAX_RESPONSE_BYTES):
                raise AdapterError('Adapter response exceeds the response size limit or has invalid Content-Length.')
            chunks = []
            size = 0
            deadline = time.monotonic() + READ_TIMEOUT
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise AdapterError('Adapter response exceeded the read deadline.')
                if connection.sock is not None:
                    connection.sock.settimeout(remaining)
                chunk = response.read1(min(65536, MAX_RESPONSE_BYTES + 1 - size))
                if not chunk:
                    break
                size += len(chunk)
                if size > MAX_RESPONSE_BYTES:
                    raise AdapterError('Adapter response exceeds the response size limit.')
                chunks.append(chunk)
            return b''.join(chunks)
        except AdapterError:
            raise
        except Exception:
            # Never expose remote bodies, headers, URLs, or library exception details.
            raise AdapterError('Adapter connection failed or timed out; check endpoint, TLS, and service availability.') from None
        finally:
            connection.close()
