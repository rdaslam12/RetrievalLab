import io
import socket

import pytest

from src.remote_transport import AdapterError, HTTPTransport, resolve_addresses, validate_endpoint


@pytest.mark.parametrize('url', [
    'file:///etc/passwd', 'ftp://example.com', 'http://example.com',
    'https://user:password@example.com', 'https://example.com?token=secret',
    'https://example.com/#secret', 'https://127.0.0.1', 'https://10.0.0.1',
    'https://169.254.169.254', 'https://192.168.0.1', 'https://172.16.0.1',
    'https://[::1]', 'https://[fe80::1]', 'https://[ff02::1]', 'https://224.0.0.1',
    'https://0.0.0.0', 'https://[::ffff:127.0.0.1]', 'https://localhost',
    'https://example.com\n', 'https://example.com\\@127.0.0.1',
    'https://[64:ff9b::7f00:1]',
])
def test_unsafe_endpoints(url, monkeypatch):
    monkeypatch.delenv('RETRIEVALLAB_ALLOW_LOCAL_ADAPTERS', raising=False)
    with pytest.raises(AdapterError):
        validate_endpoint(url)


def test_localhost_only_with_explicit_flag(monkeypatch):
    monkeypatch.delenv('RETRIEVALLAB_ALLOW_LOCAL_ADAPTERS', raising=False)
    with pytest.raises(AdapterError):
        validate_endpoint('http://localhost:8000')
    monkeypatch.setenv('RETRIEVALLAB_ALLOW_LOCAL_ADAPTERS', '1')
    assert validate_endpoint('http://localhost:8000').port == 8000
    assert validate_endpoint('http://127.0.0.1:8000').host == '127.0.0.1'
    with pytest.raises(AdapterError):
        validate_endpoint('http://10.0.0.1:8000')
    with pytest.raises(AdapterError):
        validate_endpoint('https://169.254.169.254')


def test_dns_checks_all_answers_and_rebinding(monkeypatch):
    endpoint = validate_endpoint('https://example.org')
    answers = ['8.8.8.8', '127.0.0.1']
    monkeypatch.setattr(socket, 'getaddrinfo', lambda *args, **kwargs: [
        (socket.AF_INET, socket.SOCK_STREAM, 6, '', (ip, 443)) for ip in answers])
    with pytest.raises(AdapterError):
        resolve_addresses(endpoint)
    answers[:] = ['8.8.8.8']
    assert resolve_addresses(endpoint) == ['8.8.8.8']
    answers[:] = ['169.254.169.254']
    with pytest.raises(AdapterError):
        resolve_addresses(endpoint)


class FakeSocket:
    def __init__(self, response):
        self.response = response
        self.sent = []
        self.timeouts = []
    def makefile(self, *args):
        return io.BytesIO(self.response)
    def sendall(self, data):
        self.sent.append(data)
    def settimeout(self, value):
        self.timeouts.append(value)
    def close(self):
        pass


def run_transport(monkeypatch, response):
    sock = FakeSocket(response)
    addresses, hosts = [], []
    monkeypatch.setattr('src.remote_transport.resolve_addresses', lambda endpoint: ['8.8.8.8'])
    def connect(address, timeout):
        addresses.append((address, timeout))
        return sock
    monkeypatch.setattr(socket, 'create_connection', connect)
    class TLS:
        def wrap_socket(self, sock, server_hostname):
            hosts.append(server_hostname)
            return sock
    monkeypatch.setattr('src.remote_transport.ssl.create_default_context', lambda: TLS())
    result = HTTPTransport().request(validate_endpoint('https://example.org/prefix'), 'GET', '/metadata', None, 'fake-token')
    return result, sock, addresses, hosts


def test_transport_pins_ip_and_verifies_original_tls_host(monkeypatch):
    result, sock, addresses, hosts = run_transport(monkeypatch,
        b'HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 2\r\n\r\n{}')
    assert result == b'{}' and addresses == [(('8.8.8.8', 443), 5.0)]
    assert hosts == ['example.org']
    assert b'GET /prefix/metadata' in sock.sent[0]
    assert b'Authorization: Bearer fake-token' in sock.sent[0]
    assert 30.0 in sock.timeouts


@pytest.mark.parametrize('response,match', [
    (b'HTTP/1.1 302 Found\r\nLocation: https://127.0.0.1\r\n\r\n', 'redirects'),
    (b'HTTP/1.1 401 Unauthorized\r\n\r\nfake-token', 'HTTP 401'),
    (b'HTTP/1.1 200 OK\r\nContent-Type: text/html\r\n\r\n', 'application/json'),
    (b'HTTP/1.1 200 OK\r\nContent-Encoding: gzip\r\n\r\n', 'compressed'),
    (b'HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 999999999\r\n\r\n', 'size limit'),
])
def test_transport_rejects_redirect_status_and_unbounded_response(monkeypatch, response, match):
    with pytest.raises(AdapterError, match=match) as caught:
        run_transport(monkeypatch, response)
    assert 'fake-token' not in str(caught.value)


def test_chunked_response_limit(monkeypatch):
    monkeypatch.setattr('src.remote_transport.MAX_RESPONSE_BYTES', 3)
    with pytest.raises(AdapterError, match='size limit'):
        run_transport(monkeypatch, b'HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nTransfer-Encoding: chunked\r\n\r\n4\r\nxxxx\r\n0\r\n\r\n')
