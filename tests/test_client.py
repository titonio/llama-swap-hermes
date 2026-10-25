"""``client.get_json`` / URL helpers / env readers.

Nothing here reaches outside this machine: the module-level opener is stubbed per
test, and the two tests that need a real transport bind a loopback socket (the
live server is Task 9's smoke script).
"""

import http.server
import socket
import threading
import urllib.error

import client


class _Response:
    """Stand-in for ``http.client.HTTPResponse`` (read + status + context)."""

    def __init__(self, body: bytes, status: int = 200):
        self._body = body
        self.status = status

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _Opener:
    """Replays one canned response or error and records every Request."""

    def __init__(self, result):
        self._result = result
        self.requests = []
        self.timeouts = []

    def open(self, request, timeout=None):
        self.requests.append(request)
        self.timeouts.append(timeout)
        if isinstance(self._result, BaseException):
            raise self._result
        return self._result


def _stub_opener(monkeypatch, result):
    opener = _Opener(result)
    monkeypatch.setattr(client, "_OPENER", opener)
    return opener


def test_get_json_returns_parsed_body_on_200(monkeypatch):
    opener = _stub_opener(monkeypatch, _Response(b'{"data": []}'))

    assert client.get_json(client.DEFAULT_BASE_URL, "/models") == {"data": []}

    request = opener.requests[0]
    assert request.full_url == "http://192.168.31.7:8080/v1/models"
    assert request.get_method() == "GET"
    # The caller's timeout reaches the transport; it is never a literal here.
    assert opener.timeouts == [3.0]


def test_get_json_returns_none_on_404(monkeypatch):
    _stub_opener(
        monkeypatch,
        urllib.error.HTTPError("http://192.168.31.7:8080/v1/models", 404, "Not Found", {}, None),
    )

    assert client.get_json(client.DEFAULT_BASE_URL, "/models") is None


def test_get_json_returns_none_on_non_json_body(monkeypatch):
    _stub_opener(monkeypatch, _Response(b"<html><head><title>502 Bad Gateway</title></head></html>"))

    assert client.get_json(client.DEFAULT_BASE_URL, "/models") is None


def test_get_json_sends_bearer_only_when_key_given(monkeypatch):
    opener = _stub_opener(monkeypatch, _Response(b'{"data": []}'))

    client.get_json(client.DEFAULT_BASE_URL, "/models")
    assert not opener.requests[0].has_header("Authorization")

    client.get_json(client.DEFAULT_BASE_URL, "/models", api_key="secret")
    assert opener.requests[1].get_header("Authorization") == "Bearer secret"


def test_root_url_derives_server_root_from_base():
    assert client.root_url("http://192.168.31.7:8080/v1") == "http://192.168.31.7:8080"
    assert client.root_url("http://192.168.31.7:8080/") == "http://192.168.31.7:8080"
    assert client.root_url("http://host:9000/api") == "http://host:9000/api"


def test_env_readers_fall_back_on_junk():
    assert client.env_int({"LLAMA_SWAP_CATALOG_TTL_S": "soon"}, "LLAMA_SWAP_CATALOG_TTL_S", 60) == 60
    assert client.env_int({}, "LLAMA_SWAP_CATALOG_TTL_S", 60) == 60  # absent → default, never 0
    assert client.env_int({"LLAMA_SWAP_MIN_OUTPUT_TOKENS": "1024"}, "LLAMA_SWAP_MIN_OUTPUT_TOKENS", 1024) == 1024
    assert client.env_str({"LLAMA_SWAP_BASE_URL": "  http://h.invalid:1/v1/  "}, "LLAMA_SWAP_BASE_URL", "") == "http://h.invalid:1/v1/"


def test_get_json_returns_none_on_unreachable_or_malformed_base():
    # Regression guard for the try boundary: urllib raises ValueError for several
    # of these at Request-construction time, so moving that construction back
    # outside the try turns this red instead of leaking into a Hermes hook.
    closed = socket.socket()
    closed.bind(("127.0.0.1", 0))
    closed_port = closed.getsockname()[1]
    closed.close()

    # `.invalid` hosts (RFC 2606: never resolvable) so getaddrinfo answers at
    # once — a bare "h" costs a real resolver round trip that `timeout` cannot
    # bound, and DNS is not what this test is guarding.
    for base in (
        f"http://127.0.0.1:{closed_port}/v1",  # nothing listening
        "not a url",
        "http://h.invalid:99999/v1",  # port out of range
        "h.invalid:8080/v1",  # no scheme
        "",
        "ftp://h.invalid/v1",  # no urllib handler
    ):
        assert client.get_json(base, "/models", api_key="secret", timeout=0.5) is None, base


def test_get_json_refuses_a_redirect_without_replaying_the_credential():
    seen = []

    class _Redirector(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            seen.append((self.path, self.headers.get("Authorization")))
            if self.path == "/v1/models":
                self.send_response(302)
                self.send_header("Location", "/stolen")
                self.end_headers()
            else:
                body = b'{"leaked": true}'
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        def log_message(self, *args):  # silence the stderr access log
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), _Redirector)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{server.server_port}/v1"
        assert client.get_json(base, "/models", api_key="secret") is None
    finally:
        server.shutdown()
        server.server_close()
        thread.join()

    # Exactly one request, and it is the one that carried the credential: /stolen
    # was never fetched, so there was no second request to replay it to.
    assert seen == [("/v1/models", "Bearer secret")]
