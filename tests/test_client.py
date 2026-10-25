"""``client.get_json`` / URL helpers / env readers, with the opener stubbed.

Nothing here touches the network: the module-level opener is replaced per test,
so the suite runs anywhere (the live server is Task 9's smoke script).
"""

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


def test_malformed_env_int_falls_back_to_default():
    assert client.env_int({"LLAMA_SWAP_CATALOG_TTL_S": "soon"}, "LLAMA_SWAP_CATALOG_TTL_S", 60) == 60
