"""The one HTTP surface this plugin uses: bounded stdlib JSON GETs.

Every read of a llama-swap server (``/v1/models``, ``/running``,
``/api/metrics/stats``) goes through :func:`get_json`. It never raises and never
returns anything but parsed JSON or ``None``, so a dead, wrong-shaped or
HTML-serving endpoint degrades to "no models / not enabled" instead of taking
down a Hermes hook. Stdlib only — nothing here imports Hermes.
"""

from __future__ import annotations

import http.client
import json
import socket
import urllib.error
import urllib.request
from collections.abc import Mapping
from typing import Any

DEFAULT_BASE_URL = "http://192.168.31.7:8080/v1"

# Everything a GET can fail with that means "no data this time", so get_json can
# keep its never-raises promise. Every entry earns its place:
#   URLError          transport failures, and every non-2xx status (HTTPError)
#   socket.timeout    its own class on 3.9, an alias of TimeoutError from 3.10
#   HTTPException     IncompleteRead / BadStatusLine out of response.read()
#   ValueError        a malformed base URL, plus JSONDecodeError on a bad body
#   OSError           anything the socket layer raises directly
_TOLERATED_ERRORS: tuple[type[BaseException], ...] = (
    urllib.error.URLError,
    socket.timeout,
    http.client.HTTPException,
    ValueError,
    OSError,
)


class _NoRedirects(urllib.request.HTTPRedirectHandler):
    """Refuse every redirect, so a credential can never be replayed elsewhere.

    Hermes enforces the same policy centrally in
    ``hermes_cli/urllib_security.py`` (``open_credentialed_url``), which keeps
    headers only while a redirect stays on one origin. That module is not
    importable from a standalone plugin, so this uses the stricter, simpler
    variant: a llama-swap endpoint does not redirect, and a 3xx is treated as a
    failed read rather than a free hop to an unvalidated URL.
    """

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise urllib.error.HTTPError(req.full_url, code, msg, headers, fp)


# No custom HTTPSHandler: urllib's default already verifies against the OS trust
# store, which is the same conclusion Hermes reaches in
# ``urllib_security._resolved_https_context``.
_OPENER = urllib.request.build_opener(_NoRedirects())


def normalize_base_url(value: str) -> str:
    """Trim whitespace and trailing slashes: ``http://h:8080/v1/`` → ``…/v1``."""
    return value.strip().rstrip("/")


def root_url(base_url: str) -> str:
    """Server root behind an OpenAI-style base URL (``http://h:8080/v1`` → ``http://h:8080``).

    A base without the ``/v1`` suffix is already a root and is returned as-is,
    so a misconfigured URL yields a miss, not a mangled one.
    """
    normalized = normalize_base_url(base_url)
    return normalized[: -len("/v1")] if normalized.endswith("/v1") else normalized


def get_json(
    base_url: str,
    path: str,
    *,
    api_key: str | None = None,
    timeout: float = 3.0,
) -> Any | None:
    """GET ``{normalize_base_url(base_url)}{path}`` and parse the JSON body.

    ``None`` on any transport error, non-2xx status or non-JSON body (an HTML
    proxy page in front of the server is the common real case). ``timeout`` is
    always the caller's — a catalog probe and a usage read want different
    budgets, and neither should inherit a literal from here.
    """
    headers = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        # Inside the try: a malformed operator-supplied base URL raises ValueError
        # here ("unknown url type"), and that is a miss like any other.
        request = urllib.request.Request(normalize_base_url(base_url) + path, headers=headers)
        with _OPENER.open(request, timeout=timeout) as response:
            if not 200 <= response.status < 300:
                return None
            body = response.read()
    except _TOLERATED_ERRORS:
        return None

    try:
        return json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None


def env_str(env: Mapping[str, str], name: str, default: str) -> str:
    """Env value, trimmed; empty or unset falls back to ``default``."""
    raw = (env.get(name) or "").strip()
    return raw or default


def env_int(env: Mapping[str, str], name: str, default: int) -> int:
    """Env value as an int; a malformed value falls back instead of raising.

    Hermes reads these at startup — a bad ``LLAMA_SWAP_CATALOG_TTL_S`` must not
    crash it.
    """
    raw = (env.get(name) or "").strip()
    try:
        return int(raw)
    except ValueError:
        return default
