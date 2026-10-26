"""Inject stub ``providers`` / ``providers.base`` / ``agent.account_usage`` modules.

Hermes ships ``providers.base.ProviderProfile`` (a dataclass),
``providers.register_provider`` and ``agent.account_usage.AccountUsageSnapshot`` at
runtime; the plugin must import cleanly without Hermes installed so its units run
anywhere. When the real package IS importable (tests running inside a Hermes checkout),
the stubs are skipped and the same tests run against the real types.

Below the stub: the fixtures every plugin test shares. The stubbed server is the
captured ``tests/fixtures`` payload, so no test resolves a host.
"""

import json
import pathlib
import sys
import time
import types
from dataclasses import dataclass, replace
from datetime import datetime

import pytest

# The repo root on sys.path, so `import client` / `catalog` / `profile` work under any
# invocation. It cannot live in a root conftest.py: when this directory's name is a legal
# identifier (a clone into `llama_swap`), pytest imports that conftest as part of the
# package, which runs __init__.py — and its `from providers import register_provider`
# executes before this file has injected the stub below.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

try:
    import providers  # noqa: F401
    import providers.base  # noqa: F401
except ImportError:
    # By design: a sibling-module ImportError (``providers`` importable but
    # ``providers.base`` broken) lands here too, and the stubs replace both.
    _providers = types.ModuleType("providers")
    _providers.REGISTRY = {}  # name/alias -> profile, mirrors register_provider()

    def _register_provider(profile):
        _providers.REGISTRY[profile.name] = profile
        for alias in getattr(profile, "aliases", ()):
            _providers.REGISTRY[alias] = profile

    _providers.register_provider = _register_provider

    def _get_provider_profile(name):
        """Mirror of the PUBLIC lookup in ``providers/__init__.py:192``.

        The real module keeps ``_REGISTRY`` private (:55); asserting through this is
        what lets one registration test pass with and without Hermes installed.
        """
        return _providers.REGISTRY.get(name)

    _providers.get_provider_profile = _get_provider_profile

    _base = types.ModuleType("providers.base")

    class ProviderProfile:
        """Stand-in for the real dataclass: kwargs become attributes."""

        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

        # No-op defaults matching providers/base.py's contract, so a subclass
        # that leans on super() still gets sensible values.
        def get_max_tokens(self, model):
            return getattr(self, "default_max_tokens", None)

        def get_model_context_length(self, model):
            return getattr(self, "model_context_length", None)

    _base.ProviderProfile = ProviderProfile
    _providers.base = _base

    sys.modules["providers"] = _providers
    sys.modules["providers.base"] = _base

try:
    import agent.account_usage  # noqa: F401
except ImportError:
    # Same deal as ``providers.base`` above: ``profile.fetch_account_usage`` builds
    # Hermes' ``AccountUsageSnapshot`` (agent/account_usage.py:41), and the hook's own
    # units must run offline. The stub restates only what the hook relies on — the
    # required fields, ``details``/``windows``, and ``available`` as a READ-ONLY
    # property, because assigning to it is the mistake this shape makes impossible.
    # Under a Hermes checkout the real class is used and the same tests re-run on it.
    _agent = types.ModuleType("agent")
    # An empty package, not a module: `agent.error_classifier` must still fail to import
    # offline, or the Hermes-only tests in test_errors/test_wire would run against
    # nothing instead of skipping.
    _agent.__path__ = []
    _account_usage = types.ModuleType("agent.account_usage")

    @dataclass(frozen=True)
    class AccountUsageSnapshot:
        provider: str
        source: str
        fetched_at: datetime
        title: str = "Account limits"
        windows: tuple = ()
        details: tuple = ()
        unavailable_reason: str | None = None

        @property
        def available(self):
            return bool(self.windows or self.details) and not self.unavailable_reason

    _account_usage.AccountUsageSnapshot = AccountUsageSnapshot
    sys.modules["agent"] = _agent
    sys.modules["agent.account_usage"] = _account_usage


# After the stub block: profile.py imports providers.base.
import client  # noqa: E402
import profile as plugin  # noqa: E402
from catalog import ModelInfo  # noqa: E402

FIX = pathlib.Path(__file__).resolve().parent / "fixtures"

CORPUS = json.loads((FIX / "errors.json").read_text())

# The live counter set, captured verbatim from the server (Task 8 Step 0). Tests derive
# every numeral from this dict rather than restating one: it is a running total, so any
# literal copied in is stale the moment the fixture is recaptured.
STATS = json.loads((FIX / "metrics_stats.json").read_text())


def _kwargs(case):
    """Mirror the kwargs Hermes' ``_profile_verdict`` passes to the hook."""
    body = case["body"]
    err = body.get("error") or {}
    return {
        "error": None,                   # the exception object; our rules read the body
        "status_code": case["status"],
        "error_code": err.get("code") or err.get("type") or "",
        "message": str(err.get("message") or "").lower(),   # _Ctx.msg is lowercased
        "body": body,
        "model": "strixite",
    }


def _info(model_id, **facts):
    """A ``ModelInfo`` whose unset facts stay ``None`` — the server said nothing."""
    return ModelInfo(
        id=model_id,
        context_length=None,
        vision=None,
        tools=None,
        efforts=None,
        reasoning_enabled=None,
        **facts,
    )


def _stated(info, **facts):
    """Restate one fact on a stub ``_info``: its five core fields are already passed,
    so ``_info(id, context_length=…)`` would hand ``ModelInfo`` the keyword twice.
    """
    return replace(info, **facts)


def stub_models(swap, infos):
    """Repoint ``swap``'s cached catalog at ``infos`` (a ``{id: ModelInfo}``).

    Writes the cache entry directly, because the point is to reach answers the
    captured server cannot produce — an empty catalog, a model with no facts — and
    no JSON payload through the stub would parse back into a hand-built ModelInfo.
    A TTL refresh after this re-reads the captured fixtures, so keep it last.
    """
    url = client.normalize_base_url(swap.base_url)
    plugin._CATALOG_CACHE[url] = {"at": time.monotonic(), "infos": dict(infos)}


class _Served(list):
    """The URLs the stub served, in order, plus ``timeouts`` — the ``timeout`` each GET
    was given. The budget is invisible everywhere else in the suite, so recording it
    here is what lets a test pin ``LLAMA_SWAP_PROBE_TIMEOUT_S`` to the socket.
    """

    def __init__(self) -> None:
        super().__init__()
        self.timeouts: list[float | None] = []


@pytest.fixture
def calls(monkeypatch):
    """Every URL the stubbed ``client.get_json`` served, in order.

    Patched on the ``client`` module because ``catalog.fetch_catalog`` reaches it as
    ``client.get_json`` — that attribute lookup is the seam that keeps the suite
    offline; a from-import in catalog.py would be unpatchable. The returned list also
    records ``.timeouts`` — see :class:`_Served`.
    """
    served = _Served()
    payloads = {
        "/models": json.loads((FIX / "models.json").read_text()),
        "/running": json.loads((FIX / "running.json").read_text()),
    }

    def fake_get_json(base_url, path, *, api_key=None, timeout=None):
        served.append(client.normalize_base_url(base_url) + path)
        served.timeouts.append(timeout)
        return payloads.get(path)

    monkeypatch.setattr(client, "get_json", fake_get_json)
    return served


@pytest.fixture
def swap(calls):
    """A llama-swap profile whose server is the captured fixtures, starting cold.

    The catalog cache is module-global (keyed by base URL), so a test must not
    inherit another test's warmed shard.
    """
    plugin._CATALOG_CACHE.clear()
    return plugin.build_profile({})


@pytest.fixture
def usage_served(swap, monkeypatch):
    """Repoint ``swap``'s server at the captured stats payload; returns the URLs served.

    ``/api/version`` answers ``{}`` on purpose: the build line is the one thing a
    snapshot must be able to do without, so a snapshot that needed it would fail here.
    """
    served = _Served()

    def fake_get_json(base_url, path, *, api_key=None, timeout=None):
        served.append(client.normalize_base_url(base_url) + path)
        served.timeouts.append(timeout)
        return STATS if path == "/api/metrics/stats" else {}

    monkeypatch.setattr(client, "get_json", fake_get_json)
    return served


@pytest.fixture
def usage_snapshot(swap, usage_served):
    """``swap.fetch_account_usage()`` against the captured counters (see :func:`usage_served`)."""
    return swap.fetch_account_usage()


@pytest.fixture
def clock(monkeypatch):
    """A monotonic clock the test advances, so TTL expiry needs no sleeping.

    Patches ``time.monotonic`` itself — the cache stamps ``at`` through the module
    attribute, so the fake is what the cache reads.
    """

    class Clock:
        def __init__(self):
            self.now = time.monotonic()

        def monotonic(self):
            return self.now

        def advance(self, seconds):
            self.now += seconds

    clock = Clock()
    monkeypatch.setattr(time, "monotonic", clock.monotonic)
    return clock
