"""``profile`` — the assembly: identity from the environment, facts from the server.

The rules these tests pin are the ones the design chose: nothing on the request hot
path may block or raise, a context window the server never stated stays unknown, and
every fact Hermes learns about a model came from the server that serves it.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import pytest

import client
import errors
import providers
from conftest import CORPUS, _info, _kwargs, stub_models
from errors import classify_llama_swap_error
from profile import build_profile, catalog_for
from test_fixtures import ready_ids

ROOT = pathlib.Path(__file__).resolve().parents[1]

# Gated to the stub: under a real Hermes checkout the registration below lands in
# the process-wide registry and leaks into every other test that classifies as
# provider="llama-swap" — including Task 4's, whose whole point is measuring core
# without this plugin in the way.
needs_stub_registry = pytest.mark.skipif(
    not hasattr(providers, "REGISTRY"),
    reason="real Hermes providers package in use; stub registry absent",
)


def test_identity_fields():
    p = build_profile({})
    assert (p.name, p.aliases) == ("llama-swap", ("llamaswap", "swap"))
    assert p.api_mode == "chat_completions"
    assert p.env_vars == ("LLAMA_SWAP_API_KEY", "LLAMA_SWAP_BASE_URL")
    assert p.base_url == "http://192.168.31.7:8080/v1"
    assert p.auth_type == "api_key" and p.supports_vision is True
    assert p.fallback_models == ()                    # ids are the server's to know


def test_base_url_env_override():
    assert build_profile({"LLAMA_SWAP_BASE_URL": "http://box:8080/v1/"}).base_url == "http://box:8080/v1"


def test_env_knob_reaches_the_profile():
    p = build_profile({"LLAMA_SWAP_CATALOG_TTL_S": "soon", "LLAMA_SWAP_DEFAULT_MAX_TOKENS": "4096"})
    assert p.catalog_ttl == 60            # junk TTL → default, never 0 (which would mean no caching)
    assert p.default_max_tokens == 4096


def test_fetch_models_returns_server_order_and_warms(swap, calls):
    assert swap.fetch_models() == ["crucible", "gufo", "halogen", "strixite"]
    assert catalog_for(swap, ready_ids()[0]).loaded is True      # whichever id the fixture has ready


def test_fetch_models_returns_none_when_server_unreachable(swap, monkeypatch):
    monkeypatch.setattr(client, "get_json", lambda *a, **k: None)
    assert swap.fetch_models() is None                # None, never []: keep Hermes' fallback path


def test_the_url_hermes_passes_wins_and_becomes_the_active_shard(swap, calls):
    # spec "Address authority": fetch_models' argument is the credential-pool route,
    # not $LLAMA_SWAP_BASE_URL, and the read hooks must answer from that same shard
    swap.fetch_models(base_url="http://other-box:9/v1")
    assert calls == ["http://other-box:9/v1/models", "http://other-box:9/running"]
    assert swap.get_model_context_length(ready_ids()[0]) == 262144   # a model whose window the
                                                                     # captured server reported


def test_context_length_probe_happens_once_at_first_use(swap, calls):
    # the sanctioned probe point; exactly one fetch, on first use only
    assert swap.get_model_context_length("strixite") == 262144
    assert len(calls) == 2                            # /models + /running, nothing more
    assert swap.get_model_context_length("strixite") == 262144
    assert len(calls) == 2


def test_unreachable_server_probes_once_and_degrades(swap, monkeypatch):
    monkeypatch.setattr(client, "get_json", lambda *a, **k: None)
    assert swap.get_model_context_length("strixite") is None    # probe 1: fails
    assert swap.get_model_context_length("strixite") is None    # cached failure: no probe 2


def test_model_that_reports_no_window_has_no_context_length(swap):
    assert swap.get_model_context_length("gufo") is None        # → Hermes' own chain, 256K last


def test_capabilities_refreshed_in_place_on_the_registered_profile(swap):
    caps = swap.model_capabilities                    # hold the reference: Hermes reads it live
    swap.fetch_models()
    assert swap.model_capabilities is caps                   # mutated in place, never rebound
    assert caps["halogen"]["supports_vision"] is True
    assert caps["halogen"]["context_window"] == 262144
    assert "gufo" not in caps                                # nothing reported → no entry


def test_capabilities_refresh_drops_a_model_the_server_stopped_reporting(swap, clock, monkeypatch):
    # The clear() half of "in place": a stale entry would keep describing a model
    # llama-swap no longer serves, and Hermes would trust it.
    swap.fetch_models()
    assert "halogen" in swap.model_capabilities
    monkeypatch.setattr(
        client, "get_json", lambda *a, **k: {"data": [{"id": "only", "context_length": 1024}]}
    )
    clock.advance(61)          # the TTL is what turns the next call into a real fetch
    swap.fetch_models()
    assert swap.model_capabilities == {"only": {"context_window": 1024}}


def test_ttl_expiry_triggers_a_second_fetch(swap, calls, clock):
    swap.fetch_models(); n = len(calls)
    clock.advance(61)
    swap.fetch_models()
    assert len(calls) > n


def test_catalog_for_answers_from_the_cache_only(swap):
    # Tasks 6-8 call this per request: it must never warm, so a replaced shard is
    # answered as-is and an id that is not in it is unknown rather than an error.
    stub_models(swap, {"only": _info("only", loaded=True)})
    assert catalog_for(swap, "only").loaded is True
    assert catalog_for(swap, "strixite") is None     # the stub server still serves it; the shard does not


def test_error_hook_delegates_and_never_raises(swap, monkeypatch):
    case = CORPUS["swap_no_router"]
    assert swap.classify_api_error(None, **_kwargs(case)) == classify_llama_swap_error(**_kwargs(case))
    monkeypatch.setattr(errors, "classify_llama_swap_error", lambda **kw: 1 / 0)
    assert swap.classify_api_error(None, status_code=400, error_code="", message="", body={}, model="x") is None


# Mirrors the sibling's test_package_import_registers_provider
# (halogen-hermes/tests/test_profile.py:152-169) but asserts through the PUBLIC API, and is gated
# to the stub: under a real Hermes checkout the same registration would land in the process-wide
# registry and leak into every other test that classifies as provider="llama-swap" — including
# Task 4's, whose whole point is measuring core without this plugin in the way.
@needs_stub_registry
def test_package_import_registers_provider():
    # Mirror Hermes' user-plugin loader: __init__.py as a package whose submodule search path is
    # the plugin dir (providers/__init__.py:479-493 uses spec_from_file_location + a per-home name).
    module_name = "llama_swap_plugin_under_test"
    spec = importlib.util.spec_from_file_location(
        module_name, ROOT / "__init__.py", submodule_search_locations=[str(ROOT)]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    saved = dict(providers.REGISTRY)          # snapshot for the restore below
    try:
        spec.loader.exec_module(module)
        # NOT providers.REGISTRY — that name exists only in the test stub; the real module keeps
        # _REGISTRY private (providers/__init__.py:55) and exposes get_provider_profile (:192).
        from providers import get_provider_profile
        assert get_provider_profile("llama-swap") is not None
        assert get_provider_profile("llama-swap") is get_provider_profile("swap")
        # Every submodule of the package must reach its siblings relatively: under Hermes the
        # plugin dir is not on sys.path, so an absolute `import client` / `import catalog`
        # would have raised here instead of registering. The flat copies on sys.path make the
        # absolute form succeed in tests, so the proof is which copy the package ended up with.
        assert sys.modules[f"{module_name}.catalog"].client.__name__ == f"{module_name}.client"
    finally:
        for name in [n for n in sys.modules if n.startswith(module_name)]:
            sys.modules.pop(name, None)
        # unregister: the stub registry is module-global, so restore the snapshot rather
        # than clearing it — a later test may have registered its own profile.
        for key in [k for k, v in providers.REGISTRY.items() if v is module.llama_swap]:
            del providers.REGISTRY[key]
        providers.REGISTRY.update(saved)
