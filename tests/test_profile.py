"""``profile`` — the assembly: identity from the environment, facts from the server.

The rules these tests pin are the ones the design chose: nothing on the request hot
path may block or raise, a context window the server never stated stays unknown, and
every fact Hermes learns about a model came from the server that serves it.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys
from dataclasses import replace

import pytest

import client
import errors
import providers
from conftest import CORPUS, _info, _kwargs, stub_models
from errors import classify_llama_swap_error
from profile import build_profile, catalog_for, output_cap
from test_fixtures import ready_ids

ROOT = pathlib.Path(__file__).resolve().parents[1]


@pytest.fixture
def warm_swap(swap):
    """``swap`` with its catalog already read once (the state a live session runs in).

    The Task 6 hooks answer from the cache only, so a cold ``swap`` cannot show what
    the server actually claimed; ``fetch_models`` is the setup-path read, never the
    request path. ``test_cold_shard_declares_nothing`` pins the other half.
    """
    swap.fetch_models()
    return swap


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


# --- Task 6: declared budget and per-model vocabulary -------------------------
#
# The rule these pin is that a declared budget is a CAP the server can serve, never a
# raise, and that a fact llama-swap did not state stays unstated: ``None`` means "the
# server said nothing" (Hermes keeps its own default), ``()`` means "reasoning is off"
# (no reasoning field may go on the wire), and the two must never collapse into one.


def test_output_cap_is_a_cap_never_a_raise():
    assert output_cap(262144, default_max=8192, floor=1024) == 8192   # big window keeps the default
    assert output_cap(8192, default_max=8192, floor=1024) == 2048     # window // 4
    assert output_cap(2048, default_max=8192, floor=1024) == 1024     # floor
    assert output_cap(512, default_max=8192, floor=1024) == 1024      # floor, not window//4


def test_unknown_window_declares_no_cap():
    # Not a guess: Hermes' own reactive path (compress on overflow) is better served by
    # no claim than by a number this plugin invented.
    assert output_cap(None, default_max=8192, floor=1024) is None


def _stated(info, **facts):
    """Restate one fact on a stub ``_info``: its five core fields are already passed,
    so ``_info(id, context_length=…)`` would hand ``ModelInfo`` the keyword twice.
    """
    return replace(info, **facts)


def test_get_max_tokens_follows_the_models_window(warm_swap):
    assert warm_swap.get_max_tokens("strixite") == 8192               # 262144 // 4 > default
    stub_models(warm_swap, {"tiny": _stated(_info("tiny"), context_length=8192)})
    assert warm_swap.get_max_tokens("tiny") == 2048                   # window // 4 wins
    assert warm_swap.get_max_tokens("gufo") is None                   # never reported


def test_cold_shard_declares_nothing(swap):
    # Before the first probe every hot-path hook must decline without fetching: an
    # invented budget or vocabulary is the failure mode this plugin exists to avoid.
    assert swap.get_max_tokens("strixite") is None
    assert swap.supported_reasoning_efforts("strixite") is None
    assert swap.resolve_aux_model() == ""
    assert swap.default_vision_model() is None
    assert swap.default_reasoning_config("strixite") is None


def test_hot_path_hooks_never_touch_the_network(warm_swap, calls):
    before = len(calls)
    warm_swap.supported_reasoning_efforts("halogen")
    warm_swap.get_max_tokens("halogen")
    warm_swap.resolve_aux_model()
    warm_swap.default_vision_model()
    warm_swap.default_reasoning_config("halogen")
    assert len(calls) == before


def test_supported_reasoning_efforts_is_per_model_not_server_wide(warm_swap):
    assert warm_swap.supported_reasoning_efforts("halogen") == (
        "none", "minimal", "low", "medium", "high", "xhigh", "max")
    assert warm_swap.supported_reasoning_efforts("strixite")[-1] == "xhigh"


def test_model_publishing_nothing_has_no_declared_vocabulary(warm_swap):
    stub_models(warm_swap, {"quiet": _info("quiet")})     # no reasoning block at all
    assert warm_swap.supported_reasoning_efforts("quiet") is None


def test_reasoning_switched_off_declares_an_empty_vocabulary(warm_swap):
    # The other half of the tri-state: the server SAID reasoning is off, so the
    # transport must send no reasoning field. ``None`` here would keep the default
    # vocabulary on the wire, which is the 400 this plugin is avoiding.
    stub_models(warm_swap, {"flat": _stated(_info("flat"), reasoning_enabled=False, efforts=("low",))})
    assert warm_swap.supported_reasoning_efforts("flat") == ()
    assert warm_swap.default_reasoning_config("flat") is None


def test_an_unknown_id_is_undeclared_not_disabled(warm_swap):
    assert warm_swap.supported_reasoning_efforts("no-such-model") is None
    assert warm_swap.get_max_tokens("no-such-model") is None


# NOT gufo: the captured server publishes a full reasoning vocabulary for every model,
# including gufo (verified against tests/fixtures/models.json). "Reports no capabilities"
# and "reports no reasoning vocabulary" are different facts, and only the second is
# unknown here — hence the stubbed ``quiet``/``flat`` models above.


def test_aux_prefers_an_already_loaded_model(warm_swap):
    assert warm_swap.resolve_aux_model() == ready_ids()[0]   # the fixture's one ready model


def test_aux_for_a_vision_task_skips_a_model_that_denied_vision(warm_swap):
    # ``vision=True`` is Hermes asking for a model that can take an image; an unknown
    # vision (``None``) is not a yes, so it is not picked either. Server order breaks
    # the tie, and a plain aux task still takes the first loaded model.
    stub_models(warm_swap, {
        "text": _info("text", loaded=True),                                # vision unknown
        "eyes": _stated(_info("eyes"), loaded=True, vision=True),
        "blind": _stated(_info("blind"), loaded=True, vision=False),
    })
    assert warm_swap.resolve_aux_model(vision=True) == "eyes"
    assert warm_swap.default_vision_model() == "eyes"
    assert warm_swap.resolve_aux_model() == "text"


def test_a_model_that_denied_vision_is_not_the_vision_default(warm_swap):
    stub_models(warm_swap, {"blind": _stated(_info("blind"), loaded=True, vision=False)})
    assert warm_swap.default_vision_model() is None
    assert warm_swap.resolve_aux_model(vision=True) == ""


def test_vision_default_comes_from_the_catalog(warm_swap):
    assert warm_swap.default_vision_model() == "halogen"


def test_catalog_with_nothing_loaded_answers_no_aux(warm_swap):
    stub_models(warm_swap, {"gufo": _info("gufo")})      # unloaded, no capabilities: nothing to pick
    assert warm_swap.resolve_aux_model() == ""
    assert warm_swap.default_vision_model() is None


def test_a_cold_no_answer_is_not_cached_as_an_answer(swap):
    # A cold shard must not poison the memo: aux/vision can resolve before the first
    # context probe, and a cached "" would then outlive the warm-up by the whole TTL.
    assert swap.resolve_aux_model() == ""
    assert swap.default_vision_model() is None
    swap.fetch_models()
    assert swap.resolve_aux_model() == ready_ids()[0]
    assert swap.default_vision_model() == "halogen"


def test_aux_answer_is_cached_for_the_ttl(warm_swap, clock):
    # providers/base.py:176 makes caching mandatory on this hook; the pick is a scan,
    # so a TTL refresh must be what changes the answer, not the next call.
    first = warm_swap.resolve_aux_model()
    stub_models(warm_swap, {"other": _info("other", loaded=True)})
    assert warm_swap.resolve_aux_model() == first        # still the cached pick
    clock.advance(61)
    assert warm_swap.resolve_aux_model() == "other"      # TTL expiry re-reads the shard


# NOTE: these do not re-stub client.get_json — the swap fixture already filled the
# cache, so a warm catalog would still answer. stub_models replaces the catalog itself.


def test_unset_reasoning_defaults_to_medium_not_the_engine_ceiling(warm_swap):
    assert warm_swap.default_reasoning_config("strixite") == {"enabled": True, "effort": "medium"}


def test_unset_reasoning_stays_unset_for_a_model_without_medium(warm_swap):
    # Never invent an effort a model did not publish: the server's own default is the
    # honest answer when "medium" is not in its vocabulary.
    thin = _stated(_info("thin"), efforts=("none", "minimal"), reasoning_enabled=True)
    stub_models(warm_swap, {"thin": thin})
    assert warm_swap.default_reasoning_config("thin") is None
    assert warm_swap.default_reasoning_config(None) is None   # Hermes may pass no model


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
