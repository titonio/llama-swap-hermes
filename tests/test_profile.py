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
from conftest import CORPUS, STATS, _info, _kwargs, _stated, stub_models
from errors import classify_llama_swap_error
from profile import build_profile, catalog_for, output_cap, usage_details
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
    p = build_profile({
        "LLAMA_SWAP_CATALOG_TTL_S": "soon",
        "LLAMA_SWAP_DEFAULT_MAX_TOKENS": "4096",
        "LLAMA_SWAP_PROBE_TIMEOUT_S": "7",
        "LLAMA_SWAP_MIN_OUTPUT_TOKENS": "512",
    })
    assert p.catalog_ttl == 60            # junk TTL → default, never 0 (which would mean no caching)
    assert p.default_max_tokens == 4096
    assert p.probe_timeout == 7           # the probe budget, proven at the socket below
    assert p.min_output_tokens == 512


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


def test_unreachable_server_probes_once_and_degrades(swap, clock, monkeypatch):
    probes: list[int] = []

    def dead(*args, **kwargs):
        probes.append(1)
        return None

    monkeypatch.setattr(client, "get_json", dead)
    assert swap.get_model_context_length(ready_ids()[0]) is None  # probe 1: fails
    assert swap.get_model_context_length(ready_ids()[0]) is None  # cached failure: no probe 2
    # The count is the whole point: an uncached failure makes EVERY hook call in a
    # session wait out the probe budget on a dead server. A failed /models short-circuits
    # before /running (catalog.fetch_catalog), so one probe is one GET here.
    assert len(probes) == 1
    # The joint property lives here, not split across two tests: the SAME cached failure
    # that absorbs reads inside the TTL must be retried once the TTL expires — the cache
    # is a TTL cache, not a latch (Task 5's re-review P2).
    clock.advance(61)
    assert swap.get_model_context_length(ready_ids()[0]) is None  # still dead
    assert len(probes) == 2                                       # TTL expiry re-probed


def test_a_failed_probe_is_retried_after_the_ttl(swap, clock, monkeypatch):
    # The cache is a TTL cache, not a once-per-process latch: a server that was down for
    # five seconds at session start must be back within a TTL, not silent for the rest
    # of the session — and the refresh must reach model_capabilities, not just the shard.
    probes: list[int] = []

    def dead(*args, **kwargs):
        probes.append(1)
        return None

    monkeypatch.setattr(client, "get_json", dead)
    assert swap.get_model_context_length(ready_ids()[0]) is None
    assert len(probes) == 1
    clock.advance(61)
    monkeypatch.setattr(
        client, "get_json", lambda *a, **k: {"data": [{"id": "only", "context_length": 1024}]}
    )
    assert swap.get_model_context_length("only") == 1024                    # recovered
    assert swap.model_capabilities == {"only": {"context_window": 1024}}    # capabilities too


def test_probe_timeout_reaches_the_socket(swap, calls):
    # LLAMA_SWAP_PROBE_TIMEOUT_S is the only bound between a stalled llama-swap and a
    # hung per-turn path (agent/model_metadata.py:2295). A non-default number, so a probe
    # that hardcoded the default — or dropped the timeout and waited forever — fails here.
    swap.probe_timeout = 7
    swap.get_model_context_length(ready_ids()[0])
    assert calls.timeouts == [7, 7]


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
# (the transport must omit the reasoning field — the clamp itself is Task 7's, since
# ``supported_reasoning_efforts`` only reaches the Responses transport today, see
# agent/transports/codex.py:460,463), and the two must never collapse into one.


def test_output_cap_is_a_cap_never_a_raise():
    assert output_cap(262144, default_max=8192, floor=1024) == 8192   # big window keeps the default
    assert output_cap(8192, default_max=8192, floor=1024) == 2048     # window // 4
    assert output_cap(2048, default_max=8192, floor=1024) == 1024     # floor
    assert output_cap(512, default_max=8192, floor=1024) == 1024      # floor, not window//4
    # Contradictory operator config (DEFAULT_MAX_TOKENS < MIN_OUTPUT_TOKENS): the floor
    # is nested inside the cap (spec min(default_max, max(floor, window//4))), so the
    # declared number can never EXCEED the configured default — a cap, never a raise.
    assert output_cap(262144, default_max=512, floor=1024) == 512


def test_unknown_window_declares_no_cap():
    # Not a guess: Hermes' own reactive path (compress on overflow) is better served by
    # no claim than by a number this plugin invented.
    assert output_cap(None, default_max=8192, floor=1024) is None


def test_get_max_tokens_follows_the_models_window(warm_swap):
    assert warm_swap.get_max_tokens("strixite") == 8192               # 262144 // 4 > default
    stub_models(warm_swap, {"tiny": _stated(_info("tiny"), context_length=8192)})
    assert warm_swap.get_max_tokens("tiny") == 2048                   # window // 4 wins
    assert warm_swap.get_max_tokens("gufo") is None                   # never reported


def test_cold_shard_declares_nothing(swap, calls):
    # Before the first probe every hot-path hook must decline without fetching: an
    # invented budget or vocabulary is the failure mode this plugin exists to avoid.
    # ``calls`` is the direct proof — a hook warming behind the TTL branch would put a
    # GET on every request path, and the answers alone would not show it.
    assert swap.get_max_tokens("strixite") is None
    assert swap.supported_reasoning_efforts("strixite") is None
    assert swap.resolve_aux_model() == ""
    assert swap.default_vision_model() is None
    assert swap.default_reasoning_config("strixite") is None
    assert not calls


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
    # declaration the transport reads (codex today, Task 7's clamp on this wire) must
    # be ``()`` — omit the field. ``None`` here would keep the default vocabulary in
    # play, which is the 400 the clamp is there to avoid.
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


def test_a_vision_ask_takes_the_vision_ladder_not_the_loaded_gate(warm_swap):
    # The ladder is chosen by NEED, not by which hook asked, so ``resolve_aux_model`` given a
    # vision ask can name an UNLOADED model — deliberately: declining a vision ask is not
    # free (see :meth:`default_vision_model`). No core caller passes the kwarg today, so the
    # behaviour is pinned here rather than left to the docstring. A plain aux ask over the
    # same catalog still declines, because its "" lands on "use the main model".
    stub_models(warm_swap, {"eyes": _stated(_info("eyes"), vision=True)})   # image input, unloaded
    assert warm_swap.resolve_aux_model(vision=True) == "eyes"
    assert warm_swap.default_vision_model() == "eyes"
    assert warm_swap.resolve_aux_model() == ""


def test_a_model_that_denied_vision_is_not_the_vision_default(warm_swap):
    stub_models(warm_swap, {"blind": _stated(_info("blind"), loaded=True, vision=False)})
    assert warm_swap.default_vision_model() is None
    assert warm_swap.resolve_aux_model(vision=True) == ""


def test_vision_default_comes_from_the_catalog(warm_swap):
    # NOT "halogen": the loaded id is runtime state (it has been two different models
    # during this plan), and the fixtures happen to make the one loaded model the
    # vision-capable one.
    assert warm_swap.default_vision_model() == ready_ids()[0]


def test_an_unloaded_vision_model_is_still_the_vision_answer(warm_swap):
    # The aux/vision asymmetry is deliberate (design doc "Auxiliary and vision
    # defaults"): declining for aux is free, declining for vision sends images to the
    # main chat model (agent/auxiliary_client.py:5613), which is usually text-only
    # here. So aux requires `loaded`; vision prefers it and falls back to any model
    # that reported image input — naming it costs one llama-swap auto-load.
    stub_models(warm_swap, {"eyes": _stated(_info("eyes"), vision=True)})   # image input, unloaded
    assert warm_swap.resolve_aux_model() == ""                     # aux: loaded-only, declines
    assert warm_swap.default_vision_model() == "eyes"              # vision: names it anyway


def test_a_loaded_vision_model_is_preferred_over_an_unloaded_one(warm_swap):
    # The ladder is prefer-loaded-then-any, not any-first: a loaded pick serves the
    # image now, an unloaded one pays a cold start for no reason.
    stub_models(warm_swap, {
        "early": _stated(_info("early"), vision=True),                 # reports vision, unloaded
        "late": _stated(_info("late"), loaded=True, vision=True),  # loaded, later in server order
    })
    assert warm_swap.default_vision_model() == "late"


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
    assert swap.default_vision_model() == ready_ids()[0]


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


# --- Task 8: llama-swap's own counters in /usage ------------------------------
#
# llama-swap publishes no quota, no reset and no plan (and its totals are the activity
# rows the store still holds, not server lifetime — see the note on ``_USAGE_TOKEN_KEYS``),
# so the rule these pin is that every number shown is one the server stated, read from the key
# it came from, and that nothing computed — no percentage, no "used X of Y" — appears.
# Every numeral is DERIVED from ``STATS``: the fixture is a live counter set, so a
# literal copied into a test is stale the next time the fixture is recaptured.


def test_usage_details_are_absolute_counters():
    version = {"version": "v264-9-g4a61d5"}
    lines = usage_details(STATS, version)
    # Derived from the captured payload, never re-stated by hand (see the note after this block).
    assert any(f"requests {STATS['total_requests']}" in ln for ln in lines)
    assert any(f"in {STATS['total_input_tokens']:,}" in ln
               and f"out {STATS['total_output_tokens']:,}" in ln for ln in lines)
    assert any("prompt p50" in ln and "gen p50" in ln for ln in lines)
    # The same line pinned to its keys: the loose check above survives a prompt/gen swap
    # and survives dropping the decimals; this one survives neither.
    prompt, gen = STATS["prompt_histogram"], STATS["gen_histogram"]
    assert any(f"prompt p50 {prompt['p50']:.1f} p95 {prompt['p95']:.1f}" in ln
               and f"gen p50 {gen['p50']:.1f} p95 {gen['p95']:.1f}" in ln for ln in lines)
    assert any(f"cached {STATS['total_cache_tokens']:,}" in ln for ln in lines)
    assert any(version["version"] in ln for ln in lines)


def test_no_quota_is_invented():
    assert all("%" not in ln for ln in usage_details(STATS, {}))


def test_usage_details_omits_a_counter_the_server_did_not_report():
    # A zero is a claim, an omission is honest. llama-swap can drop a key between builds
    # (a histogram is nil, not zero, when no request reported a speed), and printing 0
    # for a key the server never sent would be inventing a fact about this server.
    stripped = {key: value for key, value in STATS.items() if key != "gen_histogram"}
    lines = usage_details(stripped, {})
    assert not any("gen" in ln for ln in lines)
    assert any("prompt p50" in ln for ln in lines)
    # A key sent as null is the same omission (the Go side marshals a nil histogram).
    assert not any("gen" in ln for ln in usage_details(dict(STATS, gen_histogram=None), {}))
    assert usage_details({}, {}) == ()


def test_a_counter_that_is_not_a_number_is_omitted_not_coerced():
    # ``isinstance(True, int)`` is True in Python, so a gate that only asked "is this a
    # number?" would print `requests 1` for a server that sent a boolean — a claim about
    # this server that the server never made. A digit string is the same trap sideways:
    # it is not what the endpoint's own type says the field is, and a value that has to be
    # guessed at is not a counter. Both must render as an omission.
    lie = dict(STATS, total_requests=True, total_input_tokens="2434114")
    lines = usage_details(lie, {})
    assert not any(ln.startswith("requests") for ln in lines)
    assert not any(f"in {STATS['total_input_tokens']:,}" in ln for ln in lines)
    assert any(f"out {STATS['total_output_tokens']:,}" in ln for ln in lines)  # honest keys remain
    # Inside a histogram the same rule applies, where a coercion would print a speed of 1.0.
    assert not any("gen" in ln for ln in usage_details(
        dict(STATS, gen_histogram={"p50": True, "p95": 1.0}), {}))
    # A non-finite float is the same class of lie: Go cannot marshal one, so seeing `p50 nan`
    # would mean something between here and the server invented it.
    assert not any("requests" in ln for ln in usage_details(
        dict(STATS, total_requests=float("nan")), {}))
    assert not any("prompt" in ln for ln in usage_details(
        dict(STATS, prompt_histogram={"p50": float("inf"), "p95": 1.0}), {}))


def test_snapshot_uses_details_not_windows(usage_snapshot, usage_served):
    assert usage_snapshot.windows == ()
    assert usage_snapshot.details and usage_snapshot.available
    # The identity fields are what /usage prints and what core keys its cache by, so they
    # are part of the contract, not decoration: `provider` is the registered slug (pinned
    # at test_profile.py:51), and a naive `fetched_at` would age wrongly against a UTC stamp.
    assert (usage_snapshot.provider, usage_snapshot.source) == ("llama-swap", "llama_swap_api")
    assert usage_snapshot.title == "llama-swap server"
    assert usage_snapshot.fetched_at.tzinfo is not None
    # The route Hermes hands the hook is the route it reads, and /api/* lives at the
    # server ROOT, not under the /v1 OpenAI base (client.root_url).
    assert usage_served == [
        "http://192.168.31.7:8080/api/metrics/stats",
        "http://192.168.31.7:8080/api/version",
    ]


def test_unreachable_server_yields_no_snapshot(swap, monkeypatch):
    monkeypatch.setattr(client, "get_json", lambda *a, **k: None)
    assert swap.fetch_account_usage() is None


def test_a_server_that_states_nothing_yields_no_snapshot(swap, monkeypatch):
    # The fetch_models rule again: "nothing to say" is None, never an empty snapshot —
    # an empty one renders a /usage header over no facts, which reads as a server that
    # served zero requests.
    monkeypatch.setattr(client, "get_json", lambda *a, **k: {})
    assert swap.fetch_account_usage() is None


def test_a_version_read_that_fails_keeps_the_counters(swap, monkeypatch):
    # get_json's failure shape is None, so the version read can come back empty while the
    # counters stand: the build line is optional, the snapshot is not. This is the ordinary
    # partial failure (stats is the heavier read), and it must not blank /usage.
    monkeypatch.setattr(client, "get_json",
                        lambda base_url, path, **kw: STATS if path.endswith("stats") else None)
    assert swap.fetch_account_usage().details == usage_details(STATS, {})


def test_no_hermes_snapshot_type_means_no_snapshot(swap, usage_served, monkeypatch):
    # The hook is Hermes-only, so a missing account_usage module is a decline, not an error:
    # raising here would surface inside core's /usage path. None in sys.modules makes the
    # import itself fail the way an absent module does.
    monkeypatch.setitem(sys.modules, "agent.account_usage", None)
    assert swap.fetch_account_usage() is None


def test_profile_imports_with_no_agent_module_present(monkeypatch):
    """The lazy import above is a rule, not a style choice.

    ``conftest`` installs a stand-in for ``agent.account_usage`` so the hook's units run
    offline — which also means hoisting that import to module scope would leave every other
    test green while breaking the shipped rule that this plugin imports where Hermes'
    account module does not exist. So load profile.py with every ``agent`` name absent.
    """
    for name in [n for n in sys.modules if n == "agent" or n.startswith("agent.")]:
        monkeypatch.delitem(sys.modules, name)
    spec = importlib.util.spec_from_file_location("profile_without_agent", ROOT / "profile.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "profile_without_agent", module)
    spec.loader.exec_module(module)          # ImportError if the snapshot type is hoisted
    assert module.usage_details({}, {}) == ()


def test_usage_reads_share_one_budget(swap, clock, monkeypatch):
    # Core bounds this hook at PLUGIN_USAGE_HOOK_DEADLINE_S (10 s) and does NOT stop the
    # worker thread when it expires, so the plugin's own 5 s has to bound the whole hook
    # — two GETs, one budget — or a slow stats read spends the entire deadline before
    # the version read is even asked for.
    asked: list[float] = []

    def slow_stats(base_url, path, *, api_key=None, timeout=None):
        asked.append(timeout)
        if path.endswith("stats"):
            clock.advance(4.5)                      # a stats read that eats the budget
        return STATS if path.endswith("stats") else {}

    monkeypatch.setattr(client, "get_json", slow_stats)
    assert swap.fetch_account_usage() is not None
    assert asked[0] == 5.0
    assert asked[1] == pytest.approx(0.5)


def test_a_spent_budget_skips_the_version_read(swap, clock, monkeypatch):
    # A 0 timeout is not a timeout (urllib raises on it), so a spent budget means do not
    # ask at all — and the counters still stand on their own without a build line.
    asked: list[str] = []

    def slow_stats(base_url, path, *, api_key=None, timeout=None):
        asked.append(path)
        clock.advance(5.0)                          # the whole budget, gone
        return STATS

    monkeypatch.setattr(client, "get_json", slow_stats)
    snapshot = swap.fetch_account_usage()
    assert asked == ["/api/metrics/stats"]
    assert snapshot.available


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
        assert sys.modules[f"{module_name}.profile"].client.__name__ == f"{module_name}.client"
    finally:
        for name in [n for n in sys.modules if n.startswith(module_name)]:
            sys.modules.pop(name, None)
        # unregister: the stub registry is module-global, so restore the snapshot rather
        # than clearing it — a later test may have registered its own profile.
        for key in [k for k, v in providers.REGISTRY.items() if v is module.llama_swap]:
            del providers.REGISTRY[key]
        providers.REGISTRY.update(saved)
