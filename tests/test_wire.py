"""Task 7 — what actually reaches llama-swap's wire.

Two promises live here. First: a reasoning level only goes on the wire if *that model*
published it, because llama-swap 400s on a level its engine never stated and inventing
one is the failure this whole plugin exists to avoid. Second: the plugin never names an
output budget in its wire extras, so a caller's budget (compression, recovery) always
survives the plugin's declared cap.

The unit half runs offline against stubbed vocabularies; the guarantee half drives
Hermes' real transport and therefore skips where Hermes is absent.
"""

import importlib.util

import pytest

from conftest import _info, _stated, stub_models


def _hermes_available():
    try:
        return importlib.util.find_spec("agent.transports.chat_completions") is not None
    except (ImportError, ValueError):   # a partial checkout must skip, never error
        return False


needs_hermes = pytest.mark.skipif(
    not _hermes_available(), reason="Hermes' chat_completions transport is not importable"
)


@pytest.fixture
def swap_wire(swap):
    """``swap`` over three vocabularies: a narrow one, a wider one, and reasoning off.

    Plain, not parametrized: every case below names its own model, so a ``params=``
    list would only multiply identical runs. ``strixite`` keeps its real id and gets a
    stubbed four-level ladder, so the clamp assertions cannot be accidents of whatever
    the captured server happened to publish.
    """
    stub_models(swap, {
        "strixite": _stated(_info("strixite"), efforts=("none", "minimal", "low", "medium")),
        "low_only": _stated(_info("low_only"), efforts=("none", "minimal", "low")),
        "no_reasoning": _stated(_info("no_reasoning"), efforts=None, reasoning_enabled=False),
    })
    return swap


def test_reasoning_effort_travels_top_level(swap_wire):
    extra, top = swap_wire.build_api_kwargs_extras(reasoning_config={"enabled": True, "effort": "high"},
                                                   model="strixite")
    assert extra == {} and top == {"reasoning_effort": "medium"}   # clamped to the model's ceiling


def test_disabled_reasoning_sends_none_not_omission(swap_wire):
    _, top = swap_wire.build_api_kwargs_extras(reasoning_config={"enabled": False}, model="strixite")
    assert top == {"reasoning_effort": "none"}


def test_disabled_reasoning_omits_a_none_the_model_never_published(swap_wire):
    # The ruling: "off" is only sayable in a vocabulary that has a word for it. Hermes'
    # own clamp_effort refuses "none" as a degradation target for the same reason.
    stub_models(swap_wire, {"no_none": _stated(_info("no_none"), efforts=("minimal", "low", "medium"))})
    _, top = swap_wire.build_api_kwargs_extras(reasoning_config={"enabled": False}, model="no_none")
    assert top == {}


def test_a_vocabulary_the_model_never_published_omits_the_field(swap_wire):
    _, top = swap_wire.build_api_kwargs_extras(reasoning_config={"enabled": True, "effort": "high"},
                                               model="no_reasoning")
    assert top == {}


def test_an_undeclared_vocabulary_passes_the_requested_level_through(swap_wire):
    # ``None`` is not "off": the server said nothing, so Hermes' own wire vocabulary
    # (which contains xhigh) stays in charge — the pre-plugin answer.
    stub_models(swap_wire, {"quiet": _info("quiet")})
    _, top = swap_wire.build_api_kwargs_extras(reasoning_config={"enabled": True, "effort": "xhigh"},
                                               model="quiet")
    assert top == {"reasoning_effort": "xhigh"}


def test_a_clamp_never_escalates(swap_wire):
    # A weak ask stays weak: clamping "low" onto a ladder that reaches "medium" must not
    # answer with more thinking than the user asked for.
    _, top = swap_wire.build_api_kwargs_extras(reasoning_config={"enabled": True, "effort": "low"},
                                               model="strixite")
    assert top == {"reasoning_effort": "low"}
    _, top = swap_wire.build_api_kwargs_extras(reasoning_config={"enabled": True, "effort": "minimal"},
                                               model="low_only")
    assert top == {"reasoning_effort": "minimal"}
    # And the case that actually distinguishes "nearest weaker" from "the ceiling": a level
    # the model did not publish, sitting mid-ladder, must come DOWN to the nearest published
    # level below it rather than up to the top of the vocabulary.
    stub_models(swap_wire, {"gaps": _stated(_info("gaps"), efforts=("none", "minimal", "medium"))})
    _, top = swap_wire.build_api_kwargs_extras(reasoning_config={"enabled": True, "effort": "low"},
                                               model="gaps")
    assert top == {"reasoning_effort": "minimal"}


def test_an_ask_below_the_floor_lands_on_the_weakest_published_level(swap_wire):
    # Hermes' documented policy (``clamp_effort``): when nothing weaker exists, the provider's
    # floor is the closest honest match. Pinned so the no-Hermes fallback cannot drift from it.
    _, top = swap_wire.build_api_kwargs_extras(reasoning_config={"enabled": True, "effort": "minimal"},
                                               model="low_only")
    assert top == {"reasoning_effort": "minimal"}
    stub_models(swap_wire, {"floor": _stated(_info("floor"), efforts=("none", "low", "medium"))})
    _, top = swap_wire.build_api_kwargs_extras(reasoning_config={"enabled": True, "effort": "minimal"},
                                               model="floor")
    assert top == {"reasoning_effort": "low"}


def test_unset_reasoning_stays_unset(swap_wire):
    # ``default_reasoning_config`` is the hook that decides the unset case; inventing
    # medium here would spend thinking tokens nobody asked for.
    _, top = swap_wire.build_api_kwargs_extras(reasoning_config=None, model="strixite")
    assert top == {}
    _, top = swap_wire.build_api_kwargs_extras(reasoning_config={"enabled": True}, model="strixite")
    assert top == {}


def test_unknown_effort_levels_are_dropped_before_the_wire(swap_wire):
    stub_models(swap_wire, {"future": _stated(_info("future"), efforts=("low", "turbo", "ultra"))})
    assert swap_wire.supported_reasoning_efforts("future") == ("low", "ultra")


def test_a_level_outside_the_published_vocabulary_is_dropped(swap_wire):
    # Hermes clamps to its own wire vocabulary before we are called; a level that
    # survives (``ultra``) and is not in this model's ladder must still not be sent.
    _, top = swap_wire.build_api_kwargs_extras(reasoning_config={"enabled": True, "effort": "ultra"},
                                               model="strixite")
    assert top == {"reasoning_effort": "medium"}


def test_a_level_outside_hermes_own_ladder_is_dropped(swap_wire):
    # ``clamp_effort`` passes bespoke names through untouched (another provider may speak
    # them). Forwarding one here would be a level this server never stated, so the guard
    # is on the answer, not just on the request.
    stub_models(swap_wire, {"odd": _stated(_info("odd"), efforts=("low", "turbo", "ultra"))})
    _, top = swap_wire.build_api_kwargs_extras(reasoning_config={"enabled": True, "effort": "turbo"},
                                               model="odd")
    assert top == {}


def test_hooks_ignore_the_extra_kwargs_hermes_passes(swap_wire):
    _, top = swap_wire.build_api_kwargs_extras(reasoning_config=None, model="strixite",
                                               base_url="http://x.invalid/v1", session_id="s",
                                               cache_scope_id="c", ollama_num_ctx=4096,
                                               supports_reasoning=True, qwen_session_metadata=None)
    assert top == {}
    assert swap_wire.build_extra_body(session_id="s", provider_preferences={"x": 1}, model="strixite",
                                      base_url="http://x.invalid/v1", reasoning_config=None,
                                      openrouter_min_coding_score=0.5) == {}


def test_the_extras_never_name_an_output_budget(swap_wire):
    # The guarantee, unit half: Hermes applies these dicts AFTER the caller's budget,
    # so a ``max_tokens`` here would silently overwrite a compression or recovery budget.
    for config in (None, {}, {"enabled": True}, {"enabled": True, "effort": "high"},
                   {"enabled": False}, {"enabled": True, "effort": "none"}):
        for model in ("strixite", "no_reasoning", "quiet"):
            extra, top = swap_wire.build_api_kwargs_extras(reasoning_config=config, model=model)
            assert not {"max_tokens", "max_completion_tokens"} & (set(extra) | set(top))
    assert "max_tokens" not in swap_wire.build_extra_body(model="strixite")


def test_the_wire_hooks_never_touch_the_network(swap_wire, calls):
    swap_wire.build_api_kwargs_extras(reasoning_config={"enabled": True, "effort": "high"}, model="strixite")
    swap_wire.build_extra_body(model="strixite")
    swap_wire.build_client_kwargs_extras(base_url=swap_wire.base_url)
    assert list(calls) == []


def test_no_client_timeout_is_overridden(swap_wire):
    assert "timeout" not in swap_wire.build_client_kwargs_extras()


@needs_hermes
def test_the_no_hermes_clamp_fallback_agrees_with_hermes():
    """``_nearest_weaker`` is a second copy of a policy, so it is pinned to the first.

    Hermes is not importable in the suite that runs the fallback, so the copy is checked
    here instead: same answer for every ladder level (plus a bespoke name) against several
    vocabularies. If Hermes' policy ever moves, this goes red rather than the two clamps
    quietly disagreeing per install.
    """
    from agent.reasoning_effort import EFFORT_LADDER, clamp_effort

    from profile import _nearest_weaker

    vocabularies = [("none", "minimal", "low", "medium"), ("none", "low"),
                    ("minimal", "low", "medium"), ("none",), ("low", "turbo", "ultra"),
                    ("high", "max")]
    mismatches = [(asked, levels, clamp_effort(asked, levels), _nearest_weaker(asked, levels))
                  for asked in (*EFFORT_LADDER, "turbo", "") for levels in vocabularies
                  if clamp_effort(asked, levels) != _nearest_weaker(asked, levels)]
    assert mismatches == []


def test_the_client_key_is_the_placeholder_when_unset(swap_wire):
    # Hermes still needs a key-shaped value; the server does not check one.
    assert swap_wire.build_client_kwargs_extras() == {"api_key": "llama-swap"}
    assert swap_wire.build_client_kwargs_extras(base_url="http://x.invalid/v1")["api_key"] == "llama-swap"
    swap_wire.api_key = "secret"
    assert swap_wire.build_client_kwargs_extras() == {"api_key": "secret"}


MSGS = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]


def _params(swap, max_tokens):
    """The params dict ``ChatCompletionsTransport.build_kwargs`` receives, reduced to the keys
    this test exercises (``_base_kwargs`` reads the rest with ``.get()``, so an absent key takes
    its default). ``max_tokens_param_fn`` is the chat_completions wire shape, and it is what
    ``_apply_max_tokens`` requires before it will place any budget at all.
    """
    return {
        "model_lower": "strixite",
        "base_url": swap.base_url,
        "max_tokens": max_tokens,
        "max_tokens_param_fn": lambda n: {"max_tokens": n},
        "reasoning_config": {"enabled": True, "effort": "medium"},
    }


@needs_hermes
@pytest.mark.parametrize("caller_max, expected", [(64000, 64000), (None, 8192)])
def test_the_callers_budget_survives_the_profile_and_the_cap_applies_without_one(swap, caller_max, expected):
    """The spec's central negative requirement, measured through the real transport.

    Hermes applies the profile's extras dict AFTER ``_apply_max_tokens``, so the only thing
    keeping a caller's budget alive is this plugin never naming one: 64000 (a compression or
    recovery budget core owns) must reach the wire untouched, while with no caller budget the
    declared cap — a quarter of the window the server stated, 8192 for strixite's 262144 — is
    supposed to land. Reasoning rides top level, clamped by the model's own ladder.
    """
    from agent.transports.chat_completions import ChatCompletionsTransport

    swap.fetch_models()          # the setup-path read; get_max_tokens is cache-only
    kwargs = ChatCompletionsTransport()._build_kwargs_from_profile(
        swap, "strixite", MSGS, None, _params(swap, caller_max)
    )
    assert kwargs["max_tokens"] == expected
    assert kwargs["reasoning_effort"] == "medium"


@needs_hermes
def test_a_model_published_as_non_reasoning_gets_no_reasoning_field(swap):
    """The other half of the spec's integration promise, through the same real code path."""
    from agent.transports.chat_completions import ChatCompletionsTransport

    swap.fetch_models()
    stub_models(swap, {"flat": _stated(_info("flat"), reasoning_enabled=False, efforts=("low",))})
    kwargs = ChatCompletionsTransport()._build_kwargs_from_profile(
        swap, "flat", MSGS, None, _params(swap, 64000)
    )
    assert "reasoning_effort" not in kwargs
    assert kwargs["max_tokens"] == 64000
