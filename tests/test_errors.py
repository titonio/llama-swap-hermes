"""``errors`` — rules the captured corpus earned, and only those.

The corpus in ``tests/fixtures/errors.json`` is the whole evidence base. Every
rule in ``errors.py`` names the case it came from; a case with no rule is a case
Hermes' built-in pipeline already answers correctly, and a rule that reproduced
core's own verdict would be dead code (``agent/error_classifier.py`` runs the
profile hook *before* its own stages, so such a rule only shadows the built-in
verdict and drops the hints it attached).

``EXPECTED`` records both verdicts per case: ``hermes`` is what core's
``classify_api_error`` actually returned for the captured body (measured, not
assumed — see ``test_hermes_still_answers_the_recorded_way``, which re-checks it
wherever Hermes is importable), and ``plugin`` is what this module must return.
"""

from __future__ import annotations

import json

import pytest

from errors import classify_llama_swap_error
from test_fixtures import FIX

CORPUS = json.loads((FIX / "errors.json").read_text())

# "hermes" = measured verdict of core's classify_api_error (provider="llama-swap",
# model="strixite", context_length left at the signature's default 200000; none of
# these verdicts are token-count dependent — none reach the disconnect stage that
# reads large_session()). "plugin" = what classify_llama_swap_error returns.
EXPECTED = {
    # core: format_error (retryable=False, should_fallback=True) via
    # _INVALID_MESSAGE_BODY_PATTERNS — the "no user query found" signal is already
    # Hermes' own poisoned-transcript verdict. Nothing to correct.
    "engine_chat_template": {"hermes": "format_error", "plugin": None},
    # core: format_error, same request-validation path. Correct: compressing
    # cannot fix an empty messages array.
    "engine_empty_messages": {"hermes": "format_error", "plugin": None},
    # core is WRONG here: "'max_tokens' must be >= 1, got -5" reaches
    # context_overflow through the bare "max_tokens" entry in
    # _CONTEXT_OVERFLOW_PATTERNS, so Hermes would compress history over a
    # parameter it never sent as negative. Deliberately NOT implemented: Hermes
    # never sends max_tokens < 1, so a rule for this body is unreachable code,
    # and unreachable code is a defect rather than a safety margin.
    "engine_negative_max_tokens": {"hermes": "context_overflow", "plugin": None},
    # core is WRONG here: a real overflow (360052 tokens against a 262144 window)
    # classifies as format_error, so Hermes aborts the turn and walks the fallback
    # chain instead of compressing and retrying. None of _CONTEXT_OVERFLOW_PATTERNS
    # occurs in the engine's sentence — "the context is 262144" is not "context
    # window", "prompt does not fit" is not "prompt is too long". Earned rule 2.
    # Captured live: the engine ERRORS on overflow, it does not clamp.
    "engine_input_overflow": {
        "hermes": "format_error",
        "plugin": {"reason": "context_overflow", "should_compress": True},
    },
    # core is WRONG here: _MODEL_NOT_FOUND_PATTERNS has no "no router" wording and
    # _model_id_missing_known_prefix("strixite", "llama-swap") is False by design,
    # so a wrong model id classifies as retryable `unknown` and burns the whole
    # retry budget instead of aborting and walking the fallback chain. This is the
    # one divergence the corpus proves, so it is the one rule.
    "swap_no_router": {
        "hermes": "unknown",
        "plugin": {"reason": "model_not_found", "retryable": False, "should_fallback": True},
    },
}


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


@pytest.mark.parametrize("case", sorted(CORPUS))
def test_plugin_verdict_matches_the_recorded_expectation(case):
    assert classify_llama_swap_error(**_kwargs(CORPUS[case])) == EXPECTED[case]["plugin"]


def test_every_corpus_case_has_a_recorded_expectation():
    """A new capture must be measured against core before it can earn a rule."""
    assert sorted(EXPECTED) == sorted(CORPUS)


class _ApiStatusError(Exception):
    """What ``openai.APIStatusError`` looks like to Hermes: a status, a body, and
    a str() that already embeds the body (which is what ``_build_error_msg`` reads)."""

    def __init__(self, status, body):
        super().__init__(f"Error code: {status} - {json.dumps(body)}")
        self.status_code, self.body = status, body


@pytest.mark.parametrize("case", sorted(CORPUS))
def test_hermes_still_answers_the_recorded_way(case):
    """Re-measure ``EXPECTED[...]["hermes"]`` wherever Hermes is importable.

    The plugin suite must run offline, so the recorded verdicts are the contract
    here; this test keeps them honest in a Hermes checkout instead of letting
    them silently rot.
    """
    pytest.importorskip("agent.error_classifier")
    from agent.error_classifier import classify_api_error

    classified = classify_api_error(
        _ApiStatusError(CORPUS[case]["status"], CORPUS[case]["body"]),
        provider="llama-swap", model="strixite", base_url="http://192.168.31.7:8080/v1",
    )
    assert classified.reason.value == EXPECTED[case]["hermes"]
