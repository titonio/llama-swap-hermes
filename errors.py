"""Error classification for the llama-swap provider profile.

Hermes runs this hook *before* its own pipeline (``agent/error_classifier.py``
stage order: ``_profile_verdict`` precedes ``_by_status``) and feeds the returned
dict straight into ``ClassifiedError(**{**base, **verdict})``, whose dataclass
defaults are ``retryable=True, should_compress=False,
should_rotate_credential=False, should_fallback=False``. So a rule that merely
reproduces a verdict core already reaches is dead code that *also* discards the
hint flags the built-in had attached — a strictly weaker verdict than the one it
shadowed. Rules here are claimed only where measurement showed core wrong.

Measured, not assumed: every case in ``tests/fixtures/errors.json`` was fed to
the real ``classify_api_error`` (see ``tests/test_errors.py``, which records both
verdicts and re-checks core's wherever Hermes is importable). Two of the five
captured bodies — ``engine_empty_messages`` and ``engine_chat_template`` — already
classify as ``format_error`` with the right hints, so this module declines them and
leaves them to core. ``engine_negative_max_tokens`` is misread by core but is
unreachable from anything Hermes sends, so it is declined too (see below). Two
rules are earned:

* **Context overflow the built-in patterns miss.** Each engine rejects an
  over-long prompt in its own words. Two of them dodged every
  ``_CONTEXT_OVERFLOW_PATTERNS`` entry (corpus cases ``engine_input_overflow``
  and ``engine_input_overflow_strixite``, both captured live): *"the prompt does
  not fit: it is 360052 tokens and the context is 262144, 97908 tokens over."*
  and *"the prompt is 281164 tokens; the context holds 262144 (prompt + at least
  1 generated token)"*. Neither "the context is 262144" nor "context holds" is
  "context window", and "prompt does not fit" is not "prompt is too long", so
  core answers ``format_error`` and Hermes treats a recoverable overflow as a
  terminal client error: it aborts the turn and walks the fallback chain instead
  of compressing and retrying. A third engine, crucible, words it as "exceeds
  the context window of 262144 tokens", which core already recognises — so it
  gets no rule, because a rule that reproduced core's own verdict would be dead
  code that also discarded the hints core attached. The verdict here is core's
  own ``_V_CONTEXT_OVERFLOW`` (``context_overflow`` + ``should_compress=True``),
  which is what core would have said had it recognised the wording.

* **llama-swap's own dispatch failure.** A model id llama-swap has no router for
  comes back 404 with ``src: "llama-swap"`` and ``code: "not_found"`` (corpus
  case ``swap_no_router``). Core answers ``unknown``: ``_MODEL_NOT_FOUND_PATTERNS``
  carries no "no router" wording, and ``_model_id_missing_known_prefix`` is False
  for this provider by design ("an id absent from the curated catalogue returns
  False so real endpoint problems keep their retryable unknown classification").
  A wrong model id is therefore *retryable*, so Hermes burns the entire retry
  budget with backoff on an id that can never resolve instead of aborting and
  walking the fallback chain. The verdict mirrors core's own
  ``_V_MODEL_NOT_FOUND`` (``model_not_found`` + ``_ABORT_FALLBACK``), which is
  what it would have said had it recognised the body. The ``src`` marker is what
  keeps this narrow: it is llama-swap's own envelope, so a generic 404 from an
  engine or a proxy keeps core's classification. Requiring the 404 too costs nothing
  (llama-swap derives ``not_found`` from the status, never the reverse) and makes
  the rule correct for llama-swap's whole dispatch family — "no model id could be
  identified", "no peer found for requested model", "no local server found for
  requested model", "no router for requested model" — not just the one captured
  sentence.

Deliberately NOT implemented:

* ``engine_negative_max_tokens`` (``"'max_tokens' must be >= 1, got -5"``) — core
  misreads it as ``context_overflow`` through the bare ``max_tokens`` entry in
  ``_CONTEXT_OVERFLOW_PATTERNS``, but Hermes never sends ``max_tokens < 1``, so a
  rule for that body would be unreachable. Unreachable code is a defect, not a
  safety margin.
* 401/403/429/5xx and every unlisted status — core maps these with the correct
  hint flags (credential rotation, fallback, reset windows); claiming them could
  only lose them.

**No output-cap rule.** Only the input path was captured. The output path clamps
instead of failing: an over-large ``max_tokens`` (262144) came back HTTP 200 with
the engine having silently clamped it (``strix.max_tokens_clamped: true``), so
there is no output-cap body to earn a rule from and none is invented. The declared
cap is the defence there. Add wordings here only against a captured body.

Signature matches Hermes' ``classify_api_error`` profile-field call convention;
unexpected extra keyword arguments are absorbed by ``**_ignored``. Stdlib only.
"""

from __future__ import annotations

# The engine's own overflow wording, matched on the lowercased message (corpus
# cases `engine_input_overflow` / `engine_input_overflow_strixite`). One phrase
# per captured sentence, never a broader marker ("context", "tokens"): that would
# claim unrelated 400s and compress history that no compaction can fix.
#
#   "prompt does not fit"  -- halogen: "the prompt does not fit: it is 360052
#                             tokens and the context is 262144, 97908 tokens over."
#   "context holds"        -- strixite: "the prompt is 281164 tokens; the context
#                             holds 262144 (prompt + at least 1 generated token)"
#
# Both captured live at 400 against a 262144 window. A third engine, crucible,
# words the same refusal as "exceeds the context window of 262144 tokens", which
# core's own _CONTEXT_OVERFLOW_PATTERNS already recognises, so it needs no rule
# here and gets none. Measured, not assumed: the strixite sentence came back
# format_error (retryable=False, should_fallback=True) from the real
# classify_api_error -- a recoverable overflow turned into a terminal client error
# that aborts the turn instead of compressing and retrying.
_OVERFLOW_MARKERS: tuple[str, ...] = ("prompt does not fit", "context holds")


# llama-swap stamps `src` on envelopes it produced itself, as opposed to bodies
# forwarded from the engine. Only its own dispatch 404s are claimed; see the module
# docstring. The status is part of the rule rather than decoration: llama-swap
# derives the `not_found` code from the 404 status alone (swaputil/httperror.go
# `errorTypeForStatus` is the code's only producer), and its whole 404 family is
# model dispatch — "no model id could be identified", "no peer found for requested
# model", "no local server found for requested model", "no router for requested
# model" (swaputil/http.go). So the rule is right for the family, not just for the
# one captured string, and any other status keeps core's classification.
_SWAP_ENVELOPE = "llama-swap"
_SWAP_NOT_FOUND = "not_found"


def classify_llama_swap_error(
    error: object = None,
    *,
    status_code: int | None = None,
    error_code: str | None = None,
    message: str | None = None,
    body: object = None,
    model: str | None = None,
    **_ignored: object,
) -> dict | None:
    """Claim this stack's overflow wording and llama-swap's unknown-model 404.

    Returns ``{"reason": "context_overflow", "should_compress": True}`` for a 400
    whose lowercased ``message`` carries the engine's overflow wording, and
    ``{"reason": "model_not_found", "retryable": False, "should_fallback": True}``
    for a 404 on a llama-swap envelope (``body["src"] == "llama-swap"``) carrying
    ``code == "not_found"``. Everything else returns ``None`` — core decides. Only
    ``status_code``, ``message``, ``body`` and ``error_code`` drive the decision;
    the remaining named parameters exist for Hermes' call convention.
    """
    if status_code == 400 and any(m in (message or "").lower() for m in _OVERFLOW_MARKERS):
        return {"reason": "context_overflow", "should_compress": True}
    if status_code != 404 or not isinstance(body, dict) or body.get("src") != _SWAP_ENVELOPE:
        return None
    if (error_code or "").lower() != _SWAP_NOT_FOUND:
        return None
    return {"reason": "model_not_found", "retryable": False, "should_fallback": True}
