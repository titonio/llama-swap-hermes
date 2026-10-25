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
verdicts and re-checks core's wherever Hermes is importable). Three of the five
captured bodies — the two engine request-validation 400s and the chat-template
400 — already classify as ``format_error`` with the right hints, so this module
declines them and leaves them to core. Two rules are earned:

* **Context overflow the built-in patterns miss.** The engine rejects an
  over-long prompt with its own sentence (corpus case ``engine_input_overflow``,
  captured live): *"the prompt does not fit: it is 360052 tokens and the context
  is 262144, 97908 tokens over."* Core answers ``format_error`` — none of
  ``_CONTEXT_OVERFLOW_PATTERNS`` occurs in that sentence ("the context is
  262144" is not "context window", "prompt does not fit" is not "prompt is too
  long") — so Hermes treats a recoverable overflow as a terminal client error:
  it aborts the turn and walks the fallback chain instead of compressing and
  retrying. The verdict is core's own ``_V_CONTEXT_OVERFLOW``
  (``context_overflow`` + ``should_compress=True``), which is what core would
  have said had it recognised the wording.

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
  engine or a proxy keeps core's classification.

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
# case `engine_input_overflow`). One phrase on purpose: a broader marker
# ("context", "tokens") would claim unrelated 400s and compress history that no
# compaction can fix.
_OVERFLOW_MARKERS: tuple[str, ...] = ("prompt does not fit",)


# llama-swap stamps `src` on envelopes it produced itself, as opposed to bodies
# forwarded from the engine. Only its own "no router for requested model" 404 is
# claimed (corpus case `swap_no_router`); see the module docstring.
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
    for a llama-swap envelope (``body["src"] == "llama-swap"``) carrying ``code ==
    "not_found"``. Everything else returns ``None`` — core decides. Only
    ``status_code``, ``message``, ``body`` and ``error_code`` drive the decision;
    the remaining named parameters exist for Hermes' call convention.
    """
    if status_code == 400 and any(m in (message or "").lower() for m in _OVERFLOW_MARKERS):
        return {"reason": "context_overflow", "should_compress": True}
    if not isinstance(body, dict) or body.get("src") != _SWAP_ENVELOPE:
        return None
    if (error_code or "").lower() != _SWAP_NOT_FOUND:
        return None
    return {"reason": "model_not_found", "retryable": False, "should_fallback": True}
