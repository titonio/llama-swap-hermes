#!/usr/bin/env python3
"""Live wire smoke for the llama-swap Hermes provider.

Runs the plugin's own read path against a real llama-swap and prints what the
server actually said: build, catalog, per-model output cap and reasoning
vocabulary, thinking on/off on the loaded model, two captured error bodies (an
oversized prompt and an unpublished reasoning level), the ``/usage`` counters,
and the aux/vision picks. Doubles as the operator diagnostic for a new
deployment, and as the way ``tests/fixtures/errors.json`` gets refreshed when
llama-swap or its engines change.

**Never forces an engine swap.** Every inference request names the model
``/running`` reports as ready, read at runtime; nothing here hardcodes a model
id, and if no model is loaded the run fails rather than waking one up.

Stdlib only. Run from anywhere:

    LLAMA_SWAP_BASE_URL=http://<host>:8080/v1 python3 scripts/smoke.py

Run it under a Hermes ``PYTHONPATH`` and two Hermes-only checks light up as well:
the ``fetch_account_usage`` snapshot and the reasoning-rejection classification
of the captured ``ultra`` body. Exit code is 0 only when every check passes; the
first failure stops the run, because later checks read what earlier ones captured.
"""

import json
import os
import sys
import types
import urllib.error
import urllib.request
from pathlib import Path

# Make `import client` / `catalog` / `profile` work from any CWD by putting the repo
# root (the parent of this script's directory) on sys.path — ahead of the stdlib,
# whose own `profile` module would otherwise win the name.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import providers.base  # noqa: F401
except ImportError:
    # `profile.py` needs Hermes' ProviderProfile. The same minimal stub
    # tests/conftest.py installs, so this script runs with or without Hermes on
    # sys.path; the Hermes-only hooks then report "unavailable" instead of being faked.
    _providers = types.ModuleType("providers")
    _base = types.ModuleType("providers.base")

    class _ProviderProfile:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

        def get_max_tokens(self, model):
            return getattr(self, "default_max_tokens", None)

        def get_model_context_length(self, model):
            return getattr(self, "model_context_length", None)

    _base.ProviderProfile = _ProviderProfile
    _providers.base = _base
    sys.modules["providers"] = _providers
    sys.modules["providers.base"] = _base

import catalog  # noqa: E402
import client  # noqa: E402
import profile as plugin  # noqa: E402

CHAT_TIMEOUT_S = 120
GET_TIMEOUT_S = 15.0
# Used only when the loaded model reports no window (and so has no declared cap).
# Must stay above llama-swap's own minimum thinking budget (1280 measured on
# v268-14-gea63ff), or the thinking-on check measures the server's gate instead
# of this plugin's reasoning path.
THINKING_BUDGET_FALLBACK = 2048
# The same oversized prompt Task 4 Step 3 captured: far past any reported window,
# so the engine's own rejection is what comes back.
OVERSIZED_PROMPT = "the quick brown fox " * 90000

_swap = plugin.build_profile(os.environ)
BASE_URL = _swap.base_url
ROOT = client.root_url(BASE_URL)

# Filled by check_loaded: the one model id inference may name.
_state: dict = {}


def _snippet(body, limit=600):
    text = body if isinstance(body, str) else json.dumps(body, ensure_ascii=False)
    return text if len(text) <= limit else text[:limit] + "..."


def _post_chat(payload):
    """POST to {BASE_URL}/chat/completions → (status, parsed body).

    Non-2xx comes back through HTTPError with its body parsed, because the body IS
    the deliverable for the two capture checks. Transport errors raise; the runner
    turns them into a failed check.
    """
    request = urllib.request.Request(
        f"{BASE_URL}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Accept": "application/json", "Content-Type": "application/json"},
    )
    if _swap.api_key:
        request.add_header("Authorization", f"Bearer {_swap.api_key}")
    try:
        with urllib.request.urlopen(request, timeout=CHAT_TIMEOUT_S) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", "replace")
        try:
            return exc.code, json.loads(raw)
        except json.JSONDecodeError:
            return exc.code, raw


def _chat(extra, max_tokens=64):
    payload = {
        "model": _state["loaded"],
        "messages": [{"role": "user", "content": "Reply with exactly: LLAMASWAP_SMOKE_OK"}],
        "stream": False,
        "max_tokens": max_tokens,
    }
    payload.update(extra)
    return _post_chat(payload)


def _first_message(body):
    choices = body.get("choices") if isinstance(body, dict) else None
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        message = choices[0].get("message")
        if isinstance(message, dict):
            return message
    return {}


def _reasoning_tokens(body):
    """`usage.completion_tokens_details.reasoning_tokens`, or ``None`` when unstated."""
    usage = body.get("usage") if isinstance(body, dict) else None
    details = usage.get("completion_tokens_details") if isinstance(usage, dict) else None
    value = details.get("reasoning_tokens") if isinstance(details, dict) else None
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _skipped_by(body):
    """`usage.completion_tokens_details.reasoning_skipped_by`, or ``None``.

    Newer llama-swap builds (measured: v268-14-gea63ff) answer 200 with thinking
    silently *off* when ``max_tokens`` is below the server's own minimum thinking
    budget, and name the reason here — ``min_thinking_tokens``.
    """
    usage = body.get("usage") if isinstance(body, dict) else None
    details = usage.get("completion_tokens_details") if isinstance(usage, dict) else None
    value = details.get("reasoning_skipped_by") if isinstance(details, dict) else None
    return value if isinstance(value, str) and value else None


def _thinking_budget():
    """The output budget Hermes itself sends on a turn: the plugin's declared cap.

    Hermes starts its reserve-and-shorten recovery from ``get_max_tokens``, so that
    number — not an arbitrarily small probe budget — is the shape that has to keep
    thinking alive. A model with no reported window has no declared cap, and then
    ``THINKING_BUDGET_FALLBACK`` stands in: still above the server-side thinking
    gate measured on v268 (1280), so a pass here means thinking works, not that the
    probe was too small to reach it.
    """
    cap = _swap.get_max_tokens(_state.get("loaded"))
    return cap if isinstance(cap, int) and cap > 0 else THINKING_BUDGET_FALLBACK


def check_build_and_models():
    version = client.get_json(ROOT, "/api/version", timeout=GET_TIMEOUT_S)
    build = str(version.get("version") or "") if isinstance(version, dict) else ""
    if not build:
        return False, f"GET {ROOT}/api/version -> no version field: {_snippet(version)}"
    models = client.get_json(BASE_URL, "/models", timeout=GET_TIMEOUT_S)
    listed = models.get("data") if isinstance(models, dict) else None
    if not isinstance(listed, list) or not listed:
        return False, f"GET {BASE_URL}/models -> no model list: {_snippet(models)}"
    commit = str(version.get("commit") or "")
    return True, (f"llama-swap {build} (commit {commit}, built "
                  f"{version.get('build_date') or '?'}), {len(listed)} models listed")


def check_catalog():
    ids = _swap.fetch_models()
    if not ids:
        return False, f"fetch_models() -> {ids!r} (None means the read failed; [] is never returned)"
    infos = _swap.warm()
    facts = []
    for model_id in ids:
        info = infos[model_id]
        facts.append(
            f"{model_id}: ctx={info.context_length} vision={info.vision} "
            f"tools={info.tools} efforts={list(info.efforts or ())} loaded={info.loaded}"
        )
    return True, f"{len(ids)} ids — " + " | ".join(facts)


def check_budget_and_vocabulary():
    infos = _swap.warm()
    if not infos:
        return False, "catalog shard is empty; run the catalog check first"
    lines = []
    for model_id, info in infos.items():
        cap = _swap.get_max_tokens(model_id)
        efforts = _swap.supported_reasoning_efforts(model_id)
        if info.context_length is None:
            # The one case where "no claim" is the only honest answer.
            if cap is not None:
                return False, f"{model_id}: window unknown but cap {cap} was invented"
            lines.append(f"{model_id}: cap=None(no window) efforts={list(efforts) if efforts is not None else None}")
            continue
        expected = min(_swap.default_max_tokens, max(_swap.min_output_tokens, info.context_length // 4))
        if cap != expected:
            return False, (f"{model_id}: cap {cap} != min(default_max={_swap.default_max_tokens}, "
                           f"max(floor={_swap.min_output_tokens}, {info.context_length}//4)) = {expected}")
        lines.append(f"{model_id}: cap={cap} efforts={list(efforts) if efforts is not None else None}")
    return True, f"min({_swap.default_max_tokens}, max({_swap.min_output_tokens}, window//4)) held — " + " | ".join(lines)


def check_loaded():
    ready = catalog.parse_running(client.get_json(ROOT, "/running", timeout=GET_TIMEOUT_S))
    loaded = [model_id for model_id in _swap.warm() if model_id in ready]
    if not loaded:
        return False, (f"no model is ready ({sorted(ready) or 'none reported'}); refusing to "
                       f"start an engine — load one in llama-swap and re-run")
    _state["loaded"] = loaded[0]
    return True, f"ready={sorted(ready)} → inference targets {loaded[0]!r} (no swap forced)"


def check_thinking_on():
    budget = _thinking_budget()
    status, body = _chat({"reasoning_effort": "high"}, max_tokens=budget)
    if status != 200:
        return False, f"reasoning_effort=high (max_tokens={budget}) -> HTTP {status}: {_snippet(body)}"
    message = _first_message(body)
    tokens = _reasoning_tokens(body)
    reasoning = str(message.get("reasoning_content") or "")
    if not (tokens or reasoning.strip()):
        skipped = _skipped_by(body)
        why = (f"; the server skipped thinking (reasoning_skipped_by={skipped!r}) — its "
               f"min_thinking_tokens gate sits above max_tokens={budget}"
               if skipped else "")
        return False, (f"reasoning_effort=high -> 200 but thinking is OFF "
                       f"(reasoning_tokens={tokens}, reasoning_content={reasoning[:60]!r})"
                       f"{why}: {_snippet(body)}")
    content = str(message.get("content") or "").strip()
    return True, (f"reasoning_effort=high (max_tokens={budget}) -> 200, thinking ON "
                  f"(reasoning_tokens={tokens}, reasoning_content={reasoning[:40]!r}), "
                  f"content={content[:40]!r}")


def check_thinking_off():
    status, body = _chat({"reasoning_effort": "none"}, max_tokens=256)
    if status != 200:
        return False, f"reasoning_effort=none -> HTTP {status}: {_snippet(body)}"
    message = _first_message(body)
    tokens = _reasoning_tokens(body)
    reasoning = str(message.get("reasoning_content") or "")
    if tokens or reasoning.strip():
        return False, (f"reasoning_effort=none -> 200 but thinking is ON "
                       f"(reasoning_tokens={tokens}, reasoning_content={reasoning[:60]!r})")
    content = str(message.get("content") or "").strip()
    return True, (f"reasoning_effort=none -> 200, thinking OFF (reasoning_tokens={tokens}, "
                  f"no reasoning_content), content={content[:40]!r}")


def check_capture_oversized():
    """Prints the engine's rejection verbatim — the input to tests/fixtures/errors.json."""
    status, body = _post_chat({
        "model": _state["loaded"],
        "max_tokens": 8,
        "messages": [{"role": "user", "content": OVERSIZED_PROMPT}],
        "stream": False,
    })
    return True, f"HTTP {status} body={_snippet(body, 900)}"


def check_capture_unpublished_effort():
    """`ultra` is Hermes-only: no /v1/models ladder on this server publishes it.

    The body decides whether Hermes degrades the session (``is_reasoning_field_rejection``)
    or aborts the turn (``_V_FORMAT_ERROR``), so it is captured, not judged here.
    """
    published = {
        level
        for info in _swap.warm().values()
        for level in (info.efforts or ())
    }
    if "ultra" in published:
        return True, "skipped: this server publishes `ultra`, so it is not an unpublished level here"
    status, body = _chat({"reasoning_effort": "ultra"}, max_tokens=64)
    line = f"asked effort=ultra (published ladder union={sorted(published)}) -> HTTP {status} body={_snippet(body, 900)}"
    verdict = _classify_reasoning_rejection(status, body)
    return True, f"{line} || {verdict}"


def _classify_reasoning_rejection(status, body):
    """What Hermes' own gate makes of this body, when Hermes is importable.

    Mirrors the string core feeds it: ``_build_error_msg`` lowercases ``str(error)``
    (an ``APIStatusError`` renders as ``Error code: 400 - {json}``) and appends the
    body message; ``agent/auxiliary_client.py:3440`` passes ``str(exc)``.
    """
    try:
        from agent.error_classifier import is_reasoning_field_rejection
    except ImportError:
        return "is_reasoning_field_rejection: not measured (run under a Hermes PYTHONPATH)"
    message = "" if not isinstance(body, dict) else str((body.get("error") or {}).get("message") or "")
    error_msg = f"Error code: {status} - {json.dumps(body)}".lower()
    if message and message.lower() not in error_msg:
        error_msg += " " + message.lower()
    return f"is_reasoning_field_rejection -> {is_reasoning_field_rejection(error_msg)}"


def check_usage():
    """The counters, formatted by the plugin's own :func:`profile.usage_details`.

    ``usage_details`` is Hermes-free, so the lines render in both modes; the snapshot
    itself needs Hermes' ``AccountUsageSnapshot`` and is reported separately rather
    than faked when it is absent.
    """
    stats = client.get_json(ROOT, "/api/metrics/stats", timeout=GET_TIMEOUT_S)
    if not isinstance(stats, dict) or not stats:
        return False, f"GET {ROOT}/api/metrics/stats -> {_snippet(stats)}"
    version = client.get_json(ROOT, "/api/version", timeout=GET_TIMEOUT_S)
    lines = plugin.usage_details(stats, version if isinstance(version, dict) else {})
    if not lines:
        return False, f"server stated no counter /usage can render: {_snippet(stats, 300)}"
    snapshot = _swap.fetch_account_usage()
    shown = " / ".join(lines)
    if snapshot is None:
        return True, f"{shown} || fetch_account_usage() -> None (no Hermes snapshot type on sys.path)"
    # Compared against nothing: these are live server-lifetime totals and another client
    # is hitting the same box, so a second read legitimately differs from the first.
    return True, f"{shown} || fetch_account_usage() -> {snapshot.source!r}, available={snapshot.available}, {len(snapshot.details)} detail lines, windows={snapshot.windows}"


def check_vision_pick():
    """A vision ask must land on a model the server said can see, preferring a loaded one."""
    infos = _swap.warm()
    capable = [i for i in infos.values() if i.vision is True]
    if not capable:
        return False, f"no model reported vision, but default_vision_model() should still decline: {plugin.LlamaSwapProfile.default_vision_model(_swap)!r}"
    picked = _swap.default_vision_model()
    if picked is None:
        return False, f"vision-capable models {sorted(i.id for i in capable)} but default_vision_model() -> None"
    if picked not in {info.id for info in capable}:
        return False, f"default_vision_model() -> {picked!r}, which never reported vision"
    loaded_capable = {i.id for i in capable if i.loaded}
    if loaded_capable and picked not in loaded_capable:
        return False, f"default_vision_model() -> {picked!r} while {sorted(loaded_capable)} is loaded"
    return True, f"default_vision_model() -> {picked!r} (capable={sorted(i.id for i in capable)}, loaded-capable={sorted(loaded_capable)})"


def check_aux_pick():
    """Aux may only name a loaded engine — a side task must not block on a model load."""
    infos = _swap.warm()
    picked = _swap.resolve_aux_model()
    loaded = {i for i, info in infos.items() if info.loaded}
    if picked and picked not in loaded:
        return False, f"resolve_aux_model() -> {picked!r}, which is not loaded ({sorted(loaded)})"
    if not picked and loaded:
        return False, f"resolve_aux_model() -> '' while {sorted(loaded)} is loaded"
    vision_ask = _swap.resolve_aux_model(vision=True)
    if vision_ask and vision_ask not in {i for i, info in infos.items() if info.vision is True}:
        return False, f"resolve_aux_model(vision=True) -> {vision_ask!r}, which never reported vision"
    return True, f"resolve_aux_model() -> {picked!r} (loaded={sorted(loaded)}) | vision=True -> {vision_ask!r}"


def main():
    print(f"llama-swap smoke against {BASE_URL} (root {ROOT})", file=sys.stderr)
    checks = [
        ("build-and-models", check_build_and_models),
        ("catalog", check_catalog),
        ("output-cap-and-efforts", check_budget_and_vocabulary),
        ("loaded-model", check_loaded),
        ("thinking-on", check_thinking_on),
        ("thinking-off", check_thinking_off),
        ("capture-oversized", check_capture_oversized),
        ("capture-unpublished-effort", check_capture_unpublished_effort),
        ("usage", check_usage),
        ("vision-pick", check_vision_pick),
        ("aux-pick", check_aux_pick),
    ]
    for label, check in checks:
        try:
            ok, detail = check()
        except Exception as exc:  # transport, timeout, undecodable body
            ok, detail = False, f"{type(exc).__name__}: {exc}"
        print(f"{'PASS' if ok else 'FAIL'} {label}: {detail}")
        if not ok:
            print(f"{label} failed — later checks depend on it", file=sys.stderr)
            return 1
    print(f"all {len(checks)} checks passed", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
