# llama-swap Hermes Provider Plugin Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a Hermes `kind: model-provider` plugin that routes inference through a configurable llama-swap server, discovers every model with the metadata that server reports, and makes the context window and output budget follow what llama-swap actually loaded.

**Architecture:** A single `ProviderProfile` subclass plus four stdlib-only modules (HTTP client, catalog parser, error classifier, profile assembly). The plugin **declares** (context window, per-model output cap, capabilities, reasoning vocabulary) and lets Hermes' own recovery paths act on it — it never sends `max_tokens` on the request path, because no profile hook can see the caller's budget and a profile-supplied value would overwrite Hermes' internal ones.

**Tech Stack:** Python 3.9+ (Hermes runs 3.14 here), stdlib only (`urllib`, `json`, `dataclasses`, `contextvars`), pytest. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-10-25-llama-swap-hermes-provider-design.md` — read it before any task; every value below is drawn from it.

## Global Constraints

- Stdlib only. No third-party imports, no `requests`, no `httpx`. (Spec: "No Python dependencies beyond the standard library" pattern of the sibling `halogen-hermes`.)
- Hermes-only imports (`agent.*`, `hermes_cli.*`, `providers.*` beyond `providers.base`) must be **guarded** — lazy inside the function with a documented fallback — so the plugin's unit tests run with no Hermes installed.
- No hook may raise, and no hot-path hook (`get_max_tokens`, `supported_reasoning_efforts`, `build_api_kwargs_extras`, `get_model_context_length`) may block on network I/O. Only `fetch_models` and the first context-length probe do HTTP, and both are bounded.
- Only `GET` requests. Never call `/unload`, `/api/models/unload`, `/api/profiles/active`, or `/api/inflight/*/cancel`.
- No model id, context window, or capability is hardcoded anywhere. The literal `192.168.31.7:8080` appears once, as the `LLAMA_SWAP_BASE_URL` default.
- Every endpoint beyond `/v1/models` is optional: a 404/410/5xx or an unparseable body degrades to "unknown", never to an error. (Verified: the running build has no `/props` and no `/v1/messages`.)
- Env knobs and defaults, verbatim: `LLAMA_SWAP_BASE_URL` = `http://192.168.31.7:8080/v1`, `LLAMA_SWAP_API_KEY` (unset), `LLAMA_SWAP_CATALOG_TTL_S` = `60`, `LLAMA_SWAP_PROBE_TIMEOUT_S` = `3`, `LLAMA_SWAP_DEFAULT_MAX_TOKENS` = `8192`, `LLAMA_SWAP_MIN_OUTPUT_TOKENS` = `1024`. A malformed numeric value falls back to its default.
- Test command: `python3 -m pytest <path> -q`. The system `python3` has **no** pytest; pytest 9 is installed only for the Hermes-managed interpreter, so run tests as
  `/home/hermes/.hermes/tools/python-3.14.7+20260901-linux-x64/bin/python3 -m pytest …`, and add
  `PYTHONPATH="/home/hermes/.hermes/hermes-agent:/home/hermes/.hermes/installs/019d0d114d22b7e6/environments/948e0e5d652f48a5af57a34335d9fa11/venv/lib/python3.14/site-packages"`
  for the tests that import Hermes modules (Tasks 4 and 7). Without that path the Hermes-dependent tests skip, which is a pass — so the Hermes-backed run is the one that counts.
- Commit after every task with an imperative subject (`feat:`/`fix:`/`test:`/`docs:`).

## Review Focus

These are the inputs the spec implies that no single task's happy-path test covers. Each line is pinned by a named test in the task that owns the code.

- **Server unreachable or slow at session start** — a session must start after one bounded (`LLAMA_SWAP_PROBE_TIMEOUT_S`) wait, then keep working on Hermes' own context default; a dead server must not be re-probed on every hook call. → Task 5 `test_unreachable_server_probes_once_and_degrades`.
- **A 200 whose body is not the shape we expect** (`[...]` instead of `{"data": [...]}`, entries that are not dicts, missing `data`, HTML from a proxy) — discovery returns `None` so Hermes falls back to `fallback_models`, and never raises inside the picker. → Task 3 `test_parse_models_tolerates_junk_payloads`.
- **A model reporting nothing at all** (live example: `gufo` — no `capabilities`, no `context_length`) — no `False`/`0` is ever declared on its behalf; `model_capabilities` omits the absent fields and `get_max_tokens` returns `None`. → Task 3 `test_capabilities_omit_unreported_fields`, Task 6 `test_unknown_window_declares_no_cap`.
- **A reasoning vocabulary containing levels Hermes does not know** (a future `ultra`/`turbo`, or a non-ladder string) — dropped, never forwarded verbatim; a list that filters to empty is *unknown* (`None`), not *disabled* (`()`), because `()` tells the transport to omit reasoning fields entirely. → Task 6 `test_unknown_effort_levels_are_dropped_not_forwarded`.
- **`LLAMA_SWAP_BASE_URL` pointed at the wrong shape** — no `/v1` suffix, a trailing slash, an `https://` host that answers HTML, or a plain non-llama-swap OpenAI endpoint — must produce a clean "not enabled / no models" outcome, not a crash, and inference must still be attempted at the URL the user configured. → Task 2 `test_root_url_derives_server_root_from_base` and `test_get_json_returns_none_on_unreachable_or_malformed_base`, Task 9 `README.md` "Troubleshooting".
- **A redirect in the path to the server** — must not be followed, and must not replay the credential to the redirect target. → Task 2 `test_get_json_refuses_a_redirect_without_replaying_the_credential`. Consequence to document in Task 9: a deployment behind a proxy that issues a same-origin 3xx will read as "no models", by design.

---

## File Structure

```
llama-swap-hermes/
├── plugin.yaml              # manifest: kind: model-provider
├── __init__.py              # register_provider(build_profile(os.environ))
├── client.py                # one bounded stdlib JSON GET; never raises
├── catalog.py               # /v1/models + /running -> {id: ModelInfo}; Hermes-shaped facts
├── errors.py                # classify_llama_swap_error(...) -> dict | None
├── profile.py               # LlamaSwapProfile hooks + build_profile(env)
├── tests/
│   ├── conftest.py          # providers / providers.base stub when Hermes is absent, plus the
│   │                        # sys.path bootstrap. NO root conftest.py: pytest imports it as part of
│   │                        # the plugin package, which runs __init__.py (imports ``providers``)
│   │                        # before that stub exists, so collection dies in any legal-named clone.
│   ├── fixtures/            # payloads captured from the live server (Task 1)
│   └── test_client.py test_catalog.py test_errors.py test_profile.py test_wire.py
├── scripts/smoke.py         # live verification against a real llama-swap
└── README.md
```

`__init__.py` stays 2 lines of logic: Hermes imports it for the registration side effect, and everything testable lives in the named modules (the `halogen-hermes` shape).

---

### Task 1: Scaffold and live fixtures

**Files:**
- Create: `plugin.yaml`, `__init__.py`, `conftest.py`, `tests/conftest.py`, `tests/fixtures/models.json`, `tests/fixtures/running.json`, `tests/fixtures/errors.json`, `tests/test_fixtures.py`
- Modify: `.gitignore` (create if absent: `__pycache__/`, `*.pyc`, `.pytest_cache/`, `venv/`)

**Interfaces:**
- Consumes: nothing.
- Produces: the fixture files every later test loads, and the `providers`/`providers.base` stub used by Tasks 3-8.

- [ ] **Step 1: Write the manifest and package entry point**

`plugin.yaml`, mirroring the sibling's field set:

```yaml
name: llama-swap-provider
kind: model-provider
version: 0.1.0
description: llama-swap — model discovery, context windows and reasoning vocabularies read from a llama-swap server
author: dshagent
```

`__init__.py` (final form; it imports `profile`, which Task 5 creates, so make it import-tolerant until then by writing it in Task 5 instead — in this task create it empty with a `# implemented in Task 5` comment):

```python
"""llama-swap provider profile plugin for Hermes Agent."""
```

- [ ] **Step 2: Capture the live fixtures**

Against the configured server (operator-approved read-only GETs):

```bash
mkdir -p tests/fixtures
curl -sS http://192.168.31.7:8080/v1/models > tests/fixtures/models.json
curl -sS http://192.168.31.7:8080/running  > tests/fixtures/running.json
```

Capture what the server says **today** and commit it verbatim — never hand-edit a fixture, and never ask the operator to load a particular model so a test reads nicer. Which model happens to be loaded changes between design time and now (it was `strixite`, then `halogen`), so the tests below are written to derive that id from the fixture. If `"running"` ever comes back with two ready entries, that is a real change worth a plan edit — report it, do not work around it.

Then append to `tests/fixtures/errors.json` the four bodies captured during design (they are the classifier's corpus; Task 4 asserts against them):

```json
{
  "engine_empty_messages": {"status": 400, "body": {"error": {"message": "'messages' must be a non-empty array", "type": "invalid_request_error", "param": "messages", "code": null}}},
  "engine_negative_max_tokens": {"status": 400, "body": {"error": {"message": "'max_tokens' must be >= 1, got -5", "type": "invalid_request_error", "param": "max_tokens", "code": null}}},
  "engine_chat_template": {"status": 400, "body": {"error": {"message": "render_chat (/src/serve/chat_template.cpp:230): check `last_query >= 0` failed: chat template: No user query found in messages.", "type": "invalid_request_error", "param": "messages", "code": null}}},
  "swap_no_router": {"status": 404, "body": {"src": "llama-swap", "error": {"message": "no router for requested model", "type": "invalid_request_error", "param": null, "code": "not_found"}}}
}
```

- [ ] **Step 3: Write the failing fixture test**

`tests/test_fixtures.py` — the fixtures must contain the shapes the rest of the plan asserts on, so a silent server change is caught here (`import json, pathlib`, `FIX = pathlib.Path(__file__).parent / "fixtures"` at the top of the file). Note what is **not** asserted here: *which* model is loaded. That is transient runtime state (it is whichever model the last request touched, and pi itself runs against this server), so no test in this plan may name a loaded id — derive it from the fixture instead.

```python
def ready_ids():
    return [r["model"] for r in json.load(open(FIX / "running.json"))["running"] if r["state"] == "ready"]

def test_models_fixture_covers_every_reported_shape():
    ids = {m["id"] for m in json.load(open(FIX / "models.json"))["data"]}
    assert ids == {"crucible", "gufo", "halogen", "strixite"}   # crucible: context, no caps
                                                              # gufo: neither; halogen: vision+tools

def test_running_fixture_marks_exactly_one_model_ready():
    ready = ready_ids()
    assert len(ready) == 1 and ready[0] in {                     # the count is the contract,
        m["id"] for m in json.load(open(FIX / "models.json"))["data"]}   # the id is whatever the server had loaded
```

- [ ] **Step 4: Add the `providers` stub conftest**

Copy the mechanism from `/home/hermes/halogen-hermes/tests/conftest.py` verbatim in intent: try `import providers` / `import providers.base`, and on `ImportError` inject a `types.ModuleType` pair whose `ProviderProfile.__init__` copies kwargs to attributes and whose `register_provider` fills a module-level `REGISTRY` dict. This is what lets Tasks 3-8 run without Hermes installed.

- [ ] **Step 5: Run the tests and confirm they pass**

Run: `python3 -m pytest tests/test_fixtures.py -v`
Expected: 2 passed.

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "feat: scaffold llama-swap provider plugin with live fixtures"
```

---

### Task 2: Bounded JSON client

**Files:**
- Create: `client.py`, `tests/test_client.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `DEFAULT_BASE_URL: str` = `"http://192.168.31.7:8080/v1"`
  - `normalize_base_url(value: str) -> str` — strips whitespace and trailing `/`
  - `root_url(base_url: str) -> str` — `http://h:8080/v1` → `http://h:8080` (no `/v1` suffix → unchanged)
  - `get_json(base_url: str, path: str, *, api_key: str | None = None, timeout: float = 3.0) -> Any | None` — `GET {normalize_base_url(base_url)}{path}`; returns parsed JSON, or `None` on any transport error, non-2xx, or non-JSON body. Never raises.
  - `Env` readers used by Task 5: `env_int(env, name, default) -> int`, `env_str(env, name, default) -> str` (defined in `client.py` next to the URL helpers, mirroring how `halogen-hermes/profile.py` keeps env parsing beside the defaults).

- [ ] **Step 1: Write the failing tests**

In `tests/test_client.py`:

```python
def test_get_json_returns_parsed_body_on_200(...)      # stub opener → {"data": []}
def test_get_json_returns_none_on_404(...)             # urllib.error.HTTPError → None, no raise
def test_get_json_returns_none_on_non_json_body(...)   # an HTML proxy page → None
def test_get_json_sends_bearer_only_when_key_given(...)# captured Request has Authorization iff api_key
def test_get_json_refuses_a_redirect_without_replaying_the_credential():
    # a real 302 from a loopback http.server: None, the target never fetched, no second
    # Authorization header. This is the credential-leak path; it is tested, not reasoned about.
def test_get_json_returns_none_on_unreachable_or_malformed_base():
    # a closed loopback port, "not a url", "http://h.invalid:99999/v1", "h.invalid:8080/v1", "",
    # "ftp://h.invalid/v1" — every one returns None. Use `.invalid` (RFC 2606: never resolvable):
    # a bare "h" costs a real getaddrinfo round trip (~4 s here, and `timeout` does not bound DNS),
    # which would make the whole suite 200× slower for no reason. The malformed ones are the
    # regression guard: urllib raises ValueError for some of them at Request-construction time,
    # outside the try.
def test_env_readers_fall_back_on_junk():
    assert env_int({"LLAMA_SWAP_CATALOG_TTL_S": "soon"}, "LLAMA_SWAP_CATALOG_TTL_S", 60) == 60
    assert env_int({}, "LLAMA_SWAP_CATALOG_TTL_S", 60) == 60          # absent → default, never 0
    assert env_int({"LLAMA_SWAP_MIN_OUTPUT_TOKENS": "1024"}, "LLAMA_SWAP_MIN_OUTPUT_TOKENS", 1024) == 1024
    assert env_str({"LLAMA_SWAP_BASE_URL": "  http://h.invalid:1/v1/  "}, "LLAMA_SWAP_BASE_URL", "") == "http://h.invalid:1/v1/"
def test_root_url_derives_server_root_from_base():
    assert root_url("http://192.168.31.7:8080/v1") == "http://192.168.31.7:8080"
    assert root_url("http://192.168.31.7:8080/") == "http://192.168.31.7:8080"
    assert root_url("http://host:9000/api") == "http://host:9000/api"
```

The two `...`-argument tests are written out in full by the implementer (stub the opener the way the first four do); the point of naming them here is that the **redirect and malformed-URL paths are covered by committed tests**, not by a throwaway loopback script — they are the paths the spec's anti-goals turn on, and a later refactor of the `try` boundary is exactly what would break them silently. This list is the complete set: **8 test functions, 8 passed** — do not leave a superseded test alongside its replacement to hit a number.

- [ ] **Step 2: Run them to verify they fail**

Run: `python3 -m pytest tests/test_client.py -v`
Expected: collection error, `ModuleNotFoundError: No module named 'client'`.

- [ ] **Step 3: Implement `client.py`**

`urllib.request` with an `HTTPSHandler`-free default opener; a redirect must not carry credentials across origins, so build the opener with `urllib.request.build_opener(urllib.request.HTTPRedirectHandler())` replaced by a handler that raises on 3xx (Hermes does the same thing centrally with `hermes_cli.urllib_security.open_credentialed_url`, which is not importable from a standalone plugin — note that in a comment). Add `socket.timeout`/`urllib.error.URLError` to the tolerated-exception tuple. Timeout is the caller's argument, never a literal.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python3 -m pytest tests/test_client.py -v`
Expected: 8 passed, in well under a second. If it takes seconds, a test is doing a real DNS lookup — fix the host name, do not accept the wait.

- [ ] **Step 5: Commit**

```bash
git add client.py tests/test_client.py && git commit -m "feat: add bounded stdlib JSON client for llama-swap"
```

---

### Task 3: Catalog parsing

**Files:**
- Create: `catalog.py`, `tests/test_catalog.py`

**Interfaces:**
- Consumes: `client.get_json`, `client.root_url` (Task 2).
- Produces:
  - `@dataclass ModelInfo`: `id: str`, `context_length: int | None`, `vision: bool | None`, `tools: bool | None`, `efforts: tuple[str, ...] | None`, `reasoning_enabled: bool | None`, `loaded: bool = False`, `aliases: tuple[str, ...] = ()`, `input_modalities: tuple[str, ...] = ()`, `owned_by: str = ""`, `name: str = ""`, `description: str = ""`
  - `parse_models(payload: Any) -> dict[str, ModelInfo]` — accepts the `{"data": [...]}` shape **or** a bare list; preserves server order; skips non-dict entries and entries with no non-empty `id`.
  - `parse_running(payload: Any) -> frozenset[str]` — ids whose entry has `state == "ready"`.
  - `mark_loaded(infos: dict[str, ModelInfo], ready: frozenset[str]) -> None` — sets `.loaded` in place (also true for an alias of a ready id, matching `meta.llamaswap.aliases`).
  - `capabilities_entry(info: ModelInfo) -> dict[str, Any]` — only the keys the server reported, using Hermes' `model_overrides` names: `supports_tools`, `supports_vision`, `context_window`.
  - `EFFORT_LADDER_FALLBACK: tuple[str, ...]` = `("none", "minimal", "low", "medium", "high", "xhigh", "max", "ultra")` — used **only** when `agent.reasoning_effort.EFFORT_LADDER` cannot be imported (the plugin must stay importable without Hermes). `supported_efforts` imports the real ladder inside the function and prefers it; the local tuple is the fallback, with a comment naming that file as the source of truth.
  - `supported_efforts(info: ModelInfo, ladder: Sequence[str] | None = None) -> tuple[str, ...] | None` — `None` selects the imported-or-fallback ladder. The tri-state contract from `providers/base.py`: `None` = model said nothing (or nothing it said is on the ladder), `()` = reasoning disabled, else the filtered tuple in ladder order.
  - `fetch_catalog(base_url: str, api_key: str | None, timeout: float) -> tuple[dict[str, ModelInfo], str | None]` — one `/models` GET plus one best-effort `{root}/running` GET; returns `(infos, error)` where `error` is a short string when `/models` failed and `infos` is empty. Never raises.

- [ ] **Step 1: Write the failing tests against the captured fixtures**

`tests/test_catalog.py` loads `models.json` once at module scope through `parse_models` and exposes it as the `infos` fixture, so every assertion below is about the live server's real shape, not a hand-written one.

```python
def test_parses_every_field_the_server_reports(infos):
    hal = infos["halogen"]
    assert (hal.context_length, hal.vision, hal.tools) == (262144, True, True)
    assert hal.efforts[-1] == "max" and hal.reasoning_enabled is True
    assert infos["strixite"].efforts[-1] == "xhigh"          # per-model, not server-wide

def test_reports_context_without_capabilities(infos):        # crucible
    assert infos["crucible"].context_length == 262144
    assert infos["crucible"].vision is None and infos["crucible"].tools is None

def test_capabilities_omit_unreported_fields(infos):         # gufo: nothing reported
    assert infos["gufo"].context_length is None
    assert capabilities_entry(infos["gufo"]) == {}

def test_capabilities_entry_uses_hermes_field_names(infos):
    assert capabilities_entry(infos["halogen"]) == {
        "supports_tools": True, "supports_vision": True, "context_window": 262144}

def test_unknown_effort_levels_are_dropped_not_forwarded():
    info = ModelInfo(id="x", context_length=None, vision=None, tools=None,
                     efforts=("low", "turbo", "xhigh"), reasoning_enabled=True)
    assert supported_efforts(info) == ("low", "xhigh")

def test_disabled_reasoning_is_the_empty_tuple_not_none():
    info = ModelInfo(id="x", context_length=None, vision=None, tools=None,
                     efforts=("low", "high"), reasoning_enabled=False)
    assert supported_efforts(info) == ()

def test_all_unknown_levels_read_as_unknown_not_disabled():
    info = ModelInfo(id="x", context_length=None, vision=None, tools=None,
                     efforts=("turbo",), reasoning_enabled=True)
    assert supported_efforts(info) is None

@pytest.mark.parametrize("junk", [None, [], {}, {"data": None}, {"data": [1, "x", None]}, "<html>"])
def test_parse_models_tolerates_junk_payloads(junk):
    assert parse_models(junk) == {}

def test_parse_running_requires_ready_state():
    assert parse_running({"running": [{"model": "a", "state": "ready"},
                                      {"model": "b", "state": "starting"}]}) == frozenset({"a"})
```

- [ ] **Step 2: Run them to verify they fail**

Run: `python3 -m pytest tests/test_catalog.py -v`
Expected: collection error, `ModuleNotFoundError: No module named 'catalog'`.

- [ ] **Step 3: Implement `catalog.py`**

`parse_models` reads `payload.get("data")` when it is a dict, else `payload` when it is a list, else `[]`. Nested reads go through one small `_dig(mapping, *keys)` helper (as the bundled `router` plugin does) so a missing level yields `None`, never a `KeyError`. `efforts` comes from `meta.llamaswap.reasoning.efforts`, `reasoning_enabled` from the sibling `enabled`; both stay `None` when absent. `context_length` accepts `context_length` then `context_window` then `meta.n_ctx`, first positive int wins.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python3 -m pytest tests/test_catalog.py -v`
Expected: all passed.

- [ ] **Step 5: Commit**

```bash
git add catalog.py tests/test_catalog.py && git commit -m "feat: parse llama-swap model catalog into Hermes-shaped facts"
```

---

### Task 4: Error classification

**Files:**
- Create: `errors.py`, `tests/test_errors.py`

**Interfaces:**
- Consumes: `tests/fixtures/errors.json` (Task 1).
- Produces: `classify_llama_swap_error(error, *, status_code, error_code, message, body, model) -> dict | None` — the exact `ProviderProfile.classify_api_error` contract (`{"reason": <FailoverReason name>, ...flags}` or `None` to decline).

- [ ] **Step 1: Write the corpus test that measures Hermes first**

This task's rules are earned, not invented. The test file's first job is to record what Hermes' own classifier says about each captured body, so a rule is only added where core is wrong:

```python
CORPUS = json.load(open(FIX / "errors.json"))

def _kwargs(case):                      # mirrors agent/error_classifier._Ctx
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
```

`EXPECTED` maps each case to `{"hermes": <recorded in Step 2>, "plugin": <our verdict or None>}`. Document every non-`None` "plugin" entry with the reason it diverges, in a comment.

- [ ] **Step 2: Run Hermes' classifier over the corpus and record the truth**

`classify_api_error(error, *, provider, model, approx_tokens, context_length, num_messages, base_url, api_key)` takes an **exception**, and derives status/body/code from it (`_extract_status_code`, `_extract_error_body`, `_extract_error_code`). Reproduce that shape so the probe measures the real path:

```bash
cat > /tmp/corpus_probe.py <<'PY'
import json
from agent.error_classifier import classify_api_error

class _Resp(Exception):                      # what openai.APIStatusError looks like to Hermes
    def __init__(self, status, body):
        super().__init__(f"Error code: {status} - {json.dumps(body)}")
        self.status_code, self.body = status, body

corpus = json.load(open(__import__("os").environ["CORPUS"]))
for name in sorted(corpus):
    case = corpus[name]
    if "status" not in case:
        print(name, "-> n/a (engine banner, not an HTTP error)")
        continue
    v = classify_api_error(_Resp(case["status"], case["body"]), provider="llama-swap",
                           model="strixite", base_url="http://192.168.31.7:8080/v1")
    print(name, "->", getattr(getattr(v, "reason", None), "value", v))
PY
REPO=$(pwd)                      # capture before cd: the probe reads fixtures by absolute path
cp /tmp/corpus_probe.py /home/hermes/.hermes/hermes-agent/_corpus_probe.py
( cd /home/hermes/.hermes/hermes-agent && CORPUS="$REPO/tests/fixtures/errors.json" \
    /home/hermes/.local/bin/hermes --run-module _corpus_probe )
rm -f /home/hermes/.hermes/hermes-agent/_corpus_probe.py
```

Write each `name -> reason` line into `EXPECTED[...]["hermes"]` in the test file.

- [ ] **Step 3: Capture the two overflow bodies the corpus is still missing**

Operator-approved (this starts engines; expect minutes): one output-cap and one input-overflow error from the real stack, appended to `tests/fixtures/errors.json` under `engine_output_cap` and `engine_input_overflow`.

```bash
# Input overflow: a prompt far past the reported window.
python3 - <<'PY'
import json, urllib.request
big = "the quick brown fox " * 90000
req = urllib.request.Request("http://192.168.31.7:8080/v1/chat/completions",
    data=json.dumps({"model": "strixite", "max_tokens": 8,
                     "messages": [{"role": "user", "content": big}]}).encode(),
    headers={"Content-Type": "application/json"})
try:
    print(200, urllib.request.urlopen(req, timeout=900).read()[:400])
except urllib.error.HTTPError as exc:
    print(exc.code, exc.read().decode()[:600])
PY
```

Then re-run Step 2 for the new cases. If the server **silently clamps** instead of erroring (the behaviour already observed at `max_tokens: 262144`, which returned `200` with `strix.max_tokens_clamped: true`), record that fact in `errors.json` as `"engine_clamps_instead_of_erroring": true` and do **not** invent a rule for a body that does not exist — the declared cap in Task 6 is the defence, exactly as the spec reasons.

- [ ] **Step 4: Implement `errors.py`**

Two rules, both narrow (the `halogen-hermes` precedent: it claims only the 400 budget branch and leaves 401/403/429/5xx to core). Add a rule **only if Step 2 or Step 3 shows core getting that case wrong** — a rule that reproduces core's own verdict is dead code and Hermes' precedence comment says so:

1. llama-swap's own dispatch failures — identified by `body["src"] == "llama-swap"` with `code == "not_found"` — return `{"reason": "model_not_found", "retryable": False, "should_fallback": True}`, mirroring core's own `_V_MODEL_NOT_FOUND` / `_ABORT_FALLBACK`. Step 2 **measured** core to get this case wrong: `_MODEL_NOT_FOUND_PATTERNS` (`agent/error_classifier.py:273-280`) has no "no router" wording, and `_model_id_missing_known_prefix` is False for a provider the curated catalogue doesn't know (`:1471` — deliberately, so genuine endpoint problems stay retryable `unknown`). A wrong or renamed model id is therefore retried to exhaustion instead of aborting to the fallback chain.
   _This corrects an earlier draft of this plan, which asserted core's verdict was already right. It was asserted, not measured — which is what Step 2 is for._
2. Context-overflow wordings the corpus proves core misses → `{"reason": "context_overflow", "should_compress": True}`. Only add the specific substrings measured in Step 2/3, each with a comment naming the captured case it came from. If Step 3 shows the stack **truncating or silently clamping** instead of erroring, there is no body to earn a rule from and this rule is **not implemented** — record that outcome in the corpus and rely on the declared cap (Task 6) plus Hermes' own reactive path.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `python3 -m pytest tests/test_errors.py -v`
Expected: all passed, with the Hermes-divergence cases either skipped (no Hermes) or asserting the recorded verdict.

- [ ] **Step 6: Commit**

```bash
git add errors.py tests/test_errors.py tests/fixtures/errors.json && git commit -m "feat: classify llama-swap error bodies only where Hermes is wrong"
```

---

### Task 5: Profile assembly, discovery and warming

**Files:**
- Create: `profile.py`, `tests/test_profile.py`; Modify: `__init__.py`

**Interfaces:**
- Consumes: `client.*` (Task 2), `catalog.*` (Task 3).
- Produces:
  - `class LlamaSwapProfile(ProviderProfile)` with the fields and hooks of Tasks 5-8
  - `build_profile(env: Mapping[str, str]) -> LlamaSwapProfile`
  - `profile.warm(*, base_url: str | None = None, timeout: float | None = None, force: bool = False) -> dict[str, ModelInfo]` — the single place that does HTTP for the catalog; caches by normalized base URL; on failure caches the failure for the TTL.
  - Module-level cache accessor `catalog_for(profile, model: str) -> ModelInfo | None` (used by the tests and by Tasks 6-8 hooks).
  - `LlamaSwapProfile.classify_api_error(error, *, status_code, error_code, message, body, model)` — thin delegation to `errors.classify_llama_swap_error` (Task 4), wrapped so a raised exception inside the rules degrades to `None`.

- [ ] **Step 1: Write the failing tests**

Three fixtures, added to `tests/conftest.py` in this task: `swap` (a profile built by `build_profile({})` whose catalog comes from the captured fixtures), `calls` (a list of every URL the stubbed `client.get_json` served), and `clock` (monkeypatches `time.monotonic`, which is what the cache stamps `at` with, so TTL expiry is testable without sleeping). Two helpers, also in `tests/conftest.py`: `stub_models(swap, infos)` repoints the stub at an arbitrary `{id: ModelInfo}` catalog (use it wherever the captured server publishes nothing relevant to a test) and `_info(id, **facts)` builds a `ModelInfo` with the unset facts left `None`.

One trap in that signature: `_info` already passes all five core fields to `ModelInfo`, so `_info("x", context_length=…)` hands the keyword twice and raises `TypeError`. Task 6 added `_stated(info, **facts)` (a `dataclasses.replace`) for exactly this; it lives in `tests/test_profile.py` for now, and any later task that needs it moves it into `tests/conftest.py` beside `_info` rather than redefining it.

```python
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

def test_ttl_expiry_triggers_a_second_fetch(swap, calls, clock):
    swap.fetch_models(); n = len(calls)
    clock.advance(61)
    swap.fetch_models()
    assert len(calls) > n

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
    finally:
        for name in [n for n in sys.modules if n.startswith(module_name)]:
            sys.modules.pop(name, None)
        # unregister: the stub registry is module-global, so restore the snapshot rather
        # than leaving this test's profile registered for the rest of the session.
        providers.REGISTRY.clear()
        providers.REGISTRY.update(saved)
```

The stub in `tests/conftest.py` must therefore grow a third export alongside `ProviderProfile` and
`register_provider`: `get_provider_profile(name)` returning `REGISTRY.get(name)`, plus the
`needs_stub_registry` marker the sibling already uses. Comment both: they mirror
`providers/__init__.py:192` and `:55` so the same assertion passes with and without Hermes installed.

`profile.py` keeps that name (it is what the sibling ships and what Hermes' loader expects to find
through the package). Note in `__init__.py` why the import order is `try: from .profile import …`
FIRST: under Hermes the plugin dir is a package, so the relative import is the only correct one —
the absolute `from profile import …` fallback is for flat test imports only, and `profile` is a real
stdlib module in this interpreter (`lib/python3.14/profile.py`), so if it ever won the race the
failure would be a confusing `AttributeError` on a profiler module rather than a clean ImportError.


```python
# And LlamaSwapProfile declares `api_mode: str = "chat_completions"` explicitly, with a one-line
# comment that it restates the built-in rather than overriding it: the stub's ProviderProfile is a
# plain kwargs-copying class with no dataclass defaults (providers/base.py:53-56 holds the real ones),
# so a test that read an inherited default would pass under Hermes and AttributeError under the stub.

def test_error_hook_delegates_and_never_raises(swap, monkeypatch):
    from errors import classify_llama_swap_error
    case = CORPUS["swap_no_router"]
    assert swap.classify_api_error(None, **_kwargs(case)) == classify_llama_swap_error(**_kwargs(case))
    monkeypatch.setattr(errors, "classify_llama_swap_error", lambda **kw: 1 / 0)
    assert swap.classify_api_error(None, status_code=400, error_code="", message="", body={}, model="x") is None
```

`_kwargs` and `CORPUS` are Task 4's; move them into `tests/conftest.py` here so both files share them.

- [ ] **Step 2: Run them to verify they fail**

Run: `python3 -m pytest tests/test_profile.py -v`
Expected: collection error, `ModuleNotFoundError: No module named 'profile'`.

- [ ] **Step 3: Implement `warm`, `fetch_models`, `get_model_context_length`, `classify_api_error` and the capability refresh in `profile.py`**

Cache shape: `{base_url: {"at": float, "infos": dict[str, ModelInfo]}}` guarded by one `threading.Lock`, and per-`$HERMES_HOME` only if the module-level dict is shared across profiles in a way the bundled `router` plugin guards against — it is not here (one profile instance per process), so a plain module dict plus the lock is enough; note that in a comment. There is deliberately **no** `failed` flag: the entry's *presence* is what distinguishes "never probed" (`_shard_for` → `None`) from "reached, nothing usable" (`{}`), and both answer "no claim", so a down server and a server that reports no models share one representation. `fetch_models` returns `None` on failure (never `[]`, so Hermes keeps its own fallback path), `list(infos)` in server order on success. `get_model_context_length` calls `warm()` when the shard is absent **or past the TTL** (bounded by `LLAMA_SWAP_PROBE_TIMEOUT_S`), and answers cache-only otherwise. The TTL governs the read path too: a failed probe is cached like a successful one, so two reads inside the TTL still cost one probe, but a server that was down at session start is retried once the TTL expires instead of being latched off for the life of the process. The capability refresh replaces the contents of the existing `model_capabilities` dict in place (`clear()` + `update()`) — `agent/models_dev.py` reads it by reference on every lookup, so a rebound dict would be missed.

- [ ] **Step 4: Write `__init__.py`**

```python
"""llama-swap provider profile plugin for Hermes Agent."""
import os

from profile import build_profile  # Hermes imports this module for the registration side effect

from providers import register_provider

register_provider(build_profile(os.environ))
```

Use the same relative/absolute import fallback the sibling uses (`try: from .profile import ...` / `except ImportError:`) so the package imports both as a Hermes plugin directory and as a flat repo.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `python3 -m pytest tests -v`
Expected: all passed.

- [ ] **Step 6: Commit**

```bash
git add profile.py __init__.py tests/test_profile.py && git commit -m "feat: discover llama-swap models and probe context at session start"
```

---

### Task 6: Declared budget and per-model vocabulary

**Files:**
- Modify: `profile.py`, `tests/test_profile.py`

**Interfaces:**
- Consumes: `catalog_for`, `ModelInfo`, `supported_efforts` (Tasks 3, 5).
- Produces: `LlamaSwapProfile.get_max_tokens`, `.get_model_context_length` (Task 5), `.supported_reasoning_efforts`, `.resolve_aux_model`, `.default_vision_model`, `.default_reasoning_config`, and `output_cap(window: int | None, default_max: int, floor: int) -> int | None` (module-level in `profile.py` so it is testable without a profile).

- [ ] **Step 1: Write the failing tests**

```python
def test_output_cap_is_a_cap_never_a_raise():
    assert output_cap(262144, default_max=8192, floor=1024) == 8192      # big window keeps the default
    assert output_cap(8192, default_max=8192, floor=1024) == 2048        # window // 4
    assert output_cap(2048, default_max=8192, floor=1024) == 1024        # floor
    assert output_cap(512, default_max=8192, floor=1024) == 1024         # floor, not window//4

def test_unknown_window_declares_no_cap():
    assert output_cap(None, default_max=8192, floor=1024) is None

def test_get_max_tokens_follows_the_models_window(swap):
    assert swap.get_max_tokens("strixite") == 8192                       # 262144 // 4 > default
    stub_models(swap, {"tiny": _stated(_info("tiny"), context_length=8192)})
    assert swap.get_max_tokens("tiny") == 2048                           # window // 4 wins
    assert swap.get_max_tokens("gufo") is None                           # never reported

def test_hot_path_hooks_never_touch_the_network(swap, calls):
    before = len(calls)
    swap.supported_reasoning_efforts("halogen")
    swap.get_max_tokens("halogen")
    swap.resolve_aux_model()
    swap.default_vision_model()
    assert len(calls) == before

def test_supported_reasoning_efforts_is_per_model_not_server_wide(swap):
    assert swap.supported_reasoning_efforts("halogen") == (
        "none", "minimal", "low", "medium", "high", "xhigh", "max")
    assert swap.supported_reasoning_efforts("strixite")[-1] == "xhigh"

def test_model_publishing_nothing_has_no_declared_vocabulary(swap):
    stub_models(swap, {"quiet": _info("quiet")})       # no reasoning block at all
    assert swap.supported_reasoning_efforts("quiet") is None

# NOT gufo: the captured server publishes a full reasoning vocabulary for every model,
# including gufo (verified against tests/fixtures/models.json). "Reports no capabilities"
# and "reports no reasoning vocabulary" are different facts, and only the second is unknown here.

def test_aux_prefers_an_already_loaded_model(swap):
    assert swap.resolve_aux_model() == ready_ids()[0]   # the fixture's one ready model

def test_vision_default_comes_from_the_catalog(warm_swap):
    # NOT "halogen": the loaded id is runtime state (it has been two different models during
    # this plan), and the fixtures happen to make the one loaded model the vision-capable one.
    assert warm_swap.default_vision_model() == ready_ids()[0]

def test_an_unloaded_vision_model_is_still_the_vision_answer(warm_swap):
    # The asymmetry is deliberate (design doc "auxiliary defaults"): declining for aux is free,
    # declining for vision sends images to the main chat model (agent/auxiliary_client.py:5613),
    # which is usually text-only here. So aux requires `loaded`; vision prefers it.
    stub_models(warm_swap, {"eyes": _stated(_info("eyes"), vision=True)})   # reports image input, unloaded
    assert warm_swap.resolve_aux_model() == ""
    assert warm_swap.default_vision_model() == "eyes"

def test_catalog_with_nothing_loaded_answers_no_aux(warm_swap):
    stub_models(warm_swap, {"gufo": _info("gufo")})      # unloaded, no capabilities: nothing to pick
    assert warm_swap.resolve_aux_model() == ""
    assert warm_swap.default_vision_model() is None

# NOTE for the implementer: do not assert this by re-stubbing client.get_json — the swap
# fixture has already filled the cache, so the warm catalog would still answer. Replace the
# catalog itself, as stub_models does.

def test_unset_reasoning_defaults_to_medium_not_the_engine_ceiling(swap):
    assert swap.default_reasoning_config("strixite") == {"enabled": True, "effort": "medium"}
```

`stub_models(swap, infos)` (Task 5's conftest helper) replaces the cached catalog outright, which is the only way to reach the empty-answer branches: every model in the captured fixtures other than `gufo` is loaded, and `halogen` reports vision.

- [ ] **Step 2: Run them to verify they fail**

Run: `python3 -m pytest tests/test_profile.py -k "output_cap or max_tokens or hot_path or reasoning_efforts or aux or vision or unset" -v`
Expected: FAIL with `AttributeError` on each new hook.

- [ ] **Step 3: Implement the hooks**

`get_max_tokens(model)` looks the id up through `catalog_for` (which never does HTTP from here — the shard is already warm by the time Hermes asks, and if it is not, `None` is the honest answer) and returns `output_cap(window, self.default_max_tokens, self.min_output_tokens)`, those two fields read from `LLAMA_SWAP_DEFAULT_MAX_TOKENS` / `LLAMA_SWAP_MIN_OUTPUT_TOKENS` in `build_profile`. `resolve_aux_model` must cache its answer for the TTL — `providers/base.py:176` states the contract outright ("cheap to call (implementations must cache — this runs on client-resolution paths), never raises, and returns `\"\"` when it has no answer"). `default_vision_model` carries no such requirement in its docstring (`:250-258`), but it is resolved on the same paths, so cache it the same way; say that in the comment rather than claiming base mandates it. Return `""` / `None` when the catalog says nothing — but the two hooks do not decline on the same conditions: `resolve_aux_model` picks only among models llama-swap reports as **loaded** (a titling call must not cold-start an engine, and `""` is free), while `default_vision_model` **prefers** a loaded image-capable model and falls back to any reported one, because a `None` answer sends image input to the main chat model.

Which of these hooks actually reach the wire is uneven, and it was measured rather than assumed: `get_max_tokens` is called on **every** request (`agent/transports/chat_completions.py:559` → `_apply_max_tokens`, and the agent always supplies `max_tokens_param_fn` at `agent/chat_completion_helpers.py:1502`), as are `default_reasoning_config` (`agent/reasoning_params.py:65`) and `default_vision_model` (`agent/auxiliary_client.py:858`); `resolve_aux_model` runs only on `prefer_fast` paths (`agent/auxiliary_client.py:796`). `supported_reasoning_efforts` is consulted **only** by the Responses transport (`agent/transports/codex.py:460,463`) — on this chat-completions wire it is a declaration whose clamp lands in Task 7's `build_api_kwargs_extras`, so nothing here may describe it as protecting the llama-swap request path today.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python3 -m pytest tests -v`
Expected: all passed.

- [ ] **Step 5: Commit**

```bash
git add profile.py tests/test_profile.py && git commit -m "feat: declare per-model output cap and reasoning vocabulary"
```

---

### Task 7: Wire hooks — and the no-`max_tokens` guarantee

**Read this before writing anything: the guarantee is about the plugin's own wire hooks, not about
`get_max_tokens`.** Hermes' precedence already implements it — `_apply_max_tokens`
(`agent/transports/chat_completions.py:339-347`) returns as soon as it finds `ephemeral_max_output_tokens`
or `max_tokens` in `params`, and only then applies `profile_max`. So the requirement is: (a)
`build_api_kwargs_extras` / `build_extra_body` must never add `max_tokens` (or `max_completion_tokens`) of
their own, and (b) the integration test asserts that a caller-supplied budget **survives** — Hermes'
reactive clamp and compression budgets must not be overridden by the plugin's declared cap. Returning
`None` from `get_max_tokens` to "never send `max_tokens`" would be wrong: that is Task 6's declared cap
and it is supposed to reach the wire when the caller named none.

**Files:**
- Modify: `profile.py`; Create: `tests/test_wire.py`

**Interfaces:**
- Consumes: `supported_efforts` (Task 3), `catalog_for` (Task 5); Hermes' `agent.reasoning_effort.clamp_effort` when importable, else a local nearest-weaker implementation of the same policy.
- Produces: `LlamaSwapProfile.build_api_kwargs_extras`, `.build_extra_body`, `.build_client_kwargs_extras`.

- [ ] **Step 1: Write the failing unit tests**

```python
EFFORTS = {"low_only": ("none", "minimal", "low"), "no_reasoning": ()}

@pytest.fixture(params=["strixite"])
def swap_wire(swap):
    stub_models(swap, {"strixite": _stated(_info("strixite"), efforts=("none", "minimal", "low", "medium")),
                       "low_only": _stated(_info("low_only"), efforts=EFFORTS["low_only"]),
                       "no_reasoning": _stated(_info("no_reasoning"), efforts=None, reasoning_enabled=False)})
    return swap

def test_reasoning_effort_travels_top_level(swap_wire):
    extra, top = swap_wire.build_api_kwargs_extras(reasoning_config={"enabled": True, "effort": "high"},
                                                    model="strixite")
    assert extra == {} and top == {"reasoning_effort": "medium"}   # clamped to the model's ceiling

def test_disabled_reasoning_sends_none_not_omission(swap_wire):
    _, top = swap_wire.build_api_kwargs_extras(reasoning_config={"enabled": False}, model="strixite")
    assert top == {"reasoning_effort": "none"}

def test_a_vocabulary_the_model_never_published_omits_the_field(swap_wire):
    _, top = swap_wire.build_api_kwargs_extras(reasoning_config={"enabled": True, "effort": "high"},
                                                model="no_reasoning")
    assert top == {}

def test_unknown_effort_levels_are_dropped_before_the_wire(swap_wire):
    stub_models(swap_wire, {"future": _stated(_info("future"), efforts=("low", "turbo", "ultra"))})
    assert swap_wire.supported_reasoning_efforts("future") == ("low", "ultra")

def test_hooks_ignore_the_extra_kwargs_hermes_passes(swap_wire):
    _, top = swap_wire.build_api_kwargs_extras(reasoning_config=None, model="strixite",
                                               base_url="http://x/v1", session_id="s",
                                               ollama_num_ctx=4096, supports_reasoning=True)
    assert top == {}

def test_no_client_timeout_is_overridden(swap_wire):
    assert "timeout" not in swap_wire.build_client_kwargs_extras()
```

- [ ] **Step 2: Write the failing guarantee test against the real transport**

This is the spec's central negative requirement — the plugin must never overwrite the caller's budget. It needs Hermes, so follow the sibling's skip pattern (`needs_hermes = pytest.mark.skipif(not _hermes_available(), ...)`), and drive `ChatCompletionsTransport._build_kwargs_from_profile` the way the design prototype did:

```python
def _params(swap, max_tokens):
    """The params dict ChatCompletionsTransport.build_kwargs receives, reduced to the keys
    this test exercises (agent/transports/chat_completions.py::_base_kwargs reads the rest
    with .get(), so absent keys take their defaults)."""
    return {
        "model_lower": "strixite",
        "base_url": swap.base_url,
        "max_tokens": max_tokens,
        "max_tokens_param_fn": lambda n: {"max_tokens": n},      # chat_completions wire shape
        "reasoning_config": {"enabled": True, "effort": "medium"},
    }

MSGS = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]

@needs_hermes
@pytest.mark.parametrize("caller_max, expected", [(64000, 64000), (None, 8192)])
def test_the_callers_budget_survives_the_profile_and_the_cap_applies_without_one(swap, caller_max, expected):
    kw = ChatCompletionsTransport()._build_kwargs_from_profile(
        swap, "strixite", MSGS, None, _params(swap, caller_max))
    assert kw["max_tokens"] == expected      # 64000 untouched (core owns it) / 8192 from get_max_tokens
    assert kw["reasoning_effort"] == "medium"             # and reasoning rides top level
```

- [ ] **Step 3: Run them to verify they fail**

Run: `python3 -m pytest tests/test_wire.py -v`
Expected: FAIL (hooks still return the `ProviderProfile` defaults).

- [ ] **Step 4: Implement the three hooks**

`build_api_kwargs_extras(self, *, reasoning_config=None, **context)` resolves the effort against `supported_efforts(catalog_for(...))`, returns `({}, {"reasoning_effort": level})`, and returns `({}, {})` for a model with `()` vocabulary. It must accept and ignore the extra kwargs Hermes passes (`model`, `base_url`, `session_id`, `cache_scope_id`, `supports_reasoning`, `ollama_num_ctx`, ...) via `**context`, and it must be **pure of HTTP**. `build_extra_body` returns `{}` (unknown body fields are ignored by this server, and every field we would send is already top-level). `build_client_kwargs_extras` returns `{"api_key": self._api_key or PLACEHOLDER_API_KEY}` with `PLACEHOLDER_API_KEY = "llama-swap"`, mirroring `halogen-hermes`.

- [ ] **Step 5: Run the whole suite**

Run: `python3 -m pytest tests -v`
Expected: all passed (the two Hermes tests skipped where Hermes is absent, run where it is present — and they must be run here, see Task 9).

- [ ] **Step 6: Commit**

```bash
git add profile.py tests/test_wire.py && git commit -m "feat: send reasoning_effort on the wire and never touch max_tokens"
```

---

### Task 8: Server-side usage in `/usage`

**Files:**
- Modify: `profile.py`, `tests/test_profile.py`; Create: `tests/fixtures/metrics_stats.json` (capture with `curl -sS http://192.168.31.7:8080/api/metrics/stats`)

**Interfaces:**
- Consumes: `client.get_json`, `client.root_url`, `catalog_for` (Task 2, 5).
- Produces: `LlamaSwapProfile.fetch_account_usage(*, base_url=None, api_key=None)` → `AccountUsageSnapshot | None`, and `usage_details(stats: Mapping, version: Mapping) -> tuple[str, ...]` (module-level, Hermes-free and therefore unit-testable).

- [ ] **Step 1: Write the failing tests**

`STATS` is `tests/fixtures/metrics_stats.json` parsed; `usage_snapshot` (add to `tests/conftest.py`) calls `swap.fetch_account_usage()` with `client.get_json` serving that file for `/api/metrics/stats` and `{}` for `/api/version`.

- [ ] **Step 0: Capture the fixture**

`curl -sS http://192.168.31.7:8080/api/metrics/stats > tests/fixtures/metrics_stats.json` — a GET; the only
live call outside Task 4's approved probe. Commit the payload verbatim, unedited, so the tests below are
pinned to what the server actually said.

```python
def test_usage_details_are_absolute_counters():
    lines = usage_details(STATS, {"version": "v264-9-g4a61d5"})
    # Derived from the captured payload, never re-stated by hand (see the note after this block).
    assert any(f"requests {STATS['total_requests']}" in ln for ln in lines)
    assert any(f"in {STATS['total_input_tokens']:,}" in ln
               and f"out {STATS['total_output_tokens']:,}" in ln for ln in lines)
    assert any("prompt p50" in ln and "gen p50" in ln for ln in lines)

def test_no_quota_is_invented():
    assert all("%" not in ln for ln in usage_details(STATS, {}))

def test_snapshot_uses_details_not_windows(usage_snapshot):
    assert usage_snapshot.windows == ()
    assert usage_snapshot.details and usage_snapshot.available

def test_unreachable_server_yields_no_snapshot(swap, monkeypatch):
    monkeypatch.setattr(client, "get_json", lambda *a, **k: None)
    assert swap.fetch_account_usage() is None
```

Why the numerals are derived rather than written out: `/api/metrics/stats` is a live counter set, so any
literal copied into this plan is stale before the fixture is even captured — the values in the first draft
of this task (`requests 200`, `in 598,723`, `out 144,784`) had already quadrupled by the time of the re-check
that produced the shape above. Deriving them still tests the thing that matters, which is **which key each
number is read from and how it is formatted**: a `usage_details` that pulls `total_output_tokens` for the
`in` field, or drops the `{:,}` separators, goes red. Mutation-check both of those.

- [ ] **Step 2: Implement**

`fetch_account_usage` imports `AccountUsageSnapshot` lazily inside the function and returns `None` if the import fails (no Hermes ⇒ no snapshot type, and the hook is Hermes-only anyway). It uses `source="llama_swap_api"`, `title="llama-swap server"`, and its own timeout of 5 s — shorter than core's 10 s hook deadline, per the spec. `usage_details` formats with `{:,}` and skips any key the server omitted, so a future llama-swap that drops the histograms still renders counters.

Verified against the real dispatcher (`agent/account_usage.py`), so nobody re-derives it: the hook is called as `profile.fetch_account_usage(base_url=…, api_key=…)` at `:726` — `read_only` is a `fetch_account_usage()` parameter for *built-in* fetchers and never reaches a profile, so the signature above is complete. The deadline is `PLUGIN_USAGE_HOOK_DEADLINE_S = 10.0` (`:712`), enforced by `run_bounded_sync`, which returns `None` to `/usage` on a timeout **but does not stop the worker thread** — which is exactly why the plugin needs its own 5 s socket timeout rather than relying on core's bound. Core also skips the thread entirely when `type(profile).fetch_account_usage is ProviderProfile.fetch_account_usage` (`:723`), i.e. overriding the hook is what makes the call happen at all. On the snapshot side (`agent/account_usage.py:42`): `provider`, `source` and `fetched_at` are required fields, `details` is a `tuple[str, ...]`, and `available` is a **property** (`bool(windows or details) and not unavailable_reason`) — never assign to it.

- [ ] **Step 3: Run the tests to verify they pass**

Run: `python3 -m pytest tests/test_profile.py -k usage -v`
Expected: all passed.

- [ ] **Step 4: Commit**

```bash
git add profile.py tests/ && git commit -m "feat: surface llama-swap server counters in Hermes /usage"
```

---

### Task 9: Live verification and docs

**Files:**
- Create: `scripts/smoke.py`, `README.md`

**Interfaces:**
- Consumes: everything above.
- Produces: a runnable check and the operator documentation.

- [ ] **Step 1: Write `scripts/smoke.py`**

Mirror `halogen-hermes/scripts/wire_smoke.py`: put the repo root on `sys.path`, read `LLAMA_SWAP_BASE_URL`, exit non-zero on the first failed check, and print one line per check. Steps, in order:

1. `GET /api/version` and `GET /v1/models` — print the build; assert ≥ 1 model.
2. `build_profile(os.environ).fetch_models()` — print ids, each with `context_length` / `vision` / `tools` / efforts / loaded.
3. `get_max_tokens` and `supported_reasoning_efforts` for each id.
4. One real completion with `reasoning_effort: "high"` and one with `"none"` against a **loaded** model, asserting from the response body that thinking was on and off (the `strix.thinking`/`reasoning_content` evidence proven during design).
5. One deliberately oversized request (same shape as Task 4 Step 3) and print the status + body — this is how the error corpus is refreshed when llama-swap or its engines change.
6. `fetch_account_usage()` — print the details lines, or "unavailable".

- [ ] **Step 2: Run it against the live server**

Run: `LLAMA_SWAP_BASE_URL=http://192.168.31.7:8080/v1 python3 scripts/smoke.py`
Expected: every check green, exit 0. Any failure is a real bug in Tasks 2-8 — fix it there, not here.

- [ ] **Step 3: Verify inside Hermes itself**

```bash
mkdir -p ~/.hermes/plugins/model-providers/llama-swap
tar cf - --exclude=./tests --exclude=./scripts --exclude=./docs --exclude=./.git \
    --exclude='__pycache__' . \
  | (cd ~/.hermes/plugins/model-providers/llama-swap && tar xf -)
hermes doctor
hermes -z "reply with the single word ok" --provider llama-swap -m strixite
```

Expected: doctor lists `llama-swap` with a passing `/models` probe; the one-shot reply succeeds. If doctor reports the profile registered no `ProviderProfile`, `__init__.py`'s import fallback is wrong — fix Task 5 Step 4.

- [ ] **Step 4: Write `README.md`**

Sections, in this order: what it does (discovery, context windows, reasoning vocabulary, budget posture); install (drop-in copy, `hermes plugins install`, Docker bind-mount) — copy the working commands from `halogen-hermes/README.md` and adapt the names; selecting the provider; configuration table (every `LLAMA_SWAP_*` knob with its default and what raising or lowering it does); **budget posture** — the plugin declares the window and a per-model cap and lets Hermes' own recovery shorten a too-large budget, so an explicitly large `model.max_tokens` is corrected after one rejection, and on engines that clamp silently (observed on this server) the cap is the only shield; the **endpoint matrix** from the spec, including the two endpoints the running build lacks, so a future version can be checked against it; troubleshooting (server down ⇒ bounded startup probe then Hermes' own context default; wrong URL shape; `apiKeys` configured but `LLAMA_SWAP_API_KEY` unset); known limits, verbatim from the spec's Open risks (never-loaded models report no window, aux calls carry no cap, host-isolation mode would not see the in-place capability refresh).

- [ ] **Step 5: Commit**

```bash
git add scripts/smoke.py README.md && git commit -m "docs: install, configuration and live verification for the llama-swap provider"
```
