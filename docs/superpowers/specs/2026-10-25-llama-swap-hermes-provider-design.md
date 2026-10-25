# llama-swap provider plugin for Hermes Agent — design

Date: 2026-10-25
Status: for approval
Repo: `/home/hermes/llama-swap-hermes` (repo root **is** the plugin directory)

## Intent and the why

Hermes today reaches the llama-swap server at `192.168.31.7:8080` through the
generic `custom` provider (a cache row for
`custom:http://192.168.31.7:8080/v1` already exists in this install). The
generic route treats llama-swap as an anonymous OpenAI-compatible chat endpoint:
it does not learn the models' real context windows, vision/tool support or
reasoning vocabulary, and it sends an output budget that can push a request past
the window the engine actually loaded.

llama-swap publishes all of that, per model, from the live engines behind it. This
plugin makes Hermes read it instead of guessing, and makes the request budget
respect it.

## Goals

1. A first-class `llama-swap` provider whose endpoint is configurable without
   editing code.
2. Model discovery: every model llama-swap lists, with the metadata that server
   reports for it (context window, vision, tools, reasoning levels, load state).
   Nothing about the models is hardcoded — when the llama-swap config changes,
   Hermes follows it.
3. Output budget that cannot exceed the window: the plugin declares the true
   window and a per-model output cap, and feeds Hermes' own reserve-and-shorten
   recovery path so a too-large budget is corrected instead of failing the turn.
4. Thinking control on the wire, clamped to what each model publishes.
5. llama-swap-specific error recovery where Hermes gets it wrong.
6. llama-swap's own operational numbers surfaced in Hermes (`/usage`).

## Non-goals

- The Anthropic wire (`POST /v1/messages`): absent from the running build, so
  out of scope. `api_mode` stays `chat_completions`.
- Controlling the server: no loading, unloading, profile switching or
  cancelling (`/unload`, `/api/models/unload`, `/api/profiles/active`,
  `/api/inflight/*/cancel` are deliberately untouched).
- Push-based freshness: no SSE (`/api/events`), no background threads, no disk
  mirror of the catalog. Poll-on-demand only. (Considered and rejected — see
  Approaches.)
- A live speed/telemetry meter in the Hermes UI. That is a harness feature, not
  a provider one.
- Model-load latency handling. llama-swap blocks while it swaps engines; Hermes'
  own per-call timeout already covers it (see Constraints), and this plugin
  cannot make loading faster.

## Anti-goals

- Claiming a context window llama-swap did not report. A wrong big number is the
  exact failure mode this plugin exists to avoid.
- Turning a would-be overflow into a silently useless request (a 12-token output
  budget that produces nothing).
- Swallowing Hermes' own error classification for statuses it already maps
  correctly (401/403/429/5xx). Only claim what it gets wrong.
- Re-implementing request-time budget maths that Hermes already owns. Core
  clamps output budgets on an output-cap error; the plugin's job is to give it
  the inputs, not to duplicate the logic.
- A plugin-owned `api_mode`. Decided against after review — see Approaches.

## Constraints (verified, not assumed)

Hermes side (v0.21.6+529.g66605471e9f0 at `/home/hermes/.hermes/hermes-agent`):

- A model provider plugin is `plugin.yaml` (`kind: model-provider`) +
  `__init__.py` calling `register_provider(profile)` at import. Drop it in
  `$HERMES_HOME/plugins/model-providers/<name>/`; picker, `hermes doctor`,
  `hermes auth`, aux-model routing, and the transport auto-wire from the profile.
  `halogen-hermes` (already installed here) is the working sibling and the
  shape to follow: thin `__init__.py`, decision logic in pure modules, stdlib only.
- **`ProviderProfile.get_max_tokens()` is ignored when the caller sets
  `max_tokens`** — `agent/transports/chat_completions.py::_apply_max_tokens`
  returns early on `params["max_tokens"]`, and only honours a profile value when
  the caller sent none. Confirmed by running the real transport
  (`hermes --run-module`): a top-level `max_tokens: 777` injected after that
  point overrode the caller's `64000`, which is exactly why a profile must not
  do it (next bullet).
- **No profile hook can compute a lower-only clamp of the caller's budget.**
  `build_api_kwargs_extras` receives `model`, `base_url`, `session_id`,
  `cache_scope_id`, `supports_reasoning`, `reasoning_config` — no `max_tokens`,
  no messages, no `tools` (verified against the live transport). Returning a
  top-level `max_tokens` from a profile therefore *replaces* whatever the caller
  chose, including Hermes' internal budgets — overflow-recovery `safe_out`
  (`agent/turn_overflow.py:299`) and the 1500-token micro-compaction cap
  (`agent/micro_compaction.py:126`). **A profile must not send `max_tokens` at
  all.**
- Core already owns reserve-and-shorten, reactively:
  `agent/turn_overflow.py::_clamp_output_cap` sets
  `agent._ephemeral_max_output_tokens = min(provider_available, window -
  estimated_request) - 64`, and `_apply_max_tokens` honours that ahead of the
  caller's value. It needs two inputs from us: the correct `context_length` and
  an error body it classifies as an output-cap error *with a parseable figure*
  (`parse_available_output_tokens_from_error`, `agent/model_metadata.py:1293`).
- The provider-side output surface is `get_max_tokens(model)`, used when the
  caller sent no budget — which is the normal case, since `agent.max_tokens`
  defaults to `None`.
- `get_model_context_length()` returning a positive int beats Hermes' persistent
  cache and its HTTP probes; returning `None` falls through to the normal chain,
  whose last rung is the 256K `DEFAULT_FALLBACK_CONTEXT`.
- `model_capabilities` is read live off the registered profile object
  (`agent/models_dev.py::_provider_model_capabilities` →
  `profile.model_capabilities.get(model)`), and `register_provider` stores the
  profile by reference — so the dict can be refreshed in place after a catalog
  fetch. (Bundled profiles treat it as static; this is the dynamic-catalog use,
  the same reason `router` overrides `supported_reasoning_efforts`.)
- Plugins are discovered per `$HERMES_HOME`; the registry is per-process and is
  built lazily on first `get_provider_profile()`.

llama-swap side (running build `v264-9-g4a61d5`, checked live):

| Endpoint | Present here | What it gives |
|---|---|---|
| `GET /v1/models` | yes | per-model `context_length` + `context_window`, `capabilities` (`vision`, `function_calling`), `supported_parameters`, `architecture.input_modalities`, `meta.llamaswap.reasoning.{enabled,effort,efforts,tag}`, `meta.n_ctx`, `meta.llamaswap.aliases`, `status.value` = `loaded`/`unloaded`, `owned_by`, `name`, `description` |
| `GET /running` | yes | `running[]` with `model`, `state` (`ready`), `proxy`, `ttl`, `cmd` |
| `GET /api/version` | yes | `version`, `commit`, `build_date` |
| `GET /api/metrics/stats` | yes | `total_requests`, `total_input_tokens`, `total_output_tokens`, `total_cache_tokens`, latency histograms |
| `GET /api/hardware`, `GET /api/performance` | yes | host CPU/GPU/VRAM inventory and telemetry |
| `GET /health` | yes | literal `OK` |
| `GET /metrics` | yes | Prometheus `llamaswap_*` |
| `GET /api/events` | yes | SSE, `modelStatus` with `state` + `context_length` (not used) |
| `GET /props?model=` | **no** (newer builds only) | engine's own `n_ctx`, `n_predict` |
| `POST /v1/messages` | **no** (newer builds only) | Anthropic wire |

- The `context_length` llama-swap reports is documented as *the window one request
  can actually use* (per-slot for llama.cpp, `max_model_len` for vLLM, engine
  report for halogen) — the right number to clamp against, and not the trained
  window.
- A model that has never been loaded reports **no** `context_length` (live
  example: `gufo`). llama-swap only learns it by probing the engine once it runs.
- Unknown request-body fields are ignored, not rejected (verified: `foo_bar: 1`
  returned 200).
- Thinking control verified live on the loaded model: top-level
  `reasoning_effort: "none"` → engine reports `thinking: off`; `"high"` →
  `thinking: on`. `meta.llamaswap.reasoning` itself is operator-authored
  descriptive metadata in the llama-swap config, not a server-enforced contract,
  so it is a vocabulary to clamp to, not a guarantee.
- Overflow behaviour is **engine-dependent**: `max_tokens: 262144` against the
  loaded model returned **HTTP 200** with the engine having silently clamped it
  (`strix.max_tokens_clamped: true`) and the turn ending on `finish_reason: stop`.
  llama.cpp instead answers 400. On a silent-clamp engine there is no error for
  core to react to, so the *declared* cap is the only defence — which is why
  `get_max_tokens` is sized conservatively below.
- Per-call timeout is already generous for engine swaps: Hermes' own default is
  1800 s (`_resolved_api_call_timeout`), and the RFC-1918 address of this server
  makes `is_local_endpoint` true, which disables the non-stream stale detector
  when the user configured nothing. Nothing here needs to raise a timeout; a
  client-level `timeout` from `build_client_kwargs_extras` is applied with
  `setdefault` and would be shadowed anyway.

## Approaches considered

**A — Pure profile plugin** (chosen). Declarative profile + pure helper modules,
catalog polled on demand, every non-core endpoint probed opportunistically and
skipped silently when missing. Matches `halogen-hermes` and the in-tree
`custom`/`lmstudio` profiles; nothing runs between the request and the wire
except the documented hooks. Trades: catalog freshness is bounded by Hermes' own
refresh points (model switch, picker, doctor) plus a TTL, not by push.

**B — Profile + background warmer** (rejected). The `router` plugin's shape:
memory + disk mirror, daemon warmer thread, per-`$HERMES_HOME` state, and — for
push freshness — an SSE consumer on `/api/events`. Rejected because inference
reads these values per request from the profile anyway, so push buys freshness
nothing observes, at the cost of a thread, a cache file, a TTL policy and
multiplexed-profile scoping. Also the only approach that would make `/props`
probing safe to put on a timer — and `/props` dispatches the model, i.e. it can
start an engine.

**A′ — A + a plugin-owned transport dialect** (considered, rejected). The
reviewer's finding is that no profile hook sees the caller's `max_tokens`, so a
prompt-aware lower-only clamp requires subclassing `ChatCompletionsTransport`,
registering an `api_mode`, and overriding `build_kwargs` — which does see
`messages`, `tools` and `params`, and was prototyped successfully (64000 lowered
to the room left, 1500 left alone). Rejected: **no bundled provider registers a
dialect**, every one of the 40 profiles reuses a built-in `api_mode`, and the
equivalent problem in the closest analogue (LM Studio) was solved by core
probing and handing cached values to the profile. Prevention would have bought
accuracy this plugin does not need, at the cost of an `api_mode` in the user's
config that nothing else in Hermes understands. Chosen instead: declare, and let
core react — the way the existing providers do.

**C — A + ops surface** (partially chosen). On top of A, map
`/api/metrics/stats` into Hermes' `/usage` via `fetch_account_usage`. Chosen:
~40 lines, uses the documented hook, and it is exactly the "stays up to date with
the server" property the project is about. Rejected: the richer ops endpoints
(`/api/performance`, `/api/hardware`, `/metrics`) have no profile hook to render
them, and inventing one would mean a second plugin kind.

## The design

```
llama-swap-hermes/
├── plugin.yaml          # kind: model-provider, name llama-swap
├── __init__.py          # builds + registers the profile; config from env; no logic
├── client.py            # HTTP GET helper (stdlib urllib), base URL + optional bearer
├── catalog.py           # /v1/models + /running → dict[model_id, ModelInfo], TTL cache
├── errors.py            # classify_api_error: llama-swap/llama.cpp specific branches
├── conftest.py, tests/  # pytest; stubs providers.base when Hermes is absent
└── scripts/smoke.py     # live check against a real llama-swap
```

### Discovery and per-model facts

`fetch_models()` (override) fetches `/v1/models`, parses it into
`ModelInfo(id, context_length, vision, tools, efforts, reasoning_enabled,
loaded, aliases, modalities, owned_by, name, description)`, stores it in a
module cache, and returns the model ids **in llama-swap's own form** (raw ids,
no prefix — that is what the server expects on the wire). `/running` is fetched
in the same refresh to mark which ids are loaded; a failure to read `/running`
never blocks discovery.

On the same refresh, `profile.model_capabilities` is updated in place, one entry
per id, only with fields the server actually reported:

```python
{
  "supports_tools":   capabilities.get("function_calling"),   # when present
  "supports_vision":  bool("image" in input_modalities),       # when present
  "context_window":    context_length,                         # when present
}
```

Absent fields stay absent, so Hermes keeps its own catalog/heuristic answer
rather than being told `False`.

`get_model_context_length(model)` returns the cached `context_length`, or `None`
when the server has not reported one — which lands on Hermes' normal chain and
its 256K fallback (**the chosen behaviour**). Consequence, stated plainly: for a
model llama-swap has never probed, the declared cap and the compaction maths run
against the assumed window, so its first request can still overflow; it
self-corrects once the engine has run and the next catalog refresh sees the
number.

The cache is refreshed when it is empty, when `fetch_models` is called, or when
it is older than a TTL (default 60 s, env-tunable). Nothing about a specific
model — not one id, not one window — is written into this repo.

### Output budget: declare, and let core react

The plugin sends **no `max_tokens` on the request path** (constraint above). It
supplies the three inputs Hermes' own recovery consumes:

1. **The window** — `get_model_context_length(model)` → the `context_length`
   llama-swap reported, or `None`. This is what `local_available_out = window -
   estimated_request` is computed against, and what compaction triggers on
   (Hermes compresses at 50 % of the window by default,
   `ContextCompressor(threshold_percent=0.50)`).
2. **A declared default output cap** — `get_max_tokens(model)`:

   ```
   cap = min(LLAMA_SWAP_DEFAULT_MAX_TOKENS, max(1024, window // 4))
   ```

   `LLAMA_SWAP_DEFAULT_MAX_TOKENS` is 8192, so a 256 K model keeps 8192 and a
   8 K model gets 2048 — a cap, never a raise. The `window // 4` term is chosen
   against the 50 % compaction point: prompt ≤ ½ window plus ¼ window of output
   leaves a quarter of the window as slack, which is the only protection a
   silent-clamp engine gets.
3. **A classifiable error** — so `_clamp_output_cap` (shorten and retry) and
   `context_overflow` (compress and retry) fire instead of a dead turn.

A user-set `model.max_tokens` therefore reaches the wire unchanged and is
corrected only after the engine rejects it; on llama.cpp that is one round-trip
and a clean retry, on a silent-clamp engine it is invisible. Accepted, and
documented in the README next to the knob that avoids it.

### Reasoning, and the rest of the wire

- `supported_reasoning_efforts(model)` returns the model's published
  `meta.llamaswap.reasoning.efforts` filtered to Hermes' `EFFORT_LADDER`
  (levels Hermes does not know are dropped, as `router` does), `()` when
  `reasoning.enabled` is false, `None` when the model says nothing. Cache-only,
  never blocking — it is on the request hot path.
- `build_api_kwargs_extras` sends top-level `reasoning_effort` (verified to be
  the field this server honours), clamped to that vocabulary, and omits it
  entirely for a model published as non-reasoning.
- `default_reasoning_config` → `{"enabled": True, "effort": "medium"}` so an
  unset effort does not inherit an engine-side ceiling.
- `build_client_kwargs_extras` → the placeholder api_key described under
  Address authority. No timeout: Hermes' per-call default already covers a swap.
- `build_extra_body` sends nothing: unknown body fields are ignored by this
  server, and every field we might send is either already top-level
  (`reasoning_effort`) or engine-specific guesswork.
- `supports_vision = True` provider-wide (the wire accepts image parts);
  per-model truth comes from `model_capabilities`.

### Catalog warming

`fetch_models()` is not on the inference path — only the picker, setup and doctor
call it — so a session that never opened `/model` would otherwise run its entire
life on the 256 K fallback and the whole design would be decorative.

Warming therefore hangs off the one place Hermes already expects a provider
probe: `get_model_context_length()`, called at agent init and on model switch
(`agent/agent_init.py:1973`, `agent/context_compressor.py:2210`). The first call
for a server performs **one** catalog fetch with a short timeout (default 3 s,
env-tunable), caches the result, and remembers failure so a dead server costs
one bounded wait rather than a stall on every hook. The hot-path hooks —
`get_max_tokens`, `supported_reasoning_efforts` — are cache-only and never block,
per the contract in `providers/base.py`.

The cache also expires on a TTL (default 60 s), so a llama-swap config change is
picked up on the next session without any background machinery.

### Address authority

`fetch_models` and `fetch_account_usage` are handed a `base_url` (it comes from
the credential-pool row `hermes auth add` writes), while the hooks that *read*
the catalog get no URL. Rule: the URL Hermes passes wins, is used for the fetch,
and becomes the shard the read hooks answer from; `LLAMA_SWAP_BASE_URL` is the
fallback when nothing was passed. Single-server per profile home is the documented
assumption — llama-swap is one box, and sharding a cache with no URL-carrying
reader would be invention.

With no `LLAMA_SWAP_API_KEY` set, `build_client_kwargs_extras` supplies a
placeholder key so Hermes' credential gate is satisfied (as `halogen-hermes`
does); llama-swap ignores an Authorization header when no `apiKeys` are
configured, and a real key is required when they are — stated in the README.

### Auxiliary and vision defaults

- `resolve_aux_model()` → the first model llama-swap reports as **loaded**, else
  `""`. Compaction and summarisation then land on an engine that is already up,
  instead of forcing a swap in the middle of a long session. Proposed; if you'd
  rather pin a specific model, say which and it becomes `default_aux_model`.
- `default_vision_model()` → the first model reporting image input, else `None`.
- Auxiliary calls keep their own budgets: because the profile sends no
  `max_tokens`, nothing the aux path chose can be overwritten. Consequence of
  `_forwards_max_tokens` (a LAN llama-swap URL is not a forwarding route) is that
  aux requests go out with no output cap at all — identical to what `custom`
  does on this server today, so it is parity rather than regression, and is
  listed under Open risks.

### Errors

`classify_api_error` claims only the branches Hermes gets wrong, everything else
returns `None`:

- llama.cpp / llama-swap context wordings the built-in patterns miss →
  `context_overflow` with `should_compress=True`.
- output-cap wordings carrying a usable figure → left to the built-in output-cap
  branch, *unless* measurement shows Hermes parsing them wrong; the goal is that
  `parse_available_output_tokens_from_error` returns a number for real
  llama-swap bodies, since that number is what core retries with.
- 401/403/429/5xx: untouched, per the anti-goal.

The rule set is established by measurement, not by reading: capture **real**
error bodies by posting deliberately oversized requests to the configured server
(operator-approved, engine loads included), feed them to Hermes'
`classify_api_error`, and add a plugin rule only where its verdict is wrong.
This also settles whether llama-swap forwards the engine's status and body
verbatim or rewrites them — which decides whether the plugin has to parse two
layers of message or one.

### Usage surface

`fetch_account_usage` reads `/api/metrics/stats` (plus `/api/version` for a source
label) under a short timeout and returns an `AccountUsageSnapshot`. llama-swap
counts server-lifetime totals and publishes no quota, no reset time and no plan,
so the snapshot uses `details` lines — absolute counters (`requests`, input /output
/ cached tokens, p50/p95 prompt and generation throughput) — and **no
`AccountUsageWindow` entries**, because `used_percent` would be a fabricated
number. `AccountUsageSnapshot.available` is satisfied by `details` alone.
Failures return `None`; the hook runs under a core deadline, so give it a shorter
one.

### Configuration

Env-driven, like the sibling plugin. `LLAMA_SWAP_BASE_URL` (default
`http://192.168.31.7:8080/v1` — this machine's LAN target; any Hermes config
surface, a `providers:` entry or an env override replaces it),
`LLAMA_SWAP_API_KEY` (optional), and four tuning knobs:
`LLAMA_SWAP_CATALOG_TTL_S` (60), `LLAMA_SWAP_PROBE_TIMEOUT_S` (3),
`LLAMA_SWAP_DEFAULT_MAX_TOKENS` (8192), `LLAMA_SWAP_MIN_OUTPUT_TOKENS` (1024).
Slow engine swaps are handled by Hermes' own `HERMES_API_TIMEOUT`, documented in
the README rather than overridden here. A malformed value falls back to its
default — Hermes startup must not crash on a bad environment.

### Install and docs

`README.md` with the three install paths (drop-in copy to
`~/.hermes/plugins/model-providers/llama-swap/`, catalog install, Docker
bind-mount), how to select it (`hermes model`, `--provider llama-swap -m <id>`,
or `config.yaml`), and the endpoint matrix above so a future llama-swap version
can be checked against it.

## Testing

- Unit (pytest, Hermes-optional via a `providers.base` stub): catalog parsing from
  captured real `/v1/models` + `/running` payloads (including `gufo` with no
  context and a model with no `capabilities`); the `cap` maths (unknown window →
  `None`; tiny window → floor; big window → the default, never above it); effort
  vocabulary filtering; each error-classifier branch.
- One integration test against the real transport, the way the prototype ran:
  assert the plugin **sends no `max_tokens`** even when the caller supplied one,
  that `get_max_tokens` supplies the declared cap when the caller did not, and
  that `reasoning_effort` is absent for a model published as non-reasoning.
- A classifier test that feeds captured llama-swap/llama.cpp bodies through
  Hermes' own `classify_api_error` and asserts the plugin only overrides where
  core's verdict is wrong.
- `scripts/smoke.py` against a live llama-swap: discover, parse, and one real
  short completion each with reasoning on and off, asserting the server's own
  `thinking` report agrees with what we asked for; plus one deliberately
  oversized request (operator-approved, engine loads included) to capture the
  real error body, which is the input to the classifier rules above.
- `hermes doctor` shows the provider with a passing `/models` probe.

## Left to the builder

Module boundaries may collapse if that reads better (three tiny files vs two);
naming of internal helpers; the wording of the `/usage` detail lines;
whether `ModelInfo` is a dataclass or a plain dict. Nothing in "Constraints" may
be assumed away — re-verify with a prototype before deviating from it.

## Open risks

- A model never loaded on llama-swap cannot be sized; the 256K fallback is a
  chosen, documented exposure, not a solved problem.
- `model_capabilities` mutated after import is a use the bundled profiles do not
  make; under `plugins.isolation: host` (not this install's mode) the profile
  would live in another process and the mutation would not propagate. Worth one
  README line.
- A user-set `model.max_tokens` larger than the room left is corrected only after
  the engine rejects it — one wasted round-trip on llama.cpp, invisible on a
  silent-clamp engine.
- Aux requests carry no output cap on this route (parity with `custom` today).
  If it bites, the fix is an opt-in aux-only cap, not a transport.
- Hermes' per-model compaction threshold (`model_thresholds`) is a user setting;
  the `window // 4` output term assumes the 50 % default and gets tighter, not
  looser, if the user raises it.
