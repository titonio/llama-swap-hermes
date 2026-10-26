# llama-swap provider plugin for Hermes Agent

A [Hermes Agent](https://github.com/NousResearch/hermes-agent) model-provider
plugin for a [llama-swap](https://github.com/mostlygeek/llama-swap) server.

The repo root **is** the plugin directory (`plugin.yaml` + `__init__.py`), so any
of the three install paths below works directly. No Python dependencies beyond
the standard library.

## What it does

Without this plugin Hermes reaches llama-swap through the generic `custom`
route: an anonymous OpenAI-compatible endpoint whose models' real windows,
capabilities and reasoning vocabulary are unknown, so the output budget it sends
can push a request past the window the engine actually loaded. This plugin makes
Hermes read those facts from the server that serves them, and never invents one.

- **Discovery** — every model llama-swap lists, in llama-swap's own id form,
  from `GET /v1/models` plus `GET /running` (which ids are actually up). Nothing
  about a specific model — not one id, not one window — is written into this
  repo: change the llama-swap config and Hermes follows it.
- **Context windows** — `get_model_context_length` returns the `context_length`
  llama-swap reported for that model (the window one request can actually use:
  per-slot for llama.cpp, `max_model_len` for vLLM, the engine's own report for
  halogen). A model llama-swap has never loaded reports **no** window, and the
  plugin says so rather than guessing — Hermes' own chain (catalog, config
  override, then its 256 K default) stays in charge. See
  [Troubleshooting](#troubleshooting).
- **Reasoning vocabulary** — each model's published
  `meta.llamaswap.reasoning.efforts`, filtered to Hermes' effort ladder, is what
  Hermes may send. `build_api_kwargs_extras` puts the requested level on the
  wire as top-level `reasoning_effort` (the field this server honours), clamped
  to that model's own ladder and never upward; a model published as
  non-reasoning gets no reasoning field at all. Measured on the live server:
  `reasoning_effort: "high"` → `usage.completion_tokens_details.reasoning_tokens`
  26 and a non-empty `message.reasoning_content`; `"none"` → 0 and no
  `reasoning_content`. A level the server does not publish is rejected — see
  [Reasoning rejections](#reasoning-rejections).
- **Vision / tools** — per-model `supports_vision` and `supports_tools` from the
  server's own `capabilities`, refreshed in place on every catalog read, so
  Hermes' per-model answers are the server's answers.
- **Budget posture** — the plugin declares the window and a per-model output cap
  and lets Hermes' own recovery do the arithmetic.
  [Details below](#budget-posture).
- **Server counters in `/usage`** — `GET /api/metrics/stats` + `/api/version`
  rendered as absolute counters (requests, in/out/cached tokens, p50/p95 prompt
  and generation throughput, build). No percentage: llama-swap publishes no
  quota to divide against.

## Install

Pick one.

**1. Drop-in copy** (local Hermes install):

```bash
git clone <this-repo> && cd llama-swap-hermes
mkdir -p ~/.hermes/plugins/model-providers/llama-swap
tar cf - --exclude=./tests --exclude=./scripts --exclude=./docs \
    --exclude=./.git --exclude='__pycache__' . \
  | (cd ~/.hermes/plugins/model-providers/llama-swap && tar xf -)
```

(From a working tree rather than a fresh clone, also exclude the gitignored
scratch dirs — `--exclude=./.pi --exclude=./.superpowers --exclude=./.pytest_cache
--exclude=./.worktrees` — or they land inside the plugin directory.)

**2. Catalog install** (once the repo is published):

```bash
hermes plugins install <owner>/llama-swap-hermes
```

**3. Docker bind-mount**: stage the plugin at
`$HERMES_HOME/plugins/model-providers/llama-swap/` inside the mounted data dir
(`/opt/data` in the official image), i.e. the same file set the drop-in copy
produces, then run the container with `LLAMA_SWAP_BASE_URL` and
`LLAMA_SWAP_API_KEY` passed through.

Then give Hermes a credential — **required**, even though llama-swap checks no
key. Hermes' credential gate refuses an `api_key` provider with nothing
configured (`No usable credentials found for provider 'llama-swap'. Set
LLAMA_SWAP_API_KEY.`), and the profile's placeholder key only fills the OpenAI
client's kwargs, not that gate. Any non-empty value works:

```bash
echo 'LLAMA_SWAP_API_KEY=llama-swap' >> ~/.hermes/.env
```

## Selecting the provider

Any Hermes config surface works. The provider name is `llama-swap`, with the
aliases `llamaswap` and `swap`; the model ids are the server's, so list them
first (`hermes model`, or `python3 scripts/smoke.py`).

```bash
hermes -z "reply with the single word ok" --provider llama-swap -m <an id the server lists>
```

```yaml
# ~/.hermes/config.yaml
model:
  provider: llama-swap
  name: <an id the server lists>   # `name` is an alias of `default`
```

`hermes doctor` then shows the provider with a passing `/models` probe:

```
◆ API Connectivity
  ✓ llama-swap
```

## Configuration

Environment variables, read once when Hermes constructs the profile at process
start. A malformed value falls back to its default — a bad environment never
crashes Hermes startup. Changing any of them means restarting Hermes.

| Variable | Default | Meaning — and what moving it does |
| --- | --- | --- |
| `LLAMA_SWAP_BASE_URL` | `http://192.168.31.7:8080/v1` | The OpenAI-style base, `/v1` included. A credential-pool route (`hermes auth add`) outranks it and becomes the shard the read hooks answer from. Wrong shape (missing `/v1`, typo) reads as "no models", never as a crash. |
| `LLAMA_SWAP_API_KEY` | unset → **provider unusable** | Hermes' credential gate needs a non-empty value; llama-swap ignores it unless `apiKeys` are configured server-side, where it must be the real key. Sent as `Authorization: Bearer …`. |
| `LLAMA_SWAP_DEFAULT_MAX_TOKENS` | `8192` | The output cap ceiling. Raising it lets a small-window model be asked for more output; lowering it shrinks every declared cap. It is a ceiling, never a floor — the declared number is `min(this, max(LLAMA_SWAP_MIN_OUTPUT_TOKENS, window // 4))`. |
| `LLAMA_SWAP_MIN_OUTPUT_TOKENS` | `1024` | The floor inside that ceiling, so a tiny window still gets an output budget worth paying for (a 12-token budget produces nothing). Nested under the ceiling: a contradictory config (`DEFAULT_MAX_TOKENS=512`, `MIN_OUTPUT_TOKENS=1024`) cannot push the declared cap above 512. |
| `LLAMA_SWAP_CATALOG_TTL_S` | `60` | How long a catalog read is reused, and how long a *failed* probe is remembered. Raise it for a busy box (fewer `GET /models` + `/running` pairs, staler capability/window updates); lower it to follow llama-swap config edits sooner. A probe costs two GETs, so this bounds how often the plugin asks. Probing happens at most once per TTL per route — on startup, on a model switch, and whenever a request-path hook finds the shard expired, which is not guaranteed every turn. **Asking for the model list always probes, regardless of TTL** (`hermes model`, the model picker, `hermes doctor`), and the values it refreshes are the ones the request-path hooks then serve. So between two probes those hooks answer from the last one: an `n_ctx` edit in a llama-swap profile lands at the next probe, never mid-request, and the way to force one now is the model list or a model switch. |
| `LLAMA_SWAP_PROBE_TIMEOUT_S` | `3` | The bound on one catalog probe. One probe is **two** GETs (`/models`, then `/running`), each bounded by the full value, so `3` means up to about 6 s — and a dead server pays only the first. This is the only place the **request path** blocks on the network; every hook Hermes calls per turn is cache-only. The two reads that are not on the request path bound themselves instead: `fetch_models` (the picker, setup, doctor) allows 8 s, and `fetch_account_usage` — `/usage`, two GETs sharing one budget — 5 s. |

Slow engine swaps are Hermes' business, not this plugin's: its own per-call
timeout (`HERMES_API_TIMEOUT`, default 1800 s) already absorbs a load, and the
profile deliberately sends no client-level `timeout` that would shadow it.

## Budget posture

The plugin **sends no `max_tokens` on the request path**. No profile hook sees
the caller's budget — `build_api_kwargs_extras` receives `model`, `base_url`,
`session_id`, `reasoning_config` and friends, but no `max_tokens` and no
messages — so a profile that returned one would *replace* Hermes' internal
budgets (overflow-recovery `safe_out`, the micro-compaction cap) instead of
clamping them. Instead it supplies the three inputs Hermes' own
reserve-and-shorten recovery consumes:

1. **the window** — `get_model_context_length`, the number
   `local_available_out = window - estimated_request` is computed against and
   what compaction triggers on (Hermes compresses at 50 % of the window by
   default);
2. **a declared default output cap** — `get_max_tokens(model)`, used when the
   caller sent no budget, which is the normal case:

   ```
   cap = min(LLAMA_SWAP_DEFAULT_MAX_TOKENS, max(LLAMA_SWAP_MIN_OUTPUT_TOKENS, window // 4))
   ```

   A cap, never a raise. The `window // 4` term is chosen against that 50 %
   compaction point: prompt ≤ ½ window plus ¼ window of output leaves a quarter
   of the window as slack.
3. **a classifiable error** — so Hermes shortens and retries (output-cap branch)
   or compresses and retries (`context_overflow`) instead of losing the turn.

Consequences an operator should know:

- **An explicitly large `model.max_tokens` reaches the wire unchanged** and is
  corrected only *after* the engine rejects it — one wasted round-trip on an
  engine that errors (llama.cpp), invisible on one that clamps silently. On a
  silent-clamp engine there is no error for Hermes to react to, so **the declared
  cap is the only shield there**. That is why the cap is sized conservatively
  and why raising `LLAMA_SWAP_DEFAULT_MAX_TOKENS` is the knob to think about
  before raising `model.max_tokens`.
- **Auxiliary calls carry no output cap on this route** (see
  [Known limits](#known-limits)).

## Endpoint matrix

The running build this plugin was verified against is **`v264-9-g4a61d5`**
(`GET /api/version`: `version v264-9-g4a61d5`, `commit 4a61d5e+`, build date
`2026-10-17T23:05:34Z`). Reproduced from the design spec so a future llama-swap
can be checked against it:

| Endpoint | Present in `v264-9-g4a61d5` | What it gives |
|---|---|---|
| `GET /v1/models` | yes | per-model `context_length` + `context_window`, `capabilities` (`vision`, `function_calling`), `supported_parameters`, `architecture.input_modalities`, `meta.llamaswap.reasoning.{enabled,effort,efforts,tag}`, `meta.n_ctx`, `meta.llamaswap.aliases`, `status.value` = `loaded`/`unloaded`, `owned_by`, `name`, `description` |
| `GET /running` | yes | `running[]` with `model`, `state` (`ready`), `proxy`, `ttl`, `cmd` |
| `GET /api/version` | yes | `version`, `commit`, `build_date` |
| `GET /api/metrics/stats` | yes | `total_requests`, `total_input_tokens`, `total_output_tokens`, `total_cache_tokens`, throughput histograms (`prompt_histogram` / `gen_histogram`, each with `p50` and `p95` — a speed, not a latency) |
| `GET /api/hardware`, `GET /api/performance` | yes | host CPU/GPU/VRAM inventory and telemetry |
| `GET /health` | yes | literal `OK` |
| `GET /metrics` | yes | Prometheus `llamaswap_*` |
| `GET /api/events` | yes | SSE, `modelStatus` with `state` + `context_length` (not used) |
| `GET /props?model=` | **no** (newer builds only) | engine's own `n_ctx`, `n_predict` |
| `POST /v1/messages` | **no** (newer builds only) | Anthropic wire |

Only the first four are read. The last two are the ones a future llama-swap may
add: `/props` would let a never-loaded model be sized (see
[Known limits](#known-limits)) — but note `/props` **dispatches the model**, i.e.
it can start an engine, so it must never go on a timer; and `/v1/messages` would
change `api_mode`, which this plugin pins to `chat_completions`. The plugin
deliberately touches no control endpoint: no `/unload`,
`/api/models/unload`, `/api/profiles/active` or `/api/inflight/*/cancel`.

## Reasoning rejections

A reasoning level the loaded model does not publish is rejected on the wire.
Captured live against the loaded model (`ultra` is a Hermes ladder level that
none of this server's four models publishes):

```
HTTP 400
{"error": {"message": "reasoning_effort 'ultra' unsupported; use minimal|low|medium|high|xhigh|max|none", "type": "invalid_request_error", "param": null, "code": null}}
```

That body **is** recognised by Hermes' `is_reasoning_field_rejection`
(`agent/error_classifier.py:548`, consulted at `:1175`), measured against the
captured body rather than assumed. So Hermes takes `_V_REASONING_MANDATORY` — it
drops the reasoning field and retries once — instead of falling through to
`_V_FORMAT_ERROR` (`:1211`) and aborting the turn.

The same probe against a different loaded engine — `strixite`, a build whose own
vocabulary stops at `xhigh` — returns a **different shape**, which is the reason no
wording rule belongs in this plugin:

```
HTTP 400
{"error":{"message":"'reasoning_effort' is 'ultra'; supported: xhigh, high (= xhigh), medium, low, minimal (= low), none (thinking off)","type":"invalid_request_error","param":"reasoning_effort","code":null}}
```

The first body is caught by core's proximity rule (`unsupported` next to the field
name); this one by its structured `param` recogniser — and the message on its own,
without the `param` field, is **not** recognised. Two engines, two wordings, one
generic mechanism each. `errors.py` claims only the wordings core gets wrong, and it
gets both of these right, so **no plugin rule is needed here and none was added**.

The same clamp normally keeps the level off the wire in the first place:
`build_api_kwargs_extras` clamps onto the model's own published ladder, so on this
server a request for `ultra` is sent as that model's ceiling — `max` for the three
that publish it, `xhigh` for `strixite` — never verbatim.

## Troubleshooting

**The server is down or unreachable.** Hermes pays **one** bounded probe per
`LLAMA_SWAP_CATALOG_TTL_S` (default 3 s, and a dead server pays only the first
of the two GETs), then falls back to its own context default — the 256 K
`DEFAULT_FALLBACK_CONTEXT` — for every model. Nothing hangs and nothing crashes:
a failed probe is cached exactly like a successful one, which is what stops a
dead server from being re-probed on every hook call. The practical cost: with
the server down, windows are Hermes' guess, not the engine's, so a first request
can overflow until the server answers again.

**Wrong URL shape.** `LLAMA_SWAP_BASE_URL` must be the OpenAI-style base with
`/v1` on the end (`http://host:8080/v1`); trailing slashes are trimmed. The
server-root endpoints (`/running`, `/api/metrics/stats`, `/api/version`) are
derived by stripping that `/v1`, so a base *without* `/v1` is treated as already
a root and the model list read then misses. A base that is not a URL at all, an
HTML proxy page, or a redirect all read as "no models" — never as a crash, and
never as a redirect a credential could follow (redirects are refused outright).

**`apiKeys` are configured on llama-swap but `LLAMA_SWAP_API_KEY` is unset or
wrong.** Two distinct failures: unset means Hermes refuses the provider outright
(`No usable credentials found for provider 'llama-swap'`); a value that is
non-empty but not the server's key means every request 401s, which Hermes maps as
an auth failure with credential rotation. Set the real key in that case — the
placeholder trick only works on a server that checks no key.

**`/usage` works but the model-picker usage chip never renders.** This is a
known core behaviour, not a bug here: overriding `fetch_account_usage` makes
`has_account_usage` true (`agent/account_usage_cache.py:64`), so the model
inventory adds `llama-swap` to `wanted` and polls it — then
`remember_account_usage` returns early because the snapshot has no `windows`
(`:110`), so nothing is cached and the chip has nothing to render. Net effect:
**`/usage` works; the picker chip does not.** The poll is throttled (one try per
120 s per process), runs on a daemon thread, and is bounded by this plugin's own
5 s, so it costs nothing beyond two GETs. A details-only plugin cannot render
that chip — `AccountUsageWindow` carries a `used_percent`, and llama-swap
publishes no quota to make a percentage out of — and should not pretend to.

**The `/usage` counters are not lifetime totals.** Measured on this server:
three samples minutes apart showed `total_requests` rising (1559 → 1566 → 1567)
while `total_input_tokens` *fell* (2,528,718 → 2,516,575). llama-swap's activity
store keeps a bounded set of rows, so the token totals track recent traffic and
can go down. Read them as "what the server still remembers", never as a bill.

## Known limits

Verbatim from the design spec's Open risks, plus the two measured in review:

- A model never loaded on llama-swap cannot be sized; the 256K fallback is a
  chosen, documented exposure, not a solved problem.
- `model_capabilities` mutated after import is a use the bundled profiles do not
  make; under `plugins.isolation: host` (not this install's mode) the profile
  would live in another process and the mutation would not propagate.
- A user-set `model.max_tokens` larger than the room left is corrected only after
  the engine rejects it — one wasted round-trip on llama.cpp, invisible on a
  silent-clamp engine.
- Aux requests carry no output cap on this route (parity with `custom` today).
  If it bites, the fix is an opt-in aux-only cap, not a transport.
- Hermes' per-model compaction threshold (`model_thresholds`) is a user setting;
  the `window // 4` output term assumes the 50 % default and gets tighter, not
  looser, if the user raises it.
- **Offline default context while the server is down**: see
  [Troubleshooting](#troubleshooting) — one bounded probe per TTL, then Hermes'
  own context default, so windows are Hermes' answer until the server returns.
- **`/usage` works, the model-picker usage chip does not**: see
  [Troubleshooting](#troubleshooting) — a details-only snapshot is never cached
  (`agent/account_usage_cache.py:110`).

## Verify

```bash
# 1. offline unit/integration suite (no network; Hermes-optional)
python3 -m pytest tests -q

# 2. live check against a real llama-swap — one line per check, non-zero on the first failure
LLAMA_SWAP_BASE_URL=http://<host>:8080/v1 python3 scripts/smoke.py

# 3. inside Hermes
hermes doctor        # expect: ✓ llama-swap under API Connectivity
hermes -z "reply with the single word ok" --provider llama-swap -m <a loaded id>
```

`scripts/smoke.py` reads `LLAMA_SWAP_BASE_URL`, drives the plugin's own public
surface (`build_profile`, `fetch_models`, `get_max_tokens`,
`supported_reasoning_efforts`, `resolve_aux_model`, `fetch_account_usage`), and
prints what the server actually said — including two captured error bodies (an
oversized prompt and an unpublished reasoning level), which is how
`tests/fixtures/errors.json` gets refreshed when llama-swap or its engines
change. **It never forces an engine swap**: every inference request names the
model `/running` reports as ready, and if nothing is ready it fails rather than
waking a model up. Run it under a Hermes `PYTHONPATH` and two Hermes-only checks
light up as well (the `/usage` snapshot, and the reasoning-rejection
classification of the captured body).
