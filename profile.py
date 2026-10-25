"""The llama-swap provider profile: what Hermes asks, answered by the server.

Everything Hermes asks about this provider — which models exist, how big a context
one has, whether a body is a llama-swap failure — comes from the server's own
catalog, never from a constant here: llama-swap decides its own model ids, windows
and reasoning vocabularies, and a value invented in this file would overwrite
Hermes' own answer with something the server never claimed. So the only HTTP read
in the plugin is :meth:`LlamaSwapProfile.warm`; every other hook answers from its
cache, because those hooks run on the request path where a stall is a hung turn.

Stdlib plus ``client`` / ``catalog`` / ``errors``. ``providers.base`` is Hermes' at
runtime; the test conftest stubs it when Hermes is absent. Nothing here imports
Hermes at module scope.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Mapping
from typing import Any

try:  # Hermes loads this file as a plugin-package submodule (providers/__init__.py:479-493)
    from . import client
    from . import errors
    from .catalog import ModelInfo, capabilities_entry, fetch_catalog
except ImportError:  # standalone import (tests, scripts): repo root on sys.path
    import client
    import errors
    from catalog import ModelInfo, capabilities_entry, fetch_catalog

from providers.base import ProviderProfile

DEFAULT_CATALOG_TTL_S = 60
DEFAULT_PROBE_TIMEOUT_S = 3
DEFAULT_MAX_TOKENS = 8192
DEFAULT_MIN_OUTPUT_TOKENS = 1024

# {normalized base URL: {"at": monotonic, "infos": {id: ModelInfo}, "failed": bool}}
# ``failed`` records that the probe at ``at`` could not read the catalog, which is
# what keeps a dead server from being re-probed on every hook call: the entry is
# present, so the read hooks answer from its (empty) ``infos``.
#
# One profile instance per process (Hermes registers one profile per name, and this
# plugin ships one base URL), so a plain module dict is enough — no per-home
# sharding like the bundled ``router`` plugin needs. The lock keeps a concurrent
# reader off a half-written entry; two profiles in one process share a shard by URL,
# which is the point of keying by URL rather than by profile.
_CATALOG_CACHE: dict[str, dict[str, Any]] = {}
_CACHE_LOCK = threading.Lock()


def _cache_entry(url: str) -> dict[str, Any] | None:
    with _CACHE_LOCK:
        return _CATALOG_CACHE.get(url)


def _store(url: str, infos: dict[str, ModelInfo], *, failed: bool) -> None:
    with _CACHE_LOCK:
        _CATALOG_CACHE[url] = {"at": time.monotonic(), "infos": infos, "failed": failed}


def _shard_url(profile: LlamaSwapProfile) -> str:
    """The base URL this profile's read hooks answer from (spec "Address authority")."""
    return profile._active_url or client.normalize_base_url(profile.base_url)


def _shard_for(profile: LlamaSwapProfile) -> dict[str, ModelInfo] | None:
    """The cached catalog for that URL, or ``None`` when nothing has been fetched.

    ``None`` and ``{}`` are different answers: the empty dict is a server that was
    reached and had nothing (or failed), and must not read as "never probed".
    """
    entry = _cache_entry(_shard_url(profile))
    return None if entry is None else entry["infos"]


def catalog_for(profile: LlamaSwapProfile, model: str) -> ModelInfo | None:
    """The cached :class:`ModelInfo` for ``model``, or ``None``.

    Never touches the network — the hooks that read it (budget, vocabulary, aux
    model) run per request. ``None`` means this shard does not describe the model,
    which is the caller's "no claim" signal, not an error.
    """
    shard = _shard_for(profile)
    return None if shard is None else shard.get(model)


def _refresh_capabilities(profile: LlamaSwapProfile, infos: dict[str, ModelInfo]) -> None:
    """Replace ``model_capabilities`` contents in place, one entry per reported model.

    ``agent/models_dev.py:813-820`` reads that dict by reference on every lookup, so
    a rebound dict would be missed: ``clear()`` + ``update()`` is the only shape that
    reaches Hermes. Clearing is the other half of the contract — a model the server
    stopped reporting must stop being described. Entries carry only the fields the
    server stated (``catalog.capabilities_entry``), so Hermes keeps its own answer
    for everything llama-swap did not say.

    The two steps are not atomic: a lookup landing between them sees an empty dict,
    which is Hermes' own "no profile claim" path — it answers from its catalog, the
    same answer it would have given before the refresh. A torn *entry* would be worse,
    which is why entries are replaced whole under the cache lock instead.
    """
    entries = {
        model_id: entry
        for model_id, info in infos.items()
        if (entry := capabilities_entry(info))
    }
    capabilities = profile.model_capabilities
    capabilities.clear()
    capabilities.update(entries)


def classify_api_error(
    exc: object = None,
    *,
    status_code: int | None = None,
    error_code: str | None = None,
    message: str | None = None,
    body: object = None,
    model: str | None = None,
    **ignored: object,
) -> dict | None:
    """Delegate to :func:`errors.classify_llama_swap_error`; a raise degrades to ``None``.

    Hermes reads the hook's return value inside its own classifier, so a rule that
    raised would cost the whole verdict; failing open to "core decides" is the
    cheaper mistake. Reached as ``errors.classify_llama_swap_error`` (an attribute
    lookup) so the module stays the seam tests patch.

    ``exc`` is Hermes' positional exception argument (``agent/error_classifier.py:735``);
    a caller that spells it ``error=`` names the same value, absorbed here. It is wired
    as the ``classify_api_error`` FIELD in :func:`build_profile` rather than as a method,
    because the base dataclass's ``__init__`` assigns that field and would shadow a method.
    """
    try:
        return errors.classify_llama_swap_error(
            exc if exc is not None else ignored.get("error"),
            status_code=status_code,
            error_code=error_code,
            message=message,
            body=body,
            model=model,
        )
    except Exception:
        return None


class LlamaSwapProfile(ProviderProfile):
    """llama-swap route: every model fact is read from the server that serves it.

    ``catalog_ttl`` / ``probe_timeout`` / ``min_output_tokens`` / ``api_key`` are this
    plugin's own configuration, not ``ProviderProfile`` fields, so they are stripped
    before the dataclass base constructor sees them.
    """

    # Restates the built-in (providers/base.py:140) rather than overriding it: the test
    # stub's ProviderProfile is a plain kwargs-copying class with no dataclass defaults,
    # so a test that read an inherited default would pass under Hermes and AttributeError
    # under the stub. The same reason ``build_profile`` passes model_capabilities.
    api_mode: str = "chat_completions"

    # The shard the read hooks answer from, once fetch_models has been handed a route.
    _active_url: str = ""

    def __init__(
        self,
        *,
        catalog_ttl: int = DEFAULT_CATALOG_TTL_S,
        probe_timeout: int = DEFAULT_PROBE_TIMEOUT_S,
        min_output_tokens: int = DEFAULT_MIN_OUTPUT_TOKENS,
        api_key: str = "",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.catalog_ttl = catalog_ttl
        self.probe_timeout = probe_timeout
        self.min_output_tokens = min_output_tokens
        self.api_key = api_key

    def warm(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float | None = None,
        force: bool = False,
    ) -> dict[str, ModelInfo]:
        """The one place this plugin reads the catalog over HTTP; cached per base URL.

        Returns the shard's ``{id: ModelInfo}`` — empty when the read failed, and the
        failure is cached for the TTL so a dead server costs one bounded wait instead
        of one per hook call. ``timeout`` is the caller's when it has one (setup and
        the picker can afford more than a hot-path probe); otherwise the probe budget.
        """
        url = client.normalize_base_url(base_url or self.base_url)
        self._active_url = url
        cached = _cache_entry(url)
        if cached is not None and not force and time.monotonic() - cached["at"] < self.catalog_ttl:
            return cached["infos"]

        infos, error = fetch_catalog(
            url, api_key or self.api_key, timeout if timeout is not None else self.probe_timeout
        )
        _store(url, infos, failed=error is not None)
        if error is None:
            _refresh_capabilities(self, infos)
        return infos

    def fetch_models(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 8.0,
    ) -> list[str] | None:
        """Live model ids in llama-swap's own form (raw ids, no prefix).

        ``None`` on failure, never ``[]`` — Hermes only keeps its own fallback list if
        this declines. ``base_url`` is the credential-pool route Hermes resolved, so
        it wins over ``$LLAMA_SWAP_BASE_URL`` and becomes the shard the read hooks
        answer from. The timeout default is the base hook's: this runs off the hot
        path (picker, setup, doctor), where a mid-swap server may take its time.
        """
        infos = self.warm(base_url=base_url, api_key=api_key, timeout=timeout)
        return list(infos) or None

    def get_model_context_length(self, model: str) -> int | None:
        """The window llama-swap reported for ``model``, or ``None``.

        The sanctioned warming point: Hermes calls this at agent init and on model
        switch, so the first call for a server does ONE probe bounded by
        ``LLAMA_SWAP_PROBE_TIMEOUT_S`` and every later call is cache-only. ``None``
        means the server never stated a window — a model it has not loaded yet — which
        hands Hermes its own chain (catalog, config override, then the 256 K default).
        Inventing a window here is the failure this plugin exists to avoid.
        """
        shard = _shard_for(self)
        if shard is None:
            shard = self.warm()
        info = shard.get(model)
        return None if info is None else info.context_length


def _env_positive_int(env: Mapping[str, str], name: str, default: int) -> int:
    """``client.env_int`` floored at 1: every knob here is a budget, and 0 would mean
    no caching or no output tokens — never what an operator meant by the value."""
    return max(1, client.env_int(env, name, default))


def build_profile(env: Mapping[str, str]) -> LlamaSwapProfile:
    """Construct the profile from an env mapping (injected, so tests never edit os.environ).

    A malformed knob falls back to its default: Hermes reads these at startup and must
    not crash — nor silently disable its own cache — on a bad environment.
    """
    return LlamaSwapProfile(
        name="llama-swap",
        aliases=("llamaswap", "swap"),
        display_name="llama-swap",
        description="llama-swap — models, context windows and reasoning vocabularies read from the server",
        env_vars=("LLAMA_SWAP_API_KEY", "LLAMA_SWAP_BASE_URL"),
        base_url=client.normalize_base_url(client.env_str(env, "LLAMA_SWAP_BASE_URL", client.DEFAULT_BASE_URL)),
        auth_type="api_key",
        # The wire accepts image parts; per-model truth is in model_capabilities.
        supports_vision=True,
        fallback_models=(),          # the ids are the server's to know
        # Required under the test stub (no dataclass defaults), the base's own default
        # under Hermes. Filled in place by the first catalog refresh.
        model_capabilities={},
        default_max_tokens=_env_positive_int(env, "LLAMA_SWAP_DEFAULT_MAX_TOKENS", DEFAULT_MAX_TOKENS),
        classify_api_error=classify_api_error,
        catalog_ttl=_env_positive_int(env, "LLAMA_SWAP_CATALOG_TTL_S", DEFAULT_CATALOG_TTL_S),
        probe_timeout=_env_positive_int(env, "LLAMA_SWAP_PROBE_TIMEOUT_S", DEFAULT_PROBE_TIMEOUT_S),
        min_output_tokens=_env_positive_int(env, "LLAMA_SWAP_MIN_OUTPUT_TOKENS", DEFAULT_MIN_OUTPUT_TOKENS),
        api_key=client.env_str(env, "LLAMA_SWAP_API_KEY", ""),
    )
