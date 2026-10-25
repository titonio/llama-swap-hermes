"""llama-swap's catalog, translated into facts Hermes can act on.

One ``/v1/models`` read plus one best-effort ``/running`` read become
:class:`ModelInfo` records and one ``model_capabilities`` entry per model. The
rule for every field: report what the server stated, and nothing else. ``None``
means llama-swap did not say, so Hermes keeps its own answer; ``False`` / ``0`` /
``()`` are claims, and this module only makes one when the server made it first.
A model that has never been loaded reports no ``context_length`` at all (live
example: ``gufo``), and inventing a window is the exact failure this plugin
exists to avoid.

Stdlib plus :mod:`client`; Hermes is touched only for the effort ladder, lazily.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

# The plugin dir is not on sys.path under Hermes' loader, so sibling modules are
# reached relatively there and absolutely in a flat checkout (tests, scripts).
try:
    from . import client
except ImportError:
    import client

# Fallback only. The source of truth is Hermes' ``agent/reasoning_effort.py``
# ``EFFORT_LADDER``, imported inside :func:`supported_efforts` so this module
# stays importable without Hermes; keep this tuple in sync with that file.
EFFORT_LADDER_FALLBACK: tuple[str, ...] = (
    "none",
    "minimal",
    "low",
    "medium",
    "high",
    "xhigh",
    "max",
    "ultra",
)


@dataclass
class ModelInfo:
    """What llama-swap stated about one model id. ``None`` = the server did not say."""

    id: str
    context_length: int | None
    vision: bool | None
    tools: bool | None
    efforts: tuple[str, ...] | None
    reasoning_enabled: bool | None
    loaded: bool = False
    aliases: tuple[str, ...] = ()
    input_modalities: tuple[str, ...] = ()
    owned_by: str = ""
    name: str = ""
    description: str = ""


def _dig(obj: Any, *keys: str) -> Any:
    """Nested dict lookup; ``None`` as soon as a level is missing or not a dict."""
    for key in keys:
        obj = obj.get(key) if isinstance(obj, dict) else None
    return obj


def _strings(value: Any) -> tuple[str, ...]:
    """A JSON list as a tuple of non-empty strings; any other shape is empty."""
    if not isinstance(value, list):
        return ()
    return tuple(text for item in value if (text := str(item).strip()))


def _flag(value: Any) -> bool | None:
    """A JSON boolean, or ``None`` when the field is absent or not a bool."""
    return value if isinstance(value, bool) else None


def _first_positive(*values: Any) -> int | None:
    """First value that is a real positive int (``True`` is never a window)."""
    for value in values:
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    return None


def parse_models(payload: Any) -> dict[str, ModelInfo]:
    """``{"data": [...]}`` (or a bare list) into ``{id: ModelInfo}``, server order.

    Any other shape — an HTML proxy page, ``{"error": …}``, a list of junk —
    yields ``{}`` so discovery reports "no catalog" instead of raising inside a
    Hermes picker or crashing a hook.
    """
    entries = payload.get("data") if isinstance(payload, dict) else payload
    if not isinstance(entries, list):
        return {}

    infos: dict[str, ModelInfo] = {}
    for entry in entries:
        model_id = str(entry.get("id") or "").strip() if isinstance(entry, dict) else ""
        if not model_id:
            continue
        infos[model_id] = ModelInfo(
            id=model_id,
            # llama-swap fills context_length from the engine once it has run; the
            # config's own n_ctx is the only honest stand-in before that.
            context_length=_first_positive(
                entry.get("context_length"),
                entry.get("context_window"),
                _dig(entry, "meta", "n_ctx"),
            ),
            vision=_flag(_dig(entry, "capabilities", "vision")),
            tools=_flag(_dig(entry, "capabilities", "function_calling")),
            efforts=_strings(_dig(entry, "meta", "llamaswap", "reasoning", "efforts")) or None,
            reasoning_enabled=_flag(_dig(entry, "meta", "llamaswap", "reasoning", "enabled")),
            loaded=_dig(entry, "status", "value") == "loaded",
            aliases=_strings(_dig(entry, "meta", "llamaswap", "aliases")),
            input_modalities=_strings(_dig(entry, "architecture", "input_modalities")),
            owned_by=str(entry.get("owned_by") or ""),
            name=str(entry.get("name") or ""),
            description=str(entry.get("description") or ""),
        )
    return infos


def parse_running(payload: Any) -> frozenset[str]:
    """The ids ``/running`` reports as ``ready`` — the engines actually serving.

    Any other shape yields the empty set: ``/running`` is a best-effort read, so a
    truthy non-list ``running`` (a proxy's ``{"running": 5}``) must read as "nothing
    ready", never raise out of the one place that does HTTP.
    """
    entries = _dig(payload, "running")
    if not isinstance(entries, list):
        return frozenset()
    ready = set()
    for entry in entries:
        if not isinstance(entry, dict) or entry.get("state") != "ready":
            continue
        model_id = str(entry.get("model") or "").strip()
        if model_id:
            ready.add(model_id)
    return frozenset(ready)


def mark_loaded(infos: dict[str, ModelInfo], ready: frozenset[str]) -> None:
    """Flag the ready ids — and aliases of them — as loaded, in place.

    Only ever sets ``True``: ``/running`` is a best-effort read, and a miss must
    not overwrite the load state ``/v1/models`` already stated in ``status.value``.
    """
    for info in infos.values():
        if info.id in ready or any(alias in ready for alias in info.aliases):
            info.loaded = True


def capabilities_entry(info: ModelInfo) -> dict[str, Any]:
    """One ``model_capabilities`` entry, carrying only the reported fields.

    Keys follow Hermes' ``model_overrides`` schema (``providers/base.py``). A key
    that is absent leaves Hermes' own catalog answer in charge; declaring
    ``False`` or ``0`` here would overwrite it with something the server never said.
    """
    reported = (
        ("supports_tools", info.tools),
        ("supports_vision", info.vision),
        ("context_window", info.context_length),
    )
    return {key: value for key, value in reported if value is not None}


def supported_efforts(
    info: ModelInfo, ladder: Sequence[str] | None = None
) -> tuple[str, ...] | None:
    """The model's published effort levels, filtered to ``ladder``. Tri-state:

    ``None`` — the model said nothing Hermes can act on, so the transport keeps its
    default vocabulary; ``()`` — reasoning is off, so no reasoning field may go on
    the wire at all; a non-empty tuple — clamp the requested effort onto these
    levels (the contract in ``providers/base.py::supported_reasoning_efforts``).
    Levels off the ladder are dropped, never forwarded: a gateway 400s on them, and
    a vocabulary that filters to nothing is unknown rather than disabled.
    """
    if info.reasoning_enabled is False:
        return ()
    published = set(info.efforts or ())
    if ladder is None:
        try:  # lazy: the plugin must import (and its units run) without Hermes
            from agent.reasoning_effort import EFFORT_LADDER

            ladder = EFFORT_LADDER
        except ImportError:
            ladder = EFFORT_LADDER_FALLBACK
    return tuple(level for level in ladder if level in published) or None


def fetch_catalog(
    base_url: str, api_key: str | None, timeout: float
) -> tuple[dict[str, ModelInfo], str | None]:
    """One ``/models`` GET plus one best-effort ``/running`` GET. Never raises.

    Returns ``(infos, error)``: when ``/models`` yields nothing usable, ``infos``
    is empty and ``error`` is a short string for the caller to surface. ``/running``
    only decorates the catalog, so discovery stands without it.
    """
    infos = parse_models(client.get_json(base_url, "/models", api_key=api_key, timeout=timeout))
    if not infos:
        return {}, "llama-swap /models returned no usable models"

    root = client.root_url(base_url)
    running = client.get_json(root, "/running", api_key=api_key, timeout=timeout)
    mark_loaded(infos, parse_running(running))
    return infos, None
