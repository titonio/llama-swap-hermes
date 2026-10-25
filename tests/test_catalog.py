"""``catalog`` — the captured llama-swap payloads, parsed.

Every assertion runs on the committed captures in ``tests/fixtures/``, so the
facts are the live server's rather than a hand-written wish. No test names which
model is loaded: that is runtime state, so it comes from ``running.json`` through
``ready_ids()``.
"""

from __future__ import annotations

import json

import pytest

import client
from catalog import (
    ModelInfo,
    capabilities_entry,
    fetch_catalog,
    mark_loaded,
    parse_models,
    parse_running,
    supported_efforts,
)
from test_fixtures import FIX, ready_ids

_MODELS = json.loads((FIX / "models.json").read_text())
_RUNNING = json.loads((FIX / "running.json").read_text())


@pytest.fixture(scope="module")
def infos():
    """The captured ``/v1/models`` body, parsed once for the module."""
    return parse_models(_MODELS)


def _info(model_id, **facts):
    """A ``ModelInfo`` with every untested fact left unknown (``None``)."""
    return ModelInfo(
        id=model_id,
        context_length=None,
        vision=None,
        tools=None,
        efforts=None,
        reasoning_enabled=None,
        **facts,
    )


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


@pytest.mark.parametrize("junk", [None, [], {}, {"running": None}, {"running": [1]}, "<html>"])
def test_parse_running_tolerates_junk_payloads(junk):
    assert parse_running(junk) == frozenset()


def test_parse_models_keeps_server_order_and_passthrough_fields(infos):
    assert list(infos) == ["crucible", "gufo", "halogen", "strixite"]
    assert infos["halogen"].input_modalities == ("text", "image")
    assert infos["halogen"].owned_by == "llama-swap"
    assert infos["gufo"].input_modalities == () and infos["gufo"].aliases == ()


def test_parse_models_reads_the_stated_load_state(infos):
    # The two captures came from one refresh, so /models' status.value and
    # /running's ready id agree; if they ever diverge the fixtures drifted.
    assert [mid for mid, info in infos.items() if info.loaded] == ready_ids()


def test_mark_loaded_marks_the_ready_id_and_its_alias():
    ready = ready_ids()
    infos = {
        ready[0]: _info(ready[0]),
        "alias-holder": _info("alias-holder", aliases=(ready[0],)),
        "idle": _info("idle"),
        "stated-loaded": _info("stated-loaded", loaded=True),
    }
    mark_loaded(infos, frozenset(ready))
    assert infos[ready[0]].loaded is True and infos["alias-holder"].loaded is True
    assert infos["idle"].loaded is False
    # A /running miss never reports "not loaded" for a fact /models already stated.
    assert infos["stated-loaded"].loaded is True


def test_fetch_catalog_reads_models_then_running_and_marks_loaded(monkeypatch):
    calls = []

    def fake_get_json(base_url, path, **kwargs):
        calls.append((base_url, path, kwargs))
        return _RUNNING if path == "/running" else _MODELS

    monkeypatch.setattr(client, "get_json", fake_get_json)

    parsed, error = fetch_catalog("http://box:8080/v1", "key", 2.5)

    assert error is None
    # /running sits at the server root, not under /v1, and both reads get the
    # caller's credential and budget.
    assert calls == [
        ("http://box:8080/v1", "/models", {"api_key": "key", "timeout": 2.5}),
        ("http://box:8080", "/running", {"api_key": "key", "timeout": 2.5}),
    ]
    assert sorted(info.id for info in parsed.values() if info.loaded) == sorted(ready_ids())


def test_fetch_catalog_survives_a_failed_running_read(monkeypatch):
    monkeypatch.setattr(
        client, "get_json", lambda base, path, **kw: _MODELS if path == "/models" else None
    )

    parsed, error = fetch_catalog("http://box:8080/v1", None, 3.0)

    assert error is None and len(parsed) == 4


def test_fetch_catalog_reports_a_failed_models_read(monkeypatch):
    monkeypatch.setattr(client, "get_json", lambda *a, **kw: None)

    parsed, error = fetch_catalog("http://h.invalid:1/v1", None, 0.5)

    assert parsed == {} and error
