import json
import pathlib

FIX = pathlib.Path(__file__).parent / "fixtures"


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
