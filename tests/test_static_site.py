"""The static demo's files are what the service serves, for every curated run."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from src.serving import champion as champion_mod
from src.serving import static_site
from src.serving.app import Settings, create_app
from src.serving.meter import ROOT, Meter, config
from src.serving.store import RunStore


@pytest.fixture(scope="module")
def built(tmp_path_factory, monkeypatch_module):
    out = tmp_path_factory.mktemp("site")
    champion = champion_mod.load()
    cfg = champion.apply(config())
    store = RunStore(ROOT / cfg["curated"]["out_dir"])
    summary = static_site.build(out, store, Meter.load(cfg), cfg, champion.info())
    return out / static_site.DATA, store, cfg, summary


@pytest.fixture(scope="module")
def monkeypatch_module():
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("ANALYST_REPLAY_SPEED", "1000000")
        yield mp


def sse_events(text: str) -> list[dict]:
    return [
        json.loads(line.split("data: ", 1)[1])
        for block in text.strip().split("\n\n")
        for line in block.splitlines()
        if line.startswith("data: ")
    ]


def test_meta_runs_and_streams_equal_what_the_service_returns(built, monkeypatch_module):
    data, store, _cfg, summary = built
    client = TestClient(create_app(Settings(mode="replay", request_log=None)))
    assert (
        json.loads((data / "meta.json").read_text(encoding="utf-8"))
        == client.get("/api/meta").json()
    )
    assert summary["runs"] == len(store.index()) == 20
    for entry in store.index():
        rid = entry["id"]
        served = client.get(f"/runs/{rid}").json()
        assert json.loads((data / "runs" / f"{rid}.json").read_text(encoding="utf-8")) == served
        streamed = sse_events(client.post("/ask", json={"run_id": rid}).text)
        file = json.loads((data / "streams" / f"{rid}.json").read_text(encoding="utf-8"))
        assert [f["event"] for f in file] == streamed, rid
        assert file[0]["wait_ms"] == 0 and file[-1]["event"]["type"] == "done"


def test_the_build_is_deterministic(built, tmp_path):
    _data, store, cfg, summary = built
    again = static_site.build(tmp_path, store, Meter.load(cfg), cfg, champion_mod.load().info())
    assert again == summary
