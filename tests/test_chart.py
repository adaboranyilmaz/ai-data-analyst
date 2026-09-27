"""validate_chart: valid Vega-Lite, only the result's fields, and nothing a browser would fetch."""

from __future__ import annotations

import copy

import pytest

from src.tools.chart import SCHEMA_URLS, _validator, validate_chart

COLUMNS = ["district", "n", "avg.salary"]
BAR = {
    "$schema": "https://vega.github.io/schema/vega-lite/v6.json",
    "data": {"name": "result"},
    "mark": "bar",
    "encoding": {
        "x": {"field": "district", "type": "nominal", "sort": "-y"},
        "y": {"field": "n", "type": "quantitative"},
    },
}


def with_(path: list, value, spec=BAR) -> dict:
    s = copy.deepcopy(spec)
    node = s
    for key in path[:-1]:
        node = node.setdefault(key, {})
    node[path[-1]] = value
    return s


def test_the_pinned_schema_loads():
    assert _validator().schema["$ref"] == "#/definitions/TopLevelSpec"
    assert "https://vega.github.io/schema/vega-lite/v6.json" in SCHEMA_URLS


def test_a_valid_chart():
    assert validate_chart(BAR, COLUMNS) == {"ok": True, "errors": []}


def test_transforms_may_create_fields():
    spec = with_(["transform"], [{"calculate": "datum.n * 2", "as": "double"}])
    spec = with_(["encoding", "y", "field"], "double", spec)
    assert validate_chart(spec, COLUMNS)["ok"]


def test_layers_and_data_without_a_top_level_name():
    spec = {"layer": [{"mark": "line", "encoding": {"x": {"field": "n", "type": "quantitative"}}}]}
    assert validate_chart(spec, COLUMNS)["ok"]


@pytest.mark.parametrize(
    "spec,phrase",
    [
        (with_(["data"], {"url": "https://example.com/x.csv"}), "'url' is not allowed"),
        (with_(["data"], {"values": [{"district": "x"}]}), 'data must be {"name": "result"}'),
        (with_(["data"], {"name": "other"}), 'data must be {"name": "result"}'),
        (with_(["encoding", "href"], {"field": "district"}), "'href' is not allowed"),
        (with_(["usermeta"], {"x": 1}), "'usermeta' is not allowed"),
        (with_(["datasets"], {"result": []}), "'datasets' is not allowed"),
        (with_(["mark"], "image"), "image marks"),
        (with_(["mark"], {"type": "image"}), "image marks"),
        (with_(["encoding", "x", "field"], "salary"), "not a column of the result"),
        (with_(["encoding", "x", "field"], "avg.salary"), "escape '.'"),
        (with_(["encoding", "x", "type"], "nominal-ish"), "not valid Vega-Lite"),
        (with_(["mark"], "pie-ish"), "not valid Vega-Lite"),
        (with_(["$schema"], "https://example.com/schema.json"), "$schema must be"),
        (with_(["transform"], [{"fold": ["n", "missing"]}]), "'missing' is not a column"),
        ("not an object", "must be a JSON object"),
        (with_(["description"], "x" * 20_000), "longer than"),
    ],
)
def test_refused(spec, phrase):
    out = validate_chart(spec, COLUMNS)
    assert not out["ok"]
    assert any(phrase in e for e in out["errors"]), out["errors"]


def test_an_escaped_dot_names_the_column():
    assert validate_chart(with_(["encoding", "x", "field"], "avg\\.salary"), COLUMNS)["ok"]


def test_errors_are_capped():
    spec = with_(["encoding", "x", "field"], "a")
    for ch in ("y", "color", "size", "opacity", "shape", "detail"):
        spec = with_(["encoding", ch], {"field": f"missing_{ch}"}, spec)
    out = validate_chart(spec, COLUMNS)
    assert len(out["errors"]) == 6 and out["errors"][-1].startswith("... and")
