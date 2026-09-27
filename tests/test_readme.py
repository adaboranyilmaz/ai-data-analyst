"""The README renderer (scripts/90_readme.py): number formats, value resolution, pending
values, and the checks that no number is typed into a template by hand and that every table
names its source."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location("readme", ROOT / "scripts/90_readme.py")
readme = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(readme)


@pytest.mark.parametrize(
    "value, spec, expected",
    [
        (0.17857, "f3", "0.179"),
        (-0.0357, "f2", "−0.04"),
        (-0.001, "f2", "0.00"),  # no negative zero
        (0.036, "sf3", "+0.036"),
        (-0.04, "sf2", "−0.04"),
        (0.0001, "sf2", "0.00"),
        (0.789, "pct1", "78.9%"),
        (0.0033, "pp1", "+0.3 pp"),
        (-0.01, "pp1", "−1.0 pp"),
        ([-0.01, 0.0167], "ppci1", "[−1.0, +1.7]"),
        ([0.27, 0.751], "ci2", "[0.27, 0.75]"),
        ({"mean": 0.0733, "std": 0.0092}, "pm3", "0.073 ± 0.009"),
        (1007.5, "int", "1,008"),
        (0.62, "usd2", "$0.62"),
        (8.42, "s1", "8.4 s"),
        (6120, "dur", "1.7 h"),
    ],
)
def test_formats(value, spec, expected):
    assert readme.fmt(value, spec) == expected


def test_lookup_uses_slashes_because_keys_contain_dots():
    obj = {"arms": {"claude-haiku-4-5__design.1": {"ex": [0.1, 0.2]}}}
    assert readme.lookup(obj, "arms/claude-haiku-4-5__design.1/ex/1") == 0.2
    with pytest.raises(KeyError):
        readme.lookup(obj, "arms/missing")


def test_expr_is_arithmetic_only():
    env = {"a__b": 3.0, "c": 1.0}
    assert readme._eval("a.b / (1 - c / 2)", env) == pytest.approx(6.0)
    with pytest.raises(ValueError):
        readme._eval("__import__('os')", env)


def test_values_resolve_in_dependency_order(monkeypatch):
    monkeypatch.setattr(readme, "load", lambda f: {"x": {"y": 0.25, "items": [{"k": 1}, {"k": 2}]}})
    spec = {
        "values": {
            "d": {"expr": "a + b.c", "fmt": "f2"},  # defined before its inputs
            "a": {"file": "f", "path": "x/y", "fmt": "f2"},
            "b.c": {"file": "f", "path": "x/items", "count_where": {"k": 2}, "fmt": "int"},
            "m": {"file": "f", "path": "x/items", "mean_of": "k", "fmt": "f1"},
        }
    }
    assert readme.resolve_values(spec) == {"a": "0.25", "b.c": "1", "d": "1.25", "m": "1.5"}


def test_pending_values_render_tbd_and_so_do_exprs_over_them(monkeypatch):
    monkeypatch.setattr(readme, "load", lambda f: {"y": 0.5})
    spec = {
        "values": {
            "later": {"pending": True},
            "now": {"file": "f", "path": "y", "fmt": "pct0"},
            "diff": {"expr": "later - now", "fmt": "pp1"},
        }
    }
    assert readme.resolve_values(spec) == {"later": "TBD", "now": "50%", "diff": "TBD"}


def test_a_missing_results_file_is_an_error_not_tbd(tmp_path, monkeypatch):
    monkeypatch.setattr(readme, "RESULTS", tmp_path)
    monkeypatch.setattr(readme, "_files", {})
    spec = {"values": {"x": {"file": "metrics/typo.json", "path": "a", "fmt": "f2"}}}
    with pytest.raises(FileNotFoundError):
        readme.resolve_values(spec)


def test_typed_numbers_are_caught_outside_placeholders_and_code(tmp_path, monkeypatch):
    template = tmp_path / "t.md"
    template.write_text(
        "Accuracy is {{acc}}, not 0.18 or 12%.\n"
        "Setting `k=3, rel_tol=0.01` and Python 3.12 are fine.\n"
        "[a link](results/v1.5/plot.png) is fine; 3 questions is an integer.\n",
        encoding="utf-8",
    )
    values = tmp_path / "v.yaml"
    values.write_text("values: {}\nallowed_literals: ['3.12']\n", encoding="utf-8")
    monkeypatch.setattr(readme, "VALUES", values)
    found = readme.typed_literals(template)
    assert len(found) == 2
    assert "'0.18'" in found[0] and "'12%'" in found[1]


@pytest.mark.parametrize(
    "text, n_problems",
    [
        ("| a |\n|---|\n| 1 |\n\n<sub>Source: `results/metrics/x.json`</sub>\n", 0),
        ("| a |\n|---|\n| 1 |\nSource: `pyproject.toml`\n", 0),
        ("| a |\n|---|\n| 1 |\n\nSome prose.\n", 1),
        ("| a |\n|---|\n| 1 |\n", 1),  # at the very end
        ("| a |\n|---|\n\n<sub>Source:</sub>\n", 1),  # names nothing
        ("```\n| not | a table |\n```\n", 0),  # inside a code block
        ("| a |\n|---|\n\nSource: f\n\n| b |\n|---|\n\nText.\n", 1),  # the second one
    ],
)
def test_tables_must_name_their_source(text, n_problems):
    assert len(readme.tables_without_source(text, "doc")) == n_problems


def test_committed_documents_match_a_fresh_render():
    """The same check CI runs with --check: no hand edits, no stale render, no typed
    numbers and no table without a source."""
    rendered = readme.render_all()
    assert readme.problems_in(rendered) == []
    for out, text in rendered.items():
        assert out.read_text(encoding="utf-8") == text, f"re-render {out.name}"
