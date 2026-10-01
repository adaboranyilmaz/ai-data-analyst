"""The sandbox's analyses on inputs with known answers: published worked examples, results
recomputed here with numpy, and the error rates of the tests on simulated data.

All the jobs go to the image in one call (a module fixture), as the guardrail sends them.
Published values: Wilson and Newcombe intervals from Newcombe (1998a, 1998b); Fisher's exact test
on the tea-tasting table (Fisher, 1935); Welch's test on Wikipedia's first worked example.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from src.stats.sandbox import Policy, run_container, run_jobs

pytestmark = pytest.mark.sandbox
Z = 1.959963984540054


def binary_rows(groups: dict[str, tuple[int, int]]) -> list[list]:
    """Rows [group, outcome] with k events out of n per group."""
    rows = []
    for g, (k, n) in groups.items():
        rows += [[g, 1]] * k + [[g, 0]] * (n - k)
    return rows


def compare(rows, reference=None, outcome_type="binary", strata=(), columns=("g", "y")):
    return {
        "spec": {
            "analysis": "compare_groups",
            "outcome_type": outcome_type,
            "outcome": "y",
            "group": "g",
            "reference_group": reference,
            "strata": list(strata),
        },
        "columns": list(columns),
        "rows": rows,
    }


A1 = [27.5, 21.0, 19.0, 23.6, 17.0, 17.9, 16.9, 20.1, 21.9, 22.6, 23.1, 19.6, 19.0, 21.7, 21.4]
A2 = [27.1, 22.0, 20.8, 23.4, 23.4, 23.5, 25.8, 22.0, 24.8, 20.2, 21.9, 22.1, 22.9, 20.5, 24.4]


def simpson_rows(seed: int, effect: float, n: int = 3000) -> list[list]:
    """The outcome depends on the stratum (and on the group by `effect`); the group is far more
    common in the high-risk stratum, so the crude comparison is confounded."""
    rng = np.random.default_rng(seed)
    s = rng.random(n) < 0.5
    g = rng.random(n) < np.where(s, 0.8, 0.2)
    p = np.where(s, 0.35, 0.10) + effect * g
    y = rng.random(n) < p
    return [[int(a), int(b), "high" if c else "low"] for a, b, c in zip(g, y, s, strict=True)]


def hc3(y: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xtx_inv = np.linalg.inv(X.T @ X)
    b = xtx_inv @ X.T @ y
    e = y - X @ b
    h = np.einsum("ij,jk,ik->i", X, xtx_inv, X)
    meat = X.T @ (X * (e**2 / (1 - h) ** 2)[:, None])
    return b, np.sqrt(np.diag(xtx_inv @ meat @ xtx_inv))


def trend_rows(seed: int = 3) -> tuple[list[list], np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.integers(1993, 1999, 3000).astype(float)
    y = (rng.random(3000) < 0.1 + 0.02 * (x - 1993)).astype(float)
    return [[a, b] for a, b in zip(x, y, strict=True)], x, y


def null_sims(seed: int, sims: int, n: int, p1: float, p0: float) -> list[dict]:
    rng = np.random.default_rng(seed)
    jobs = []
    for _ in range(sims):
        k1, k0 = int(rng.binomial(n, p1)), int(rng.binomial(n, p0))
        jobs.append(compare(binary_rows({"a": (k1, n), "b": (k0, n)}), reference="b"))
    return jobs


CASES = {
    "wilson": compare(binary_rows({"a": (81, 263), "b": (15, 148), "c": (0, 20), "d": (1, 29)})),
    "newcombe_a": compare(binary_rows({"x": (56, 70), "y": (48, 80)}), reference="y"),
    "newcombe_b": compare(binary_rows({"x": (9, 10), "y": (3, 10)}), reference="y"),
    "newcombe_d": compare(binary_rows({"x": (5, 56), "y": (0, 29)}), reference="y"),
    "newcombe_e": compare(binary_rows({"x": (0, 10), "y": (0, 20)}), reference="y"),
    "newcombe_f": compare(binary_rows({"x": (10, 10), "y": (0, 20)}), reference="y"),
    "fisher_tea": compare(binary_rows({"milk": (3, 4), "tea": (1, 4)}), reference="tea"),
    "welch": compare([["a1", v] for v in A1] + [["a2", v] for v in A2], "a2", "numeric"),
    "chi_square": compare(binary_rows({"a": (50, 200), "b": (70, 200), "c": (60, 200)})),
    "permutation": compare(binary_rows({"a": (1, 12), "b": (4, 10), "c": (0, 9)})),
    "simpson_null": compare(simpson_rows(1, 0.0), "0", strata=["s"], columns=("g", "y", "s")),
    "simpson_real": compare(simpson_rows(2, -0.08), "0", strata=["s"], columns=("g", "y", "s")),
    "trend": {
        "spec": {"analysis": "trend", "outcome_type": "binary", "outcome": "y", "x": "x"},
        "columns": ["x", "y"],
        "rows": trend_rows()[0],
    },
    "skewed": compare(
        [["a", v] for v in [1] * 40 + [500]] + [["b", v] for v in [1, 2] * 20], "b", "numeric"
    ),
    "missing": compare([["a", 1], ["a", None], [None, 0], ["b", 0], ["a", 0], ["b", 1]]),
    "binned": compare(
        [[i % 2, i % 3 == 0, i] for i in range(200)], "0", strata=["age"], columns=("g", "y", "age")
    ),
    "err_column": compare(binary_rows({"a": (1, 3)}), columns=("g", "z")),
    "err_one_group": compare(binary_rows({"a": (1, 3)})),
    "err_many_groups": compare([[f"g{i}", i % 2] for i in range(30)]),
    "err_outcome": compare([["a", 2], ["b", 0]]),
    "err_trend_x": {
        "spec": {"analysis": "trend", "outcome_type": "binary", "outcome": "y", "x": "x"},
        "columns": ["x", "y"],
        "rows": [[1, 0], [2, 1], [1, 1]],
    },
    "err_spec": {"spec": {"analysis": "regress"}, "columns": ["a"], "rows": []},
}
NULL = null_sims(11, 300, 150, 0.2, 0.2)
COVER = null_sims(12, 300, 150, 0.3, 0.2)


@pytest.fixture(scope="module")
def results(sandbox_ready) -> dict:
    jobs = [{"id": k, **v} for k, v in CASES.items()]
    jobs += [{"id": f"null{i}", **j} for i, j in enumerate(NULL)]
    jobs += [{"id": f"cover{i}", **j} for i, j in enumerate(COVER)]
    return {r["id"]: r for r in run_jobs(jobs)}


def ok(results, key) -> dict:
    r = results[key]
    assert r["ok"], r.get("error")
    return r["result"]


def approx(a, b, tol=5e-5) -> bool:
    return abs(a - b) <= tol


def test_wilson_intervals_match_the_published_ones(results):
    groups = {g["label"]: g for g in ok(results, "wilson")["groups"]}
    published = {
        "a": (0.2553, 0.3662),
        "b": (0.0624, 0.1605),
        "c": (0.0, 0.1611),
        "d": (0.0061, 0.1718),
    }
    for label, (lo, hi) in published.items():
        assert [round(v, 4) for v in groups[label]["ci"]] == [lo, hi], label
    assert groups["c"]["ci"][0] == 0.0  # exactly 0 for a count of zero


@pytest.mark.parametrize(
    "case,estimate,lo,hi",
    [
        ("newcombe_a", 0.2, 0.0524, 0.3339),
        ("newcombe_b", 0.6, 0.1705, 0.8090),
        ("newcombe_d", 0.0893, -0.0381, 0.1926),
        ("newcombe_e", 0.0, -0.1611, 0.2775),
        ("newcombe_f", 1.0, 0.6791, 1.0),
    ],
)
def test_newcombe_intervals_match_the_published_ones(results, case, estimate, lo, hi):
    (c,) = ok(results, case)["comparisons"]
    assert round(c["estimate"], 4) == estimate
    assert [round(v, 4) for v in c["ci"]] == [lo, hi]


def test_fisher_exact_on_the_tea_tasting_table(results):
    (c,) = ok(results, "fisher_tea")["comparisons"]
    assert approx(c["p_value"], 17 / 35, 1e-9)
    assert not ok(results, "fisher_tea")["detected"]


def test_welch_on_the_worked_example(results):
    r = ok(results, "welch")
    (c,) = r["comparisons"]
    assert approx(c["estimate"], np.mean(A1) - np.mean(A2), 1e-9)
    se = math.sqrt(np.var(A1, ddof=1) / 15 + np.var(A2, ddof=1) / 15)
    assert approx(c["estimate"] / se, -2.46, 0.005)  # t, as published
    assert approx(c["p_value"], 0.021, 0.0005)
    assert c["ci"][0] < c["estimate"] < c["ci"][1] < 0
    assert r["detected"]


def test_chi_square_for_three_groups(results):
    r = ok(results, "chi_square")
    ev, n = np.array([50, 70, 60]), np.array([200, 200, 200])
    table = np.stack([ev, n - ev], axis=1)
    exp = np.outer(table.sum(1), table.sum(0)) / table.sum()
    x2 = float(((table - exp) ** 2 / exp).sum())
    assert r["any_difference"]["method"] == "Pearson's chi-square"
    assert approx(r["any_difference"]["p_value"], math.exp(-x2 / 2), 1e-9)  # df 2
    assert r["primary"] == "any_difference"
    assert {w["kind"] for w in r["warnings"]} >= {"several_comparisons"}


def test_small_counts_use_a_permutation_test(results):
    r = ok(results, "permutation")
    assert r["any_difference"]["method"].startswith("permutation")
    assert 0 < r["any_difference"]["p_value"] < 1


def test_the_stratified_check_finds_confounding(results):
    null = ok(results, "simpson_null")
    s = null["stratified"]
    assert not (s["crude"]["ci"][0] <= 0 <= s["crude"]["ci"][1])  # crude: a difference
    assert s["adjusted"]["ci"][0] <= 0 <= s["adjusted"]["ci"][1]  # within strata: none
    assert s["change"] == "vanished" and not null["detected"] and null["primary"] == "stratified"
    real = ok(results, "simpson_real")["stratified"]
    assert real["crude"]["estimate"] > 0 > real["adjusted"]["estimate"]
    assert real["change"] == "reversed"


def test_the_stratified_estimate_is_the_fixed_effects_fit(results):
    rows = simpson_rows(1, 0.0)
    g = np.array([r[0] for r in rows], float)
    y = np.array([r[1] for r in rows], float)
    s = np.array([r[2] == "low" for r in rows], float)
    b, se = hc3(y, np.column_stack([np.ones(len(y)), g, s]))
    adj = ok(results, "simpson_null")["stratified"]["adjusted"]
    assert approx(adj["estimate"], b[1], 1e-9)
    assert approx(adj["ci"][0], b[1] - Z * se[1], 1e-7)
    assert approx(adj["ci"][1], b[1] + Z * se[1], 1e-7)


def test_trend_slope_with_robust_errors(results):
    r = ok(results, "trend")
    _, x, y = trend_rows()
    b, se = hc3(y, np.column_stack([np.ones(len(y)), x]))
    assert approx(r["slope"]["estimate"], b[1], 1e-9)
    assert approx(r["slope"]["ci"][0], b[1] - Z * se[1], 1e-7)
    assert r["detected"] and r["primary"] == "slope"
    assert [p["x"] for p in r["by_x"]] == [1993.0, 1994.0, 1995.0, 1996.0, 1997.0, 1998.0]
    assert sum(p["n"] for p in r["by_x"]) == 3000


def test_warnings(results):
    kinds = lambda key: {w["kind"] for w in ok(results, key)["warnings"]}  # noqa: E731
    assert {"few_events", "all_or_none", "small_group"} <= kinds("newcombe_d")
    assert "skewed" in kinds("skewed")
    assert "dropped_missing" in kinds("missing") and ok(results, "missing")["n_dropped"] == 2
    assert "binned_stratum" in kinds("binned")


@pytest.mark.parametrize(
    "case,kind",
    [
        ("err_column", "unknown_column"),
        ("err_one_group", "too_few_groups"),
        ("err_many_groups", "too_many_groups"),
        ("err_outcome", "invalid_outcome"),
        ("err_trend_x", "too_few_x_values"),
        ("err_spec", "invalid_spec"),
    ],
)
def test_errors_are_reported_per_job(results, case, kind):
    assert not results[case]["ok"]
    assert results[case]["error"]["kind"] == kind


def test_false_alarms_at_the_nominal_rate(results):
    detected = [ok(results, f"null{i}")["detected"] for i in range(len(NULL))]
    assert 0.02 <= np.mean(detected) <= 0.09


def test_interval_coverage(results):
    covered = []
    for i in range(len(COVER)):
        (c,) = ok(results, f"cover{i}")["comparisons"]
        covered.append(c["ci"][0] <= 0.1 <= c["ci"][1])
    assert 0.91 <= np.mean(covered) <= 0.98


def test_the_same_input_gives_the_same_bytes(sandbox_ready):
    payload = json.dumps({"version": 1, "jobs": [{"id": k, **v} for k, v in CASES.items()]})
    policy = Policy.from_config()
    first = run_container(payload.encode(), policy)
    second = run_container(payload.encode(), policy)
    assert first.exit_code == 0 and first.stdout == second.stdout


def test_host_labels_match_the_sandbox(sandbox_ready):
    from src.stats.answer import label, order_key

    values = [True, False, 3, 2.0, 2.5, "F", 10]
    rows = [[v, i % 2] for i, v in enumerate(values * 4)]
    (r,) = run_jobs([{"id": "l", **compare(rows)}])
    assert r["ok"], r.get("error")
    got = [g["label"] for g in r["result"]["groups"]]
    assert got == sorted({label(v) for v in values}, key=order_key)


def test_dates_are_time_not_categories(sandbox_ready):
    import datetime as dt

    days = [dt.date(1993, 1, 1) + dt.timedelta(days=7 * i) for i in range(300)]
    strat = [[i % 2, (i * 7) % 3 == 0, d.isoformat()] for i, d in enumerate(days)]
    trend = [[d.isoformat(), (i * 13) % 5 == 0] for i, d in enumerate(days)]
    jobs = [
        {"id": "s", **compare(strat, "0", strata=["opened"], columns=("g", "y", "opened"))},
        {
            "id": "t",
            "spec": {"analysis": "trend", "outcome_type": "binary", "outcome": "y", "x": "day"},
            "columns": ["day", "y"],
            "rows": trend,
        },
    ]
    s, t = run_jobs(jobs)
    assert s["ok"] and t["ok"], (s.get("error"), t.get("error"))
    st = s["result"]["stratified"]
    assert st["units_dropped"] == 0
    levels = {b["stratum"] for b in st["by_stratum"]}
    assert len(levels) == 5 and all(" to " in lev and lev[:4].isdigit() for lev in levels)
    assert {"kind": "binned_stratum", "variable": "opened", "bins": 5} in s["result"]["warnings"]
    lo, hi = t["result"]["x_range"]
    assert 1993 <= lo < 1993.01 and 1998.7 < hi < 1998.8  # years
