"""Planted effects: effect sizes, the draws, orientation of estimates (no database), and the copy
database's set-up and guards (with the benchmark loaded)."""

from __future__ import annotations

import numpy as np
import pytest

from src.stats import planted as pl
from src.stats.plans import Plan, PlanError

CFG = pl.config()
TEMPLATES = {t["id"]: t for t in CFG["templates"]}


def synthetic_units(n: int = 4000, seed: int = 0) -> pl.Units:
    rng = np.random.default_rng(seed)
    duration = rng.choice([12, 24, 36, 48, 60], n)
    amount = (duration * rng.integers(1000, 9000, n)).astype(np.int64)
    status = rng.choice(np.array(["A", "B", "C", "D"], dtype=object), n, p=[0.3, 0.05, 0.55, 0.1])
    cols = {
        "loan_id": np.arange(n, dtype=np.int64),
        "account_id": np.arange(n, dtype=np.int64) + 10_000,
        "client_id": np.arange(n, dtype=np.int64) + 20_000,
        "finished": np.isin(status, ["A", "B"]),
        "has_card": rng.random(n) < 0.25,
        "prague": rng.random(n) < 0.12,
        "female": rng.random(n) < 0.5,
        "year": rng.integers(1993, 1999, n),
        "age": rng.integers(18, 65, n),
        "duration": duration,
        "amount": amount,
        "payments": amount / duration,
        "status": status,
        "frequency": rng.choice(np.array([pl.WEEKLY, pl.MONTHLY], dtype=object), n),
        "gender": np.where(rng.random(n) < 0.5, "F", "M").astype(object),
    }
    cols["long_loan"] = cols["duration"] >= 48
    order = np.argsort(np.argsort(cols["amount"], kind="stable"), kind="stable")
    cols["amount_rank"] = order / (n - 1)
    return pl.Units(cols)


UNITS = synthetic_units()


def test_power_is_the_normal_approximation():
    assert pl.power(0.0, 1.0) == pytest.approx(0.05, abs=1e-6)
    assert pl.power(pl.Z, 1.0) == pytest.approx(0.5, abs=0.001)
    assert pl.power(pl.Z + 1.2815516, 1.0) == pytest.approx(0.9, abs=0.001)


@pytest.mark.parametrize("tid", ["A1", "A2", "B1", "B2", "C1", "C2"])
def test_effect_sizes_reach_the_configured_power(tid):
    sizes = pl.effect_sizes(TEMPLATES[tid], UNITS, CFG["power"])
    assert sizes["none"]["effect"] == 0
    for cond, target in CFG["power"].items():
        assert sizes[cond]["power"] == pytest.approx(target, abs=1e-6)
    assert 0 < sizes["small"]["effect"] < sizes["large"]["effect"]


def test_draws_are_reproducible_and_distinct():
    t = TEMPLATES["A1"]
    sizes = pl.effect_sizes(t, UNITS, CFG["power"])
    a = pl.draw(t, "large", UNITS, sizes, CFG, pl.seed_for(CFG, "A1", "large", 3))
    b = pl.draw(t, "large", UNITS, sizes, CFG, pl.seed_for(CFG, "A1", "large", 3))
    c = pl.draw(t, "large", UNITS, sizes, CFG, pl.seed_for(CFG, "A1", "large", 4))
    assert all(np.array_equal(a[k], b[k]) for k in a)
    assert not np.array_equal(a["status"], c["status"])


def bad(values) -> np.ndarray:
    return np.isin(values["status"], ["B", "D"])


def test_rates_plant_a_difference_and_keep_finished_loans_finished():
    t = TEMPLATES["A1"]
    sizes = pl.effect_sizes(t, UNITS, CFG["power"])
    v = pl.draw(t, "large", UNITS, sizes, CFG, np.random.default_rng(1))
    g = UNITS["has_card"]
    diff = bad(v)[g].mean() - bad(v)[~g].mean()
    assert diff == pytest.approx(sizes["large"]["effect"], abs=0.03)
    assert np.array_equal(np.isin(v["status"], ["A", "B"]), UNITS["finished"])
    # nothing else changes
    for k in ("amount", "frequency", "gender"):
        assert np.array_equal(v[k], pl.draw_real(UNITS)[k])


def test_means_shuffle_within_duration_and_keep_amount_equal_duration_times_payment():
    t = TEMPLATES["B2"]
    sizes = pl.effect_sizes(t, UNITS, CFG["power"])
    none = pl.draw(t, "none", UNITS, sizes, CFG, np.random.default_rng(2))
    for d in np.unique(UNITS["duration"]):
        idx = UNITS["duration"] == d
        assert sorted(none["amount"][idx]) == sorted(UNITS["amount"][idx])
    large = pl.draw(t, "large", UNITS, sizes, CFG, np.random.default_rng(2))
    assert np.allclose(large["payments"] * UNITS["duration"], large["amount"])
    g = UNITS["has_card"]
    shift = (large["amount"] - none["amount"])[g]
    assert np.all(shift == round(sizes["large"]["effect"]))
    assert np.all((large["amount"] - none["amount"])[~g] == 0)


def test_trend_plants_a_rising_rate():
    t = TEMPLATES["C1"]
    sizes = pl.effect_sizes(t, UNITS, CFG["power"])
    v = pl.draw(t, "large", UNITS, sizes, CFG, np.random.default_rng(3))
    x, y = UNITS["year"].astype(float), bad(v).astype(float)
    assert np.polyfit(x, y, 1)[0] == pytest.approx(sizes["large"]["effect"], abs=0.01)


@pytest.mark.parametrize(
    "tid,column,value", [("D1", "frequency", pl.WEEKLY), ("D2", "gender", "F")]
)
def test_confounded_draws(tid, column, value):
    t = TEMPLATES[tid]
    z = UNITS[t["confounder"]].astype(float)
    rng = np.random.default_rng(5)
    conf = pl.draw(t, "confounded", UNITS, {}, CFG, rng)
    none = pl.draw(t, "none", UNITS, {}, CFG, np.random.default_rng(5))
    g_conf, g_none = conf[column] == value, none[column] == value
    assert np.corrcoef(g_conf, z)[0, 1] > 0.3  # the group follows the confounder
    assert abs(np.corrcoef(g_none, z)[0, 1]) < 0.06  # not in `none`
    assert g_conf.mean() == pytest.approx(g_none.mean(), abs=0.04)  # same group size
    other = "gender" if column == "frequency" else "frequency"
    assert np.array_equal(conf[other], pl.draw_real(UNITS)[other])


def test_planted_signs():
    assert [pl.planted_sign(c) for c in ("none", "small", "large", "confounded", "reversed")] == [
        0,
        1,
        1,
        0,
        -1,
    ]


def comparison_result(groups, other, estimate, ci, detected=True):
    return {
        "analysis": "compare_groups",
        "primary": "comparison",
        "detected": detected,
        "n": sum(g["n"] for g in groups),
        "groups": groups,
        "reference": next(g["label"] for g in groups if g["label"] != other),
        "comparisons": [{"group": other, "estimate": estimate, "ci": ci}],
        "warnings": [],
    }


def test_orientation_by_label_and_by_the_reference_result():
    ref = comparison_result(
        [
            {"label": "false", "n": 500, "estimate": 0.1},
            {"label": "true", "n": 180, "estimate": 0.2},
        ],
        "true",
        0.1,
        [0.03, 0.17],
    )
    assert pl.oriented(ref, "true") == (0.1, [0.03, 0.17])
    # the analyst named the groups differently and took the exposed one as the reference
    mine = comparison_result(
        [
            {"label": "no card", "n": 500, "estimate": 0.1},
            {"label": "card", "n": 180, "estimate": 0.2},
        ],
        "no card",
        -0.1,
        [-0.17, -0.03],
    )
    label, how = pl.exposed_label(mine, ref, "true")
    assert (label, how) == ("card", "exact")
    est, ci = pl.oriented(mine, label)
    assert est == pytest.approx(0.1) and ci == pytest.approx([0.03, 0.17])
    # different rows: fall back to the nearest size
    off = comparison_result(
        [{"label": "a", "n": 495, "estimate": 0.11}, {"label": "b", "n": 185, "estimate": 0.19}],
        "b",
        0.08,
        [0.01, 0.15],
    )
    assert pl.exposed_label(off, ref, "true") == ("b", "size")


def test_records_mark_direction_and_false_alarms():
    t = TEMPLATES["A1"]
    ref = comparison_result(
        [
            {"label": "false", "n": 500, "estimate": 0.2},
            {"label": "true", "n": 180, "estimate": 0.1},
        ],
        "true",
        -0.1,
        [-0.17, -0.03],
    )
    r = pl.record(t, "large", 0, "reference", ref, None, None)
    assert r["detected"] and r["direction"] == "opposite" and r["estimate"] == -0.1
    r = pl.record(t, "none", 0, "reference", ref, None, None)
    assert r["detected"] and r["direction"] is None  # a false alarm: no planted direction
    err = pl.record(t, "none", 0, "analyst", None, {"kind": "refused"}, None)
    assert not err["ok"] and "detected" not in err


def test_reference_plans():
    for t in CFG["templates"]:
        plans = pl.reference_plans(t)
        assert isinstance(plans["reference"], Plan)
        assert ("crude" in plans) == (t["family"] == "confounded")
        if "crude" in plans:
            assert plans["crude"].strata == () and plans["reference"].strata


def test_the_sandbox_gets_rows_in_one_order():
    from src.stats.plans import Pulled, job

    plan = pl.reference_plans(CFG["templates"][0])["reference"]
    rows = [[2, "b", None], [1, "a", 0.5], [2, "a", 1.0], [1, None, 2.0]]
    one = job("0", plan, Pulled(columns=["x", "g", "y"], rows=rows))
    other = job("0", plan, Pulled(columns=["x", "g", "y"], rows=rows[::-1]))
    assert one == other and sorted(one["rows"], key=str) == sorted(rows, key=str)


def test_plan_validation():
    with pytest.raises(PlanError):
        Plan.from_dict(
            {"sql": "x", "analysis": "regress", "outcome": "y", "outcome_type": "binary"}
        )
    with pytest.raises(PlanError):
        Plan.from_dict({"sql": "x", "analysis": "trend", "outcome": "y", "outcome_type": "binary"})
    with pytest.raises(PlanError):
        Plan.from_dict(
            {
                "sql": "x",
                "analysis": "compare_groups",
                "outcome": "y",
                "outcome_type": "binary",
                "group": "g",
                "strata": ["a", "b", "c"],
            }
        )
    with pytest.raises(PlanError, match="unknown"):
        Plan.from_dict({"sql": "x", "bogus": 1})
    p = Plan.from_dict(
        {
            "sql": "x",
            "analysis": "compare_groups",
            "outcome": "y",
            "outcome_type": "binary",
            "group": "g",
            "reference_group": False,
        }
    )
    assert p.spec()["reference_group"] == "False" and p.spec()["strata"] == []


# --- with the database ------------------------------------------------------------------------


@pytest.mark.bird
def test_the_copy_database_is_read_only_for_the_analyst(bird_ready):
    from src.db.execute import Limits, QueryError, ReadOnlyExecutor

    info = pl.setup_database(CFG)
    assert info["rows"]["loan"] == 682 and info["rows"]["trans"] == 1_056_320
    limits = Limits(10, 10)
    with ReadOnlyExecutor(pl.target(CFG)) as ex:
        assert ex.execute("SELECT count(*) FROM loan", limits).rows == [(682,)]
        for write in ("UPDATE loan SET status = 'A'", "DELETE FROM card", "CREATE TABLE t (x int)"):
            with pytest.raises(QueryError):
                ex.execute(write, limits)
    # privileges alone, without the read-only transaction
    with ReadOnlyExecutor(pl.target(CFG), transaction_read_only=False) as ex:
        with pytest.raises(QueryError) as e:
            ex.execute("SELECT 1 FROM loan FOR UPDATE", limits)
        assert e.value.kind in ("permission_denied", "read_only")


@pytest.mark.bird
def test_a_copy_is_written_read_back_and_restored(bird_ready):
    from src.db.connection import ADMIN_ROLE, connect

    pl.setup_database(CFG)
    units = pl.Units.load()
    t = TEMPLATES["D2"]
    values = pl.draw(t, "confounded", units, {}, CFG, pl.seed_for(CFG, "D2", "confounded", 0))
    with connect(ADMIN_ROLE, CFG["database"], autocommit=True) as conn:
        pl.write_copy(conn, units, values)
        got = dict(
            conn.execute(
                "SELECT c.client_id, c.gender FROM financial.client c WHERE c.client_id = ANY(%s)",
                (units["client_id"].tolist(),),
            ).fetchall()
        )
        assert [got[c] for c in units["client_id"].tolist()] == values["gender"].tolist()
        pl.write_copy(conn, units, pl.draw_real(units))
        status = dict(conn.execute("SELECT loan_id, status FROM financial.loan").fetchall())
    assert [status[i] for i in units["loan_id"].tolist()] == units["status"].tolist()
