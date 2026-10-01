"""Planted effects: copies of the Czech bank database with effects of known size, to check that
the guardrail finds effects that are there and reports none where there is none.

**The copies.** One local database (`planted`, schema `financial`) holds a copy of the eight
tables. Only the columns an effect is planted in change from copy to copy: a loan's status (it
goes bad: status B if finished, D if running), its amount (and monthly payment, so that amount =
duration x payment still holds), an account's statement frequency and a loan owner's gender. Every
copy rewrites all of these for every loan, from the real values or from a seeded draw, so nothing
of one copy carries over to the next. The agent reads the copy through the same guards as the
benchmark: the query guard and the read-only executor, as a role that can read this schema and
nothing else (created here, like the benchmark's schema roles).

**The effects** (configs/guardrail.yaml `planted`), per family, on the 682 loans:
- `rates`: a loan goes bad with probability p0 + d * g, g the exposure (a card on the account; a
  branch in the Prague region), p0 the real share of bad loans;
- `means`: the real amounts shuffled within each loan duration (so an amount still fits its
  duration), plus d for the exposed group (female owners; a card);
- `trend`: a loan goes bad with probability p0 + b * (x - mean x), x the year the loan was granted
  or the owner's age then;
- `confounded`: the group (weekly statements; a female owner) is drawn with a probability that
  rises with a confounder (a loan of 48 months or more; the loan amount's rank), and so does the
  chance of going bad; `none` draws the group independently of it, `confounded` plants no effect
  within the confounder's levels, and `reversed` plants one opposite to the confounding, so the
  crude comparison points the wrong way (Simpson's paradox).
The first three families have conditions `none`, `small` and `large`: d or b sized so that the
reference plan's test has the configured power (normal approximation, from the real group sizes).

Every draw comes from a seed derived from the configured seed, the question and the copy, so the
copies can be rebuilt exactly.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from psycopg import sql

from src.data import bird
from src.db.connection import ADMIN_ROLE, AGENT_ROLE, BIRD_DB, DB_NAME, connect
from src.db.execute import Target
from src.stats.plans import Plan

ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG = ROOT / "configs/guardrail.yaml"
SCHEMA = "financial"
TABLES = ("district", "account", "client", "disp", "card", "loan", "order", "trans")
CONDITIONS = {
    "rates": ("none", "small", "large"),
    "means": ("none", "small", "large"),
    "trend": ("none", "small", "large"),
    "confounded": ("none", "confounded", "reversed"),
}
Z = 1.959963984540054
WEEKLY, MONTHLY = "POPLATEK TYDNE", "POPLATEK MESICNE"


def config(path: Path = CONFIG) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))["planted"]


def target(cfg: dict) -> Target:
    zone = bird.config()["bird_minidev"]["time_zone"]
    return Target(cfg["database"], SCHEMA, zone, cfg["role"])


# --- the copy database --------------------------------------------------------------------------


def _source_fingerprint() -> str:
    with connect(ADMIN_ROLE, BIRD_DB) as conn:
        counts = [
            conn.execute(
                sql.SQL("SELECT count(*) FROM {}").format(sql.Identifier(SCHEMA, t))
            ).fetchone()[0]
            for t in TABLES
        ]
    return hashlib.sha256(repr(list(zip(TABLES, counts, strict=True))).encode()).hexdigest()


def setup_database(cfg: dict, force: bool = False) -> dict[str, Any]:
    """Create the copy database, its tables and the read-only role, unless they already match the
    source. Returns the table sizes and whether the tables were copied now."""
    dbname, role = cfg["database"], cfg["role"]
    fingerprint = _source_fingerprint()
    with connect(ADMIN_ROLE, DB_NAME, autocommit=True) as conn:
        if not conn.execute("SELECT 1 FROM pg_database WHERE datname = %s", (dbname,)).fetchone():
            # from template1, which carries the hardening (closed functions and views)
            conn.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(dbname)))
    with connect(ADMIN_ROLE, dbname, autocommit=True) as conn:
        current = conn.execute(
            "SELECT obj_description(to_regnamespace(%s)::oid, 'pg_namespace')", (SCHEMA,)
        ).fetchone()[0]
    copied = force or current != fingerprint
    if copied:
        _copy_tables(dbname, fingerprint)
    _grant(dbname, role)
    with connect(ADMIN_ROLE, dbname) as conn:
        sizes = {
            t: conn.execute(
                sql.SQL("SELECT count(*) FROM {}").format(sql.Identifier(SCHEMA, t))
            ).fetchone()[0]
            for t in TABLES
        }
    return {"database": dbname, "role": role, "copied_now": copied, "rows": sizes}


def _copy_tables(dbname: str, fingerprint: str) -> None:
    with (
        connect(ADMIN_ROLE, BIRD_DB) as src,
        connect(ADMIN_ROLE, dbname, autocommit=True) as dst,
    ):
        dst.execute(sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(sql.Identifier(SCHEMA)))
        dst.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(SCHEMA)))
        for t in TABLES:
            cols = src.execute(
                "SELECT a.attname, format_type(a.atttypid, a.atttypmod) FROM pg_attribute a "
                "WHERE a.attrelid = %s::regclass AND a.attnum > 0 AND NOT a.attisdropped "
                "ORDER BY a.attnum",
                (f'{SCHEMA}."{t}"',),
            ).fetchall()
            pkey = src.execute(
                "SELECT pg_get_constraintdef(oid) FROM pg_constraint "
                "WHERE conrelid = %s::regclass AND contype = 'p'",
                (f'{SCHEMA}."{t}"',),
            ).fetchone()
            body = [sql.SQL("{} {}").format(sql.Identifier(n), sql.SQL(ty)) for n, ty in cols]
            if pkey:
                body.append(sql.SQL(pkey[0]))
            dst.execute(
                sql.SQL("CREATE TABLE {} ({})").format(
                    sql.Identifier(SCHEMA, t), sql.SQL(", ").join(body)
                )
            )
            copy_out = sql.SQL("COPY {} TO STDOUT (FORMAT BINARY)").format(
                sql.Identifier(SCHEMA, t)
            )
            copy_in = sql.SQL("COPY {} FROM STDIN (FORMAT BINARY)").format(
                sql.Identifier(SCHEMA, t)
            )
            with (
                src.cursor().copy(copy_out) as out,
                dst.cursor().copy(copy_in) as into,
            ):
                for block in out:
                    into.write(block)
            dst.execute(sql.SQL("ANALYZE {}").format(sql.Identifier(SCHEMA, t)))
        dst.execute(
            sql.SQL("COMMENT ON SCHEMA {} IS {}").format(
                sql.Identifier(SCHEMA), sql.Literal(fingerprint)
            )
        )


def _grant(dbname: str, role: str) -> None:
    r, s, d = sql.Identifier(role), sql.Identifier(SCHEMA), sql.Identifier(dbname)
    with connect(ADMIN_ROLE, dbname, autocommit=True) as conn:
        if not conn.execute("SELECT 1 FROM pg_roles WHERE rolname = %s", (role,)).fetchone():
            conn.execute(
                sql.SQL(
                    "CREATE ROLE {} NOLOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOINHERIT "
                    "NOREPLICATION NOBYPASSRLS"
                ).format(r)
            )
        conn.execute(
            sql.SQL("GRANT {} TO {} WITH INHERIT FALSE, SET TRUE").format(
                r, sql.Identifier(AGENT_ROLE)
            )
        )
        conn.execute(sql.SQL("REVOKE ALL ON DATABASE {} FROM PUBLIC").format(d))
        conn.execute(
            sql.SQL("GRANT CONNECT ON DATABASE {} TO {}").format(d, sql.Identifier(AGENT_ROLE))
        )
        conn.execute(sql.SQL("REVOKE ALL ON SCHEMA public FROM PUBLIC"))
        conn.execute(sql.SQL("GRANT USAGE ON SCHEMA {} TO {}").format(s, r))
        conn.execute(sql.SQL("GRANT SELECT ON ALL TABLES IN SCHEMA {} TO {}").format(s, r))


# --- the loans and their real values ------------------------------------------------------------

UNITS_SQL = """
SELECT l.loan_id, l.account_id, c.client_id, l.status IN ('A', 'B') AS finished,
       EXISTS (SELECT 1 FROM financial.disp AS d2 JOIN financial.card AS k ON k.disp_id = d2.disp_id
               WHERE d2.account_id = l.account_id) AS has_card,
       di.a3 = 'Prague' AS prague, c.gender = 'F' AS female,
       EXTRACT(YEAR FROM l.date)::int AS year,
       EXTRACT(YEAR FROM AGE(l.date, c.birth_date))::int AS age,
       l.duration, l.amount, l.payments, l.status, a.frequency, c.gender
FROM financial.loan AS l
JOIN financial.account AS a ON a.account_id = l.account_id
JOIN financial.district AS di ON di.district_id = a.district_id
JOIN financial.disp AS d ON d.account_id = l.account_id AND d.type = 'OWNER'
JOIN financial.client AS c ON c.client_id = d.client_id
ORDER BY l.loan_id
"""


@dataclass
class Units:
    """One row per loan, from the source database (the real values)."""

    columns: dict[str, np.ndarray]

    @classmethod
    def load(cls) -> Units:
        with connect(ADMIN_ROLE, BIRD_DB) as conn:
            cur = conn.execute(UNITS_SQL)
            names = [d.name for d in cur.description]
            rows = cur.fetchall()
        cols = {n: np.array([r[i] for r in rows], dtype=object) for i, n in enumerate(names)}
        for n in ("loan_id", "account_id", "client_id", "year", "age", "duration", "amount"):
            cols[n] = cols[n].astype(np.int64)
        for n in ("finished", "has_card", "prague", "female"):
            cols[n] = cols[n].astype(bool)
        cols["payments"] = cols["payments"].astype(float)
        cols["long_loan"] = cols["duration"] >= 48
        order = np.argsort(np.argsort(cols["amount"], kind="stable"), kind="stable")
        cols["amount_rank"] = order / (len(order) - 1)
        return cls(cols)

    def __len__(self) -> int:
        return len(self.columns["loan_id"])

    def __getitem__(self, name: str) -> np.ndarray:
        return self.columns[name]

    def exposure(self, name: str) -> np.ndarray:
        if name == "weekly":
            return self["frequency"] == WEEKLY
        return self[name]

    def base_rate(self) -> float:
        return float(np.isin(self["status"], ["B", "D"]).mean())


# --- effect sizes -------------------------------------------------------------------------------


def _phi(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def power(effect: float, se: float) -> float:
    """Two-sided power at 5% for an effect with this standard error (normal approximation)."""
    if se <= 0:
        return 1.0
    return _phi(effect / se - Z) + _phi(-effect / se - Z)


def _solve(target_power: float, se_of) -> float:
    lo, hi = 0.0, 1.0
    while power(hi, se_of(hi)) < target_power:
        hi *= 2
    for _ in range(200):
        mid = (lo + hi) / 2
        if power(mid, se_of(mid)) < target_power:
            lo = mid
        else:
            hi = mid
    return hi


def effect_sizes(template: dict, units: Units, powers: dict[str, float]) -> dict[str, dict]:
    """The planted effect and its power per condition, for the first three families."""
    family = template["family"]
    p0 = units.base_rate()
    out: dict[str, dict] = {"none": {"effect": 0.0, "power": 0.05}}
    if family == "rates":
        g = units.exposure(template["exposure"])
        n1, n0 = int(g.sum()), int((~g).sum())

        def se_of(d: float) -> float:
            p1 = min(p0 + d, 1.0)
            return math.sqrt(p1 * (1 - p1) / n1 + p0 * (1 - p0) / n0)

    elif family == "means":
        g = units.exposure(template["exposure"])
        n1, n0 = int(g.sum()), int((~g).sum())
        sd = float(np.std(units["amount"], ddof=1))

        def se_of(d: float) -> float:
            return sd * math.sqrt(1 / n1 + 1 / n0)

    elif family == "trend":
        x = units[template["exposure"]].astype(float)
        sxx = float(((x - x.mean()) ** 2).sum())

        def se_of(d: float) -> float:
            return math.sqrt(p0 * (1 - p0) / sxx)

    else:
        return {}
    for cond, target_power in powers.items():
        size = _solve(target_power, se_of)
        out[cond] = {"effect": size, "power": power(size, se_of(size))}
    return out


# --- drawing a copy -----------------------------------------------------------------------------


def seed_for(cfg: dict, template_id: str, condition: str, copy: int) -> np.random.Generator:
    key = hashlib.sha256(f"{template_id}/{condition}/{copy}".encode()).digest()
    words = [int.from_bytes(key[i : i + 4], "little") for i in range(0, 16, 4)]
    return np.random.default_rng(np.random.SeedSequence([cfg["seed"], *words]))


def _status(finished: np.ndarray, bad: np.ndarray) -> np.ndarray:
    return np.where(finished, np.where(bad, "B", "A"), np.where(bad, "D", "C"))


def draw(
    template: dict, condition: str, units: Units, sizes: dict, cfg: dict, rng: np.random.Generator
) -> dict[str, np.ndarray]:
    """The planted columns of one copy: status, amount, payments, frequency and gender for every
    loan (the real values where nothing is planted)."""
    n = len(units)
    out = {
        "status": units["status"].astype(str),
        "amount": units["amount"].copy(),
        "payments": units["payments"].copy(),
        "frequency": units["frequency"].astype(str),
        "gender": units["gender"].astype(str),
    }
    family = template["family"]
    p0 = units.base_rate()
    if family == "rates":
        g = units.exposure(template["exposure"])
        p = p0 + sizes[condition]["effect"] * g
        out["status"] = _status(units["finished"], rng.random(n) < p)
    elif family == "means":
        g = units.exposure(template["exposure"])
        amount = units["amount"].copy()
        for dur in np.unique(units["duration"]):
            idx = np.flatnonzero(units["duration"] == dur)
            amount[idx] = amount[rng.permutation(idx)]
        amount = amount + np.rint(sizes[condition]["effect"] * g).astype(np.int64)
        out["amount"] = amount
        out["payments"] = amount / units["duration"]
    elif family == "trend":
        x = units[template["exposure"]].astype(float)
        p = np.clip(p0 + sizes[condition]["effect"] * (x - x.mean()), 0.0, 1.0)
        out["status"] = _status(units["finished"], rng.random(n) < p)
    elif family == "confounded":
        c = template["plant"]
        z = units[template["confounder"]].astype(float)
        q = c["q_low"] + (c["q_high"] - c["q_low"]) * z
        if condition == "none":
            q = np.full(n, q.mean())
        g = rng.random(n) < q
        delta = c["delta_reversed"] if condition == "reversed" else 0.0
        p = np.clip(c["p_low"] + (c["p_high"] - c["p_low"]) * z + delta * g, 0.0, 1.0)
        out["status"] = _status(units["finished"], rng.random(n) < p)
        if template["exposure"] == "weekly":
            out["frequency"] = np.where(g, WEEKLY, MONTHLY)
        elif template["exposure"] == "female":
            out["gender"] = np.where(g, "F", "M")
        else:
            raise ValueError(f"no group column for exposure {template['exposure']!r}")
    else:
        raise ValueError(f"unknown family {family!r}")
    return out


def write_copy(conn, units: Units, values: dict[str, np.ndarray]) -> None:
    """Write one copy's planted columns (every loan, its account and its owner)."""
    conn.execute(
        "UPDATE financial.loan AS l SET status = v.status, amount = v.amount, "
        "payments = v.payments FROM unnest(%s::bigint[], %s::text[], %s::bigint[], "
        "%s::real[]) AS v(loan_id, status, amount, payments) WHERE l.loan_id = v.loan_id",
        (
            units["loan_id"].tolist(),
            values["status"].tolist(),
            values["amount"].astype(np.int64).tolist(),
            values["payments"].astype(float).tolist(),
        ),
    )
    conn.execute(
        "UPDATE financial.account AS a SET frequency = v.frequency FROM unnest(%s::bigint[], "
        "%s::text[]) AS v(account_id, frequency) WHERE a.account_id = v.account_id",
        (units["account_id"].tolist(), values["frequency"].tolist()),
    )
    conn.execute(
        "UPDATE financial.client AS c SET gender = v.gender FROM unnest(%s::bigint[], "
        "%s::text[]) AS v(client_id, gender) WHERE c.client_id = v.client_id",
        (units["client_id"].tolist(), values["gender"].tolist()),
    )


def reference_plans(template: dict) -> dict[str, Plan]:
    """The reference plan, and for the confounded family the same plan without its strata."""
    ref = {k: v for k, v in template["reference"].items() if k != "exposed_group"}
    plans = {"reference": Plan.from_dict(ref)}
    if template["family"] == "confounded":
        plans["crude"] = Plan.from_dict({**ref, "strata": []})
    return plans


# --- level 1: every copy, every plan, the sandbox's verdict --------------------------------------


def primary(result: dict) -> dict | None:
    """The estimate the verdict rests on: the stratified fit, the single comparison, or the
    slope (None for a test of several groups)."""
    kind = result.get("primary")
    if kind == "stratified":
        adjusted = result["stratified"]["adjusted"]
        return adjusted if "estimate" in adjusted else None
    if kind == "comparison":
        return result["comparisons"][0]
    if kind == "slope":
        return result["slope"]
    return None


def exposed_label(result: dict, reference: dict, reference_exposed: str) -> tuple[str | None, str]:
    """The analyst's label for the exposed group, found through the reference plan's result on
    the same copy: `exact` (the group with the same size and estimate), else `size` (the group
    nearest in size)."""
    groups = result.get("groups", [])
    ref = [g for g in reference.get("groups", []) if g["label"] == reference_exposed]
    if len(groups) != 2 or not ref:
        return None, "none"
    target_n, target_est = ref[0]["n"], ref[0]["estimate"]
    same = [g for g in groups if g["n"] == target_n and abs(g["estimate"] - target_est) < 1e-9]
    if len(same) == 1:
        return same[0]["label"], "exact"
    return min(groups, key=lambda g: abs(g["n"] - target_n))["label"], "size"


def oriented(result: dict, exposed: str | None) -> tuple[float | None, list | None]:
    """The primary estimate and interval as exposed minus unexposed (a trend's slope as is)."""
    main = primary(result)
    if main is None:
        return None, None
    if result["analysis"] == "trend":
        return main["estimate"], main["ci"]
    if exposed is None or len(result["groups"]) != 2:
        return None, None
    sign = 1.0 if result["comparisons"][0]["group"] == exposed else -1.0
    lo, hi = main["ci"]
    return sign * main["estimate"], sorted([sign * lo, sign * hi])


def planted_sign(condition: str) -> int:
    """The planted effect's direction: exposed higher (+1), lower (-1) or none (0)."""
    return {"none": 0, "confounded": 0, "reversed": -1}.get(condition, 1)


def record(
    template: dict,
    condition: str,
    copy: int,
    plan: str,
    outcome: dict | None,
    error: dict | None,
    reference: dict | None,
) -> dict:
    rec: dict[str, Any] = {
        "template": template["id"],
        "family": template["family"],
        "condition": condition,
        "copy": copy,
        "plan": plan,
        "ok": outcome is not None,
        "error": error,
    }
    if outcome is None:
        return rec
    ref_exposed = template["reference"].get("exposed_group")
    if outcome["analysis"] == "trend":
        exposed, how = None, "trend"
    elif plan in ("reference", "crude"):
        exposed, how = ref_exposed, "label"
    elif reference is not None and ref_exposed is not None:
        exposed, how = exposed_label(outcome, reference, ref_exposed)
    else:
        exposed, how = None, "none"
    estimate, ci = oriented(outcome, exposed)
    sign = planted_sign(condition)
    direction = None
    if outcome["detected"] and estimate is not None and sign != 0:
        direction = "planted" if estimate * sign > 0 else "opposite"
    rec.update(
        detected=outcome["detected"],
        primary=outcome["primary"],
        n=outcome["n"],
        estimate=estimate,
        ci=ci,
        orientation=how,
        exposed=exposed,
        direction=direction,
        change=outcome.get("stratified", {}).get("change"),
        warnings=sorted({w["kind"] for w in outcome.get("warnings", [])}),
    )
    return rec


def run_level1(
    cfg: dict,
    units: Units,
    analyst_plans: dict[str, Plan],
    copies: int,
    policy=None,
    only: set[str] | None = None,
    chunk: int = 200,
    progress=print,
    keep: int = 0,
    clock_dir: Path | None = None,
    keep_plan: str = "analyst",
) -> tuple[list[dict], dict[str, dict], dict[tuple, tuple]]:
    """Every configured question on `copies` copies per condition, with the reference plan(s)
    and the analyst's plan where there is one. Returns the records, each question's planted
    effect sizes, and for the first `keep` copies the analyst plan's (plan, rows, result) by
    (question, condition, copy), for the answers written from them. A query that reads the clock
    goes through the clock store in `clock_dir`, so a replay sees the same rows."""
    from src.agent.clock import ClockStore
    from src.db.execute import Limits, ReadOnlyExecutor
    from src.db.guard import QueryGuard
    from src.stats.guardrail import pull_stored
    from src.stats.plans import job
    from src.stats.sandbox import run_jobs

    guard = QueryGuard.for_benchmark_db(SCHEMA)
    limits = Limits(cfg["limits"]["max_rows"], cfg["limits"]["timeout_s"])
    clock = ClockStore(clock_dir or ROOT / "data/cache/llm_planted")
    records: list[dict] = []
    sizes_by_template: dict[str, dict] = {}
    kept: dict[tuple, tuple] = {}
    with (
        ReadOnlyExecutor(target(cfg)) as executor,
        connect(ADMIN_ROLE, cfg["database"], autocommit=True) as admin,
    ):
        for t in cfg["templates"]:
            if only and t["id"] not in only:
                continue
            sizes = effect_sizes(t, units, cfg["power"])
            sizes_by_template[t["id"]] = sizes
            plans = reference_plans(t)
            if t["id"] in analyst_plans:
                plans["analyst"] = analyst_plans[t["id"]]
            for condition in CONDITIONS[t["family"]]:
                pulled: dict[str, list] = {name: [] for name in plans}
                for copy in range(copies):
                    rng = seed_for(cfg, t["id"], condition, copy)
                    write_copy(admin, units, draw(t, condition, units, sizes, cfg, rng))
                    for name, plan in plans.items():
                        at = ("planted", f"{t['id']}/{condition}/{copy}/{name}")
                        pulled[name].append(
                            pull_stored(plan, guard, executor, limits, clock, cfg["database"], at)
                        )
                results: dict[str, dict[int, dict]] = {}
                for name, plan in plans.items():
                    jobs = [job(str(i), plan, p) for i, p in enumerate(pulled[name]) if p.ok]
                    out: dict[int, dict] = {}
                    for start in range(0, len(jobs), chunk):
                        for r in run_jobs(jobs[start : start + chunk], policy):
                            out[int(r["id"])] = r
                    results[name] = out
                for copy in range(copies):
                    ref = results["reference"].get(copy)
                    ref_result = ref["result"] if ref and ref["ok"] else None
                    for name in plans:
                        p = pulled[name][copy]
                        r = results[name].get(copy)
                        if not p.ok:
                            records.append(record(t, condition, copy, name, None, p.error, None))
                        elif not r["ok"]:
                            records.append(record(t, condition, copy, name, None, r["error"], None))
                        else:
                            records.append(
                                record(t, condition, copy, name, r["result"], None, ref_result)
                            )
                            if name == keep_plan and copy < keep:
                                kept[(t["id"], condition, copy)] = (plans[name], p, r["result"])
                progress(f"{t['id']} {condition}: {copies} copies, plans {sorted(plans)}")
        # leave the copy database holding the real values
        write_copy(admin, units, draw_real(units))
    return records, sizes_by_template, kept


def draw_real(units: Units) -> dict[str, np.ndarray]:
    return {
        "status": units["status"].astype(str),
        "amount": units["amount"].copy(),
        "payments": units["payments"].copy(),
        "frequency": units["frequency"].astype(str),
        "gender": units["gender"].astype(str),
    }


def claimed_direction(finding: dict, analysis: dict, exposed: str | None) -> int | None:
    """+1 if the answer claims the exposed group (or x) goes with a higher outcome, -1 lower,
    0 if it claims no effect, None if its claim cannot be placed."""
    if finding.get("claims_effect") != "yes":
        return 0
    higher = finding.get("higher")
    if analysis["analysis"] == "trend":
        return {"increasing": 1, "decreasing": -1}.get(higher)
    labels = [g["label"] for g in analysis["groups"]]
    if exposed is None or higher not in labels or len(labels) != 2:
        return None
    return 1 if higher == exposed else -1
