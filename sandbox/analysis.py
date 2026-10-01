"""The statistical analyses of the guardrail, run inside the sandbox.

The analyst never writes code for these. It describes the analysis it wants: which rows are the
units, which column is the outcome, which column holds the groups or the x of a trend, and which
variables to stratify by. One of the functions here runs it. Every figure a guarded answer
states comes from this module, so the intervals, the tests and the warnings are the same
whatever the question.

Methods (two-sided, 95%):
- a rate in a group: the Wilson score interval, which stays inside 0-1 and holds for a count of
  zero;
- a difference in rates between two groups: Newcombe's hybrid score interval (method 10 of
  Newcombe, 1998), built from the two Wilson intervals, with Fisher's exact test;
- a difference in means: Welch's t interval and test (the groups' variances may differ);
- more than two groups: a test of any difference (Pearson's chi-square for rates, with a
  permutation p-value from the hypergeometric distribution when an expected count is below 5;
  Welch's one-way ANOVA for means), and each group against the reference as above;
- a trend: the least-squares slope of the outcome on x (for a yes/no outcome, the change in the
  rate per unit of x), with heteroskedasticity-robust (HC3) standard errors;
- a stratified check: the same comparison or slope fitted with an indicator for every stratum
  (fixed effects, HC3), beside the crude one fitted the same way, with each stratum's own
  estimates, and how the estimate changed: `reversed` (both intervals exclude zero, on opposite
  sides), `vanished` (only the crude one excludes zero), `appeared` or `stable`.

An effect is reported (`detected`) when the primary interval excludes zero: the stratified one
if strata were given, the crude one otherwise. With more than two groups, the primary result is
the test of any difference, at 0.05.

Nothing here reads the clock, the network or the file system. The only randomness (the
permutation test) uses a fixed seed, so the same input always gives the same output.
"""

from __future__ import annotations

import datetime as dt
import math
import re
from collections.abc import Sequence
from typing import Any

import numpy as np
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.oneway import anova_oneway

LEVEL = 0.95
ALPHA = 1 - LEVEL
Z = float(stats.norm.ppf(1 - ALPHA / 2))
PERMUTATIONS = 20_000
SEED = 20260930
MAX_GROUPS = 20
MAX_STRATA_VARIABLES = 2
MAX_STRATUM_LEVELS = 10  # a numeric stratum with more distinct values is cut into fifths
SMALL_GROUP = 30
FEW_EVENTS = 5
SKEWED = 2.0
ANALYSES = ("compare_groups", "trend")
OUTCOME_TYPES = ("binary", "numeric")


class AnalysisError(Exception):
    """An analysis that cannot be run on this input; `kind` says why."""

    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind


# --- intervals and tests ------------------------------------------------------------------


def wilson(k: int, n: int, z: float = Z) -> tuple[float, float]:
    """Wilson score interval for k successes in n trials."""
    if n <= 0:
        raise ValueError("n must be positive")
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    lo = 0.0 if k == 0 else max(0.0, center - half)  # exact at the ends, not off by rounding
    hi = 1.0 if k == n else min(1.0, center + half)
    return lo, hi


def newcombe(k1: int, n1: int, k0: int, n0: int) -> tuple[float, float, float]:
    """The difference p1 - p0 and Newcombe's hybrid score interval (his method 10)."""
    p1, p0 = k1 / n1, k0 / n0
    l1, u1 = wilson(k1, n1)
    l0, u0 = wilson(k0, n0)
    d = p1 - p0
    lo = d - math.sqrt((p1 - l1) ** 2 + (u0 - p0) ** 2)
    hi = d + math.sqrt((u1 - p1) ** 2 + (p0 - l0) ** 2)
    return d, lo, hi


def fisher_p(k1: int, n1: int, k0: int, n0: int) -> float:
    return float(stats.fisher_exact([[k1, n1 - k1], [k0, n0 - k0]]).pvalue)


def welch(x1: np.ndarray, x0: np.ndarray) -> tuple[float, float, float, float]:
    """The difference in means x1 - x0, Welch's interval and p-value."""
    d = float(x1.mean() - x0.mean())
    if x1.var(ddof=1) == 0 and x0.var(ddof=1) == 0:
        return d, d, d, (1.0 if d == 0 else 0.0)
    res = stats.ttest_ind(x1, x0, equal_var=False)
    ci = res.confidence_interval(confidence_level=LEVEL)
    return d, float(ci.low), float(ci.high), float(res.pvalue)


def chi_square_p(events: np.ndarray, sizes: np.ndarray) -> tuple[float, str]:
    """Test that the rates of several groups are equal: Pearson's chi-square, or its permutation
    distribution (draws from the hypergeometric, the margins fixed) when an expected count is
    below 5."""
    table = np.stack([events, sizes - events], axis=1).astype(float)
    total = table.sum()
    expected = np.outer(table.sum(axis=1), table.sum(axis=0)) / total
    if (table.sum(axis=0) == 0).any():  # every unit has the same outcome: no difference exists
        return 1.0, "none (no variation in the outcome)"

    def statistic(ev: np.ndarray) -> np.ndarray:
        obs = np.stack([ev, sizes - ev], axis=-1)
        return (((obs - expected) ** 2) / expected).sum(axis=(-1, -2))

    observed = float(statistic(events.astype(float)))
    if (expected >= FEW_EVENTS).all():
        dof = len(sizes) - 1
        return float(stats.chi2.sf(observed, dof)), "Pearson's chi-square"
    rng = np.random.default_rng(SEED)
    draws = rng.multivariate_hypergeometric(
        sizes.astype(np.int64), int(events.sum()), size=PERMUTATIONS
    )
    sims = statistic(draws.astype(float))
    p = (1 + int((sims >= observed - 1e-9).sum())) / (PERMUTATIONS + 1)
    return float(p), f"permutation chi-square ({PERMUTATIONS} draws)"


def ols(y: np.ndarray, X: np.ndarray, names: list[str]):
    fit = sm.OLS(y, X).fit(cov_type="HC3")
    return fit, {n: i for i, n in enumerate(names)}


# --- input --------------------------------------------------------------------------------


def _label(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return str(int(v)) if v.is_integer() else repr(v)
    if isinstance(v, str):
        return v
    raise AnalysisError("invalid_value", f"a group or stratum value is not a scalar: {v!r}")


def _order_key(label: str) -> tuple:
    try:
        return (0, float(label), label)
    except ValueError:
        return (1, 0.0, label)


def _binary(v: Any, column: str) -> float:
    if isinstance(v, bool):
        return 1.0 if v else 0.0
    if isinstance(v, int | float) and v in (0, 1):
        return float(v)
    raise AnalysisError("invalid_outcome", f"column {column!r} is not yes/no (0/1): {v!r}")


_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}")


def _date(v: Any) -> dt.date | None:
    """An ISO date (a date or a timestamp's date part), else None."""
    if isinstance(v, str) and _DATE.match(v):
        try:
            return dt.date.fromisoformat(v[:10])
        except ValueError:
            return None
    return None


def _number(v: Any, column: str) -> float:
    """A number; a date counts as time, in years (so a trend's slope is per year)."""
    if isinstance(v, bool):
        return 1.0 if v else 0.0
    if isinstance(v, int | float) and math.isfinite(v):
        return float(v)
    d = _date(v)
    if d is not None:
        return d.year + (d.timetuple().tm_yday - 1) / 365.25
    raise AnalysisError("invalid_number", f"column {column!r} is not a number: {v!r}")


def _column_index(columns: Sequence[str], name: Any, role: str) -> int:
    if not isinstance(name, str) or name not in columns:
        raise AnalysisError("unknown_column", f"the {role} column {name!r} is not in the result")
    if list(columns).count(name) > 1:
        raise AnalysisError("ambiguous_column", f"the result has two columns named {name!r}")
    return list(columns).index(name)


class Data:
    """The rows used, with missing values dropped, and the strata as labels."""

    def __init__(self, spec: dict, columns: Sequence[str], rows: Sequence[Sequence[Any]]):
        analysis = spec.get("analysis")
        if analysis not in ANALYSES:
            raise AnalysisError("invalid_spec", f"analysis must be one of {ANALYSES}")
        outcome_type = spec.get("outcome_type")
        if outcome_type not in OUTCOME_TYPES:
            raise AnalysisError("invalid_spec", f"outcome_type must be one of {OUTCOME_TYPES}")
        strata = spec.get("strata") or []
        if not isinstance(strata, list) or len(strata) > MAX_STRATA_VARIABLES:
            raise AnalysisError(
                "invalid_spec", f"strata must be a list of at most {MAX_STRATA_VARIABLES} columns"
            )
        self.analysis, self.outcome_type = analysis, outcome_type
        iy = _column_index(columns, spec.get("outcome"), "outcome")
        other = spec.get("group") if analysis == "compare_groups" else spec.get("x")
        ig = _column_index(columns, other, "group" if analysis == "compare_groups" else "x")
        istrata = [_column_index(columns, s, "stratum") for s in strata]
        if len({iy, ig, *istrata}) != 2 + len(istrata):
            raise AnalysisError("invalid_spec", "the outcome, groups/x and strata must differ")
        self.strata_names = list(strata)

        y, g, s = [], [], []
        dropped = 0
        for row in rows:
            if not isinstance(row, list | tuple) or len(row) != len(columns):
                raise AnalysisError("invalid_rows", "every row must have one value per column")
            used = [row[iy], row[ig], *(row[i] for i in istrata)]
            if any(v is None for v in used):
                dropped += 1
                continue
            y.append(
                _binary(row[iy], columns[iy])
                if outcome_type == "binary"
                else _number(row[iy], columns[iy])
            )
            g.append(
                _label(row[ig]) if analysis == "compare_groups" else _number(row[ig], columns[ig])
            )
            s.append([row[i] for i in istrata])
        if not y:
            raise AnalysisError("no_data", "no row has every value the analysis needs")
        self.y = np.asarray(y, dtype=float)
        self.g = g
        self.n_dropped = dropped
        self.strata: list[list[str]] = []
        self.binned: list[str] = []
        for j, name in enumerate(self.strata_names):
            self.strata.append(self._stratum_labels([r[j] for r in s], name))

    def _stratum_labels(self, values: list[Any], name: str) -> list[str]:
        """A stratum's levels. A numeric or date stratum with more than MAX_STRATUM_LEVELS
        distinct values is cut into fifths (a date is ordered time, not a category)."""
        numeric = all(isinstance(v, int | float) and not isinstance(v, bool) for v in values)
        dates = None if numeric else [_date(v) for v in values]
        is_date = dates is not None and all(d is not None for d in dates)
        if (numeric or is_date) and len(set(values)) > MAX_STRATUM_LEVELS:
            raw = [d.toordinal() for d in dates] if is_date else values
            arr = np.asarray(raw, dtype=float)
            edges = np.unique(np.quantile(arr, [0.2, 0.4, 0.6, 0.8]))
            which = np.searchsorted(edges, arr, side="right")
            self.binned.append(name)
            bounds = [float(arr.min()), *map(float, edges), float(arr.max())]
            if is_date:
                names = [dt.date.fromordinal(int(b)).isoformat() for b in bounds]
            else:
                names = [_label(b) for b in bounds]
            return [f"{names[k]} to {names[k + 1]}" for k in which]
        return [_label(v) for v in values]


# --- the analyses ---------------------------------------------------------------------------


def _group_summary(data: Data, labels: list[str]) -> tuple[list[dict], dict[str, np.ndarray]]:
    by = {lab: data.y[[i for i, g in enumerate(data.g) if g == lab]] for lab in labels}
    out = []
    for lab in labels:
        v = by[lab]
        item: dict[str, Any] = {"label": lab, "n": int(len(v))}
        if data.outcome_type == "binary":
            k = int(v.sum())
            lo, hi = wilson(k, len(v))
            item.update(events=k, estimate=k / len(v), ci=[lo, hi])
        else:
            item.update(estimate=float(v.mean()), median=float(np.median(v)))
            if len(v) >= 2 and v.std(ddof=1) > 0:
                t = stats.t.ppf(1 - ALPHA / 2, len(v) - 1)
                half = float(t * v.std(ddof=1) / math.sqrt(len(v)))
                item["ci"] = [item["estimate"] - half, item["estimate"] + half]
            else:
                item["ci"] = None
            if len(v) >= 3 and v.std(ddof=1) > 0:
                item["skewness"] = float(stats.skew(v))
        out.append(item)
    return out, by


def _pair(data: Data, a: np.ndarray, b: np.ndarray) -> dict:
    """Group `a` against the reference `b`."""
    if data.outcome_type == "binary":
        k1, n1, k0, n0 = int(a.sum()), len(a), int(b.sum()), len(b)
        d, lo, hi = newcombe(k1, n1, k0, n0)
        return {
            "measure": "difference in rates",
            "estimate": d,
            "ci": [lo, hi],
            "p_value": fisher_p(k1, n1, k0, n0),
            "method": "Newcombe's hybrid score interval; Fisher's exact test",
        }
    if len(a) < 2 or len(b) < 2:
        raise AnalysisError("too_few_units", "a difference in means needs two units per group")
    d, lo, hi, p = welch(a, b)
    return {
        "measure": "difference in means",
        "estimate": d,
        "ci": [lo, hi],
        "p_value": p,
        "method": "Welch's t interval and test",
    }


def _dummies(labels: list[str], order: list[str]) -> np.ndarray:
    """Indicators for every level but the first."""
    return np.array([[1.0 if lab == lev else 0.0 for lev in order[1:]] for lab in labels])


def _design(
    data: Data, main: np.ndarray, main_names: list[str], with_strata: bool, keep: np.ndarray
) -> tuple[np.ndarray, list[str]]:
    cols = [np.ones(int(keep.sum())), *main[keep].T]
    names = ["const", *main_names]
    if with_strata:
        for name, labels in zip(data.strata_names, data.strata, strict=True):
            kept = [lab for lab, k in zip(labels, keep, strict=True) if k]
            order = sorted(set(kept), key=_order_key)
            if len(order) > 1:
                d = _dummies(kept, order)
                cols.extend(d.T)
                names.extend(f"{name}={lev}" for lev in order[1:])
    return np.column_stack(cols), names


def _singletons(data: Data) -> np.ndarray:
    """Units kept for the stratified fit: those in a stratum (of every variable) with at least
    two units, since a unit alone in its stratum has leverage 1 and no HC3 variance."""
    keep = np.ones(len(data.y), dtype=bool)
    for labels in data.strata:
        counts: dict[str, int] = {}
        for lab in labels:
            counts[lab] = counts.get(lab, 0) + 1
        keep &= np.array([counts[lab] >= 2 for lab in labels])
    return keep


def _fit_effect(
    data: Data, main: np.ndarray, main_names: list[str], with_strata: bool, keep: np.ndarray
) -> dict:
    X, names = _design(data, main, main_names, with_strata, keep)
    y = data.y[keep]
    if np.linalg.matrix_rank(X) < X.shape[1]:
        # a stratum variable that duplicates the groups (or x): drop to what is estimable
        X, names = _drop_collinear(X, names, keep=len(main_names) + 1)
    fit, idx = ols(y, X, names)
    out: dict[str, Any] = {"n": int(len(y))}
    if len(main_names) == 1:
        j = idx[main_names[0]]
        lo, hi = fit.conf_int(alpha=ALPHA)[j]
        out.update(
            estimate=float(fit.params[j]), ci=[float(lo), float(hi)], p_value=float(fit.pvalues[j])
        )
    else:
        R = np.zeros((len(main_names), X.shape[1]))
        for r, nm in enumerate(main_names):
            R[r, idx[nm]] = 1.0
        test = fit.wald_test(R, use_f=True, scalar=True)
        out.update(
            p_value=float(test.pvalue),
            estimates={nm: float(fit.params[idx[nm]]) for nm in main_names},
        )
    return out


def _drop_collinear(X: np.ndarray, names: list[str], keep: int) -> tuple[np.ndarray, list[str]]:
    """Drop stratum columns (after the first `keep`) that add no rank."""
    cols, kept = [X[:, i] for i in range(keep)], names[:keep]
    for i in range(keep, X.shape[1]):
        trial = np.column_stack([*cols, X[:, i]])
        if np.linalg.matrix_rank(trial) == trial.shape[1]:
            cols.append(X[:, i])
            kept.append(names[i])
    if np.linalg.matrix_rank(np.column_stack(cols)) < len(cols):
        raise AnalysisError("not_estimable", "the strata leave no variation to compare")
    return np.column_stack(cols), kept


def _change(crude: dict, adjusted: dict) -> str:
    """How the estimate changed within strata: `reversed` (both intervals exclude zero, on
    opposite sides: Simpson's paradox), `vanished` (only the crude one excludes zero),
    `appeared` (only the stratified one does) or `stable`."""
    c, a = crude["estimate"], adjusted["estimate"]
    c_sig = not (crude["ci"][0] <= 0 <= crude["ci"][1])
    a_sig = not (adjusted["ci"][0] <= 0 <= adjusted["ci"][1])
    if c_sig and a_sig and c * a < 0:
        return "reversed"
    if c_sig and not a_sig:
        return "vanished"
    if a_sig and not c_sig:
        return "appeared"
    return "stable"


def _by_stratum(
    data: Data, labels_of_unit: list[str], reference: str | None, other: str | None
) -> list[dict]:
    """Each stratum's own estimate: per variable, the comparison (two groups) or the slope."""
    out = []
    for name, labels in zip(data.strata_names, data.strata, strict=True):
        for lev in sorted(set(labels), key=_order_key):
            rows = [i for i, lab in enumerate(labels) if lab == lev]
            item: dict[str, Any] = {"variable": name, "stratum": lev, "n": len(rows)}
            if data.analysis == "compare_groups":
                a = data.y[[i for i in rows if labels_of_unit[i] == other]]
                b = data.y[[i for i in rows if labels_of_unit[i] == reference]]
                item.update(n_group=int(len(a)), n_reference=int(len(b)))
                if len(a) and len(b):
                    item["estimate"] = float(a.mean() - b.mean())
                    item["group_estimate"] = float(a.mean())
                    item["reference_estimate"] = float(b.mean())
            else:
                x = np.asarray([data.g[i] for i in rows], dtype=float)
                if len(set(x)) >= 2:
                    item["estimate"] = float(np.polyfit(x, data.y[rows], 1)[0])
            out.append(item)
    return out


def _warnings(data: Data, groups: list[dict] | None) -> list[dict]:
    w: list[dict] = []
    if data.n_dropped:
        w.append({"kind": "dropped_missing", "rows": data.n_dropped})
    if len(data.y) < SMALL_GROUP:
        w.append({"kind": "few_units", "n": int(len(data.y))})
    for name in data.binned:
        w.append({"kind": "binned_stratum", "variable": name, "bins": 5})
    if groups:
        small = [{"label": g["label"], "n": g["n"]} for g in groups if g["n"] < SMALL_GROUP]
        if small:
            w.append({"kind": "small_group", "groups": small})
        if data.outcome_type == "binary":
            few = [
                {"label": g["label"], "events": g["events"], "n": g["n"]}
                for g in groups
                if min(g["events"], g["n"] - g["events"]) < FEW_EVENTS
            ]
            if few:
                w.append({"kind": "few_events", "groups": few})
            none = [g["label"] for g in groups if g["events"] in (0, g["n"])]
            if none:
                w.append({"kind": "all_or_none", "groups": none})
        else:
            skewed = [
                {"label": g["label"], "skewness": g["skewness"], "median": g["median"]}
                for g in groups
                if abs(g.get("skewness", 0.0)) > SKEWED
            ]
            if skewed:
                w.append({"kind": "skewed", "groups": skewed})
        if len(groups) > 2:
            w.append({"kind": "several_comparisons", "groups": len(groups)})
    return w


def compare_groups(data: Data, spec: dict) -> dict:
    labels = sorted(set(data.g), key=_order_key)
    if len(labels) < 2:
        raise AnalysisError("too_few_groups", "the groups column has fewer than two values")
    if len(labels) > MAX_GROUPS:
        raise AnalysisError("too_many_groups", f"more than {MAX_GROUPS} groups")
    reference = spec.get("reference_group")
    if reference is None:
        reference = labels[0]
    reference = _label(reference)
    if reference not in labels:
        raise AnalysisError("unknown_group", f"the reference group {reference!r} does not occur")
    others = [lab for lab in labels if lab != reference]
    groups, by = _group_summary(data, labels)
    if data.outcome_type == "numeric" and any(len(v) < 2 for v in by.values()):
        raise AnalysisError("too_few_units", "a comparison of means needs two units per group")
    out: dict[str, Any] = {
        "reference": reference,
        "groups": groups,
        "comparisons": [{"group": lab, **_pair(data, by[lab], by[reference])} for lab in others],
    }
    if len(labels) > 2:
        if data.outcome_type == "binary":
            ev = np.array([int(by[lab].sum()) for lab in labels])
            nn = np.array([len(by[lab]) for lab in labels])
            p, method = chi_square_p(ev, nn)
        else:
            res = anova_oneway([by[lab] for lab in labels], use_var="unequal")
            p, method = float(res.pvalue), "Welch's one-way ANOVA"
        out["any_difference"] = {"p_value": p, "method": method}

    main = _dummies(data.g, [reference, *others])
    main_names = [f"group={lab}" for lab in others]
    if data.strata:
        keep = _singletons(data)
        crude = _fit_effect(data, main, main_names, False, keep)
        adjusted = _fit_effect(data, main, main_names, True, keep)
        method = (
            "least squares with an indicator per stratum (fixed effects), HC3 errors"
            if data.outcome_type == "numeric"
            else "linear probability model with an indicator per stratum, HC3 errors"
        )
        adj: dict[str, Any] = {
            "strata": data.strata_names,
            "method": method,
            "units_dropped": int((~keep).sum()),
            "crude": crude,
            "adjusted": adjusted,
        }
        if len(others) == 1:
            adj["change"] = _change(crude, adjusted)
            adj["by_stratum"] = _by_stratum(data, data.g, reference, others[0])
        out["stratified"] = adj
        primary = adjusted
    elif len(others) == 1:
        primary = out["comparisons"][0]
    else:
        primary = out["any_difference"]
    out["detected"] = _detected(primary)
    out["primary"] = (
        "stratified" if data.strata else ("comparison" if len(others) == 1 else "any_difference")
    )
    out["warnings"] = _warnings(data, groups)
    return out


def trend(data: Data, spec: dict) -> dict:
    x = np.asarray(data.g, dtype=float)
    if len(set(x.tolist())) < 3:
        raise AnalysisError("too_few_x_values", "a trend needs at least three distinct x values")
    main = x.reshape(-1, 1)
    everyone = np.ones(len(data.y), dtype=bool)
    crude = _fit_effect(data, main, ["x"], False, everyone)
    measure = (
        "change in the rate per unit of x"
        if data.outcome_type == "binary"
        else ("change in the outcome per unit of x")
    )
    out: dict[str, Any] = {
        "slope": {
            "measure": measure,
            **crude,
            "method": "least-squares slope, HC3 errors"
            + (" (linear probability model)" if data.outcome_type == "binary" else ""),
        },
        "x_range": [float(x.min()), float(x.max())],
    }
    levels = sorted(set(x.tolist()))
    if len(levels) <= 12:  # few x values (years, say): each one's own estimate
        per = []
        for v in levels:
            ys = data.y[x == v]
            item: dict[str, Any] = {"x": v, "n": int(len(ys)), "estimate": float(ys.mean())}
            if data.outcome_type == "binary":
                k = int(ys.sum())
                item["events"] = k
                item["ci"] = list(wilson(k, len(ys)))
            per.append(item)
        out["by_x"] = per
    rho = stats.spearmanr(x, data.y)
    out["spearman"] = {"rho": _finite(rho.statistic), "p_value": _finite(rho.pvalue)}
    if data.strata:
        keep = _singletons(data)
        crude_k = _fit_effect(data, main, ["x"], False, keep)
        adjusted = _fit_effect(data, main, ["x"], True, keep)
        out["stratified"] = {
            "strata": data.strata_names,
            "method": "least-squares slope with an indicator per stratum, HC3 errors",
            "units_dropped": int((~keep).sum()),
            "crude": crude_k,
            "adjusted": adjusted,
            "change": _change(crude_k, adjusted),
            "by_stratum": _by_stratum(data, [], None, None),
        }
        primary = adjusted
    else:
        primary = crude
    out["detected"] = _detected(primary)
    out["primary"] = "stratified" if data.strata else "slope"
    w = _warnings(data, None)
    if data.outcome_type == "binary":
        k = int(data.y.sum())
        if min(k, len(data.y) - k) < FEW_EVENTS:
            w.append(
                {
                    "kind": "few_events",
                    "groups": [{"label": "all", "events": k, "n": int(len(data.y))}],
                }
            )
    if len(levels) < 5:
        w.append({"kind": "few_x_values", "values": len(levels)})
    out["warnings"] = w
    return out


def _finite(v: Any) -> float | None:
    v = float(v)
    return v if math.isfinite(v) else None


def _detected(primary: dict) -> bool:
    if "ci" in primary:
        lo, hi = primary["ci"]
        return not (lo <= 0 <= hi)
    return primary["p_value"] < ALPHA


def run(spec: dict, columns: Sequence[str], rows: Sequence[Sequence[Any]]) -> dict:
    """Run one analysis. Raises AnalysisError when it cannot be run on this input."""
    if not isinstance(spec, dict):
        raise AnalysisError("invalid_spec", "the analysis spec must be an object")
    data = Data(spec, columns, rows)
    result = compare_groups(data, spec) if data.analysis == "compare_groups" else trend(data, spec)
    return {
        "analysis": data.analysis,
        "outcome_type": data.outcome_type,
        "n": int(len(data.y)),
        "n_dropped": data.n_dropped,
        "level": LEVEL,
        **result,
    }


def clean(obj: Any) -> Any:
    """JSON-safe: numpy scalars to Python, non-finite floats to None."""
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [clean(v) for v in obj]
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj
