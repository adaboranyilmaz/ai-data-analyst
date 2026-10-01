"""What the answer is written from, and what the guardrail adds to it.

Two inputs for the answer call (prompts/stat_answer_v1.md), which differ only by the guardrail:
- `numbers only`: the question, how the data was drawn, and a descriptive table computed here from
  the rows (each group's units and share or mean, or the outcome at each x): what a text-to-SQL
  analyst would show;
- `guarded`: the same, plus the statistics block rendered from the sandbox's result: intervals and
  tests, the stratified check, the warnings, the note that the records are observational, and the
  test's verdict.

After the call, deterministic checks compare what the answer claims (`claims_effect`, `higher`)
with the analysis: a claimed effect the interval does not support, an effect the answer denies,
a direction the numbers contradict, and causal wording outside a negation. The guarded answer as
delivered is the model's text followed by the statistics summary, a correction where the claim
and the test disagree, and the observational note: whatever the model writes, the reader sees the
interval and the caveat.
"""

from __future__ import annotations

import math
import re
from typing import Any

import numpy as np

from src.stats.plans import Plan, Pulled

OBSERVATIONAL = (
    "These are observational records: the groups were not assigned at random, so a difference "
    "or a trend is an association and does not show that one thing causes another."
)
AREA = (
    "The groups (or x) describe areas the units belong to, not the units themselves: a relation "
    "seen across areas need not hold for the individuals in them."
)
CHANGE_TEXT = {
    "vanished": "the difference seen overall disappears within these levels: they explain it",
    "reversed": (
        "within these levels the difference points the other way: the overall one is produced "
        "by them (Simpson's paradox)"
    ),
    "appeared": "a difference appears within these levels that the overall comparison hides",
    "stable": "the result holds within these levels",
}
CAUSAL = re.compile(
    r"\b(causes?|caused|causing|leads? to|led to|results? in|because|due to|protects?|"
    r"protected|the effect of|effect on|drives?|makes? (?:\w+ ){0,4}(?:more|less) likely)\b",
    re.IGNORECASE,
)
NEGATION = re.compile(r"\b(not|no|cannot|can't|doesn't|don't|isn't|never|nor|without)\b", re.I)


# --- labels, as the sandbox writes them ---------------------------------------------------------


def label(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return str(int(v)) if v.is_integer() else repr(v)
    return str(v)


def order_key(lab: str) -> tuple:
    try:
        return (0, float(lab), lab)
    except ValueError:
        return (1, 0.0, lab)


# --- numbers ------------------------------------------------------------------------------------


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def points(x: float) -> str:
    return f"{100 * x:+.1f} percentage points"


def num(x: float) -> str:
    return f"{x:,.0f}" if abs(x) >= 100 else f"{x:,.2f}"


def signed(x: float) -> str:
    return ("+" if x >= 0 else "-") + num(abs(x))


def slope(x: float, binary: bool) -> str:
    """A slope: percentage points (two decimals) for a yes/no outcome, else the outcome's units."""
    return f"{100 * x:+.2f}" if binary else signed(x)


def pval(p: float | None) -> str:
    if p is None:
        return "not available"
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


# --- the descriptive table (both inputs) --------------------------------------------------------


def describe(plan: Plan, pulled: Pulled) -> list[str]:
    """Each group's (or x's) units and outcome, computed from the rows: no inference."""
    cols = pulled.columns
    iy = cols.index(plan.outcome) if plan.outcome in cols else None
    key = plan.group if plan.analysis == "compare_groups" else plan.x
    ig = cols.index(key) if key in cols else None
    if iy is None or ig is None:
        return ["(the result lacks the columns the plan names)"]
    pairs = [(r[ig], r[iy]) for r in pulled.rows if r[ig] is not None and r[iy] is not None]
    binary = plan.outcome_type == "binary"

    def stat(values: list) -> str:
        ys = np.array([float(v) for v in values])
        if binary:
            return f"{int(ys.sum())} of {len(ys)} with the outcome ({pct(ys.mean())})"
        return f"{len(ys)} units, mean {num(ys.mean())}, median {num(float(np.median(ys)))}"

    lines = [f"{len(pairs):,} units with every value ({len(pulled.rows) - len(pairs)} left out)."]
    if plan.analysis == "compare_groups":
        groups: dict[str, list] = {}
        for g, y in pairs:
            groups.setdefault(label(g), []).append(y)
        for lab in sorted(groups, key=order_key):
            lines.append(f"- {plan.group} = {lab}: {stat(groups[lab])}")
        return lines
    xs = np.array([float(x) for x, _ in pairs])
    levels = sorted(set(xs.tolist()))
    if len(levels) <= 12:
        for v in levels:
            lines.append(f"- {plan.x} = {label(v)}: {stat([y for x, y in pairs if float(x) == v])}")
        return lines
    edges = np.quantile(xs, [0.2, 0.4, 0.6, 0.8])
    which = np.searchsorted(edges, xs, side="right")
    bounds = [xs.min(), *edges, xs.max()]
    for k in range(5):
        ys = [y for (x, y), w in zip(pairs, which, strict=True) if w == k]
        if ys:
            lines.append(
                f"- {plan.x} from {label(float(bounds[k]))} to {label(float(bounds[k + 1]))}: "
                f"{stat(ys)}"
            )
    return lines


def data_line(plan: Plan) -> str:
    what = "the share of units with outcome 1" if plan.outcome_type == "binary" else "its mean"
    if plan.analysis == "compare_groups":
        across = f"groups `{plan.group}` (compared with {plan.reference_group or 'the first'})"
    else:
        across = f"`{plan.x}`"
    return (
        f"The data: each row is {plan.unit or 'one unit'}; outcome `{plan.outcome}` "
        f"({what}), across {across}."
    )


# --- the statistics block (guarded input only) --------------------------------------------------


def _difference(result: dict, value: float) -> str:
    return points(value) if result["outcome_type"] == "binary" else signed(value)


def _interval(result: dict, ci: list) -> str:
    if result["outcome_type"] == "binary":
        return f"{100 * ci[0]:+.1f} to {100 * ci[1]:+.1f}"
    return f"{signed(ci[0])} to {signed(ci[1])}"


def _warning(w: dict) -> str:
    kind = w["kind"]
    if kind == "small_group":
        return "Small groups: " + ", ".join(f"{g['label']} ({g['n']} units)" for g in w["groups"])
    if kind == "few_events":
        items = ", ".join(f"{g['label']} ({g['events']} of {g['n']})" for g in w["groups"])
        return (
            f"Few units with (or without) the outcome in: {items}; counts this small rest on "
            "a few cases"
        )
    if kind == "all_or_none":
        return "All or none of the units have the outcome in: " + ", ".join(w["groups"])
    if kind == "skewed":
        items = ", ".join(f"{g['label']} (median {num(g['median'])})" for g in w["groups"])
        return f"Skewed values, so a few large ones pull the mean: {items}"
    if kind == "several_comparisons":
        return (
            f"{w['groups']} groups: the intervals for each pair are not adjusted for comparing many"
        )
    if kind == "few_units":
        return f"Only {w['n']} units in all"
    if kind == "few_x_values":
        return f"Only {w['values']} distinct values of x"
    if kind == "binned_stratum":
        return f"{w['variable']} was cut into fifths to compare within it"
    if kind == "dropped_missing":
        return f"{w['rows']} rows without every value were left out"
    return kind


def render_stats(result: dict, plan: Plan) -> list[str]:
    lines = ["Statistical analysis (computed by a tested program; 95% intervals):"]
    binary = result["outcome_type"] == "binary"
    subject = "difference"
    if result["analysis"] == "compare_groups":
        for g in result["groups"]:
            if binary:
                lines.append(
                    f"- {plan.group} = {g['label']}: {pct(g['estimate'])} of {g['n']} units, "
                    f"interval {pct(g['ci'][0])} to {pct(g['ci'][1])}"
                )
            else:
                ci = g.get("ci")
                band = f", interval {num(ci[0])} to {num(ci[1])}" if ci else ""
                lines.append(
                    f"- {plan.group} = {g['label']}: mean {num(g['estimate'])}{band}, "
                    f"median {num(g['median'])}, {g['n']} units"
                )
        for c in result["comparisons"]:
            lines.append(
                f"- {c['group']} minus {result['reference']}: {_difference(result, c['estimate'])}"
                f", interval {_interval(result, c['ci'])}; {c['method']}, p {pval(c['p_value'])}"
            )
        if "any_difference" in result:
            a = result["any_difference"]
            lines.append(
                f"- Any difference between the groups: {a['method']}, p {pval(a['p_value'])}"
            )
    else:
        subject = "trend"
        s = result["slope"]
        per = "percentage points" if binary else "units of the outcome"
        lines.append(
            f"- Slope: {slope(s['estimate'], binary)} {per} per unit of {plan.x}, interval "
            f"{slope(s['ci'][0], binary)} to {slope(s['ci'][1], binary)}; {s['method']}, "
            f"p {pval(s['p_value'])}"
        )
        for b in result.get("by_x", []):
            band = f", interval {pct(b['ci'][0])} to {pct(b['ci'][1])}" if binary else ""
            value = pct(b["estimate"]) if binary else num(b["estimate"])
            lines.append(f"- {plan.x} = {label(b['x'])}: {value} of {b['n']} units{band}")
    strat = result.get("stratified")
    if strat:
        adj, crude = strat["adjusted"], strat["crude"]
        where = ", ".join(strat["strata"])
        if "estimate" in adj:
            fmt = (
                (lambda v: slope(v, binary))
                if subject == "trend"
                else (lambda v: _difference(result, v))
            )
            band = (
                (lambda ci: f"{fmt(ci[0])} to {fmt(ci[1])}")
                if subject == "trend"
                else (lambda ci: _interval(result, ci))
            )
            text = (
                f"- Compared within levels of {where}: {fmt(adj['estimate'])}, interval "
                f"{band(adj['ci'])}; without them, on the same units: {fmt(crude['estimate'])}, "
                f"interval {band(crude['ci'])}"
            )
            if strat.get("change"):
                text += f". So {CHANGE_TEXT[strat['change']]}"
            lines.append(text)
        else:
            lines.append(
                f"- Compared within levels of {where}: test of any difference, p "
                f"{pval(adj['p_value'])}; without them, p {pval(crude['p_value'])}"
            )
    warnings = [_warning(w) for w in result.get("warnings", [])]
    if plan.x_describes == "area":
        warnings.append(AREA)
    for w in warnings:
        lines.append(f"- Warning: {w}.")
    lines.append(f"- {OBSERVATIONAL}")
    lines.append("- " + verdict(result, subject))
    return lines


def verdict(result: dict, subject: str | None = None) -> str:
    subject = subject or ("trend" if result["analysis"] == "trend" else "difference")
    basis = {
        "stratified": "compared within the strata",
        "comparison": "the difference",
        "slope": "the slope",
        "any_difference": "the test of any difference",
    }[result["primary"]]
    if result["primary"] == "any_difference" or (
        result["primary"] == "stratified" and "estimate" not in result["stratified"]["adjusted"]
    ):
        rule = "p is below 0.05" if result["detected"] else "p is not below 0.05"
    else:
        rule = "the interval excludes zero" if result["detected"] else "the interval includes zero"
    shows = "show" if result["detected"] else "do not show"
    return f"Verdict of the test ({basis}): {rule}, so the data {shows} a {subject}."


# --- the two inputs -----------------------------------------------------------------------------


def answer_text(question: str, plan: Plan, pulled: Pulled, result: dict | None) -> str:
    """The answer call's input; `result` None gives the numbers-only input."""
    lines = [f"Question: {question.strip()}", "", data_line(plan), "", "Results:"]
    lines += describe(plan, pulled)
    if result is not None:
        lines += [""] + render_stats(result, plan)
    return "\n".join(lines)


# --- checks after the call ----------------------------------------------------------------------


LEAKED_MARKUP = re.compile(r"</answer>|<parameter name=|<claims_effect>|<higher>")


def answer_body(text: str) -> str:
    """The model's answer without tool-call markup it sometimes writes into the string itself
    (a closing tag and the next fields, after the answer's last sentence)."""
    m = LEAKED_MARKUP.search(text)
    return (text[: m.start()] if m else text).strip()


def causal_sentences(text: str) -> list[str]:
    """Sentences with causal wording and no negation (for the hand check and the report)."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s for s in sentences if CAUSAL.search(s) and not NEGATION.search(s)]


def higher_label(result: dict) -> str | None:
    """The group with the higher outcome (two groups), or the trend's direction."""
    if result["analysis"] == "trend":
        est = result["slope"]["estimate"]
        return "increasing" if est > 0 else "decreasing" if est < 0 else None
    groups = result["groups"]
    if len(groups) != 2 or math.isclose(groups[0]["estimate"], groups[1]["estimate"]):
        return None
    return max(groups, key=lambda g: g["estimate"])["label"]


def checks(finding: dict, result: dict) -> dict[str, Any]:
    claims = finding.get("claims_effect")
    detected = result["detected"]
    higher = finding.get("higher")
    expected = higher_label(result)
    return {
        "unsupported_claim": claims == "yes" and not detected,
        "missed_effect": claims == "no" and detected,
        "direction_mismatch": claims == "yes"
        and higher is not None
        and expected is not None
        and str(higher) != expected,
        "causal_sentences": causal_sentences(answer_body(finding.get("answer") or "")),
    }


def delivered(finding: dict, result: dict, plan: Plan) -> str:
    """The guarded answer as the reader gets it."""
    lines = [answer_body(finding.get("answer") or ""), ""]
    stats = render_stats(result, plan)
    # The model's input ends the area warning with two periods; the reader's copy has one.
    keep = [
        s.removesuffix(".") if s.endswith("..") else s
        for s in stats[1:]
        if s.startswith(("- Verdict", "- Warning"))
    ]
    main = [s for s in stats[1:] if " minus " in s or s.startswith(("- Slope", "- Compared"))]
    lines += ["Statistics (95% intervals):", *main, *keep]
    c = checks(finding, result)
    if c["unsupported_claim"]:
        lines.append("- Note: the interval includes zero, so these data do not show an effect.")
    if c["missed_effect"]:
        lines.append("- Note: the interval excludes zero, so these data do show an effect.")
    lines.append(f"- {OBSERVATIONAL}")
    return "\n".join(lines)
