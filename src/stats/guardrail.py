"""The guardrail's runs: which questions, which calls, and the banking set's guarded answers.

The flow for one question: the classifier (src/stats/classify.py) says whether it is
statistical; if so, the analyst plans the analysis from the schema (a `plan` call), the plan's
query pulls the rows through the two guards, the sandbox runs the analysis, and the answer is
written from its result (an `answer` call). The same answer call also runs on the numbers alone,
without the statistics, to show what the guardrail changes. The winning design's own answer (its
SQL, execution accuracy and confidence) is left as it was: the guardrail adds to it.

The questions: the hand-written banking set (60), the benchmark's 500 (classified only) and the
planted-effect questions (configs/guardrail.yaml `planted.templates`). The probes
(`probes`) are other questions, used only to check the prompts' format before they are frozen.
A pull whose query reads the clock goes through the clock store (src/agent/clock.py), so a
replay sees the rows of the first run.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from src.agent.clock import ClockStore
from src.agent.critic import schema_context
from src.agent.tools import AgentTools
from src.data import bird
from src.db.connection import BIRD_DB
from src.db.execute import Limits, ReadOnlyExecutor, Target
from src.db.guard import QueryGuard
from src.db.hardening import schema_role
from src.stats.calls import Item
from src.stats.plans import Plan, PlanError, Pulled, pull

ROOT = Path(__file__).resolve().parent.parent.parent
OWN_SET = ROOT / "own_set/questions.yaml"
DB = "financial"


def question_text(question: str) -> str:
    return f"Question: {question.strip()}"


# --- the questions ------------------------------------------------------------------------------


def own_questions() -> list[dict]:
    doc = yaml.safe_load(OWN_SET.read_text(encoding="utf-8"))
    return [
        {
            "id": f"own:{q['id']}",
            "source": "own",
            "db_id": DB,
            "category": q["category"],
            "question": q["question"],
        }
        for q in doc["questions"]
    ]


def bird_questions() -> list[dict]:
    return [
        {
            "id": f"bird:{q['question_id']}",
            "source": "bird",
            "db_id": q["db_id"],
            "category": None,
            "question": q["question"],
        }
        for q in bird.questions()
    ]


def planted_questions(cfg: dict) -> list[dict]:
    return [
        {
            "id": f"planted:{t['id']}",
            "source": "planted",
            "db_id": DB,
            "category": t["family"],
            "question": t["question"],
        }
        for t in cfg["planted"]["templates"]
    ]


def probe_questions(cfg: dict) -> list[dict]:
    return [
        {"id": f"probe:{i}", "source": "probe", "db_id": DB, "category": None, "question": q}
        for i, q in enumerate(cfg["probes"]["questions"])
    ]


def classify_items(questions: list[dict]) -> list[Item]:
    return [
        Item(q["id"], f"Database: {q['db_id']}", question_text(q["question"])) for q in questions
    ]


def plan_items(questions: list[dict], toolbox) -> list[Item]:
    """The plan call reads the schema exactly as design 1 does."""
    tools = AgentTools(toolbox, [], 20)
    context = schema_context(tools, DB)
    return [Item(q["id"], context, question_text(q["question"])) for q in questions]


# --- plans --------------------------------------------------------------------------------------

PLAN_FIELDS = (
    "sql",
    "analysis",
    "outcome",
    "outcome_type",
    "group",
    "reference_group",
    "x",
    "strata",
    "x_describes",
    "unit",
)


def to_plan(submitted: dict | None) -> tuple[Plan | None, dict]:
    """The analyst's submitted plan as a Plan, or why there is none."""
    if not isinstance(submitted, dict):
        return None, {"kind": "no_plan", "message": "no plan was submitted"}
    if submitted.get("declined"):
        return None, {"kind": "declined", "message": submitted.get("decline_reason") or ""}
    if not submitted.get("sql"):
        return None, {"kind": "no_sql", "message": "the plan has no query"}
    fields = {k: submitted.get(k) for k in PLAN_FIELDS}
    if fields["analysis"] == "trend":
        fields["group"] = fields["reference_group"] = None
    else:
        fields["x"] = None
    try:
        return Plan.from_dict(fields), {}
    except PlanError as e:
        return None, {"kind": "invalid_plan", "message": str(e)}


# --- pulling through the guards -----------------------------------------------------------------


def benchmark_target() -> Target:
    zone = bird.config()["bird_minidev"]["time_zone"]
    return Target(BIRD_DB, DB, zone, schema_role(DB))


def pull_stored(
    plan: Plan,
    guard: QueryGuard,
    executor: ReadOnlyExecutor,
    limits: Limits,
    clock: ClockStore,
    db: str,
    at: tuple[str, str],
) -> Pulled:
    """`pull`, through the clock store (only a query that reads the clock is stored)."""

    def run() -> dict[str, Any]:
        p = pull(plan, guard, executor, limits)
        if p.ok:
            return {"ok": True, "columns": p.columns, "rows": p.rows}
        kind = p.error["kind"]
        return {
            "ok": False,
            "error": {
                **p.error,
                "kind": "sql_error"
                if kind in ("syntax_error", "sql_error", "permission_denied")
                else kind,
            },
        }

    r = clock.through(db, guard.tables, plan.sql, at, run)
    if r.get("ok"):
        return Pulled(columns=list(r["columns"]), rows=[list(row) for row in r["rows"]])
    return Pulled(error=dict(r["error"]))


# --- records ------------------------------------------------------------------------------------


def compact(result: dict | None) -> dict | None:
    """The parts of an analysis a record keeps."""
    if result is None:
        return None
    keep = (
        "analysis",
        "outcome_type",
        "n",
        "n_dropped",
        "detected",
        "primary",
        "reference",
        "groups",
        "comparisons",
        "any_difference",
        "slope",
        "stratified",
        "warnings",
    )
    return {k: result[k] for k in keep if k in result}


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True, default=str) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# --- the runs -----------------------------------------------------------------------------------


def split_ids(name: str) -> list[int]:
    doc = json.loads((ROOT / "results/metrics/splits.json").read_text(encoding="utf-8"))
    return list(doc["sets"][name])


def classify_questions(cfg: dict, probe: bool = False) -> list[dict]:
    """The probe: the first pilot questions. Otherwise the banking set, the 470 benchmark
    questions outside the pilot split and the planted questions."""
    by_id = {q["id"]: q for q in bird_questions()}
    pilot = split_ids("pilot")
    if probe:
        return [by_id[f"bird:{i}"] for i in pilot[: cfg["probes"]["classify_pilot"]]]
    skip = {f"bird:{i}" for i in pilot}
    bird = [q for qid, q in by_id.items() if qid not in skip]
    return own_questions() + bird + planted_questions(cfg)


def _usage(results: list[dict]) -> dict:
    tokens: dict[str, int] = {}
    for r in results:
        for k, v in r["tokens"].items():
            tokens[k] = tokens.get(k, 0) + v
    return {"tokens": tokens, "cost_usd": round(sum(r["cost_usd"] for r in results), 8)}


def _call_fields(r: dict) -> dict:
    return {
        "tokens": r["tokens"],
        "cost_usd": r["cost_usd"],
        "errors": r["errors"],
        "cache_keys": r["cache_keys"],
    }


def run_classify(
    questions: list[dict], cfg: dict, probe: bool = False, log=print, **kw
) -> list[dict]:
    from src.stats import classify
    from src.stats.calls import run_calls

    st = cfg["stages"]["classify"]
    results = run_calls(
        classify_items(questions),
        "classify",
        cfg,
        "direct" if probe else cfg["calls"]["classify"]["mode"],  # probes: no batch wait
        ROOT / st["cache"],
        cfg["phase"],
        log,
        probe,
        **kw,
    )
    patterns = classify.rules(cfg)
    records = []
    for q, r in zip(questions, results, strict=True):
        hits = classify.rule_hits(q["question"], patterns)
        model = classify.parse(r["submitted"])
        records.append(
            {
                **q,
                "rule_hits": hits,
                "rule_flag": bool(hits),
                "model": model,
                "model_flag": model["statistical"],
                "statistical": classify.combine(bool(hits), model["statistical"]),
                **_call_fields(r),
            }
        )
    return records


def run_plans(questions: list[dict], cfg: dict, probe: bool = False, log=print, **kw) -> list[dict]:
    from src.stats.calls import run_calls
    from src.tools.toolbox import Toolbox

    st = cfg["stages"]["plans"]
    with Toolbox(DB) as box:
        items = plan_items(questions, box)
    results = run_calls(
        items,
        "plan",
        cfg,
        cfg["calls"]["plan"]["mode"],
        ROOT / st["cache"],
        cfg["phase"],
        log,
        probe,
        **{"workers": 1, **kw},  # one at a time: the schema is written to the cache once
    )
    records = []
    for q, r in zip(questions, results, strict=True):
        plan, error = to_plan(r["submitted"])
        records.append(
            {
                **q,
                "submitted": r["submitted"],
                "plan": None if plan is None else {**plan.__dict__, "strata": list(plan.strata)},
                "plan_error": error or None,
                **_call_fields(r),
            }
        )
    return records


def analyze(plan_records: list[dict], cfg: dict, stage: str, log=print) -> dict[str, dict]:
    """Each plan's rows (through the guards, on the real data) and the sandbox's result."""
    from src.stats.plans import job
    from src.stats.sandbox import run_jobs

    lim = cfg["planted"]["limits"]
    limits = Limits(lim["max_rows"], lim["timeout_s"])
    guard = QueryGuard.for_benchmark_db(DB)
    clock = ClockStore(ROOT / cfg["stages"][stage]["cache"])
    out: dict[str, dict] = {}
    with ReadOnlyExecutor(benchmark_target()) as executor:
        for rec in plan_records:
            if rec["plan"] is None:
                continue
            plan = Plan.from_dict(rec["plan"])
            pulled = pull_stored(plan, guard, executor, limits, clock, DB, (stage, rec["id"]))
            out[rec["id"]] = {"plan": plan, "pulled": pulled, "result": None, "error": pulled.error}
    jobs = [job(i, o["plan"], o["pulled"]) for i, o in out.items() if o["pulled"].ok]
    for r in run_jobs(jobs) if jobs else []:
        if r["ok"]:
            out[r["id"]]["result"] = r["result"]
        else:
            out[r["id"]]["error"] = r["error"]
    log(f"analyzed {len(jobs)} of {len(plan_records)} plans")
    return out


ARMS = ("guarded", "numbers_only")


def answer_items(analyses: dict[str, dict], questions: dict[str, str]) -> list[Item]:
    from src.stats.answer import answer_text

    items = []
    for i, a in analyses.items():
        if a["result"] is None:
            continue
        for arm in ARMS:
            result = a["result"] if arm == "guarded" else None
            items.append(
                Item(
                    f"{i}#{arm}",
                    f"Database: {DB}",
                    answer_text(questions[i], a["plan"], a["pulled"], result),
                )
            )
    return items


def run_own(
    plan_records: list[dict],
    classified: list[dict],
    cfg: dict,
    stage: str = "own",
    probe: bool = False,
    log=print,
    **kw,
) -> list[dict]:
    """The guarded answers: every planned question analyzed and answered with and without the
    statistics. Questions the classifier left on the descriptive path are recorded as such."""
    from src.stats.answer import checks, delivered
    from src.stats.calls import run_calls

    analyses = analyze(plan_records, cfg, stage, log)
    questions = {r["id"]: r["question"] for r in plan_records}
    items = answer_items(analyses, questions)
    results = run_calls(
        items,
        "answer",
        cfg,
        cfg["calls"]["answer"]["mode"],
        ROOT / cfg["stages"][stage]["cache"],
        cfg["phase"],
        log,
        probe,
        **kw,
    )
    by_item = {r["id"]: r for r in results}
    planned = {r["id"]: r for r in plan_records}
    records = []
    for c in classified:
        rec: dict[str, Any] = {k: c[k] for k in ("id", "source", "db_id", "category", "question")}
        rec["statistical"] = c.get("statistical", True)
        p = planned.get(c["id"])
        if p is None:
            rec["path"] = "descriptive"
            records.append(rec)
            continue
        rec.update(path="statistical", plan=p["plan"], plan_error=p["plan_error"])
        a = analyses.get(c["id"])
        if a is None or a["result"] is None:
            rec["analysis_error"] = None if a is None else a["error"]
            records.append(rec)
            continue
        rec["analysis"] = compact(a["result"])
        rec["analysis_error"] = None
        calls = []
        for arm in ARMS:
            r = by_item[f"{c['id']}#{arm}"]
            calls.append(r)
            finding = r["submitted"] if isinstance(r["submitted"], dict) else {}
            entry = {
                "finding": finding or None,
                "checks": checks(finding, a["result"]),
                "errors": r["errors"],
                "cache_keys": r["cache_keys"],
            }
            if arm == "guarded":
                entry["delivered"] = delivered(finding, a["result"], a["plan"])
            rec[arm] = entry
        rec.update(_usage(calls))
        records.append(rec)
    return records
