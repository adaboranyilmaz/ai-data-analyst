"""Whether a rebuild of the pipeline reproduced the committed results.

Compares a results tree with a baseline copy taken before the rebuild (`dvc repro --force` from a
clean checkout), file by file: JSON and JSON-lines semantically, every other file byte for byte.
A difference is allowed only in a field matched by an EXEMPT rule (files, field path, reason, and
a numeric tolerance or none). The rules are as narrow as the evidence allows: each one names
something that records the run rather than a result (when it ran, how long it took), or a value
that is not reproducible by construction, with the reason. Every other difference is a
reproduction failure.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import re
from pathlib import Path
from typing import Any

# (files glob, field-path regex, reason, absolute tolerance or None for any value). Field paths
# look like `.runs.x.y`, with `[i]` for list items (a JSON-lines file is a list of records).
EXEMPT: list[tuple[str, str, str, float | None]] = [
    ("*", r"\.(updated_utc|run_utc|evaluated_at|utc)$", "when the run happened", None),
    (
        "*",
        r"\.(latency_s|latency_ms|seconds|elapsed_s|duration_s)$|\.(p50|p95|p99|mean)_(ms|s)$"
        r"|\.[a-z_]*_seconds$|\.(max|min)_ms$|\.latency_s\.(p50|p95|p99|mean)$"
        r"|\.seconds_per_question_cached_calls\.[a-z_]+\.(p50|p95|mean)$",
        "measured wall-clock time",
        None,
    ),
    (
        "metrics/gold_execution*.json",
        r"\.summary\.slowest\[\d+\]\.question_id$",
        "which questions were slowest is a ranking of measured time",
        None,
    ),
]

# Wall-clock time or an ephemeral address written inside a recorded string: the string is compared
# with that part masked. (files glob, field-path regex, pattern, replacement, reason)
MASKED: list[tuple[str, str, str, str, str]] = [
    (
        "metrics/mcp_security.json",
        r"\.boundary\[\d+\]\.detail$",
        r'"seconds": [0-9.e+-]+',
        '"seconds": N',
        "measured wall-clock time inside a recorded tool reply",
    ),
    (
        "metrics/security_suite.json",
        r"\.records\[\d+\]\.modes\.[a-z]+\.detail$",
        r"in [0-9.]+ s",
        "in N s",
        "measured wall-clock time inside a recorded detail",
    ),
    (
        "metrics/mcp_security.json",
        r"\.records\[\d+\]\.detail$",
        r"in [0-9.]+ s",
        "in N s",
        "measured wall-clock time inside a recorded detail",
    ),
    (
        "metrics/sandbox_security.json",
        r"\.control_records\[\d+\]\.detail$",
        r"laddr=\('[0-9.]+', \d+\), raddr=\('[0-9.]+', \d+\)",
        "laddr=ADDR, raddr=ADDR",
        "the ports and container addresses of a socket's repr",
    ),
]

# What a rebuild was seen to change that is a result, not a record of the run, with its cause. Each
# names one file and what the cause touches; the report lists these apart from the exempt ones,
# with the values seen, and a rebuild that changes anything else in them still fails. The cause
# is the order in which PostgreSQL returns the rows of a query with no ORDER BY, which is the
# physical layout of a database loaded afresh: the execution accuracy compares row sets and is not
# affected; the ordered hash and the soft-F1, which depend on row order, are.
ROW_ORDER = "the row order of an unordered query differs between loads of the database"
PARALLEL = "parallel query workers return an unordered query's rows in an order that varies"
NONDETERMINISTIC: list[tuple[str, str, str, float | None]] = [
    ("metrics/data_stats.json", r"\.gold\.public_vs_schema_layout\.[a-z ,]+$", ROW_ORDER, 1),
    (
        "metrics/gold_execution.json",
        r"\.compared_with_public_layout\.counts\.[a-z ,]+$",
        ROW_ORDER,
        1,
    ),
    (
        "metrics/gold_execution.json",
        r"\.compared_with_public_layout\.not_identical\.\d+$",
        ROW_ORDER,
        None,
    ),
    ("metrics/gold_execution.json", r"\.questions\[439\]\.ordered_hash$", ROW_ORDER, None),
    (
        "metrics/ablation.json",
        r"\.runs\.qwen2\.5:3b-instruct/d2\.soft_f1\.(estimate|low|high)$",
        ROW_ORDER,
        0.01,
    ),
    (
        "runs/ablation/ablation-d2-qwen2.5_3b-instruct-evidence.jsonl",
        r"\[50\]\.soft_f1$",
        ROW_ORDER,
        0.5,
    ),
    # the official evaluator with the database's default parallel plans: this block measures how
    # many gold queries it scores wrong because parallel workers return rows in a varying order
    ("metrics/ex_validation.json", r"\.parallel_plans\.official_ex_1$", PARALLEL, 3),
    ("metrics/ex_validation.json", r"\.parallel_plans\.scored_0$", PARALLEL, None),
]

SKIP_FILES = {
    # this report itself, and results the rebuild does not own: measured or checked against a
    # running stack by their own scripts, or figures taken from a running tracing tool
    "metrics/reproduction_check.json",
    "metrics/canary_history.json",
    "metrics/alert_check.json",
    "metrics/load_test.json",
    "metrics/static_site.json",
    "plots/trace_mlflow.png",
    "plots/trace_langfuse.png",
}
SKIP_PREFIXES = ("canary/",)  # the canary's local runs


def differences(a: Any, b: Any, path: str = "") -> list[tuple[str, Any, Any, str]]:
    """Every (field path, value a, value b, description) at which two JSON values differ."""
    numbers = isinstance(a, int | float) and isinstance(b, int | float)
    if type(a) is not type(b) and not numbers:
        return [(path, a, b, f"type {type(a).__name__} vs {type(b).__name__}")]
    if isinstance(a, dict):
        out = []
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                out.append((f"{path}.{k}", a.get(k), b.get(k), "present in only one"))
            else:
                out += differences(a[k], b[k], f"{path}.{k}")
        return out
    if isinstance(a, list):
        if len(a) != len(b):
            return [(path, a, b, f"length {len(a)} vs {len(b)}")]
        out = []
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            out += differences(x, y, f"{path}[{i}]")
        return out
    return [] if a == b else [(path, a, b, f"{a!r} vs {b!r}"[:200])]


def exemption(
    rel: str,
    path: str,
    a: Any = None,
    b: Any = None,
    rules: list[tuple[str, str, str, float | None]] | None = None,
) -> str | None:
    for glob, pattern, reason, tol in EXEMPT if rules is None else rules:
        if matches(rel, path, glob, pattern, a, b, tol):
            return reason
    if rules is None and isinstance(a, str) and isinstance(b, str):
        for glob, pattern, mask, repl, reason in MASKED:
            if fnmatch.fnmatch(rel, glob) and re.search(pattern, path):
                if re.sub(mask, repl, a) == re.sub(mask, repl, b):
                    return reason
    return None


def matches(
    rel: str, path: str, glob: str, pattern: str, a: Any, b: Any, tol: float | None
) -> bool:
    if not (fnmatch.fnmatch(rel, glob) and re.search(pattern, path)):
        return False
    if tol is None:
        return True
    return isinstance(a, int | float) and isinstance(b, int | float) and abs(a - b) <= tol


def nondeterminism(rel: str, path: str, a: Any, b: Any) -> str | None:
    for glob, pattern, reason, tol in NONDETERMINISTIC:
        if matches(rel, path, glob, pattern, a, b, tol):
            return reason
    return None


def load(path: Path) -> Any:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        return [json.loads(x) for x in text.splitlines() if x.strip()]
    return json.loads(text)


def compare_file(
    rel: str,
    base: Path,
    cur: Path,
    rules: list[tuple[str, str, str, float | None]] | None = None,
) -> dict[str, Any]:
    if base.read_bytes() == cur.read_bytes():
        return {"status": "identical"}
    if base.suffix not in (".json", ".jsonl"):
        reason = exemption(rel, "<bytes>", rules=rules)
        if reason:
            return {"status": "exempt_only", "n_fields": 1, "reasons": [reason]}
        return {"status": "different", "detail": ["bytes differ"]}
    diffs = differences(load(base), load(cur))
    if not diffs:  # the same content, formatted differently
        return {"status": "identical"}
    known = rules is None  # the committed rules also know the causes seen in a rebuild
    unexplained = [
        f"{p}: {d}"
        for p, a, b, d in diffs
        if exemption(rel, p, a, b, rules) is None and not (known and nondeterminism(rel, p, a, b))
    ]
    if unexplained:
        return {"status": "different", "n": len(unexplained), "detail": unexplained[:20]}
    seen = [
        {"field": p, "committed": a, "rebuilt": b, "reason": nondeterminism(rel, p, a, b)}
        for p, a, b, _ in diffs
        if known and exemption(rel, p, a, b, rules) is None
    ]
    if seen:
        return {"status": "known_nondeterministic", "n_fields": len(seen), "seen": seen}
    reasons = sorted({exemption(rel, p, a, b, rules) for p, a, b, _ in diffs})
    return {"status": "exempt_only", "n_fields": len(diffs), "reasons": reasons}


def files(root: Path) -> set[str]:
    return {
        p.relative_to(root).as_posix()
        for p in root.rglob("*")
        if p.is_file()
        and p.relative_to(root).as_posix() not in SKIP_FILES
        and not p.relative_to(root).as_posix().startswith(SKIP_PREFIXES)
    }


def check(
    baseline: Path,
    current: Path,
    rules: list[tuple[str, str, str, float | None]] | None = None,
) -> dict[str, Any]:
    """The report: counts by status, whether every file reproduced, and what did not."""
    base_files, cur_files = files(baseline), files(current)
    report: dict[str, dict[str, Any]] = {}
    for rel in sorted(base_files | cur_files):
        if rel not in cur_files:
            report[rel] = {"status": "missing_after_rebuild"}
        elif rel not in base_files:
            report[rel] = {"status": "new_after_rebuild"}
        else:
            report[rel] = compare_file(rel, baseline / rel, current / rel, rules)
    counts: dict[str, int] = {}
    for r in report.values():
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    ok = ("identical", "exempt_only", "known_nondeterministic")
    failures = {k: v for k, v in report.items() if v["status"] not in ok}
    return {
        "counts": counts,
        "files_compared": len(report),
        "passed": not failures,
        "reproduced_exactly": all(
            v["status"] in ("identical", "exempt_only") for v in report.values()
        ),
        "failures": failures,
        "known_nondeterministic": {
            k: v for k, v in report.items() if v["status"] == "known_nondeterministic"
        },
        "exempt_only": {k: v for k, v in report.items() if v["status"] == "exempt_only"},
        "baseline_sha256_of_file_list": hashlib.sha256(
            "\n".join(sorted(base_files)).encode()
        ).hexdigest(),
    }
