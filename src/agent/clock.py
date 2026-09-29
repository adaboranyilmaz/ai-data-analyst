"""The results of the agent's queries that read the clock, stored so a run replays exactly.

Every tool result is a function of the database and the query, so a replayed run rebuilds each
request exactly and finds its response in the cache, with one exception: a query that reads the
current time (`now()`, `current_date` and the like; src/eval/execution.py `reads_the_clock`).
Its result changes with the clock (an age computed with `AGE(NOW(), ...)` changes every
microsecond), so the next request would differ and a replay would miss the cache. The result of
such a query is therefore stored the first time it runs, keyed by the database, the tables its
guard allows (design 5 narrows them), the query text and the call it answers (the request whose
response made the call, and the call's id), and returned from then on. What the model saw on
the first run is what it sees on every replay.

Only results that came from the database are stored: a result, or the database's error for the
query. A refusal never reaches the database, and a timeout or a lost connection says nothing
about the query, so neither is stored.

Like the response cache, each stage writes its own directory (`<stage cache>/clock/`) and reads
the earlier stages' as well, so a stage that replays another's conversations sees their results.
For runs made before this store existed, `seed_from_traces` fills it from the traces, which keep
every tool result exactly as it was returned.

Scoring is unaffected: it runs the answer's query and the gold query together, at scoring time,
as the official evaluator does.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from src.eval.execution import reads_the_clock

SUBDIR = "clock"
_STORED_ERRORS = ("sql_error",)  # a database error for the query itself


def key(db: str, tables: Iterable[str], sql: str, at: tuple[str, str]) -> str:
    """`at`: the call the query answers, as (the cache key of the request whose response made
    the call, the call's id). Two conversations, or two samples, that run the same query at
    different moments saw different results, so each call keeps its own; a replay makes the
    same call from the same request."""
    text = json.dumps(
        [db, sorted(tables), sql, list(at)], ensure_ascii=False, separators=(",", ":")
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def storable(result: dict) -> bool:
    if result.get("ok"):
        return True
    return (result.get("error") or {}).get("kind") in _STORED_ERRORS


class ClockStore:
    """Writes to one stage's cache directory; reads it first, then the earlier stages'."""

    def __init__(self, write: Path, read_also: Sequence[Path] = ()):
        self.write = Path(write) / SUBDIR
        self.layers = [self.write] + [Path(p) / SUBDIR for p in read_also]

    @staticmethod
    def _path(root: Path, k: str) -> Path:
        return root / k[:2] / f"{k}.json"

    def get(self, k: str) -> dict | None:
        for root in self.layers:
            path = self._path(root, k)
            if path.exists():
                return json.loads(path.read_text(encoding="utf-8"))["result"]
        return None

    def put(
        self,
        at: tuple[str, str],
        db: str,
        tables: Iterable[str],
        sql: str,
        result: dict,
        source: str,
    ) -> None:
        """Store a result (without its timing). `source`: `run` or `trace`."""
        tables = sorted(tables)
        path = self._path(self.write, key(db, tables, sql, at))
        path.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "db": db,
            "tables": tables,
            "sql": sql,
            "at": list(at),
            "source": source,
            "result": {k: v for k, v in result.items() if k != "seconds"},
        }
        tmp = path.with_suffix(f".tmp{os.getpid()}.{threading.get_ident()}")
        tmp.write_text(
            json.dumps(entry, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n"
        )
        os.replace(tmp, path)

    def through(
        self, db: str, tables: Iterable[str], sql: str, at: tuple[str, str] | None, run
    ) -> dict:
        """The result of `run()` for this query, stored when it reads the clock: the stored one
        if there is one, else a fresh one, stored if it came from the database. Without `at`
        (a call outside a conversation) nothing is stored."""
        if at is None or not reads_the_clock(sql):
            return run()
        stored = self.get(key(db, tables, sql, at))
        if stored is not None:
            return dict(stored)
        result = run()
        if storable(result):
            self.put(at, db, tables, sql, result, "run")
        return result


def _calls_of_sample(trace: dict, sample: int, entry: Any) -> list[tuple[str, dict]]:
    """The tool calls of one sample's conversation, in order, each with the request it came
    from: [(request key, tool_use block)]. `entry(key)`: the cached {request, response}."""
    out = []
    for r in trace["requests"]:
        e = entry(r["cache_key"])
        req = e["request"]
        if any(t["name"] == "select_schema" for t in req["tools"]):
            continue  # design 5's narrowing call
        if (req["params"].get("_sample") or 0) != sample:
            continue
        for block in e["response"]["content"]:
            if block.get("type") == "tool_use":
                out.append((r["cache_key"], block))
    return out


def seed_from_traces(
    traces: Iterable[Path], store: ClockStore, tables_of: Any, entry: Any
) -> dict[str, int]:
    """Store the clock-reading `run_sql` results of earlier runs from their traces.

    A trace keeps each sample's tool results in the order they ran; the cached responses its
    requests list (`entry(key)`: the cached {request, response}) give the calls, in the same
    order, with their ids. Each result is matched to the next call of the same tool with the
    same input (calls past the tool budget have no result, and are skipped).
    `tables_of(db)`: the tables the full guard of a database allows. Returns counts."""
    counts = {"stored": 0, "already": 0, "not_storable": 0}
    for path in traces:
        t = json.loads(Path(path).read_text(encoding="utf-8"))
        db = t["question"]["db_id"]
        selection = (t.get("narrowing") or {}).get("selection")
        tables = sorted(selection) if selection else sorted(tables_of(db))
        for i, sample in enumerate(t["samples"]):
            calls = iter(_calls_of_sample(t, i, entry))
            for tr in sample["tool_results"]:
                request_key, block = next(
                    (k, b)
                    for k, b in calls
                    if b["name"] == tr["tool"] and b["input"] == tr["input"]
                )
                sql = (tr.get("input") or {}).get("sql") if tr["tool"] == "run_sql" else None
                if not isinstance(sql, str) or not reads_the_clock(sql):
                    continue
                at = (request_key, block["id"])
                if store.get(key(db, tables, sql, at)) is not None:
                    counts["already"] += 1
                elif not storable(tr["result"]):
                    counts["not_storable"] += 1
                else:
                    store.put(at, db, tables, sql, tr["result"], "trace")
                    counts["stored"] += 1
    return counts
