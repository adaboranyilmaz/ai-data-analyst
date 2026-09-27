"""All five tools bound to one benchmark database, behind one `call(name, input)`.

The limits come from configs/tools.yaml; the time zone from configs/datasets.yaml. Every call
returns a JSON-ready dict with `ok` and the call's wall-clock `seconds`; an unknown tool or a
malformed input is an error result, never an exception, so the agent can recover from it.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import yaml

from src.data import bird
from src.db.connection import BIRD_DB
from src.db.execute import Limits, ReadOnlyExecutor, Target
from src.db.guard import QueryGuard
from src.db.hardening import schema_role
from src.tools.chart import validate_chart
from src.tools.definitions import TOOL_NAMES, TOOLS
from src.tools.schema import SchemaTools
from src.tools.sql import SqlTools, refused

ROOT = Path(__file__).resolve().parent.parent.parent
TOOLS_CONFIG = ROOT / "configs/tools.yaml"


def config() -> dict[str, Any]:
    return yaml.safe_load(TOOLS_CONFIG.read_text(encoding="utf-8"))


def benchmark_target(db: str) -> Target:
    return Target(BIRD_DB, db, bird.config()["bird_minidev"]["time_zone"], schema_role(db))


def agent_limits(cfg: dict | None = None) -> Limits:
    a = (cfg or config())["agent"]
    return Limits(max_rows=a["max_rows"], timeout_s=a["timeout_s"])


def evaluation_limits(cfg: dict | None = None) -> Limits:
    e = (cfg or config())["evaluation"]
    return Limits(max_rows=e["max_rows"], timeout_s=e["timeout_s"], count_total=False)


def _check_input(name: str, args: Any) -> list[str]:
    """The input against the tool's schema: required keys, no others, basic types."""
    schema = next(t["input_schema"] for t in TOOLS if t["name"] == name)
    if not isinstance(args, dict):
        return ["the input must be a JSON object"]
    problems = [f"missing {k!r}" for k in schema.get("required", []) if k not in args]
    problems += [f"unexpected {k!r}" for k in args if k not in schema["properties"]]
    kinds = {"string": str, "integer": int, "object": dict}
    for k, v in args.items():
        want = schema["properties"].get(k, {}).get("type")
        if want and not (isinstance(v, kinds[want]) and not isinstance(v, bool)):
            problems.append(f"{k!r} must be a{'n' if want[0] in 'aeiou' else ''} {want}")
    return problems


class Toolbox:
    def __init__(self, db: str, cfg: dict | None = None):
        cfg = cfg or config()
        self.db = db
        self.cfg = cfg
        self.schema = SchemaTools(db)
        self.executor = ReadOnlyExecutor(benchmark_target(db))
        self.sql = SqlTools(
            QueryGuard(db, set(self.schema.tables)),
            self.executor,
            agent_limits(cfg),
            cfg["agent"]["max_cell_chars"],
            cfg["sample_rows"]["default_rows"],
            cfg["sample_rows"]["max_rows"],
        )

    def __enter__(self) -> Toolbox:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        self.executor.close()

    def call(self, name: str, args: Any, result_columns: list[str] | None = None) -> dict:
        """Run one tool. `result_columns`: for validate_chart, the columns of the result the
        chart draws (the caller knows which query a chart belongs to)."""
        start = time.perf_counter()
        if name not in TOOL_NAMES:
            out = refused([f"unknown tool {name!r}"])
        elif problems := _check_input(name, args):
            out = refused(problems)
        elif name == "list_tables":
            out = self.schema.list_tables()
        elif name == "describe_table":
            out = self.schema.describe_table(args["table"])
        elif name == "sample_rows":
            out = self.sql.sample_rows(args["table"], args.get("n"))
        elif name == "run_sql":
            out = self.sql.run_sql(args["sql"])
        else:  # validate_chart
            out = validate_chart(
                args["spec"], result_columns or [], self.cfg["chart"]["max_spec_chars"]
            )
        out["seconds"] = round(time.perf_counter() - start, 6)
        return out
