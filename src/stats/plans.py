"""An analysis plan, and the data it pulls: the SQL that returns one row per unit, and what the
sandbox should do with those rows.

A plan is what the analyst writes for a statistical question (or what the reference plans of the
planted-effect checks state): the query, which of its columns is the outcome, which the groups
(or the x of a trend), the reference group, the variables to stratify by, and whether x describes
the unit itself or an area it belongs to (a district's unemployment rate describes the district,
not the client). The query runs through the same two guards as every query of the analyst (the
query guard, then the read-only executor as the schema's own role), with the evaluation limits:
the rows are for the sandbox, never shown to a model, and a pull larger than the row limit is
refused rather than cut, since an analysis of a cut result would be wrong.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from src.db.execute import Limits, QueryError, ReadOnlyExecutor
from src.db.guard import QueryGuard
from src.stats.sandbox import jsonable

ANALYSES = ("compare_groups", "trend")
OUTCOME_TYPES = ("binary", "numeric")
X_DESCRIBES = ("unit", "area")
MAX_STRATA = 2


class PlanError(ValueError):
    pass


@dataclass(frozen=True)
class Plan:
    sql: str
    analysis: str
    outcome: str
    outcome_type: str
    group: str | None = None
    reference_group: str | None = None
    x: str | None = None
    strata: tuple[str, ...] = ()
    x_describes: str = "unit"
    unit: str = ""

    def __post_init__(self) -> None:
        if self.analysis not in ANALYSES:
            raise PlanError(f"analysis must be one of {ANALYSES}")
        if self.outcome_type not in OUTCOME_TYPES:
            raise PlanError(f"outcome_type must be one of {OUTCOME_TYPES}")
        if self.analysis == "compare_groups" and not self.group:
            raise PlanError("a comparison needs its groups column")
        if self.analysis == "trend" and not self.x:
            raise PlanError("a trend needs its x column")
        if len(self.strata) > MAX_STRATA:
            raise PlanError(f"at most {MAX_STRATA} strata")
        if self.x_describes not in X_DESCRIBES:
            raise PlanError(f"x_describes must be one of {X_DESCRIBES}")

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Plan:
        keys = {f for f in cls.__dataclass_fields__}
        unknown = set(d) - keys
        if unknown:
            raise PlanError(f"unknown plan fields: {sorted(unknown)}")
        args = {k: v for k, v in d.items() if v is not None}
        if "strata" in args:
            args["strata"] = tuple(args["strata"])
        if "reference_group" in args:
            args["reference_group"] = str(args["reference_group"])
        try:
            return cls(**args)
        except TypeError as e:
            raise PlanError(str(e)) from None

    def spec(self) -> dict[str, Any]:
        """The analysis spec the sandbox reads."""
        spec: dict[str, Any] = {
            "analysis": self.analysis,
            "outcome": self.outcome,
            "outcome_type": self.outcome_type,
            "strata": list(self.strata),
        }
        if self.analysis == "compare_groups":
            spec["group"] = self.group
            spec["reference_group"] = self.reference_group
        else:
            spec["x"] = self.x
        return spec


@dataclass
class Pulled:
    columns: list[str] = field(default_factory=list)
    rows: list[list[Any]] = field(default_factory=list)
    error: dict[str, str] | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


def pull(plan: Plan, guard: QueryGuard, executor: ReadOnlyExecutor, limits: Limits) -> Pulled:
    verdict = guard.check(plan.sql)
    if not verdict.allowed:
        return Pulled(error={"kind": "refused", "message": "; ".join(verdict.reasons)})
    try:
        result = executor.execute(verdict.query, limits)
    except QueryError as e:
        return Pulled(error={"kind": e.kind, "message": e.message})
    if result.truncated:
        return Pulled(
            error={
                "kind": "too_many_rows",
                "message": f"the query returned more than {limits.max_rows:,} rows",
            }
        )
    return Pulled(
        columns=[name for name, _ in result.columns],
        rows=[[jsonable(v) for v in row] for row in result.rows],
    )


def job(job_id: str, plan: Plan, pulled: Pulled) -> dict[str, Any]:
    """The sandbox's input, its rows in a fixed order: a query without ORDER BY can return them
    in any order, and a sum taken in another order can differ in its last bits."""
    rows = sorted(pulled.rows, key=json.dumps)
    return {"id": job_id, "spec": plan.spec(), "columns": pulled.columns, "rows": rows}
