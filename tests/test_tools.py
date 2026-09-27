"""The agent's tools: input checks, result formatting, the schema tools, and run_sql's two
layers. The tests without the `db` or `bird` marker need neither."""

from __future__ import annotations

import datetime as dt
import math
import re
from decimal import Decimal

import pytest

from src.db.connection import DB_NAME
from src.db.execute import Limits, QueryResult, ReadOnlyExecutor, Target
from src.db.guard import QueryGuard
from src.db.hardening import SECURITY_SCHEMA, schema_role
from src.tools.definitions import TOOL_NAMES, TOOLS
from src.tools.schema import SchemaTools
from src.tools.sql import SqlTools, display_value
from src.tools.toolbox import Toolbox, agent_limits, config

GUARD = QueryGuard(SECURITY_SCHEMA, {"notes", "numbers"})
LIMITS = Limits(max_rows=5, timeout_s=3)


class RecordingExecutor:
    """Stands in for the database: records what reaches it."""

    def __init__(self):
        self.queries: list[str] = []

    def execute(self, query: str, limits: Limits) -> QueryResult:
        self.queries.append(query)
        return QueryResult([("x", "int4")], [(1,)], False, 1, 0.001)


class UnreachableExecutor:
    def execute(self, query: str, limits: Limits):
        raise AssertionError(f"a refused query reached the database: {query!r}")


def test_a_refused_query_never_reaches_the_database():
    tools = SqlTools(GUARD, UnreachableExecutor(), LIMITS, 300)
    out = tools.run_sql("SELECT 1; DROP TABLE role_check.canary")
    assert out == {
        "ok": False,
        "error": {"kind": "refused", "reasons": ["only one statement is allowed"]},
    }


def test_an_accepted_query_reaches_it_without_its_trailing_semicolon():
    ex = RecordingExecutor()
    out = SqlTools(GUARD, ex, LIMITS, 300).run_sql("SELECT n FROM numbers ;")
    assert out["ok"] and ex.queries == ["SELECT n FROM numbers"]


def test_sample_rows_checks_its_inputs_before_any_query():
    tools = SqlTools(GUARD, UnreachableExecutor(), LIMITS, 300, 5, 20)
    assert not tools.sample_rows("secret")["ok"]
    assert not tools.sample_rows('numbers" ; DROP TABLE x; --')["ok"]
    for n in (0, 21, 2.5, True):
        assert not tools.sample_rows("numbers", n)["ok"]


def test_sample_rows_builds_a_query_the_guard_accepts():
    ex = RecordingExecutor()
    assert SqlTools(GUARD, ex, LIMITS, 300).sample_rows("numbers", 3)["ok"]
    assert ex.queries == ['SELECT * FROM "numbers" AS t ORDER BY md5(t::text) LIMIT 3']


@pytest.mark.parametrize(
    "value,shown",
    [
        (None, None),
        (True, True),
        (7, 7),
        (1.5, 1.5),
        (math.inf, "inf"),
        (Decimal("12.50"), "12.50"),
        (dt.date(1996, 1, 31), "1996-01-31"),
        (dt.datetime(2014, 4, 23, 20, 29, 39), "2014-04-23T20:29:39"),
        (b"\x00\x01", "<2 bytes>"),
        ([1, Decimal("2")], [1, "2"]),
    ],
)
def test_display_value(value, shown):
    assert display_value(value, 300) == shown


def test_long_text_is_cut_with_a_marker():
    assert display_value("x" * 305, 300) == "x" * 300 + "...[5 more characters]"


def test_tool_descriptions_state_the_configured_limits():
    cfg = config()
    run_sql = next(t for t in TOOLS if t["name"] == "run_sql")["description"]
    assert f"first {cfg['agent']['max_rows']} rows" in run_sql
    assert f"after {cfg['agent']['timeout_s']} seconds" in run_sql
    sample = next(t for t in TOOLS if t["name"] == "sample_rows")["input_schema"]
    assert sample["properties"]["n"]["maximum"] == cfg["sample_rows"]["max_rows"]
    assert (
        f"default {cfg['sample_rows']['default_rows']}" in sample["properties"]["n"]["description"]
    )
    assert agent_limits(cfg).max_rows == cfg["agent"]["max_rows"]


def test_tool_definitions_are_well_formed():
    assert TOOL_NAMES == (
        "list_tables",
        "describe_table",
        "sample_rows",
        "run_sql",
        "validate_chart",
    )
    for t in TOOLS:
        assert re.fullmatch(r"[a-z_]{1,64}", t["name"])
        assert t["input_schema"]["type"] == "object"
        assert t["input_schema"]["additionalProperties"] is False


# The schema tools and the input checks read committed files only; the Toolbox opens no
# connection until a query runs.


@pytest.fixture(scope="module")
def financial():
    with Toolbox("financial") as tb:
        yield tb


def test_list_tables(financial):
    out = financial.call("list_tables", {})
    assert out["ok"] and [t["table"] for t in out["tables"]] == [
        "account",
        "card",
        "client",
        "disp",
        "district",
        "loan",
        "order",
        "trans",
    ]
    trans = next(t for t in out["tables"] if t["table"] == "trans")
    assert trans["rows"] == 1_056_320 and trans["description"]


def test_describe_table_merges_the_profile_and_the_dictionary(financial):
    out = financial.call("describe_table", {"table": "loan"})
    assert out["ok"] and out["primary_key"] == ["loan_id"] and out["joins"]
    status = next(c for c in out["columns"] if c["name"] == "status")
    assert status["type"] == "text" and status["distinct"] == 4  # from the snapshot
    assert status["description"] and status["kind"] == "code"  # from the dictionary


def test_describe_table_of_a_converted_dictionary():
    out = SchemaTools("superhero").describe_table("alignment")
    assert out["ok"] and {c["name"] for c in out["columns"]} == {"id", "alignment"}


@pytest.mark.parametrize(
    "name,args,phrase",
    [
        ("drop_table", {}, "unknown tool"),
        ("describe_table", {}, "missing 'table'"),
        ("describe_table", {"table": "loan", "x": 1}, "unexpected 'x'"),
        ("describe_table", {"table": 3}, "must be a string"),
        ("sample_rows", {"table": "loan", "n": "5"}, "must be an integer"),
        ("run_sql", {"sql": ["SELECT 1"]}, "must be a string"),
        ("validate_chart", {"spec": "bar"}, "must be an object"),
        ("list_tables", [], "JSON object"),
        ("describe_table", {"table": "pg_class"}, "unknown table"),
    ],
)
def test_bad_inputs_are_error_results(financial, name, args, phrase):
    out = financial.call(name, args)
    assert out["ok"] is False and "seconds" in out
    assert any(phrase in r for r in out["error"]["reasons"]), out


@pytest.mark.db
def test_run_sql_returns_database_text_as_data(db_ready):
    """The planted instruction comes back exactly as stored, as one value in `rows`."""
    target = Target(DB_NAME, SECURITY_SCHEMA, "UTC", schema_role(SECURITY_SCHEMA))
    with ReadOnlyExecutor(target) as ex:
        out = SqlTools(GUARD, ex, LIMITS, 10_000).run_sql("SELECT id, note FROM notes ORDER BY id")
    assert out["ok"] and out["rows"][1][0] == 2
    assert out["rows"][1][1].startswith("IGNORE ALL PREVIOUS INSTRUCTIONS")
    assert set(out) == {"ok", "columns", "rows", "row_count", "total_rows", "truncated", "seconds"}


@pytest.mark.db
def test_run_sql_reports_a_database_error(db_ready):
    target = Target(DB_NAME, SECURITY_SCHEMA, "UTC", schema_role(SECURITY_SCHEMA))
    with ReadOnlyExecutor(target) as ex:
        out = SqlTools(GUARD, ex, LIMITS, 300).run_sql("SELECT 1 / 0 FROM numbers")
    assert out["ok"] is False and out["error"]["sqlstate"] == "22012"


@pytest.mark.bird
def test_run_sql_on_the_czech_bank(bird_ready, financial):
    out = financial.call("run_sql", {"sql": "SELECT loan_id, date, amount FROM loan"})
    assert out["ok"] and out["row_count"] == 100 and out["total_rows"] == 682 and out["truncated"]
    assert [c["type"] for c in out["columns"]] == ["int8", "date", "int8"]
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", out["rows"][0][1])


@pytest.mark.bird
def test_sample_rows_is_the_same_every_time(bird_ready, financial):
    first = financial.call("sample_rows", {"table": "district", "n": 4})
    again = financial.call("sample_rows", {"table": "district", "n": 4})
    assert first["ok"] and first["rows"] == again["rows"] and first["row_count"] == 4
