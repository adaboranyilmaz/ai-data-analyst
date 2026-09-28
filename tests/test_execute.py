"""The read-only executor: limits the query cannot lift, settings fixed per call, the schema
role's privileges, and the benchmark's gold results reproduced through it."""

from __future__ import annotations

import time

import psycopg
import pytest

from src.db.connection import ADMIN_ROLE, BIRD_DB, DB_NAME, connect
from src.db.execute import Limits, QueryError, ReadOnlyExecutor, Target
from src.db.hardening import SECURITY_SCHEMA, schema_role

pytestmark = pytest.mark.db

TARGET = Target(DB_NAME, SECURITY_SCHEMA, "UTC", schema_role(SECURITY_SCHEMA))
LIMITS = Limits(max_rows=10, timeout_s=2)


@pytest.fixture
def ex(db_ready):
    with ReadOnlyExecutor(TARGET) as e:
        yield e


def test_returns_rows_columns_and_timing(ex):
    r = ex.execute("SELECT n, n::text AS s FROM numbers WHERE n <= 3 ORDER BY n", LIMITS)
    assert r.rows == [(1, "1"), (2, "2"), (3, "3")]
    assert r.columns == [("n", "int4"), ("s", "text")]
    assert (r.truncated, r.total_rows) == (False, 3)
    assert r.seconds > 0


def test_row_cap_with_the_total_counted_on_the_server(ex):
    r = ex.execute("SELECT n FROM numbers ORDER BY n", LIMITS)
    assert len(r.rows) == 10 and r.rows[-1] == (10,)
    assert (r.truncated, r.total_rows) == (True, 10_000)


def test_row_cap_without_counting(ex):
    r = ex.execute("SELECT n FROM numbers", Limits(5, 2, count_total=False))
    assert len(r.rows) == 5 and r.truncated and r.total_rows is None


def test_a_count_that_runs_out_of_time_leaves_the_rows(ex):
    r = ex.execute(
        "SELECT a.n FROM numbers a JOIN numbers b ON true JOIN numbers c ON true",
        Limits(3, 1.5),
    )
    assert len(r.rows) == 3 and r.truncated and r.total_rows is None


def test_the_client_cancels_at_the_time_limit(ex):
    start = time.perf_counter()
    with pytest.raises(QueryError) as e:
        ex.execute(
            "SELECT count(*) FROM numbers a JOIN numbers b ON true JOIN numbers c ON true",
            Limits(1, 1.0),
        )
    assert e.value.kind == "timeout"
    assert time.perf_counter() - start < 2.5
    assert ex.execute("SELECT 1", LIMITS).rows == [(1,)]  # the session is usable afterwards


def test_queries_run_as_the_schema_role(ex):
    assert ex.execute("SELECT current_user::text", LIMITS).rows == [(TARGET.role,)]
    with pytest.raises(QueryError) as e:
        ex.execute("SELECT note FROM role_check.canary", LIMITS)  # analyst_ro could read it
    assert e.value.kind == "permission_denied"


def test_settings_are_the_targets_on_every_call(ex):
    hour = "SELECT extract(hour FROM timestamptz '2000-01-01 00:00:00+00')::int"
    assert ex.execute(hour, LIMITS).rows == [(0,)]
    assert ex.execute("SELECT 'a\\b'", LIMITS).rows == [("a\\b",)]  # a backslash is a character
    with ReadOnlyExecutor(Target(DB_NAME, SECURITY_SCHEMA, "Asia/Shanghai", TARGET.role)) as e2:
        assert e2.execute(hour, LIMITS).rows == [(8,)]
    assert ex.execute("SELECT count(*) FROM numbers", LIMITS).rows == [(10_000,)]  # search_path


def test_every_scan_starts_at_the_first_block(ex):
    # current_setting() is closed to the role, so the setting is read with SHOW inside the
    # executor's own transaction, after its per-call settings
    conn = ex._connection()
    with conn.transaction():
        ex._set_local(conn, LIMITS)
        assert conn.execute("SHOW synchronize_seqscans").fetchone() == ("off",)
        raise psycopg.Rollback


@pytest.mark.bird
def test_rows_do_not_depend_on_another_sessions_scan(bird_ready):
    """By default a scan of a large table joins a scan already under way, mid-table: a LIMIT
    without ORDER BY then returns other rows. `trans` (about 100 MB) is large enough for
    PostgreSQL to synchronize scans of it with the default shared_buffers."""
    from src.tools.toolbox import benchmark_target

    query = "SELECT trans_id FROM trans LIMIT 3"
    with ReadOnlyExecutor(benchmark_target("financial")) as e:
        alone = e.execute(query, LIMITS).rows
        with connect(ADMIN_ROLE, BIRD_DB) as other, connect(ADMIN_ROLE, BIRD_DB) as plain:
            for conn in (other, plain):
                conn.execute("SET search_path = financial")
                conn.execute("SET max_parallel_workers_per_gather = 0")
            with other.transaction(), other.cursor(name="midway") as cur:
                cur.execute("SELECT trans_id FROM trans")
                cur.fetchmany(400_000)  # a scan under way, a good part into the table
                assert plain.execute(query).fetchall() != alone  # the default joins it
                assert e.execute(query, LIMITS).rows == alone  # the executor does not


@pytest.mark.parametrize(
    "sql,kind",
    [
        ("SELECT 1; SELECT 2", "multiple_statements"),
        ("SELEC 1", "syntax_error"),
        ("SET TimeZone = 'UTC'", "syntax_error"),  # the cursor takes only a query
        ("DELETE FROM numbers", "syntax_error"),
        ("WITH d AS (DELETE FROM numbers RETURNING *) SELECT * FROM d", "not_supported"),
        ("SELECT token FROM secret", "permission_denied"),
        ("SELECT pg_sleep(1)", "permission_denied"),
        ("SELECT * FROM numbers FOR UPDATE", "read_only"),
        ("SELECT 1 / 0", "sql_error"),
    ],
)
def test_errors_are_classified(ex, sql, kind):
    with pytest.raises(QueryError) as e:
        ex.execute(sql, LIMITS)
    assert e.value.kind == kind, e.value


def test_without_the_read_only_transaction_the_privileges_refuse_locks(db_ready):
    """FOR UPDATE needs UPDATE privilege, which the schema role lacks."""
    with ReadOnlyExecutor(TARGET, transaction_read_only=False) as e:
        with pytest.raises(QueryError) as err:
            e.execute("SELECT * FROM numbers FOR UPDATE", LIMITS)
    assert err.value.kind == "permission_denied"


def test_a_missing_database_is_a_connection_error(db_ready):
    with pytest.raises(QueryError) as e:
        ReadOnlyExecutor(Target("no_such_db", "x", "UTC")).execute("SELECT 1", LIMITS)
    assert e.value.kind == "connection"


# Gold queries whose results depend on the execution settings: timestamps written at UTC+8
# (#532, #533, #563, #565, #683), and float sums and averages that change with parallel
# workers (#1473, #1482).
SENSITIVE = [532, 533, 563, 565, 683, 1473, 1482]


@pytest.mark.bird
def test_gold_results_reproduce_through_the_executor(bird_ready):
    import json
    from pathlib import Path

    from src.data import bird
    from src.db.values import set_hash
    from src.tools.toolbox import benchmark_target, evaluation_limits

    gold = {
        r["question_id"]: r
        for r in json.loads(Path("results/metrics/gold_execution.json").read_text("utf-8"))[
            "questions"
        ]
    }
    chosen = [
        q for q in bird.questions() if q["db_id"] == "financial" or q["question_id"] in SENSITIVE
    ]
    assert len(chosen) == 32 + len(SENSITIVE)
    for q in chosen:
        with ReadOnlyExecutor(benchmark_target(q["db_id"])) as e:
            for _ in range(2):  # twice: the float results must not change between runs
                r = e.execute(q["SQL"], evaluation_limits())
                g = gold[q["question_id"]]
                assert ([list(c) for c in r.columns], len(r.rows), set_hash(r.rows)) == (
                    g["columns"],
                    g["rows"],
                    g["set_hash"],
                ), q["question_id"]
