"""Scoring a prediction against a gold query in one read-only transaction, on the security
suite's schema (no benchmark data needed)."""

from __future__ import annotations

import pytest

from src.db.connection import DB_NAME
from src.db.execute import Target
from src.db.hardening import SECURITY_SCHEMA, schema_role
from src.eval.execution import PairExecutor, ScoreSettings

pytestmark = pytest.mark.db

TARGET = Target(DB_NAME, SECURITY_SCHEMA, "UTC", schema_role(SECURITY_SCHEMA))
SETTINGS = ScoreSettings(timeout_s=2, fetch_rows=100, soft_f1_max_distinct_rows=1000)


@pytest.fixture
def scorer(db_ready):
    with PairExecutor(TARGET, SETTINGS) as e:
        yield e


def test_same_result_scores_one(scorer):
    s = scorer.score(
        "SELECT n FROM numbers WHERE n <= 5 ORDER BY n DESC", "SELECT n FROM numbers WHERE n <= 5"
    )
    assert (s.ex, s.outcome, s.gold_rows, s.predicted_rows) == (1, "ok", 5, 5)
    assert s.gold_set_hash and s.seconds > 0


def test_a_prediction_streams_past_many_fetches(scorer):
    # 10,000 predicted rows over 100 fetches, three distinct values, all in the gold result
    s = scorer.score("SELECT n % 3 FROM numbers", "SELECT DISTINCT n % 3 FROM numbers")
    assert (s.ex, s.predicted_rows, s.gold_rows) == (1, 10_000, 3)


def test_both_queries_see_the_same_transaction_time(scorer):
    assert scorer.score("SELECT now()", "SELECT now()").ex == 1
    assert scorer.score("SELECT current_date", "SELECT current_date").ex == 1
    # the wall clock moves between the two queries
    assert scorer.score("SELECT clock_timestamp()", "SELECT clock_timestamp()").ex == 0


def test_a_failing_prediction_scores_zero_with_its_error(scorer):
    s = scorer.score("SELECT missing FROM numbers", "SELECT n FROM numbers")
    assert (s.ex, s.soft_f1, s.outcome, s.error_kind) == (0, 0.0, "error", "sql_error")


def test_a_failing_gold_query_is_reported_as_such(scorer):
    s = scorer.score("SELECT 1", "SELECT missing FROM numbers")
    assert (s.ex, s.outcome) == (0, "gold_error")


def test_the_pair_is_cancelled_at_the_time_limit(scorer):
    s = scorer.score(
        "SELECT count(*) FROM numbers a JOIN numbers b ON true JOIN numbers c ON true",
        "SELECT 1",
    )
    assert (s.ex, s.outcome) == (0, "timeout")
    assert s.seconds < SETTINGS.timeout_s + 3
    assert scorer.score("SELECT 1", "SELECT 1").ex == 1  # the connection is usable again


def test_documented_divergences_score_zero(scorer):
    # psycopg2 would run both statements and compare the last one's rows
    s = scorer.score("SELECT 2; SELECT 1", "SELECT 1")
    assert (s.ex, s.error_kind) == (0, "multiple_statements")
    s = scorer.score("SET search_path = public", "SELECT 1")
    assert (s.ex, s.outcome) == (0, "error")
    # a table outside the schema role's grant
    s = scorer.score("SELECT count(*) FROM secret", "SELECT 1")
    assert (s.ex, s.error_kind) == (0, "permission_denied")


def test_trailing_semicolons_and_comments_are_fine(scorer):
    assert scorer.score("SELECT count(*) FROM numbers;", "SELECT count(*) FROM numbers").ex == 1
    assert scorer.score("-- note\nSELECT count(*) FROM numbers", "SELECT 10000").ex == 1


def test_values_that_cannot_be_hashed_fail_the_comparison(scorer):
    s = scorer.score("SELECT array_agg(n) FROM numbers WHERE n < 3", "SELECT 1")
    assert (s.ex, s.soft_f1, s.outcome) == (0, 0.0, "comparison_error")
    assert "unhashable" in s.message


def test_rows_with_no_columns_fail_soft_f1_alone(scorer):
    s = scorer.score("SELECT FROM numbers LIMIT 1", "SELECT FROM numbers LIMIT 2")
    assert (s.ex, s.soft_f1, s.outcome) == (1, 0.0, "ok")
    assert s.soft_f1_error.startswith("ZeroDivisionError")


def test_no_prediction(scorer):
    s = scorer.score(None, "SELECT 1")
    assert (s.ex, s.soft_f1, s.outcome) == (0, 0.0, "no_prediction")


def test_soft_f1_gives_up_above_its_limit_while_ex_stays_exact(db_ready):
    settings = ScoreSettings(timeout_s=5, fetch_rows=1000, soft_f1_max_distinct_rows=100)
    with PairExecutor(TARGET, settings) as e:
        s = e.score("SELECT n FROM numbers", "SELECT n FROM numbers ORDER BY n DESC")
        assert (s.ex, s.soft_f1, s.predicted_rows) == (1, None, 10_000)
        # EX is settled at the first row outside the gold, and Soft-F1 was given up: the
        # rest of the prediction is not fetched
        s = e.score("SELECT n FROM numbers ORDER BY n", "SELECT n FROM numbers WHERE n = 1")
        assert (s.ex, s.soft_f1) == (0, None)
        assert s.predicted_rows < 10_000
