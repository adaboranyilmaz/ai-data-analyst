"""The query guard: what it accepts, what it refuses and why. No database needed."""

from __future__ import annotations

import pytest
import yaml

from src.db.guard import QueryGuard, scan
from src.db.security_suite import ATTACKS_FILE

GUARD = QueryGuard(
    "financial", {"account", "card", "client", "disp", "district", "loan", "order", "trans"}
)

ACCEPTED = [
    "SELECT count(*) FROM loan",
    "SELECT * FROM financial.loan",  # its own schema, qualified
    "SELECT account_id FROM \"order\" WHERE k_symbol = 'SIPO'",
    "SELECT a.account_id FROM account a JOIN loan l ON l.account_id = a.account_id",
    "SELECT a.account_id FROM account a LEFT JOIN disp d USING (account_id)",
    "WITH big AS (SELECT * FROM loan WHERE amount > 100000) SELECT count(*) FROM big",
    "SELECT district_id, rank() OVER (ORDER BY a11 DESC) FROM district",
    "SELECT account_id FROM loan UNION SELECT account_id FROM card",
    "SELECT account_id FROM loan EXCEPT SELECT account_id FROM disp",
    "SELECT (SELECT max(amount) FROM loan)",
    "SELECT * FROM district WHERE a2 = 'Praha -- not a comment; nor this'",
    "SELECT 'Jesenik', 'Hl.m. Praha', 'příjem' FROM district",  # non-ASCII inside a string
    "SELECT 'it''s' FROM loan",
    "SELECT n FROM generate_series(1, 12) AS g(n)",
    "SELECT lpad(account_id::text, 8, '0'), repeat('-', 3) FROM loan",
    "SELECT CAST(amount AS numeric) / NULLIF(duration, 0) FROM loan",
    "SELECT count(*) FROM loan;",  # one trailing semicolon
    "SELECT count(*) FROM loan ;  \n",
    "(SELECT 1)",
]

REFUSED = {  # query -> a phrase the reasons must contain
    "DROP TABLE loan": "only a SELECT",
    "DELETE FROM loan": "only a SELECT",
    "SELECT 1; SELECT 2": "only one statement",
    "SELECT 1 -- x": "comments",
    "SELECT 1 /* x */": "comments",
    "SELECT $$x$$": "$ is not allowed",
    "SELECT 'a\\b'": "backslashes",
    "SELECT E'x'": "E'' strings",
    'SELECT U&"x"': "U& escapes",
    "SELECT 1​": "U+200B",
    "SELECT '‮'": "U+202E",
    "SELECT ＳＥＬＥＣＴ": "only ASCII",
    "SELECT 'x": "unterminated",
    "SELECT * FROM secret": "unknown table",
    "SELECT * FROM formula_1.drivers": "only tables of financial",
    "SELECT * FROM bird.financial.loan": "no database qualifier",
    "SELECT * FROM pg_class": "system objects",
    "SELECT * FROM information_schema.tables": "information_schema",
    "SELECT 'loan'::regclass": "regclass",
    "SELECT pg_sleep(1)": "system objects",
    "SELECT set_config('TimeZone', 'UTC', false)": "set_config",
    "SELECT current_setting('TimeZone')": "current_setting",
    "SELECT version()": "version",
    "SELECT query_to_xml('select 1', true, false, '')": "query_to_xml",
    "SELECT lo_create(0)": "lo_create",
    "SELECT nextval('s')": "nextval",
    "SELECT has_table_privilege('loan', 'SELECT')": "has_table_privilege",
    "SELECT * FROM loan FOR UPDATE": "FOR UPDATE",
    "SELECT * INTO copy FROM loan": "SELECT INTO",
    "WITH RECURSIVE r(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM r) SELECT * FROM r": (
        "RECURSIVE"
    ),
    "WITH d AS (DELETE FROM loan RETURNING *) SELECT * FROM d": "DELETE is not allowed",
    "SELECT * FROM loan CROSS JOIN trans": "CROSS JOIN",
    "SELECT * FROM loan, trans": "comma joins",
    "SELECT * FROM loan NATURAL JOIN account": "NATURAL JOIN",
    "SELECT * FROM loan l JOIN LATERAL (SELECT 1) s ON true": "LATERAL",
    "SELECT generate_series(1, 1000000000)": "limited to",
    "SELECT generate_series(1, amount) FROM loan": "literal bounds",
    "SELECT repeat('x', 100000000)": "repeat()",
    "SELECT lpad('x', amount::int) FROM loan": "lpad()/rpad()",
    "SET TimeZone = 'UTC'": "only a SELECT",
    "SHOW TimeZone": "only a SELECT",
    "COPY loan TO STDOUT": "only a SELECT",
    "VALUES (1)": "only a SELECT",
    "": "empty",
    "   ;  ": "empty",
    "SELECT FROM WHERE (": "could not be parsed",
    "SELECT 1" + " " * 20_000: "longer than",
}


@pytest.mark.parametrize("query", ACCEPTED)
def test_accepts(query):
    v = GUARD.check(query)
    assert v.allowed, v.reasons
    assert v.query == query.strip().rstrip(";").rstrip()


@pytest.mark.parametrize("query,phrase", REFUSED.items(), ids=range(len(REFUSED)))
def test_refuses_with_a_reason(query, phrase):
    v = GUARD.check(query)
    assert not v.allowed
    assert any(phrase in r for r in v.reasons), v.reasons


def test_quoted_table_names_are_case_sensitive():
    assert GUARD.check('SELECT * FROM "order"').allowed
    assert not GUARD.check('SELECT * FROM "Order"').allowed
    assert GUARD.check("SELECT * FROM LOAN").allowed  # unquoted names fold to lower case


def test_cte_names_are_tables_only_within_their_query():
    assert GUARD.check("WITH x AS (SELECT 1 AS a) SELECT a FROM x").allowed
    assert not GUARD.check("SELECT a FROM x").allowed
    assert not GUARD.check("WITH x AS (SELECT 1 AS a) SELECT a FROM financial.x").allowed


def test_a_comment_is_skipped_as_postgresql_skips_it():
    """Reasons after a comment are about the text PostgreSQL would run: nested block comments
    end where PostgreSQL ends them, and a line comment hides the rest of its line only."""
    s = scan("SELECT 1 /* a /* b */ ; c */ FROM loan")
    assert s.problems == ["comments are not allowed"]
    s = scan("SELECT 1 -- '\nFROM loan")
    assert s.problems == ["comments are not allowed"]


def test_every_reason_is_given_at_once():
    v = GUARD.check("SELECT pg_sleep(1), version() FROM secret")
    assert not v.allowed and len(v.reasons) >= 2


def test_for_benchmark_db_reads_the_committed_snapshot():
    g = QueryGuard.for_benchmark_db("financial")
    assert g.schema == "financial"
    assert g.tables == GUARD.tables


# Resource attacks the guard is expected to accept: legitimate-looking joins whose cost no
# checker can judge. The execution layer's limits contain them (tests/test_security_suite.py).
EXPECTED_TO_PASS_THE_GUARD = {
    "resource-inequality-join",
    "resource-join-on-true",
    "resource-sort-spill",
    "resource-trans-self-join",
    "resource-trans-all-rows",
}


def _attacks():
    return yaml.safe_load(ATTACKS_FILE.read_text(encoding="utf-8"))["attacks"]


@pytest.mark.parametrize("attack", _attacks(), ids=lambda a: a["id"])
def test_guard_refuses_every_attack_it_can_judge(attack):
    guard = (
        QueryGuard.for_benchmark_db("financial")
        if attack.get("target") == "bird"
        else QueryGuard("security_check", {"notes", "numbers"})
    )
    v = guard.check(attack["sql"])
    assert v.allowed == (attack["id"] in EXPECTED_TO_PASS_THE_GUARD), v.reasons


@pytest.mark.bird
def test_accepts_every_gold_query_unchanged(bird_ready):
    from src.data import bird

    guards: dict[str, QueryGuard] = {}
    for q in bird.questions():
        g = guards.setdefault(q["db_id"], QueryGuard.for_benchmark_db(q["db_id"]))
        v = g.check(q["SQL"])
        assert v.allowed, (q["question_id"], v.reasons)
        assert v.query == q["SQL"].strip()
