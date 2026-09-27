"""The loaded BIRD benchmark: it matches its committed snapshots and published counts, the
agent's role can read it but not change it, and the dictionaries do not give answers away.

Needs the benchmark loaded in its schema layout and its files in data/raw (marker `bird`).
"""

from __future__ import annotations

import importlib.util
import json
import re

import pytest
from psycopg import errors

from src.data import bird
from src.db.connection import ADMIN_ROLE, AGENT_ROLE, BIRD_DB, connect
from src.dictionary import model
from src.dictionary.birth_number import decode_birth_number, encode_birth_number
from src.dictionary.snapshot import SNAPSHOT_DIR, snapshot_database

pytestmark = pytest.mark.bird

DATABASES = sorted(p.stem for p in SNAPSHOT_DIR.glob("*.json"))


@pytest.fixture(scope="module")
def agent(bird_ready):
    with connect(AGENT_ROLE, BIRD_DB, autocommit=True) as conn:
        yield conn


@pytest.mark.parametrize("db", DATABASES)
def test_snapshot_matches_the_database(agent, db):
    committed = model.load_snapshot(db)
    assert snapshot_database(agent, db, profile=committed["profiled"]) == committed


def test_czech_bank_row_counts_match_the_published_ones(agent):
    published = bird.config()["financial_published_rows"]
    for table, n in published.items():
        assert agent.execute(f'SELECT count(*) FROM financial."{table}"').fetchone()[0] == n, table


def test_every_table_is_in_its_databases_schema(agent):
    rows = agent.execute(
        "SELECT table_schema, table_name FROM information_schema.tables "
        "WHERE table_schema NOT IN ('pg_catalog', 'information_schema')"
    ).fetchall()
    assert {t: s for s, t in rows} == bird.table_databases()


def test_timestamps_read_in_the_benchmarks_time_zone(agent):
    """The dump's timestamps were written at UTC+8; the benchmark's questions use those times."""
    assert (
        agent.execute("SHOW timezone").fetchone()[0] == bird.config()["bird_minidev"]["time_zone"]
    )
    row = agent.execute(
        "SELECT count(*) FROM codebase_community.comments "
        "WHERE userid = 3025 AND creationdate = '2014-04-23 20:29:39'"
    ).fetchone()
    assert row == (1,)


def test_birth_number_rule_round_trips_on_every_client(agent):
    """Re-encode each client's decoded sex and birth date into a birth number and decode it."""
    clients = agent.execute("SELECT gender, birth_date FROM financial.client").fetchall()
    assert len(clients) == 5369
    for sex, born in clients:
        number = encode_birth_number(sex, born)
        assert (int(number[2:4]) > 50) == (sex == "F")
        assert decode_birth_number(number) == (sex, born)


# --- the agent's role on the benchmark database ------------------------------------------

WRITES = [
    "INSERT INTO financial.account (account_id) VALUES (999999)",
    "UPDATE financial.loan SET status = 'A'",
    "DELETE FROM financial.trans",
    "TRUNCATE financial.card",
    "ALTER TABLE financial.client ADD COLUMN extra integer",
    "DROP TABLE financial.district",
    "CREATE TABLE financial.new_table (id integer)",
    "CREATE TABLE public.new_table (id integer)",
    "CREATE SCHEMA new_schema",
    "CREATE TEMP TABLE scratch (id integer)",
    "CREATE INDEX new_index ON financial.trans (amount)",
]


def _loans() -> tuple:
    with connect(ADMIN_ROLE, BIRD_DB) as conn:
        return conn.execute(
            "SELECT count(*), md5(string_agg(status, '' ORDER BY loan_id)) FROM financial.loan"
        ).fetchone()


@pytest.fixture(scope="module")
def loans_before(bird_ready):
    return _loans()


@pytest.mark.parametrize("sql", WRITES)
def test_write_fails_in_a_default_session(agent, loans_before, sql):
    with pytest.raises((errors.ReadOnlySqlTransaction, errors.InsufficientPrivilege)):
        agent.execute(sql)
    assert _loans() == loans_before


@pytest.mark.parametrize("sql", WRITES)
def test_write_fails_on_privileges_alone(bird_ready, loans_before, sql):
    with connect(AGENT_ROLE, BIRD_DB, autocommit=True) as conn:
        conn.execute("SET default_transaction_read_only = off")
        with pytest.raises(errors.InsufficientPrivilege):
            conn.execute(sql)
    assert _loans() == loans_before


@pytest.mark.parametrize("db", DATABASES)
def test_agent_can_read_every_schema(agent, db):
    table = next(iter(model.load_snapshot(db)["tables"]))
    agent.execute(f'SELECT * FROM "{db}"."{table}" LIMIT 1')


# --- the dictionaries -------------------------------------------------------------------------


def _normalise(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def test_no_dictionary_contains_a_benchmark_question_or_its_gold_sql(bird_ready):
    """The dictionary describes columns; it must never carry a question or its answer."""
    text = {
        db: _normalise((model.DICTIONARY_DIR / f"{db}.yaml").read_text(encoding="utf-8"))
        for db in DATABASES
    }
    for q in bird.questions():
        for needle in (q["question"], q["SQL"]):
            assert _normalise(needle) not in text[q["db_id"]], f"question {q['question_id']}"


def test_converted_dictionaries_are_what_the_converter_writes(bird_ready):
    """They are regenerated, never edited by hand."""
    spec = importlib.util.spec_from_file_location(
        "convert", bird.ROOT / "scripts/14_convert_descriptions.py"
    )
    convert = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(convert)
    for db in DATABASES:
        if db in convert.HAND_WRITTEN:
            continue
        assert convert.convert(db)[0] == model.load(db), db


def test_gold_results_were_recorded_for_every_question(bird_ready):
    gold = json.loads(
        (bird.ROOT / "results/metrics/gold_execution.json").read_text(encoding="utf-8")
    )
    ids = {q["question_id"] for q in bird.questions()}
    assert {r["question_id"] for r in gold["questions"]} == ids
    assert gold["questions_sha256"] == bird.config()["bird_minidev"]["questions"]["sha256"]
