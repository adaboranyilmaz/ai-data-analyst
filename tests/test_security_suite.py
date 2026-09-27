"""The security suite: every attack is stopped by the query guard alone, by the database alone,
and by the role's privileges alone. The only accepted exception is object names, which
PostgreSQL shows to any role that can connect: the database configurations record them as
`disclosed`, and the guard must refuse them.

The suite runs once for the session (about a minute: resource attacks wait out the time
limit); each attack and configuration is then its own test. Attacks on the benchmark's tables
skip where the benchmark is not loaded (CI)."""

from __future__ import annotations

import pytest
import yaml

from src.db.security_suite import (
    ATTACKS_FILE,
    BLOCKED,
    DOCUMENTED_EXCEPTION,
    MODES,
    bird_available,
    run_suite,
)

ATTACKS = yaml.safe_load(ATTACKS_FILE.read_text(encoding="utf-8"))["attacks"]
GOALS = {"write", "read", "session", "resource", "names"}
# the kinds of attack the suite must cover, at least a few of each
REQUIRED_CATEGORIES = {
    "destructive",
    "multi_statement",
    "obfuscation",
    "catalog_snooping",
    "confinement",
    "session",
    "resource_exhaustion",
    "prompt_injection",
}


def test_the_attack_file_is_well_formed():
    ids = [a["id"] for a in ATTACKS]
    assert len(ids) == len(set(ids))
    assert {a["goal"] for a in ATTACKS} <= GOALS
    assert {a.get("target", "analyst") for a in ATTACKS} <= {"analyst", "bird"}
    counts = {c: sum(a["category"] == c for a in ATTACKS) for c in REQUIRED_CATEGORIES}
    assert all(n >= 3 for n in counts.values()), counts


@pytest.fixture(scope="session")
def suite(db_ready):
    return run_suite()


@pytest.mark.db
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("attack", ATTACKS, ids=lambda a: a["id"])
def test_attack_is_stopped(suite, attack, mode):
    record = next((r for r in suite["records"] if r["id"] == attack["id"]), None)
    if record is None:
        assert attack.get("target") == "bird" and not bird_available()
        pytest.skip("the benchmark is not loaded")
    result = record["modes"][mode]
    if attack["goal"] == "names" and mode != "parser":
        assert result["outcome"] in BLOCKED | {DOCUMENTED_EXCEPTION}, result
    else:
        assert result["outcome"] in BLOCKED, result


@pytest.mark.db
def test_the_planted_instruction_is_returned_as_data(suite):
    assert suite["injection_row"]["returned_as_a_row_value"]


@pytest.mark.db
def test_the_suite_summary_agrees_with_its_records(suite):
    assert suite["every_attack_stopped_by_each_layer"]
    for mode in MODES:
        s = suite["summary"][mode]
        assert s["breach"] == 0
        assert s["blocked"] + s["documented_exception"] == s["attacks"] == len(suite["records"])
    assert all(e["modes"] == ["database", "privileges"] for e in suite["documented_exceptions"])
