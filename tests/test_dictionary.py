"""The data dictionaries are complete: checked against the committed schema snapshots, so these
tests run without the database (tests/test_bird_data.py checks the snapshots against it).

- every table and column of every BIRD database has an entry, and no entry is stale;
- the Czech bank dictionary gives every column an English name, a kind and a description,
  translates every code value present in the data, and explains every NULL;
- the checks themselves catch each kind of gap (mutation tests);
- the birth-number rule that encodes a client's sex decodes correctly.
"""

from __future__ import annotations

import copy
import datetime as dt

import pytest

from src.dictionary import model
from src.dictionary.birth_number import decode_birth_number, encode_birth_number
from src.dictionary.snapshot import SNAPSHOT_DIR

BIRD_DATABASES = [
    "california_schools", "card_games", "codebase_community", "debit_card_specializing",
    "european_football_2", "financial", "formula_1", "student_club", "superhero",
    "thrombosis_prediction", "toxicology",
]  # fmt: skip


def test_every_bird_database_has_a_snapshot_and_a_dictionary():
    assert sorted(p.stem for p in SNAPSHOT_DIR.glob("*.json")) == BIRD_DATABASES
    assert sorted(p.stem for p in model.DICTIONARY_DIR.glob("*.yaml")) == BIRD_DATABASES


@pytest.mark.parametrize("db", BIRD_DATABASES)
def test_dictionary_is_complete(db):
    d, snap = model.load(db), model.load_snapshot(db)
    assert d["database"] == db
    assert model.structure_problems(d) == []
    assert model.coverage_problems(d, snap) == []
    assert model.code_problems(d, snap) == []


def test_only_the_czech_bank_is_hand_written_and_profiled():
    for db in BIRD_DATABASES:
        hand = db == "financial"
        assert (model.load(db)["origin"] == "hand-written") is hand
        assert model.load_snapshot(db)["profiled"] is hand


# --- the Czech bank dictionary ---------------------------------------------------------


@pytest.fixture(scope="module")
def financial():
    return model.load("financial"), model.load_snapshot("financial")


def test_money_columns_are_in_koruna(financial):
    d, _ = financial
    cols = {f"{t}.{c}": e for t, table in d["tables"].items() for c, e in table["columns"].items()}
    for name in ("trans.amount", "trans.balance", "order.amount", "loan.amount", "district.a11"):
        assert cols[name]["unit"] == "CZK", name
    assert cols["loan.payments"]["unit"] == "CZK per month"
    assert all(e.get("unit") for e in cols.values() if e["kind"] == "quantity")


def test_czech_codes_are_translated(financial):
    """Spot checks of the translations against the data set's guide."""
    d, _ = financial
    codes = lambda t, c: d["tables"][t]["columns"][c]["codes"]  # noqa: E731
    assert codes("trans", "type")["PRIJEM"].startswith("credit")
    assert codes("trans", "type")["VYDAJ"].startswith("debit")
    assert codes("trans", "operation")["VYBER KARTOU"] == "card withdrawal"
    assert codes("account", "frequency")["POPLATEK MESICNE"] == "monthly statements"
    assert codes("loan", "status")["D"] == "contract running, client in debt"
    assert set(codes("client", "gender")) == {"F", "M"}


def test_join_paths_cover_every_foreign_key(financial):
    d, _ = financial
    for t, table in d["tables"].items():
        joined = {j["to"] for j in table.get("joins", [])}
        for c, e in table["columns"].items():
            if ref := e.get("references"):
                assert ref.split(".")[0] in joined, (
                    f"{t}.{c} references {ref} but {t} has no join to it"
                )


def test_decoded_gender_and_birth_dates_are_plausible(financial):
    """BIRD decoded the birth number; what it produced must be what the rule can produce."""
    _, snap = financial
    cols = {c["name"]: c for c in snap["tables"]["client"]["columns"]}
    assert set(cols["gender"]["values"]) == {"F", "M"}
    assert cols["gender"]["nulls"] == cols["birth_date"]["nulls"] == 0
    assert "1900-01-01" <= cols["birth_date"]["min"] <= cols["birth_date"]["max"] <= "1998-12-31"


# --- the checks catch what they should (mutation tests) ----------------------------------


def _mutated(financial, change):
    d, snap = copy.deepcopy(financial[0]), financial[1]
    change(d["tables"])
    return (
        model.structure_problems(d)
        + model.coverage_problems(d, snap)
        + model.code_problems(d, snap)
    )


@pytest.mark.parametrize(
    "change, expected",
    [
        (
            lambda t: t["trans"]["columns"]["type"]["codes"].pop("VYBER"),
            "values without a translation ['VYBER']",
        ),
        (
            lambda t: t["order"]["columns"]["k_symbol"]["codes"].pop(""),
            "values without a translation ['']",
        ),
        (
            lambda t: t["trans"]["columns"]["type"]["codes"].update(X="x"),
            "translations for values not in the data ['X']",
        ),
        (
            lambda t: t["trans"]["columns"]["operation"].pop("null_meaning"),
            "NULLs but no null_meaning",
        ),
        (
            lambda t: t["account"]["columns"].pop("frequency"),
            "column account.frequency has no entry",
        ),
        (
            lambda t: t["account"]["columns"].update(
                extra={"name": "x", "kind": "text", "description": "x"}
            ),
            "which the database does not have",
        ),
        (lambda t: t.pop("card"), "table card has no entry"),
        (lambda t: t["card"]["columns"]["type"].update(kind="text"), "(code or label?)"),
        (lambda t: t["card"]["columns"]["type"].update(kind="colour"), "unknown kind 'colour'"),
        (
            lambda t: t["card"]["columns"]["type"].pop("description"),
            "needs a name, a kind and a description",
        ),
        (
            lambda t: t["card"]["columns"]["disp_id"].update(references="disp.nope"),
            "references unknown column",
        ),
        (lambda t: t["card"]["joins"][0].update(to="nowhere"), "join to unknown table"),
    ],
)
def test_checks_catch_a_gap(financial, change, expected):
    problems = _mutated(financial, change)
    assert any(expected in p for p in problems), problems


# --- the birth-number rule ------------------------------------------------------------------


@pytest.mark.parametrize(
    "number, sex, born",
    [
        ("706213", "F", dt.date(1970, 12, 13)),  # month 62 = December + 50: a woman
        ("701213", "M", dt.date(1970, 12, 13)),
        ("110820", "M", dt.date(1911, 8, 20)),
        ("875127", "F", dt.date(1987, 1, 27)),  # month 51 = January + 50
        (450101, "M", dt.date(1945, 1, 1)),  # an integer, as the original files store it
        ("000101", "M", dt.date(1900, 1, 1)),  # leading zeros
    ],
)
def test_birth_number_decodes(number, sex, born):
    assert decode_birth_number(number) == (sex, born)
    assert encode_birth_number(sex, born) == f"{int(number):06d}"


# months 13 and 63; 30 February for a man and a woman; day 0; seven digits; 29 February 1900
@pytest.mark.parametrize(
    "number", ["701313", "706313", "700230", "705230", "701200", "1234567", "005229"]
)
def test_impossible_birth_numbers_are_rejected(number):
    with pytest.raises(ValueError):
        decode_birth_number(number)


def test_birth_number_round_trips_over_every_day_of_the_century_in_the_data():
    day, end = dt.date(1911, 1, 1), dt.date(1987, 12, 31)
    while day <= end:
        for sex in ("F", "M"):
            assert decode_birth_number(encode_birth_number(sex, day)) == (sex, day)
        day += dt.timedelta(days=1)


def test_birth_number_rejects_what_it_cannot_encode():
    with pytest.raises(ValueError):
        encode_birth_number("X", dt.date(1970, 1, 1))
    with pytest.raises(ValueError):
        encode_birth_number("F", dt.date(2001, 1, 1))
