"""Load a data dictionary (dictionary/<db>.yaml) and check it: its own structure, and whether
it covers a database as recorded in the database's snapshot (dictionary/_snapshot/<db>.json).

Two origins, set by the file's `origin` key:
  hand-written      every table and column needs an English name and a description, every
                    column a kind; code columns translate every value present in the data
  bird-description  converted as-is from BIRD's description files: every column has an
                    entry, but its description may be missing (`status: missing`)
Each check returns a list of problems, empty when the dictionary passes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from src.dictionary.snapshot import SNAPSHOT_DIR, TEXT_TYPES

ROOT = Path(__file__).resolve().parent.parent.parent
DICTIONARY_DIR = ROOT / "dictionary"
ORIGINS = {"hand-written", "bird-description"}
# identifier: a key or reference number; code: values with a meaning to translate; label: a
# name or opaque code that is its own meaning (district names, anonymized bank codes);
# quantity: a number with a unit; date; text: free text
KINDS = {"identifier", "code", "label", "quantity", "date", "text"}
COLUMN_KEYS = {
    "name", "kind", "description", "unit", "codes", "null_meaning", "encoding", "quirks",
    "references", "bird_name", "data_format", "value_description", "status",
}  # fmt: skip
TABLE_KEYS = {"name", "description", "grain", "primary_key", "joins", "columns", "bird_file"}


def load(db: str, directory: Path = DICTIONARY_DIR) -> dict[str, Any]:
    return yaml.safe_load((directory / f"{db}.yaml").read_text(encoding="utf-8"))


def load_snapshot(db: str, directory: Path = SNAPSHOT_DIR) -> dict[str, Any]:
    return json.loads((directory / f"{db}.json").read_text(encoding="utf-8"))


def structure_problems(d: dict[str, Any]) -> list[str]:
    out: list[str] = []
    origin = d.get("origin")
    if origin not in ORIGINS:
        return [f"origin must be one of {sorted(ORIGINS)}, not {origin!r}"]
    strict = origin == "hand-written"
    for key in ("database", "name", "tables"):
        if not d.get(key):
            out.append(f"missing top-level {key!r}")
    tables = d.get("tables") or {}
    for t, table in tables.items():
        where = f"{t}"
        if unknown := set(table) - TABLE_KEYS:
            out.append(f"{where}: unknown keys {sorted(unknown)}")
        if strict and not (table.get("name") and table.get("description")):
            out.append(f"{where}: needs a name and a description")
        for j in table.get("joins") or []:
            if j.get("to") not in tables:
                out.append(f"{where}: join to unknown table {j.get('to')!r}")
        for c, col in (table.get("columns") or {}).items():
            where = f"{t}.{c}"
            if not isinstance(col, dict):
                out.append(f"{where}: entry is not a mapping")
                continue
            if unknown := set(col) - COLUMN_KEYS:
                out.append(f"{where}: unknown keys {sorted(unknown)}")
            kind = col.get("kind")
            if kind is not None and kind not in KINDS:
                out.append(f"{where}: unknown kind {kind!r}")
            if strict and not (col.get("name") and col.get("description") and kind):
                out.append(f"{where}: needs a name, a kind and a description")
            if not strict and not (col.get("description") or col.get("status") == "missing"):
                out.append(f"{where}: no description and not marked `status: missing`")
            codes = col.get("codes")
            if kind == "code" and not codes:
                out.append(f"{where}: a code column needs its codes")
            if codes is not None and not (
                isinstance(codes, dict) and all(isinstance(k, str) and v for k, v in codes.items())
            ):
                out.append(f"{where}: codes must map text values to meanings")
            if ref := col.get("references"):
                rt, _, rc = ref.partition(".")
                if rc not in ((tables.get(rt) or {}).get("columns") or {}):
                    out.append(f"{where}: references unknown column {ref!r}")
    return out


def coverage_problems(d: dict[str, Any], snapshot: dict[str, Any]) -> list[str]:
    """Every table and column in the database has an entry, and no entry names one that is not
    there."""
    out: list[str] = []
    tables = d.get("tables") or {}
    for t in sorted(set(snapshot["tables"]) - set(tables)):
        out.append(f"table {t} has no entry")
    for t in sorted(set(tables) - set(snapshot["tables"])):
        out.append(f"entry for table {t}, which the database does not have")
    for t in sorted(set(tables) & set(snapshot["tables"])):
        have = {c["name"] for c in snapshot["tables"][t]["columns"]}
        entries = set(tables[t].get("columns") or {})
        out += [f"column {t}.{c} has no entry" for c in sorted(have - entries)]
        out += [
            f"entry for column {t}.{c}, which the database does not have"
            for c in sorted(entries - have)
        ]
    return out


def _is_number(s: str) -> bool:
    try:
        float(s)
    except ValueError:
        return False
    return True


def code_problems(d: dict[str, Any], snapshot: dict[str, Any]) -> list[str]:
    """For a profiled database: every value present in a code column is translated, a NULL
    that occurs has a stated meaning, and no low-cardinality text column escapes being a code
    or a label."""
    if not snapshot.get("profiled"):
        return []
    out: list[str] = []
    for t, table in snapshot["tables"].items():
        entries = (d.get("tables") or {}).get(t, {}).get("columns") or {}
        for col in table["columns"]:
            entry = entries.get(col["name"]) or {}
            where = f"{t}.{col['name']}"
            if col.get("nulls") and not entry.get("null_meaning"):
                out.append(f"{where}: has {col['nulls']} NULLs but no null_meaning")
            values = col.get("values")
            if values is None:
                continue
            kind = entry.get("kind")
            if kind == "code":
                codes = entry.get("codes") or {}
                missing = sorted(v for v in values if v not in codes)
                if missing:
                    out.append(f"{where}: values without a translation {missing}")
                unused = sorted(v for v in codes if v not in values)
                if unused:
                    out.append(f"{where}: translations for values not in the data {unused}")
            elif col["type"] in TEXT_TYPES and not all(_is_number(v) for v in values):
                if kind != "label":
                    out.append(
                        f"{where}: {len(values)} distinct text values, "
                        f"but kind {kind!r} (code or label?)"
                    )
    return out
