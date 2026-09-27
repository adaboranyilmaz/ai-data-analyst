"""`list_tables` and `describe_table`: the schema and the data dictionary, without SQL.

Both read the committed schema snapshot (dictionary/_snapshot/) and data dictionary
(dictionary/), never the database's catalogs: the agent learns the structure of its own
database from the same files a person would read, and its role needs no access to the
catalogs for it. The snapshot holds each column's type and profile (nulls, distinct values,
range, and the values of low-cardinality text columns); the dictionary holds what each table
and column means.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.dictionary import model

# dictionary keys shown for a table, in this order; `columns` are merged in separately
TABLE_KEYS = ("name", "description", "grain", "primary_key", "joins")


class SchemaTools:
    def __init__(
        self,
        db: str,
        dictionary_dir: Path = model.DICTIONARY_DIR,
        snapshot_dir: Path = model.SNAPSHOT_DIR,
    ):
        self.db = db
        self.dictionary = model.load(db, dictionary_dir)
        self.snapshot = model.load_snapshot(db, snapshot_dir)

    @property
    def tables(self) -> list[str]:
        return sorted(self.snapshot["tables"])

    def list_tables(self) -> dict:
        entries = self.dictionary.get("tables") or {}
        out: dict[str, Any] = {"ok": True, "database": self.db}
        for key in ("name", "description"):
            if self.dictionary.get(key) and self.dictionary[key] != self.db:
                out[f"database_{key}"] = self.dictionary[key]
        out["tables"] = [
            {
                "table": t,
                "rows": self.snapshot["tables"][t]["rows"],
                **{k: entries[t][k] for k in ("name", "description") if entries.get(t, {}).get(k)},
            }
            for t in self.tables
        ]
        return out

    def describe_table(self, table: str) -> dict:
        if table not in self.snapshot["tables"]:
            return {
                "ok": False,
                "error": {"kind": "refused", "reasons": [f"unknown table {table!r}"]},
            }
        entry = (self.dictionary.get("tables") or {}).get(table) or {}
        described = entry.get("columns") or {}
        columns = []
        for col in self.snapshot["tables"][table]["columns"]:
            meaning = described.get(col["name"]) or {}
            columns.append({**col, **{k: v for k, v in meaning.items() if k not in col}})
        return {
            "ok": True,
            "database": self.db,
            "table": table,
            "rows": self.snapshot["tables"][table]["rows"],
            **{k: entry[k] for k in TABLE_KEYS if entry.get(k)},
            "columns": columns,
        }
