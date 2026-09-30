"""Convert BIRD's database description files into data dictionaries (dictionary/<db>.yaml) for
every BIRD database except the Czech bank (`financial`), whose dictionary is hand-written.

BIRD ships one CSV per table: original_column_name, column_name, column_description,
data_format, value_description. They are converted as they are: the text is not edited,
beyond normalizing line endings and trailing spaces. Each CSV row is matched to a column in
the database's snapshot (dictionary/_snapshot/<db>.json) by name: exactly, then ignoring
case, then ignoring case and surrounding spaces. Every database column gets an entry; one
with no description is marked `status: missing`. CSV rows that match no column are listed
under `unmatched_descriptions`. Most files are UTF-8; a few are Windows-1252.

Usage:
    uv run python scripts/14_convert_descriptions.py
"""

from __future__ import annotations

import csv
import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402

from src.data import bird  # noqa: E402
from src.dictionary.model import DICTIONARY_DIR, load_snapshot  # noqa: E402

HAND_WRITTEN = {"financial"}
HEADER = """\
# Data dictionary: the BIRD `{db}` database, converted by scripts/14_convert_descriptions.py
# from BIRD's database description files, without editing their text. Regenerate; do not
# edit by hand. Derived from BIRD (Li et al., 2023), licensed CC BY-SA 4.0.
"""


class _Dumper(yaml.SafeDumper):
    pass


def _str(dumper: yaml.SafeDumper, s: str) -> yaml.Node:
    return dumper.represent_scalar("tag:yaml.org,2002:str", s, style="|" if "\n" in s else None)


_Dumper.add_representer(str, _str)


def read_csv(path: Path) -> list[dict[str, str]]:
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("cp1252")
    rows = list(csv.DictReader(io.StringIO(text, newline="")))
    return [{(k or "").strip(): v for k, v in r.items() if k} for r in rows]


def clean(s: str | None) -> str | None:
    if s is None:
        return None
    s = "\n".join(
        line.rstrip() for line in s.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    ).strip()
    return s or None


def match(name: str, columns: list[str]) -> str | None:
    for key in (lambda x: x, str.lower, lambda x: x.strip().lower()):
        hits = [c for c in columns if key(c) == key(name)]
        if len(hits) == 1:
            return hits[0]
    return None


def convert(db: str) -> tuple[dict, int, int]:
    snap = load_snapshot(db)
    files = {p.stem.lower(): p for p in (bird.raw_dir() / bird.DESCRIPTIONS_DIR / db).glob("*.csv")}
    tables: dict = {}
    unmatched: list[dict[str, str]] = []
    described = missing = 0
    for t, info in sorted(snap["tables"].items()):
        cols = [c["name"] for c in info["columns"]]
        entries: dict[str, dict] = {}
        path = files.get(t)
        for row in read_csv(path) if path else []:
            bird_name = row.get("original_column_name", "")
            col = match(bird_name, cols)
            if col is None or col in entries:
                unmatched.append({"table": t, "bird_name": bird_name})
                continue
            entry = {
                "bird_name": bird_name.strip(),
                "name": clean(row.get("column_name")) or bird_name.strip(),
                "description": clean(row.get("column_description")),
                "data_format": clean(row.get("data_format")),
                "value_description": clean(row.get("value_description")),
            }
            entries[col] = {k: v for k, v in entry.items() if v is not None}
        columns = {}
        for c in cols:
            entry = entries.get(c) or {"name": c}
            if entry.get("description") or entry.get("value_description"):
                entry.setdefault("description", entry.get("value_description"))
                described += 1
            else:
                entry["status"] = "missing"
                missing += 1
            columns[c] = entry
        tables[t] = {"bird_file": path.name if path else None, "columns": columns}
    out = {
        "database": db,
        "origin": "bird-description",
        "name": db,
        "sources": ["BIRD mini-dev database description files (CC BY-SA 4.0)"],
        "tables": tables,
    }
    if unmatched:
        out["unmatched_descriptions"] = unmatched
    return out, described, missing


def main() -> None:
    for db in sorted(set(bird.table_databases().values()) - HAND_WRITTEN):
        d, described, missing = convert(db)
        path = DICTIONARY_DIR / f"{db}.yaml"
        body = yaml.dump(d, Dumper=_Dumper, sort_keys=False, allow_unicode=True, width=100)
        path.write_text(HEADER.format(db=db) + body, encoding="utf-8", newline="\n")
        extra = len(d.get("unmatched_descriptions", []))
        print(f"{db}: {described} described, {missing} missing, {extra} unmatched description rows")


if __name__ == "__main__":
    main()
