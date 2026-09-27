"""The BIRD mini-dev benchmark files: where they are, and how to read them.

configs/datasets.yaml pins every source by hash; scripts/10_fetch_bird.py puts the files in
`raw_dir`. The question file is JSON (an array, or one object per line), each question with
`question_id`, `db_id`, `question`, `evidence`, `SQL` (PostgreSQL dialect) and `difficulty`.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent.parent
DATASETS = ROOT / "configs/datasets.yaml"
QUESTIONS_FILE = "mini_dev_pg.json"
TABLES_FILE = "dev_tables.json"
DESCRIPTIONS_DIR = "descriptions"  # descriptions/<db>/<table>.csv, as BIRD ships them


def config() -> dict[str, Any]:
    return yaml.safe_load(DATASETS.read_text(encoding="utf-8"))


def raw_dir() -> Path:
    return ROOT / config()["bird_minidev"]["raw_dir"]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def parse_questions(text: str) -> list[dict[str, Any]]:
    text = text.strip()
    if text.startswith("["):
        return json.loads(text)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def questions(directory: Path | None = None) -> list[dict[str, Any]]:
    path = (directory or raw_dir()) / QUESTIONS_FILE
    return parse_questions(path.read_text(encoding="utf-8"))


def table_databases(directory: Path | None = None) -> dict[str, str]:
    """Each table's database, keyed by the table name as PostgreSQL stores it (lower case:
    the dump creates every table unquoted). Table names are unique across the databases."""
    path = (directory or raw_dir()) / TABLES_FILE
    out: dict[str, str] = {}
    for db in json.loads(path.read_text(encoding="utf-8")):
        for table in db["table_names_original"]:
            key = table.lower()
            if key in out:
                raise ValueError(f"table {key!r} is in both {out[key]} and {db['db_id']}")
            out[key] = db["db_id"]
    return out


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
