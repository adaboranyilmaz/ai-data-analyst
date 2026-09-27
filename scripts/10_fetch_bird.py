"""Fetch the BIRD mini-dev sources and check each against its pinned hash.

  1. The package (configs/datasets.yaml `package`): downloaded if missing; its size and
     sha256 must match. It is the output of the `download` stage in dvc.yaml, so a clone
     gets it with `dvc pull`. `--download-only` stops after this step.
  2. From the package: the table list and every database's column descriptions, extracted
     into the raw directory; the PostgreSQL dump is hashed in place (the loader streams it
     from the package, so it is never unpacked to disk).
  3. The question file at its pinned Hugging Face revision; its sha256 must match.
  4. The package's own, older question file is compared with the pinned one, question by
     question, so the differences between the two versions are on record.
Writes results/metrics/bird_sources.json.

Usage:
    uv run python scripts/10_fetch_bird.py [--download-only]
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data import bird  # noqa: E402

OUT = ROOT / "results/metrics/bird_sources.json"
FIELDS = ("db_id", "question", "evidence", "SQL", "difficulty")


def download(url: str, dest: Path) -> None:
    print(f"downloading {url}")
    part = dest.with_suffix(dest.suffix + ".part")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as r, part.open("wb") as f:
        shutil.copyfileobj(r, f, 1 << 20)
    part.replace(dest)


def check(label: str, actual: str | int, expected: str | int) -> None:
    if actual != expected:
        sys.exit(f"{label}: expected {expected}, got {actual}")


def extract(z: zipfile.ZipFile, members: dict, raw: Path) -> dict[str, int]:
    """dev_tables.json and descriptions/<db>/<table>.csv; returns CSV counts per database."""
    (raw / bird.TABLES_FILE).write_bytes(z.read(members["tables"]))
    prefix, suffix = members["descriptions"].split("{db}")
    counts: Counter[str] = Counter()
    for name in z.namelist():
        if name.startswith(prefix) and suffix in name and name.endswith(".csv"):
            db = name[len(prefix) :].split("/", 1)[0]
            dest = raw / bird.DESCRIPTIONS_DIR / db / name.rsplit("/", 1)[1]
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(z.read(name))
            counts[db] += 1
    return dict(sorted(counts.items()))


def member_sha256(z: zipfile.ZipFile, name: str) -> str:
    h = hashlib.sha256()
    with z.open(name) as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def compare_versions(stale: list[dict], pinned: list[dict]) -> dict:
    """Which questions the pinned file changes, by id and field (never the question text)."""
    ids = Counter(q["question_id"] for q in stale)
    by_id = {q["question_id"]: q for q in pinned}
    stale_by_id: dict[int, list[dict]] = {}
    for q in stale:
        stale_by_id.setdefault(q["question_id"], []).append(q)
    changed = []
    for qid, new in sorted(by_id.items()):
        olds = stale_by_id.get(qid, [])
        if not olds:
            changed.append({"question_id": qid, "db_id": new["db_id"], "change": "added"})
            continue
        fields = sorted({f for old in olds for f in FIELDS if old[f] != new[f]})
        if fields:
            changed.append({"question_id": qid, "db_id": new["db_id"], "fields": fields})
    removed = sorted(set(stale_by_id) - set(by_id))
    return {
        "stale_questions": len(stale),
        "stale_duplicate_ids": sorted(k for k, v in ids.items() if v > 1),
        "pinned_questions": len(pinned),
        "removed_ids": removed,
        "changed": changed,
        "changed_sql": sum("SQL" in c.get("fields", []) for c in changed),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--download-only", action="store_true", help="fetch and verify the package only")
    download_only = p.parse_args().download_only
    cfg = bird.config()["bird_minidev"]
    pkg, members, qcfg = cfg["package"], cfg["members"], cfg["questions"]

    zip_path = ROOT / pkg["path"]
    if not zip_path.exists():
        download(pkg["url"], zip_path)
    check("package size", zip_path.stat().st_size, pkg["size_bytes"])
    check("package sha256", bird.sha256_file(zip_path), pkg["sha256"])
    if download_only:
        print(f"{pkg['path']} matches its pinned size and sha256")
        return
    raw = bird.raw_dir()
    raw.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path) as z:
        descriptions = extract(z, members, raw)
        print("hashing the PostgreSQL dump")
        dump_sha = member_sha256(z, members["dump"])
        check("dump sha256", dump_sha, members["dump_sha256"])
        dump_bytes = z.getinfo(members["dump"]).file_size
        stale = bird.parse_questions(z.read(members["stale_questions"]).decode("utf-8"))

    url = qcfg["url"].format(revision=qcfg["revision"])
    qpath = raw / bird.QUESTIONS_FILE
    download(url, qpath)
    check("questions sha256", bird.sha256_file(qpath), qcfg["sha256"])
    pinned = bird.questions(raw)
    check("question ids unique", len({q["question_id"] for q in pinned}), len(pinned))

    bird.write_json(
        OUT,
        {
            "package": {
                "url": pkg["url"],
                "size_bytes": pkg["size_bytes"],
                "sha256": pkg["sha256"],
            },
            "dump": {"member": members["dump"], "bytes": dump_bytes, "sha256": dump_sha},
            "questions": {
                "url": url,
                "revision": qcfg["revision"],
                "sha256": qcfg["sha256"],
                "count": len(pinned),
                "per_database": dict(sorted(Counter(q["db_id"] for q in pinned).items())),
                "per_difficulty": dict(sorted(Counter(q["difficulty"] for q in pinned).items())),
            },
            "description_files": descriptions,
            "package_questions_vs_pinned": compare_versions(stale, pinned),
            "licence": cfg["licence"],
        },
    )
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
