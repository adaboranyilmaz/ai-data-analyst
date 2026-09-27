"""Fetch BIRD's official evaluator at its pinned commit and check every file's hash.

The files (configs/eval.yaml `official_evaluator`) go to data/raw/bird_eval/, the output of the
`bird_eval` stage in dvc.yaml. They are run for one purpose: to check that the project's
scoring reproduces their verdicts (scripts/31_validate_ex.py). They are not committed, since
the upstream repository has no licence file. A file already present with the right hash is
kept.

Usage:
    uv run python scripts/30_fetch_bird_eval.py
"""

from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.eval.config import config, official_evaluator_dir  # noqa: E402


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    cfg = config()
    official = cfg["official_evaluator"]
    out = official_evaluator_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    for name, expected in official["sha256"].items():
        dest = out / name
        if dest.exists() and sha256(dest.read_bytes()) == expected:
            print(f"{name}: present, hash matches")
            continue
        url = official["url"].format(commit=official["commit"], file=name)
        print(f"downloading {url}")
        with urllib.request.urlopen(url) as r:
            data = r.read()
        if sha256(data) != expected:
            sys.exit(f"{name}: sha256 {sha256(data)}, expected {expected}")
        dest.write_bytes(data)
    print(f"the official evaluator is in {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
