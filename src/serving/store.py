"""Where evidence records live: the curated runs (committed) and the live runs (local data).

A run is a JSON file named by its id, so a permalink is the id. Ids are checked against a strict
pattern before they touch the file system.
"""

from __future__ import annotations

import json
import re
import uuid
from pathlib import Path
from typing import Any

ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


def valid_id(run_id: str) -> bool:
    return bool(ID.match(run_id)) and ".." not in run_id


class RunStore:
    def __init__(self, curated_dir: Path, live_dir: Path | None = None):
        self.curated_dir = curated_dir
        self.live_dir = live_dir

    def index(self) -> list[dict[str, Any]]:
        path = self.curated_dir / "index.json"
        if not path.exists():
            return []
        return json.loads(path.read_text(encoding="utf-8"))["runs"]

    def get(self, run_id: str) -> dict[str, Any] | None:
        if not valid_id(run_id):
            return None
        for d in (self.curated_dir / "runs", self.live_dir):
            if d is None:
                continue
            path = d / f"{run_id}.json"
            if path.is_file():
                return json.loads(path.read_text(encoding="utf-8"))
        return None

    def new_live_id(self) -> str:
        return f"live-{uuid.uuid4().hex[:12]}"

    def prune_live(self, keep: int) -> int:
        """Keep the newest `keep` live runs; older permalinks stop working. Returns how many
        were removed."""
        if self.live_dir is None or not self.live_dir.is_dir():
            return 0
        files = sorted(self.live_dir.glob("live-*.json"), key=lambda p: p.stat().st_mtime)
        old = files[: max(len(files) - keep, 0)]
        for p in old:
            p.unlink(missing_ok=True)
        return len(old)

    def save_live(self, ev: dict[str, Any]) -> None:
        if self.live_dir is None:
            raise RuntimeError("this store keeps no live runs")
        if not valid_id(ev["id"]):
            raise ValueError(f"bad run id {ev['id']!r}")
        self.live_dir.mkdir(parents=True, exist_ok=True)
        path = self.live_dir / f"{ev['id']}.json"
        path.write_text(
            json.dumps(ev, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n"
        )
