"""The recorded runs as plain files, for a page that has no server behind it.

The static demo is the same interface the service serves, reading these files instead of
calling the API: `data/meta.json` (what `/api/meta` returns), `data/runs/<id>.json` (an
evidence record, what `/runs/<id>` returns) and `data/streams/<id>.json` (a run's events, each
with the wait before it, what `/ask` streams). The streams come from the same function the
service replays with, so the page shows what the service would. No model, database or key.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from src.serving import replay
from src.serving.meter import Meter
from src.serving.store import RunStore

DATA = "data"


def stream_of(ev: dict[str, Any], cfg: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"wait_ms": round(wait * 1000), "event": event} for event, wait in replay.events(ev, cfg)
    ]


def meta_of(store: RunStore, meter: Meter, cfg: dict[str, Any], agent: dict[str, Any]) -> dict:
    """`/api/meta` of a replay-mode service, as the page reads it."""
    return {
        "mode": "replay",
        "local_mode": False,
        "agent": agent,
        "database": cfg["live"]["database"],
        "meter": meter.summary(),
        "connection": None,
        "guardrail": {"enabled": False},
        "suggested": store.index(),
        "max_question_chars": cfg["live"]["max_question_chars"],
    }


def _write(path: Path, doc: Any) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(doc, ensure_ascii=False, separators=(",", ":")) + "\n"
    path.write_text(text, encoding="utf-8", newline="\n")
    return len(text.encode("utf-8"))


def build(
    out: Path, store: RunStore, meter: Meter, cfg: dict[str, Any], agent: dict[str, Any]
) -> dict[str, Any]:
    """Write the data files under `out/data` and describe them."""
    root = out / DATA
    size = _write(root / "meta.json", meta_of(store, meter, cfg, agent))
    ids = [e["id"] for e in store.index()]
    for run_id in ids:
        ev = store.get(run_id)
        assert ev is not None, run_id
        size += _write(root / "runs" / f"{run_id}.json", ev)
        size += _write(root / "streams" / f"{run_id}.json", stream_of(ev, cfg["replay"]))
    digest = hashlib.sha256()
    for p in sorted(root.rglob("*.json")):
        digest.update(p.relative_to(root).as_posix().encode())
        digest.update(p.read_bytes())
    return {
        "runs": len(ids),
        "files": 1 + 2 * len(ids),
        "bytes": size,
        "sha256": digest.hexdigest(),
    }
