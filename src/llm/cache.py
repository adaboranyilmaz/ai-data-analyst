"""The on-disk response cache, and the replay-only switch.

One JSON file per request, `<root>/<key[:2]>/<key>.json`, holding the request and the
response. Writes go through a temporary file and a rename, so an interrupted run never
leaves a half-written entry that later reads as a valid response.

Replay-only mode (`ANALYST_REPLAY_ONLY=1`, set in CI and for pipeline rebuilds): every
response must come from the cache, and a miss raises `CacheMiss` instead of calling a model,
so a rebuild costs nothing and cannot silently produce new answers. An empty
`ANTHROPIC_API_KEY` is no substitute: on Windows an empty variable is removed from the
environment, and loading `.env` afterwards restores the real key.
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import asdict
from pathlib import Path

from src.llm.types import LLMRequest, LLMResponse

CACHE_DIR = Path("data/cache/llm")


class CacheMiss(RuntimeError):
    """A request absent from the response cache while replay-only mode is on."""


def replay_only() -> bool:
    return os.environ.get("ANALYST_REPLAY_ONLY") == "1"


class ResponseCache:
    def __init__(self, root: Path = CACHE_DIR):
        self.root = Path(root)

    def _path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.json"

    def has(self, key: str) -> bool:
        return self._path(key).exists()

    def get(self, key: str) -> LLMResponse | None:
        path = self._path(key)
        if not path.exists():
            return None
        entry = json.loads(path.read_text(encoding="utf-8"))
        return LLMResponse(**entry["response"])

    def put(self, request: LLMRequest, response: LLMResponse) -> None:
        path = self._path(request.cache_key)
        path.parent.mkdir(parents=True, exist_ok=True)
        entry = {"request": asdict(request), "response": asdict(response)}
        tmp = path.with_suffix(f".tmp{os.getpid()}.{threading.get_ident()}")
        tmp.write_text(
            json.dumps(entry, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n"
        )
        os.replace(tmp, path)
