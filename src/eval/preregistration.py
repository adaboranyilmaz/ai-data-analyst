"""The pre-registration: what was fixed before any design was run, and the check that holds
the analysis to it.

results/metrics/preregistration.md states in prose the designs, the question sets, the
selection rule, the calibration and decline rules, how the hand-written banking set is scored,
and the predictions. One fenced YAML block in it carries what the code checks:

    splits_sha256: <the question sets in results/metrics/splits.json>
    own_set_sha256: <own_set/questions.yaml>
    predictions:
      - id: ...
        claim: ...        # what is predicted about
        prediction: ...   # the predicted outcome

The file is frozen by its hash in configs/eval.yaml (`preregistration.sha256`). `require()`
refuses (raises) unless the file is frozen and unchanged, every prediction is filled in, and
the question sets and the banking set are exactly the ones it names. Every analysis of
evaluation runs calls it first.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import yaml

from src.eval.config import ROOT, config

_YAML_BLOCK = re.compile(r"^```yaml\n(.*?)^```", re.MULTILINE | re.DOTALL)
BLANK = {"", "tbd", "todo", "?", "..."}


class PreregistrationError(RuntimeError):
    pass


def file_sha256(path: Path) -> str:
    """sha256 of a text file with LF line endings, whatever the checkout wrote."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def is_blank(value: Any) -> bool:
    return value is None or (isinstance(value, str) and value.strip().lower() in BLANK)


def parse(text: str) -> dict[str, Any]:
    blocks = _YAML_BLOCK.findall(text)
    if len(blocks) != 1:
        raise PreregistrationError(f"expected one ```yaml block, found {len(blocks)}")
    data = yaml.safe_load(blocks[0])
    if not isinstance(data, dict):
        raise PreregistrationError("the yaml block is not a mapping")
    return data


def problems(cfg: dict[str, Any] | None = None, root: Path = ROOT) -> list[str]:
    cfg = cfg or config()
    pre = cfg["preregistration"]
    path = root / pre["path"]
    if not path.exists():
        return [f"{pre['path']} does not exist"]
    found = []
    if pre.get("sha256") is None:
        found.append("the pre-registration is not frozen (configs/eval.yaml has no hash)")
    elif file_sha256(path) != pre["sha256"]:
        found.append(f"{pre['path']} changed since it was frozen")
    try:
        data = parse(path.read_text(encoding="utf-8"))
    except (PreregistrationError, yaml.YAMLError) as e:
        return [*found, str(e)]

    predictions = data.get("predictions")
    if not isinstance(predictions, list) or not predictions:
        found.append("no predictions")
    else:
        for i, p in enumerate(predictions, 1):
            if not isinstance(p, dict):
                found.append(f"prediction {i} is not a mapping")
                continue
            for key in ("id", "claim", "prediction"):
                if is_blank(p.get(key)):
                    found.append(f"prediction {p.get('id') or i}: {key} is blank")

    splits = json.loads((root / cfg["splits_file"]).read_text(encoding="utf-8"))
    if data.get("splits_sha256") != splits["sets_sha256"]:
        found.append("the question sets differ from the ones pre-registered")
    own = cfg["own_set"]
    if data.get("own_set_sha256") != file_sha256(root / own["path"]):
        found.append("the banking set differs from the one pre-registered")
    return found


def require(cfg: dict[str, Any] | None = None, root: Path = ROOT) -> None:
    if found := problems(cfg, root):
        raise PreregistrationError("the analysis refuses to run: " + "; ".join(found))
