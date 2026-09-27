"""The analysis refuses to run unless the pre-registration is frozen, complete, and names the
question sets and the banking set actually used."""

from __future__ import annotations

import json

import pytest

from src.eval.config import ROOT, config
from src.eval.preregistration import (
    PreregistrationError,
    file_sha256,
    parse,
    problems,
    require,
)

PRE = """# Pre-registration

Prose.

```yaml
splits_sha256: {splits}
own_set_sha256: {own}
predictions:
  - id: P1
    claim: design 4 against design 1, EX
    prediction: {p1}
  - id: P2
    claim: AUROC of the winner's confidence
    prediction: above 0.7
```
"""


@pytest.fixture
def root(tmp_path):
    (tmp_path / "results/metrics").mkdir(parents=True)
    (tmp_path / "own_set").mkdir()
    (tmp_path / "results/metrics/splits.json").write_text(json.dumps({"sets_sha256": "abc"}))
    (tmp_path / "own_set/questions.yaml").write_text("questions: []\n")
    return tmp_path


def write(root, p1="higher by 5 points", splits="abc", own=None, freeze=True) -> dict:
    own = own or file_sha256(root / "own_set/questions.yaml")
    path = root / "results/metrics/preregistration.md"
    text = PRE.format(splits=splits, own=own, p1=p1)
    path.write_bytes(text.encode("utf-8"))  # LF, as a checkout writes it
    return {
        "splits_file": "results/metrics/splits.json",
        "own_set": {"path": "own_set/questions.yaml", "sha256": None},
        "preregistration": {
            "path": "results/metrics/preregistration.md",
            "sha256": file_sha256(path) if freeze else None,
        },
    }


def test_a_complete_frozen_preregistration_passes(root):
    cfg = write(root)
    assert problems(cfg, root) == []
    require(cfg, root)


@pytest.mark.parametrize("blank", ["", "TBD", "tbd", '"?"', "null"])
def test_a_blank_prediction_stops_the_analysis(root, blank):
    cfg = write(root, p1=blank)
    with pytest.raises(PreregistrationError, match="P1: prediction is blank"):
        require(cfg, root)


def test_not_frozen_or_changed_after_freezing(root):
    assert any("not frozen" in p for p in problems(write(root, freeze=False), root))
    cfg = write(root)
    path = root / "results/metrics/preregistration.md"
    path.write_text(path.read_text(encoding="utf-8") + "\nAn edit.\n", encoding="utf-8")
    assert any("changed since it was frozen" in p for p in problems(cfg, root))


def test_line_endings_do_not_change_the_hash(root):
    cfg = write(root)
    path = root / "results/metrics/preregistration.md"
    path.write_bytes(path.read_bytes().replace(b"\n", b"\r\n"))
    assert problems(cfg, root) == []


def test_other_question_sets_or_banking_set_are_refused(root):
    assert any("question sets" in p for p in problems(write(root, splits="other"), root))
    cfg = write(root)
    (root / "own_set/questions.yaml").write_text("questions: [changed]\n")
    assert any("banking set" in p for p in problems(cfg, root))


def test_the_yaml_block_is_required_once():
    with pytest.raises(PreregistrationError):
        parse("no block here")
    with pytest.raises(PreregistrationError):
        parse("```yaml\na: 1\n```\n```yaml\nb: 2\n```\n")


def test_the_committed_preregistration_parses_and_names_the_committed_sets():
    cfg = config()
    path = ROOT / cfg["preregistration"]["path"]
    if not path.exists():
        pytest.skip("the pre-registration is not written yet")
    data = parse(path.read_text(encoding="utf-8"))
    splits = json.loads((ROOT / cfg["splits_file"]).read_text(encoding="utf-8"))
    assert data["splits_sha256"] == splits["sets_sha256"]
    if cfg["preregistration"]["sha256"] is not None:  # once frozen, nothing may be missing
        assert problems(cfg) == []
