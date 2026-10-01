"""The sandbox security suite: how outcomes are decided (no Docker), and the whole suite against
the image (marker `sandbox`)."""

from __future__ import annotations

import json

import pytest

from src.stats import security
from src.stats.sandbox import Completed, Policy


def completed(stdout=b"", stderr=b"", code=0, timed_out=False, capped=False, oom=False):
    return Completed(code, stdout, stderr, 1.0, timed_out, capped, oom)


def verdict(achieved, how="denied", detail="x") -> bytes:
    return json.dumps({"achieved": achieved, "how": how, "detail": detail}).encode() + b"\n"


def test_code_outcomes():
    assert security.code_outcome(completed(verdict(False, "absent")), 0, []) == ("absent", "x")
    assert security.code_outcome(completed(verdict(True)), 0, [])[0] == "breach"
    assert security.code_outcome(completed(timed_out=True), 0, [])[0] == "contained"
    assert security.code_outcome(completed(oom=True, code=137), 0, [])[0] == "contained"
    assert security.code_outcome(completed(code=137), 0, [])[0] == "contained"
    # outside checks win over what the program says
    assert security.code_outcome(completed(verdict(False)), 1, [])[0] == "breach"
    leaked = completed(verdict(False) + b"sk-secret-value")
    assert security.code_outcome(leaked, 0, ["sk-secret-value"])[0] == "breach"


def test_input_outcomes():
    err = json.dumps({"version": 1, "error": {"kind": "invalid_input", "message": "m"}})
    assert security.input_outcome(completed(err.encode()))[0] == "rejected"
    job_err = {"version": 1, "results": [{"id": 1, "ok": False, "error": {"kind": "k"}}]}
    assert security.input_outcome(completed(json.dumps(job_err).encode()))[0] == "rejected"
    fine = {"version": 1, "results": [{"id": 1, "ok": True, "result": {}}]}
    assert security.input_outcome(completed(json.dumps(fine).encode()))[0] == "handled"
    assert security.input_outcome(completed(b"Traceback", code=1))[0] == "failed"
    assert security.input_outcome(completed(timed_out=True))[0] == "contained"


def test_program_fills_the_placeholders():
    policy = Policy.from_config().relaxed(timeout_s=5)
    code = security.program(
        "x = (LISTENER_PORT, TIMEOUT_S, SCRATCH_MB, IMAGE_ENV)", policy, 4321, ["PATH"]
    )
    assert f'x = (4321, 5.0, {policy.scratch_mb}, ["PATH"])' in code
    assert code.startswith(security.PRELUDE)
    compile(code, "attack", "exec")


def test_every_attack_is_well_formed():
    cfg = security.load_attacks()
    ids = [a["id"] for a in cfg["attacks"]]
    assert len(ids) == len(set(ids))
    for a in cfg["attacks"]:
        assert ("code" in a) != ("input" in a), a["id"]
        if "code" in a:
            compile(security.program(a["code"], Policy.from_config(), 1, []), a["id"], "exec")
        else:
            assert security.input_payload(a["input"])
        for key in a.get("control", {}):
            assert hasattr(Policy.from_config(), key), (a["id"], key)
    categories = {a["category"] for a in cfg["attacks"]}
    assert categories == {"network", "file_system", "privileges", "secrets", "resources", "input"}


@pytest.mark.sandbox
def test_the_suite_blocks_every_attack_and_every_control_succeeds(sandbox_ready):
    result = security.run_suite()
    failed = [r for r in result["records"] if not r["passed"]]
    assert not failed, failed
    missed = [c for c in result["control_records"] if not c["achieved"]]
    assert not missed, missed
    assert result["attacks"] == len(security.load_attacks()["attacks"])
