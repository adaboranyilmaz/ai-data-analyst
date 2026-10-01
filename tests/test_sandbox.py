"""The sandbox client: the container flags, the image tag, the pinned packages (no Docker)."""

from __future__ import annotations

import datetime as dt
import decimal
import re
import shutil
import tomllib
from pathlib import Path

import pytest

from src.stats import sandbox as sb

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def policy() -> sb.Policy:
    return sb.Policy.from_config()


@pytest.fixture(autouse=True)
def _fake_docker(monkeypatch):
    monkeypatch.setattr(sb, "docker", lambda: "docker")


def _flag(args: list[str], name: str) -> str:
    return args[args.index(name) + 1]


def test_run_args_lock_the_container_down(policy):
    args = sb.run_args(policy, "img:tag", "n1")
    assert args[:2] == ["docker", "run"]
    assert _flag(args, "--network") == "none"
    assert "--read-only" in args
    assert _flag(args, "--user") == sb.USER
    assert _flag(args, "--cap-drop") == "ALL"
    assert _flag(args, "--security-opt") == "no-new-privileges"
    assert _flag(args, "--pids-limit") == str(policy.pids)
    assert _flag(args, "--memory") == _flag(args, "--memory-swap") == f"{policy.memory_mb}m"
    assert _flag(args, "--cpus") == str(policy.cpus)
    assert _flag(args, "--pull") == "never"
    assert _flag(args, "--log-driver") == "none"
    scratch = _flag(args, "--tmpfs").split(",")
    assert scratch[0] == "/scratch:rw"
    assert {"noexec", "nosuid", "nodev", f"size={policy.scratch_mb}m"} <= set(scratch)
    assert args[-1] == "img:tag"


def test_run_args_pass_nothing_from_the_host(policy):
    args = sb.run_args(policy, "img:tag", "n1")
    for forbidden in (
        "--env",
        "-e",
        "--env-file",
        "-v",
        "--volume",
        "--mount",
        "--privileged",
        "--device",
        "--cap-add",
        "--ipc",
        "--pid",
    ):
        assert forbidden not in args
    assert not any(a.startswith("--env") for a in args)


def test_relaxed_controls_change_only_their_flag(policy):
    base = sb.run_args(policy, "t", "n")
    exec_ok = sb.run_args(policy.relaxed(scratch_noexec=False), "t", "n")
    assert "exec" in _flag(exec_ok, "--tmpfs").split(",")
    assert "noexec" not in _flag(exec_ok, "--tmpfs").split(",")
    bridge = sb.run_args(policy.relaxed(network="bridge"), "t", "n")
    assert _flag(bridge, "--network") == "bridge"
    i = base.index("--network") + 1
    assert bridge[:i] + bridge[i + 1 :] == base[:i] + base[i + 1 :]
    env = sb.run_args(policy.relaxed(env=("A=1",)), "t", "n")
    assert _flag(env, "--env") == "A=1"


def test_entrypoint_and_command_come_last(policy):
    args = sb.run_args(policy, "t", "n", entrypoint="python", command=("-c", "pass"))
    assert _flag(args, "--entrypoint") == "python"
    assert args[-3:] == ["t", "-c", "pass"]


def test_policy_reads_the_config(policy):
    assert policy.network == "none" and policy.read_only and policy.drop_capabilities
    assert policy.timeout_s > 0 and policy.memory_mb > 0 and policy.pids > 0
    assert policy.env == ()


def test_image_tag_follows_the_sources(tmp_path):
    for name in sb.SOURCES:
        shutil.copy(sb.SANDBOX_DIR / name, tmp_path / name)
    assert sb.sources_hash(tmp_path) == sb.sources_hash()
    crlf = tmp_path / "runner.py"
    crlf.write_bytes(crlf.read_bytes().replace(b"\n", b"\r\n"))
    assert sb.sources_hash(tmp_path) == sb.sources_hash()  # line endings do not count
    (tmp_path / "analysis.py").write_bytes(b"# changed\n")
    assert sb.sources_hash(tmp_path) != sb.sources_hash()
    assert re.fullmatch(rf"{sb.IMAGE}:[0-9a-f]{{12}}", sb.image_tag())


def test_requirements_are_the_locked_versions():
    lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
    locked = {p["name"]: p["version"] for p in lock["package"]}
    text = (sb.SANDBOX_DIR / "requirements.txt").read_text(encoding="utf-8")
    pins = dict(re.findall(r"^([a-z0-9_.-]+)==([^ ;\\\n]+)", text, flags=re.M))
    assert {"numpy", "scipy", "pandas", "statsmodels", "patsy"} <= set(pins)
    for name, version in pins.items():
        assert locked[name] == version, name
    # every requirement carries hashes
    blocks = re.split(r"\n(?=[a-z])", text.split("\n", 2)[2])
    assert all("--hash=sha256:" in b for b in blocks if "==" in b)


def test_dockerfile_pins_the_base_and_runs_as_non_root():
    text = (sb.SANDBOX_DIR / "Dockerfile").read_text(encoding="utf-8")
    assert re.search(r"^FROM python:3\.12-slim@sha256:[0-9a-f]{64}$", text, flags=re.M)
    assert re.search(r"^USER 10001:10001$", text, flags=re.M)
    assert "--require-hashes" in text and "--only-binary=:all:" in text
    assert "OMP_NUM_THREADS=1" in text


def test_jsonable_values():
    assert sb.jsonable(decimal.Decimal("1.50")) == 1.5
    assert sb.jsonable(decimal.Decimal("NaN")) is None
    assert sb.jsonable(float("inf")) is None
    assert sb.jsonable(dt.date(1998, 12, 31)) == "1998-12-31"
    assert sb.jsonable(True) is True and sb.jsonable(3) == 3 and sb.jsonable(None) is None
    assert sb.jsonable("F") == "F"


def test_run_jobs_refuses_a_missing_image(monkeypatch):
    monkeypatch.setattr(sb, "image_present", lambda tag=None: False)
    with pytest.raises(sb.SandboxUnavailable, match="not built"):
        sb.run_jobs([{"id": 1}])
