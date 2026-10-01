"""The host side of the statistics sandbox: its image, and one fresh container per call.

The analyses (sandbox/analysis.py) run in a Docker container that is created for one call and
removed after it, started with:

- `--network none`: no network interface but loopback, so no data or secret can leave it and no
  service (the database included) can be reached from it;
- `--read-only`, with a `--tmpfs` scratch directory (small, `noexec`, `nosuid`, `nodev`), the
  only place it can write;
- a non-root user, `--cap-drop ALL` and `no-new-privileges`, so nothing in it can gain rights;
- limits on processes (`--pids-limit`, against fork bombs), memory (swap included), CPUs and open
  files, and a wall-clock limit enforced from here: the container is killed when it runs out;
- no environment from the host (only the image's own), no mounts, and no log storage (so a flood
  of output fills nothing). At most `output_mb` of its output (and 1 MB of its error stream) is
  read; more stops it;
- `--pull never`: only the locally built image runs; a missing one is never fetched.

The input goes in on stdin and the result comes back on stdout, as JSON (sandbox/runner.py).
The image is tagged with a hash of the sandbox directory, so a call never runs an image built from
other sources: a missing image is an error, not a silent rebuild (scripts/70_build_sandbox.py
builds it).
"""

from __future__ import annotations

import datetime as dt
import decimal
import hashlib
import json
import shutil
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent.parent
SANDBOX_DIR = ROOT / "sandbox"
CONFIG = ROOT / "configs/guardrail.yaml"
SOURCES = ("Dockerfile", "requirements.txt", "runner.py", "analysis.py")
IMAGE = "ai-data-analyst-sandbox"
LABEL = "analyst.sandbox"
USER = "10001:10001"
PROTOCOL_VERSION = 1
KILL_GRACE_S = 10.0  # after the kill, how long the docker client may take to return
ERROR_CAP = 1024 * 1024  # bytes of the error stream read; more stops the container


class SandboxUnavailable(RuntimeError):
    """Docker, or the sandbox image, is not there."""


class SandboxError(RuntimeError):
    def __init__(self, kind: str, message: str):
        super().__init__(f"{kind}: {message}")
        self.kind = kind


@dataclass(frozen=True)
class Policy:
    """How a container is started. The defaults come from configs/guardrail.yaml; the security
    suite's controls relax one field at a time."""

    timeout_s: float
    memory_mb: int
    cpus: float
    pids: int
    scratch_mb: int
    output_mb: float
    nofile: int
    network: str = "none"
    read_only: bool = True
    user: str | None = USER
    drop_capabilities: bool = True
    no_new_privileges: bool = True
    scratch_noexec: bool = True
    # environment variables passed in: only the security suite's control sets one, to show its
    # check would catch a variable from the host; the guardrail never passes any
    env: tuple[str, ...] = ()
    # extra host names: only the network controls set one (host.docker.internal on Linux)
    add_host: tuple[str, ...] = ()

    @classmethod
    def from_config(cls, path: Path = CONFIG) -> Policy:
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))["sandbox"]
        return cls(
            **{
                k: cfg[k]
                for k in (
                    "timeout_s",
                    "memory_mb",
                    "cpus",
                    "pids",
                    "scratch_mb",
                    "output_mb",
                    "nofile",
                )
            }
        )

    def relaxed(self, **changes: Any) -> Policy:
        return replace(self, **changes)


def sources_hash(directory: Path = SANDBOX_DIR) -> str:
    h = hashlib.sha256()
    for name in SOURCES:
        data = (directory / name).read_bytes().replace(b"\r\n", b"\n")
        h.update(name.encode() + b"\0" + hashlib.sha256(data).digest())
    return h.hexdigest()


def image_tag(directory: Path = SANDBOX_DIR) -> str:
    return f"{IMAGE}:{sources_hash(directory)[:12]}"


def docker() -> str:
    exe = shutil.which("docker")
    if exe is None:
        raise SandboxUnavailable("the docker command is not on the PATH")
    return exe


def image_present(tag: str | None = None) -> bool:
    tag = tag or image_tag()
    try:
        done = subprocess.run(
            [docker(), "image", "inspect", "--format", "{{.Id}}", tag],
            capture_output=True,
            timeout=60,
        )
    except (SandboxUnavailable, subprocess.TimeoutExpired):
        return False
    return done.returncode == 0


def run_args(
    policy: Policy,
    tag: str,
    name: str,
    entrypoint: str | None = None,
    command: tuple[str, ...] = (),
) -> list[str]:
    """The `docker run` arguments for one call (the container is removed after it)."""
    # Docker's tmpfs defaults include noexec, so relaxing it (the suite's control) needs `exec`
    scratch = (
        f"/scratch:rw,{'noexec' if policy.scratch_noexec else 'exec'},nosuid,nodev,"
        f"size={policy.scratch_mb}m,mode=0700"
    )
    if policy.user:
        uid, gid = policy.user.split(":")
        scratch += f",uid={uid},gid={gid}"
    args = [
        docker(),
        "run",
        "-i",
        "--pull",
        "never",
        "--name",
        name,
        "--label",
        f"{LABEL}=1",
        "--network",
        policy.network,
        "--hostname",
        "sandbox",
        "--tmpfs",
        scratch,
        "--pids-limit",
        str(policy.pids),
        "--memory",
        f"{policy.memory_mb}m",
        "--memory-swap",
        f"{policy.memory_mb}m",
        "--cpus",
        str(policy.cpus),
        "--ulimit",
        f"nofile={policy.nofile}:{policy.nofile}",
        "--ulimit",
        "core=0",
        "--log-driver",
        "none",
    ]
    if policy.read_only:
        args.append("--read-only")
    if policy.user:
        args += ["--user", policy.user]
    if policy.drop_capabilities:
        args += ["--cap-drop", "ALL"]
    if policy.no_new_privileges:
        args += ["--security-opt", "no-new-privileges"]
    for var in policy.env:
        args += ["--env", var]
    for host in policy.add_host:
        args += ["--add-host", host]
    if entrypoint is not None:
        args += ["--entrypoint", entrypoint]
    return [*args, tag, *command]


@dataclass
class Completed:
    exit_code: int | None
    stdout: bytes
    stderr: bytes
    seconds: float
    timed_out: bool
    output_capped: bool
    oom_killed: bool
    name: str = field(repr=False, default="")


def _drain(stream, buf: bytearray, cap: int, on_cap) -> None:
    """Read a stream to its end, keeping at most `cap` bytes; at the cap, stop the container. The
    rest is read and dropped: Docker stops a container only once its output has been taken, so
    a stream left unread would hold up the kill."""
    capped = False
    while True:
        chunk = stream.read1(65536)
        if not chunk:
            return
        room = cap - len(buf)
        if room > 0:
            buf.extend(chunk[:room])
        if len(chunk) > room and not capped:
            capped = True
            threading.Thread(target=on_cap, daemon=True).start()


def _quiet(*args: str, timeout: float = 30) -> subprocess.CompletedProcess | None:
    try:
        return subprocess.run([docker(), *args], capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None


def run_container(
    stdin: bytes,
    policy: Policy,
    tag: str | None = None,
    entrypoint: str | None = None,
    command: tuple[str, ...] = (),
) -> Completed:
    """Run one container to the end, the time limit or the output cap, whichever comes first,
    then remove it."""
    tag = tag or image_tag()
    name = f"analyst-sandbox-{uuid.uuid4().hex[:12]}"
    args = run_args(policy, tag, name, entrypoint, command)
    out, err = bytearray(), bytearray()
    cap = int(policy.output_mb * 1024 * 1024)
    state = {"timed_out": False, "capped": False}
    kill_lock = threading.Lock()

    def kill(reason: str) -> None:
        with kill_lock:
            if state["timed_out"] or state["capped"]:
                return
            state[reason] = True
        _quiet("kill", name)

    start = time.perf_counter()
    proc = subprocess.Popen(
        args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )

    def feed() -> None:
        try:
            proc.stdin.write(stdin)
        except (BrokenPipeError, OSError):
            pass
        finally:
            try:
                proc.stdin.close()
            except OSError:
                pass

    threads = [
        threading.Thread(target=feed, daemon=True),
        threading.Thread(
            target=_drain, args=(proc.stdout, out, cap, lambda: kill("capped")), daemon=True
        ),
        threading.Thread(
            target=_drain, args=(proc.stderr, err, ERROR_CAP, lambda: kill("capped")), daemon=True
        ),
    ]
    for t in threads:
        t.start()
    timer = threading.Timer(policy.timeout_s, kill, args=("timed_out",))
    timer.start()
    try:
        try:
            code = proc.wait(timeout=policy.timeout_s + KILL_GRACE_S)
        except subprocess.TimeoutExpired:
            kill("timed_out")
            proc.kill()
            code = proc.wait()
    finally:
        timer.cancel()
        for t in threads:
            t.join(timeout=5)
        seconds = time.perf_counter() - start
        inspected = _quiet("inspect", "--format", "{{.State.OOMKilled}}", name)
        oom = bool(inspected and inspected.returncode == 0 and inspected.stdout.strip() == b"true")
        _quiet("rm", "-f", name)
    return Completed(
        code, bytes(out), bytes(err), seconds, state["timed_out"], state["capped"], oom, name
    )


def jsonable(v: Any) -> Any:
    """A database value as the sandbox reads it: numbers, yes/no, text or null."""
    if v is None or isinstance(v, bool | int | str):
        return v
    if isinstance(v, float):
        return v if v == v and v not in (float("inf"), float("-inf")) else None
    if isinstance(v, decimal.Decimal):
        return float(v) if v.is_finite() else None
    if isinstance(v, dt.datetime | dt.date | dt.time):
        return v.isoformat()
    return str(v)


def run_jobs(jobs: list[dict], policy: Policy | None = None, tag: str | None = None) -> list[dict]:
    """Run analysis jobs in one container; one result per job, in order."""
    policy = policy or Policy.from_config()
    tag = tag or image_tag()
    if not image_present(tag):
        raise SandboxUnavailable(
            f"the sandbox image {tag} is not built (uv run python scripts/70_build_sandbox.py)"
        )
    payload = json.dumps(
        {"version": PROTOCOL_VERSION, "jobs": jobs}, sort_keys=True, allow_nan=False
    ).encode("utf-8")
    done = run_container(payload, policy, tag)
    if done.timed_out:
        raise SandboxError("timeout", f"no result within {policy.timeout_s} s")
    if done.output_capped:
        raise SandboxError("output_too_large", f"more than {policy.output_mb} MB of output")
    if done.oom_killed:
        raise SandboxError("memory", f"the analysis needed more than {policy.memory_mb} MB")
    try:
        doc = json.loads(done.stdout.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        tail = done.stderr.decode("utf-8", "replace")[-500:]
        raise SandboxError("failed", f"exit code {done.exit_code}: {tail}") from None
    if "error" in doc:
        raise SandboxError(doc["error"]["kind"], doc["error"]["message"])
    results = doc.get("results")
    if (
        doc.get("version") != PROTOCOL_VERSION
        or not isinstance(results, list)
        or [r.get("id") for r in results] != [j.get("id") for j in jobs]
    ):
        raise SandboxError("protocol", "the results do not match the jobs")
    return results
