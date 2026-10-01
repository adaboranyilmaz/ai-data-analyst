"""What is delivered is pinned: images by digest, third-party actions by commit."""

from __future__ import annotations

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
DIGEST = re.compile(r"@sha256:[0-9a-f]{64}$")
COMMIT = re.compile(r"@[0-9a-f]{40}$")
FIRST_PARTY = ("actions/", "astral-sh/")


def test_every_image_the_compose_file_pulls_is_pinned_by_digest():
    services = yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))["services"]
    pulled = {n: s["image"] for n, s in services.items() if "build" not in s}
    assert {"postgres", "mlflow", "langfuse-web", "langfuse-worker"} <= set(pulled)
    unpinned = {n: i for n, i in pulled.items() if not DIGEST.search(i)}
    assert unpinned == {}


def test_the_dockerfile_builds_from_pinned_base_images():
    froms = [
        line.split()[1]
        for line in (ROOT / "Dockerfile").read_text(encoding="utf-8").splitlines()
        if line.startswith("FROM ")
    ]
    assert len(froms) == 2 and all(DIGEST.search(f) for f in froms)


def test_third_party_actions_are_pinned_by_commit():
    unpinned = []
    for wf in (ROOT / ".github/workflows").glob("*.yml"):
        for line in wf.read_text(encoding="utf-8").splitlines():
            stripped = line.strip().removeprefix("- ")
            if stripped.startswith("uses:"):
                ref = stripped.split()[1]
                if not ref.startswith(FIRST_PARTY) and not COMMIT.search(ref):
                    unpinned.append((wf.name, ref))
    assert unpinned == []


def test_the_canary_workflow_is_manual_and_capped():
    wf = yaml.safe_load((ROOT / ".github/workflows/canary.yml").read_text(encoding="utf-8"))
    triggers = wf.get("on") or wf.get(True)
    assert set(triggers) == {"workflow_dispatch"}
    assert "max_usd" in triggers["workflow_dispatch"]["inputs"]


def test_the_publish_workflow_runs_the_gate_before_it_pushes():
    wf = yaml.safe_load((ROOT / ".github/workflows/publish.yml").read_text(encoding="utf-8"))
    assert wf["jobs"]["image"]["needs"] == "gate"
    assert wf["jobs"]["image"]["permissions"]["packages"] == "write"
    assert wf["permissions"] == {"contents": "read"}


def test_the_pages_workflow_deploys_only_what_it_built_and_pins_what_holds_the_permission():
    path = ROOT / ".github/workflows/pages.yml"
    wf = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert wf["jobs"]["deploy"]["needs"] == "build"
    assert wf["permissions"] == {"contents": "read"}
    assert wf["jobs"]["deploy"]["permissions"] == {"pages": "write", "id-token": "write"}
    for line in path.read_text(encoding="utf-8").splitlines():
        ref = line.strip().removeprefix("- ")
        if ref.startswith("uses: actions/") and "pages" in ref:
            assert COMMIT.search(ref.split()[1]), ref
