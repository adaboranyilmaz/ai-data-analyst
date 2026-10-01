"""Build the statistics sandbox's image.

The packages come from the project's lock file: sandbox/requirements.txt is exported from the
`sandbox` dependency group (versions and hashes), then the image is built from sandbox/ and
tagged with a hash of its files (src/stats/sandbox.py), so the code that runs a call can tell
whether the image matches the sources. Building needs the network (the base image and the
packages); running the image never has it.

Usage:
    uv run python scripts/70_build_sandbox.py            # export, then build
    uv run python scripts/70_build_sandbox.py --check    # the export matches the lock file
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.stats.sandbox import LABEL, SANDBOX_DIR, docker, image_tag, sources_hash  # noqa: E402

REQUIREMENTS = SANDBOX_DIR / "requirements.txt"
EXPORT = [
    "uv",
    "export",
    "--only-group",
    "sandbox",
    "--format",
    "requirements-txt",
    "--no-emit-project",
    "--frozen",
]


def exported() -> str:
    done = subprocess.run(EXPORT, cwd=ROOT, capture_output=True, text=True, check=True)
    return done.stdout.replace("\r\n", "\n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--check", action="store_true", help="only check the requirements export")
    args = p.parse_args()

    text = exported()
    current = REQUIREMENTS.read_text(encoding="utf-8") if REQUIREMENTS.exists() else None
    if args.check:
        if current != text:
            sys.exit(f"{REQUIREMENTS.relative_to(ROOT)} differs from the lock file's export")
        print("the sandbox requirements match the lock file")
        return
    if current != text:
        REQUIREMENTS.write_bytes(text.encode("utf-8"))
        print(f"wrote {REQUIREMENTS.relative_to(ROOT)}")

    tag = image_tag()
    subprocess.run(
        [
            docker(),
            "build",
            "--pull=false",
            "--label",
            f"{LABEL}.sources={sources_hash()}",
            "-t",
            tag,
            str(SANDBOX_DIR),
        ],
        check=True,
    )
    print(f"built {tag}")


if __name__ == "__main__":
    main()
