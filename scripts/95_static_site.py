"""Write the data files of the static demo (the recorded runs as plain files).

Run after `npm run build:static` in ui/, which builds the page; this adds its data. Writes
results/metrics/static_site.json. Needs no database and makes no model call.

Usage:
    uv run python scripts/95_static_site.py [--out ui/dist-static]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.bird import write_json  # noqa: E402
from src.serving import champion as champion_mod  # noqa: E402
from src.serving import static_site  # noqa: E402
from src.serving.meter import Meter, config  # noqa: E402
from src.serving.store import RunStore  # noqa: E402

OUT = ROOT / "results/metrics/static_site.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="ui/dist-static")
    out = ROOT / ap.parse_args().out
    champion = champion_mod.load()
    cfg = champion.apply(config())
    store = RunStore(ROOT / cfg["curated"]["out_dir"])
    summary = static_site.build(out, store, Meter.load(cfg), cfg, champion.info())
    write_json(
        OUT,
        {
            "note": "the recorded runs as plain files for the static demo; parity with the "
            "service is tested (tests/test_static_site.py)",
            "replay": cfg["replay"],
            **summary,
        },
    )
    print(f"{summary['runs']} runs, {summary['files']} files, {summary['bytes']:,} bytes -> {out}")


if __name__ == "__main__":
    main()
