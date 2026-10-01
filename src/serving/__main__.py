"""Run the service.

    uv run python -m src.serving                    # replay mode: recorded runs, no key
    uv run python -m src.serving --mode live        # live questions (ANTHROPIC_API_KEY in .env)

Bound to this machine by default. Connections to a person's own database are allowed only when
the service is bound to a loopback address; bind elsewhere (`--host 0.0.0.0`, as the container
does) and they are off.
"""

from __future__ import annotations

import argparse
import os

import uvicorn

from src.serving.app import Settings, create_app
from src.serving.meter import ROOT

LOOPBACK = ("127.0.0.1", "localhost", "::1")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--mode", choices=("replay", "live"), default=None)
    a = p.parse_args()
    try:  # the container takes its environment from the runtime; a checkout has a .env
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env")
    except ImportError:
        pass
    settings = Settings.from_env()
    if a.mode:
        settings.mode = a.mode
    settings.local_mode = a.host in LOOPBACK and settings.mode == "live"
    if os.environ.get("ANALYST_SERVING_MODE") not in (None, settings.mode):
        os.environ["ANALYST_SERVING_MODE"] = settings.mode
    uvicorn.run(create_app(settings), host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
