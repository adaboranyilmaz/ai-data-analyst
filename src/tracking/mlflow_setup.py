"""Where MLflow records runs.

The default store is local SQLite (`sqlite:///mlflow.db`, gitignored). It is a view, not a
source: runs are rebuilt from the committed results files, so no number exists only in
MLflow. `MLFLOW_TRACKING_URI` points it elsewhere, such as a tracking server.
"""

from __future__ import annotations

import os
import sys

DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"


def tracking_uri() -> str:
    return os.environ.get("MLFLOW_TRACKING_URI") or DEFAULT_TRACKING_URI


def configure(experiment: str) -> str:
    """Point MLflow at the tracking store and select (or create) an experiment."""
    import mlflow

    uri = tracking_uri()
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment)
    return uri


def tolerant_console() -> None:
    """MLflow's client prints an emoji when a run ends on a tracking server; a console whose code
    page cannot encode it (Windows) would raise in the middle of a logging call. Write what the
    console cannot show as a replacement character instead."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
