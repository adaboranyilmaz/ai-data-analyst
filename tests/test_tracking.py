"""MLflow configuration: the default store, the environment override, and a run recorded to
and read back from a SQLite store."""

from __future__ import annotations

from src.tracking.mlflow_setup import DEFAULT_TRACKING_URI, configure, tracking_uri


def test_default_is_local_sqlite(monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    assert tracking_uri() == DEFAULT_TRACKING_URI


def test_empty_variable_means_the_default(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "")
    assert tracking_uri() == DEFAULT_TRACKING_URI


def test_run_is_recorded_and_read_back(tmp_path, monkeypatch):
    import mlflow

    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    mlflow.set_tracking_uri(uri)
    mlflow.create_experiment("scaffold", artifact_location=(tmp_path / "artifacts").as_uri())
    assert configure("scaffold") == uri
    with mlflow.start_run() as run:
        mlflow.log_param("design", "single_shot")
        mlflow.log_metric("execution_accuracy", 0.5)
    got = mlflow.get_run(run.info.run_id).data
    assert got.params == {"design": "single_shot"}
    assert got.metrics == {"execution_accuracy": 0.5}
