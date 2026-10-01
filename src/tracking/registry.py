"""The agent registry: what is deployed, as a hashed configuration with a champion and a challenger.

The deployable unit is an *agent configuration* (configs/agents/): a design on a model, the
prompts it uses, the calibration of its confidence and its decline threshold, an optional router
to a larger model, and whether the statistical guardrail runs. `resolve` turns the file into
everything that determines the system's behavior: each prompt's sha256, the model's request
settings, the fitted calibrators and the threshold, read from committed files. The sha256 of that
resolved form identifies the version, so a changed prompt, model, design or calibrator is a new
version, never a silent edit of an old one.

`results/registry/state.json` is the committed record of the registry: every registered version
(with its resolved form and its evaluation file) and which version holds each alias, `champion` and
`challenger`. MLflow's model registry mirrors it (`sync`), as a view the service can ask for
the champion; the committed state is what CI checks and what the service image carries.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent.parent
AGENTS_DIR = "configs/agents"
STATE = "results/registry/state.json"
PROMOTIONS = "results/registry/promotions.jsonl"
EVALUATIONS_DIR = "results/registry/evaluations"
MODEL_NAME = "analyst-agent"
ALIASES = ("champion", "challenger")


def _yaml(path: Path) -> Any:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def prompt_sha256(path: Path) -> str:
    """The sha256 of a prompt file, line endings normalized as the agent hashes it."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def sha256_of(obj: Any) -> str:
    return hashlib.sha256(canonical(obj).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def config_names(root: Path = ROOT) -> list[str]:
    return sorted(p.stem for p in (root / AGENTS_DIR).glob("*.yaml"))


def _prompts(cfg: dict[str, Any], agent: dict[str, Any], guardrail: dict[str, Any], root: Path):
    design = agent["designs"][cfg["design"]]
    roles = {"single_shot" if not design["tools"] else "agent": None}
    if design.get("narrow"):
        roles["narrow"] = None
    files = {role: agent["prompts"][role] for role in roles}
    if cfg.get("guardrail"):
        for call in guardrail["calls"].values():
            files[Path(call["prompt"]).stem] = call["prompt"]
    return {
        role: {"file": f, "sha256": prompt_sha256(root / f)} for role, f in sorted(files.items())
    }


def _opus_calibrator(root: Path, platt_source: str) -> dict[str, Any]:
    """A Platt fit of the larger model's stated confidence on the calibration split's answers
    (the same fit as the primary model's: declined answers at confidence 0)."""
    from src.eval.calibrate import Platt
    from src.eval.records import answered_correct, read_records
    from src.eval.reports import split_ids
    from src.eval.summary import answered_confidence

    path = root / platt_source
    ids = split_ids("ablation")
    records = [r for r in read_records(path) if r["question_id"] in ids]
    if len(records) != len(ids):
        raise ValueError(f"{platt_source} must hold the whole calibration split")
    x = answered_confidence([r["confidence"] for r in records], [r["declined"] for r in records])
    fit = Platt.fit(x, [answered_correct(r) for r in records])
    return {**fit.to_dict(), "fitted_on": "calibration split", "questions": len(records)}


def resolve(name: str, root: Path = ROOT) -> dict[str, Any]:
    """Everything that determines the configuration's behavior, from committed files."""
    cfg = _yaml(root / AGENTS_DIR / f"{name}.yaml")
    if cfg["name"] != name:
        raise ValueError(f"{name}.yaml names itself {cfg['name']!r}")
    agent = _yaml(root / "configs/agent.yaml")
    guardrail = _yaml(root / "configs/guardrail.yaml")
    confidence = _yaml(root / "configs/confidence.yaml")
    calibration = json.loads(
        (root / "results/metrics/calibration.json").read_text(encoding="utf-8")
    )
    design = agent["designs"][cfg["design"]]
    out: dict[str, Any] = {
        "name": name,
        "description": " ".join(cfg["description"].split()),
        "design": cfg["design"],
        "model": cfg["model"],
        "samples": design["samples"],
        "evidence": cfg["evidence"],
        "request_settings": agent["models"][cfg["model"]],
        "prompts": _prompts(cfg, agent, guardrail, root),
        "calibration": {
            "source": "results/metrics/calibration.json",
            "calibrator": calibration["calibration_split"]["calibrators"]["platt"],
            "decline_threshold": calibration["decline"]["chosen_on_calibration_split"]["threshold"],
            "target_accuracy": calibration["decline"]["target_accuracy"],
            "fitted_on_questions": calibration["calibration_split"]["questions"],
        },
        # the threshold below which the system withholds an answer's confidence as too low to
        # state: the primary model's own, or, with a router, the whole system's (below)
        "decline_threshold": calibration["decline"]["chosen_on_calibration_split"]["threshold"],
        "router": None,
        "guardrail": bool(cfg.get("guardrail")),
    }
    if cfg.get("router"):
        r = cfg["router"]
        esc = confidence[r["settings_from"]]
        if esc["model"] != r["model"]:
            raise ValueError("the router's model is not the escalation arm's model")
        out["router"] = {
            "model": r["model"],
            "request_settings": esc["settings"],
            "routes_below": r["routes_below"],
            "calibrator": _opus_calibrator(
                root,
                f"results/runs/escalation/ablation-{cfg['design']}-{r['model']}-evidence.jsonl",
            ),
        }
        from src.eval import promotion

        out["decline_threshold"] = promotion.decline_threshold(out, root, confidence["decline"])[
            "threshold"
        ]
    return out


# ---------------------------------------------------------------- the committed state


def read_state(root: Path = ROOT) -> dict[str, Any]:
    return json.loads((root / STATE).read_text(encoding="utf-8"))


def write_state(state: dict[str, Any], root: Path = ROOT) -> None:
    path = root / STATE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(state, indent=1, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def read_promotions(root: Path = ROOT) -> list[dict[str, Any]]:
    path = root / PROMOTIONS
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def champion_name(state: dict[str, Any]) -> str:
    return state["aliases"]["champion"]


def champion(state: dict[str, Any]) -> dict[str, Any]:
    """The resolved configuration the champion alias points to."""
    return state["versions"][champion_name(state)]["config"]


def problems(root: Path = ROOT) -> list[str]:
    """What is wrong with the registry as committed: a configuration that differs from its
    registered version, one that was never registered, a version without an evaluation of exactly
    that configuration, an alias that points nowhere, a promotion log that does not match."""
    out: list[str] = []
    state = read_state(root)
    versions = state["versions"]
    for name in config_names(root):
        if name not in versions:
            out.append(f"{name}: not registered (register it and evaluate it as a challenger)")
            continue
        now = sha256_of(resolve(name, root))
        if now != versions[name]["config_sha256"]:
            out.append(
                f"{name}: its prompts, model, design or calibration changed since it was "
                "registered; register the new version with a fresh evaluation"
            )
    for name, v in versions.items():
        if sha256_of(v["config"]) != v["config_sha256"]:
            out.append(f"{name}: the registered configuration does not match its hash")
        ev_path = root / v["evaluation"]
        if not ev_path.exists():
            out.append(f"{name}: no evaluation file at {v['evaluation']}")
        elif (
            json.loads(ev_path.read_text(encoding="utf-8")).get("config_sha256")
            != v["config_sha256"]
        ):
            out.append(f"{name}: its evaluation file is for another version of the configuration")
    for alias in ALIASES:
        target = state["aliases"].get(alias)
        if target is not None and target not in versions:
            out.append(f"alias {alias} points to {target}, which is not registered")
    if "champion" not in state["aliases"]:
        out.append("no champion")
    log = read_promotions(root)
    passed = [p for p in log if p["decision"]["promote"]]
    expected = passed[-1]["challenger"] if passed else state.get("initial_champion")
    if expected != champion_name(state):
        out.append(
            f"the champion is {champion_name(state)} but the promotion log says {expected}: "
            "an alias moves only through a promotion"
        )
    return out


# ---------------------------------------------------------------- MLflow's view


def sync_mlflow(tracking_uri: str, root: Path = ROOT) -> dict[str, Any]:
    """Mirror the committed registry in MLflow's model registry: one model version per registered
    configuration (its resolved form logged as an artifact, its hash as a tag) and the aliases
    on the versions they point to."""
    import mlflow
    from mlflow import MlflowClient

    from src.tracking.mlflow_setup import tolerant_console

    tolerant_console()
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri)
    state = read_state(root)
    try:
        client.create_registered_model(MODEL_NAME, description="Agent configurations")
    except Exception:  # it exists
        pass
    if client.get_experiment_by_name("analyst-registry") is None:
        client.create_experiment("analyst-registry")
    exp = client.get_experiment_by_name("analyst-registry").experiment_id
    existing = {
        mv.tags.get("config_sha256"): mv
        for mv in client.search_model_versions(f"name='{MODEL_NAME}'")
    }
    by_name: dict[str, str] = {}
    for name, v in state["versions"].items():
        mv = existing.get(v["config_sha256"])
        if mv is None:
            run = client.create_run(exp, run_name=f"register {name}")
            client.log_dict(run.info.run_id, v["config"], "agent_config/config.json")
            client.log_param(run.info.run_id, "config_sha256", v["config_sha256"])
            client.set_terminated(run.info.run_id)
            mv = client.create_model_version(
                MODEL_NAME,
                source=f"runs:/{run.info.run_id}/agent_config",
                run_id=run.info.run_id,
                tags={"config_sha256": v["config_sha256"], "config_name": name},
            )
        by_name[name] = mv.version
    for alias in ALIASES:
        target = state["aliases"].get(alias)
        if target is None:
            try:
                client.delete_registered_model_alias(MODEL_NAME, alias)
            except Exception:
                pass
        else:
            client.set_registered_model_alias(MODEL_NAME, alias, by_name[target])
    return {"model": MODEL_NAME, "versions": by_name, "aliases": dict(state["aliases"])}


def mlflow_alias(tracking_uri: str, alias: str = "champion") -> str:
    """The configuration name MLflow's registry holds an alias on."""
    from mlflow import MlflowClient

    mv = MlflowClient(tracking_uri).get_model_version_by_alias(MODEL_NAME, alias)
    return mv.tags["config_name"]


def load_champion(root: Path = ROOT, tracking_uri: str | None = None) -> dict[str, Any]:
    """The resolved champion configuration. With a tracking URI the alias is read from MLflow's
    registry and must agree with the committed state; without one, the committed state decides
    (the service image carries only that)."""
    state = read_state(root)
    name = champion_name(state)
    if tracking_uri:
        held = mlflow_alias(tracking_uri, "champion")
        if held != name:
            raise RuntimeError(
                f"MLflow's champion is {held} but the committed registry says {name}; "
                "run scripts/86_registry.py sync"
            )
    return state["versions"][name]["config"]
