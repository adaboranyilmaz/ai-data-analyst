"""Choose the local model for the free arm: which candidates fit the 4 GB GPU at a useful
context, and which make valid tool calls.

For each candidate in configs/local_models.yaml that is already downloaded (the script never
downloads one):
  1. its capabilities, size, quantisation, licence and weights digest, from Ollama;
  2. GPU fit: the model is loaded at each context size and Ollama's own report of how much
     of it sits in GPU memory is recorded; it fits when all of it does;
  3. tool calling: five fixed questions about a small shop database, each to be answered by
     calling one of three tools. A call is valid if it names a known tool and its arguments
     match that tool's schema (required keys present, non-empty strings, no unknown keys).
     Temperature 0 and a fixed seed; responses go through the response cache.
The selection rule, fixed in the config before anything is measured, is then applied; while
any candidate is still to be downloaded, the choice is recorded as provisional.
Writes results/metrics/local_model_selection.json.

Usage:
    uv run python scripts/00_local_model_check.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.llm.backends import OllamaBackend  # noqa: E402
from src.llm.cache import ResponseCache, replay_only  # noqa: E402
from src.llm.generate import generate_cached  # noqa: E402
from src.llm.types import LLMRequest  # noqa: E402

CONFIG = ROOT / "configs/local_models.yaml"
OUT = ROOT / "results/metrics/local_model_selection.json"
MIB = 2**20

SYSTEM = (
    "You are a data analyst with read-only access to a shop database with two tables: "
    "customers(id, name, city) and orders(id, customer_id, amount, ordered_at). Answer the "
    "user's question by calling exactly one of your tools."
)


def _schema(properties: dict[str, str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {k: {"type": "string", "description": d} for k, d in properties.items()},
        "required": list(properties),
        "additionalProperties": False,
    }


TOOLS = [
    {
        "name": "list_tables",
        "description": "List the tables in the database.",
        "input_schema": _schema({}),
    },
    {
        "name": "describe_table",
        "description": "Show the columns of one table and their types.",
        "input_schema": _schema({"table": "the table's name"}),
    },
    {
        "name": "run_sql",
        "description": "Run one read-only SQL SELECT statement and return the rows.",
        "input_schema": _schema({"sql": "a single SELECT statement"}),
    },
]
PROBES = [  # (question, the tool a correct first step calls)
    ("Which tables are in the database?", "list_tables"),
    ("What columns does the orders table have?", "describe_table"),
    ("How many customers are there?", "run_sql"),
    ("What is the total order amount per city?", "run_sql"),
    ("Show the five largest orders.", "run_sql"),
]


def valid_call(call: dict[str, Any]) -> bool:
    tool = next((t for t in TOOLS if t["name"] == call["name"]), None)
    args = call.get("input")
    if tool is None or not isinstance(args, dict):
        return False
    schema = tool["input_schema"]
    return (
        set(args) <= set(schema["properties"])
        and set(schema["required"]) <= set(args)
        and all(isinstance(v, str) and v.strip() for v in args.values())
    )


def ollama_host() -> str:
    from dotenv import dotenv_values

    host = os.environ.get("OLLAMA_HOST") or dotenv_values(ROOT / ".env").get("OLLAMA_HOST")
    return host or "http://localhost:11434"


def ollama_version(host: str) -> str | None:
    try:
        with urllib.request.urlopen(f"{host}/api/version", timeout=5) as r:
            return json.load(r).get("version")
    except OSError:
        return None


def gpu_name() -> str | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def model_facts(client: Any, model: str, digests: dict[str, str]) -> dict[str, Any]:
    show = client.show(model)
    info = show.modelinfo or {}
    max_ctx = next((v for k, v in info.items() if k.endswith(".context_length")), None)
    return {
        "digest": digests.get(model),
        "capabilities": list(show.capabilities or []),
        "parameter_size": show.details.parameter_size if show.details else None,
        "quantization": show.details.quantization_level if show.details else None,
        "max_context": max_ctx,
        "license_first_line": (show.license or "").strip().splitlines()[0]
        if show.license
        else None,
    }


def gpu_fit(client: Any, model: str, sizes: list[int]) -> list[dict[str, Any]]:
    rows = []
    for n in sizes:
        client.generate(model=model, prompt="", options={"num_ctx": n}, keep_alive="1m")
        loaded = next((m for m in client.ps().models if model in (m.model, m.name)), None)
        if loaded is None:
            raise RuntimeError(f"{model} did not appear in `ollama ps` after loading")
        rows.append(
            {
                "num_ctx": n,
                "context_reported": loaded.context_length,
                "size_mib": round(loaded.size / MIB),
                "gpu_mib": round(loaded.size_vram / MIB),
                "on_gpu": loaded.size_vram >= loaded.size,
            }
        )
        client.generate(model=model, prompt="", keep_alive=0)  # unload before the next size
    return rows


def run_probes(
    backend: OllamaBackend, model: str, cfg: dict, cache: ResponseCache
) -> list[dict[str, Any]]:
    params = OllamaBackend.request_params(
        cfg["options"]["temperature"], cfg["options"]["seed"], cfg["probe_context"]
    )
    rows = []
    for question, expected in PROBES:
        request = LLMRequest.single(
            "ollama", model, SYSTEM, question, cfg["max_tokens"], params, TOOLS
        )
        response, was_cached, _ = generate_cached(backend, request, cache)
        calls = response.tool_calls()
        first = calls[0] if calls else None
        eval_s = (response.extra.get("eval_ms") or 0) / 1000
        rows.append(
            {
                "question": question,
                "expected_tool": expected,
                "tool_calls": calls,
                "text": response.text,
                "valid_call": first is not None and valid_call(first),
                "expected_tool_called": first is not None and first["name"] == expected,
                "output_tokens": response.tokens.output,
                "tokens_per_s": round(response.tokens.output / eval_s, 1) if eval_s else None,
                "cache_key": request.cache_key,
                "from_cache": was_cached,
            }
        )
    return rows


def select(cfg: dict, candidates: dict[str, dict]) -> dict[str, Any]:
    eligible = {
        m: r
        for m, r in candidates.items()
        if r["status"] == "measured"
        and "tools" in r["capabilities"]
        and r["max_context_on_gpu"] >= cfg["min_context"]
    }

    def gpu_at_min(r: dict) -> int:
        return next(f["gpu_mib"] for f in r["gpu_fit"] if f["num_ctx"] == cfg["min_context"])

    selected = max(
        eligible,
        key=lambda m: (
            eligible[m]["valid_call_rate"],
            eligible[m]["max_context_on_gpu"],
            -gpu_at_min(eligible[m]),
        ),
        default=None,
    )
    not_downloaded = [m for m, r in candidates.items() if r["status"] == "not_downloaded"]
    return {
        "selected": selected,
        "provisional": bool(not_downloaded),
        "eligible": sorted(eligible),
        "not_downloaded": not_downloaded,
    }


def main() -> None:
    if replay_only():
        sys.exit("this script measures the local GPU; it does not run in replay-only mode")
    import ollama

    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    host = ollama_host()
    client = ollama.Client(host=host)
    backend = OllamaBackend(client=client)
    cache = ResponseCache(ROOT / "data/cache/llm")
    digests = {m.model: m.digest for m in client.list().models}

    candidates: dict[str, dict[str, Any]] = {}
    for model in cfg["candidates"]:
        if model not in digests:
            candidates[model] = {"status": "not_downloaded"}
            print(f"{model}: not downloaded, skipped")
            continue
        facts = model_facts(client, model, digests)
        if "tools" not in facts["capabilities"]:
            candidates[model] = {"status": "no_tools_capability", **facts}
            print(f"{model}: no tools capability, skipped")
            continue
        sizes = [
            n for n in cfg["context_sizes"] if not facts["max_context"] or n <= facts["max_context"]
        ]
        fit = gpu_fit(client, model, sizes)
        probes = run_probes(backend, model, cfg, cache)
        valid = sum(p["valid_call"] for p in probes)
        candidates[model] = {
            "status": "measured",
            **facts,
            "gpu_fit": fit,
            "max_context_on_gpu": max((f["num_ctx"] for f in fit if f["on_gpu"]), default=0),
            "probes": probes,
            "n_probes": len(probes),
            "n_valid_calls": valid,
            "valid_call_rate": valid / len(probes),
            "n_expected_tool": sum(p["expected_tool_called"] for p in probes),
        }
        client.generate(model=model, prompt="", keep_alive=0)
        print(
            f"{model}: {valid}/{len(probes)} valid tool calls; entirely on the GPU up to "
            f"num_ctx {candidates[model]['max_context_on_gpu']}"
        )

    result = {
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "ollama_version": ollama_version(host),
        "gpu": gpu_name(),
        "config": {k: cfg[k] for k in ("min_context", "probe_context", "max_tokens", "options")},
        "selection_rule": cfg["selection_rule"],
        "probe_system_prompt": SYSTEM,
        "probe_tools": [t["name"] for t in TOOLS],
        "candidates": candidates,
        "selection": select(cfg, candidates),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    s = result["selection"]
    print(f"selected: {s['selected']}{' (provisional)' if s['provisional'] else ''}")
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
