"""Read training provenance and mounted checkpoint metadata without an Agent."""

from __future__ import annotations

import hashlib
import json
import math
import re
import sys
import tempfile
from pathlib import Path
from typing import Any

from analysis_helpers import TRAINING_STEP_KEYS
from data_sources.wandb import RUN_URL_RE, WandbDataSource
from train_analysis import PlatformClient, _default_api_base, _find_nodes
from training_artifacts import inspect_checkpoint, resolve_artifacts, storage_path


def observe_files(root: Path, output_path: str, base_model: str | None) -> list[dict]:
    directory = (root / storage_path(output_path).lstrip("/")).resolve()
    if not directory.is_relative_to(root.resolve()):
        raise ValueError("training output escapes Storage mount")
    if not directory.exists():
        return []
    results = []
    for path in sorted(directory.iterdir()):
        match = re.fullmatch(r"checkpoint-(\d+)", path.name)
        if not match or not path.is_dir():
            continue
        step = int(match[1])
        reference = output_path.rstrip("/") + "/" + path.name
        artifact = {
            "step": step,
            "path": reference,
            "complete": False,
            "kind": "unknown",
            "issues": [],
        }
        signature = None
        try:
            if any(
                not file.resolve().is_relative_to(root.resolve())
                for file in [path, *path.iterdir()]
            ):
                raise ValueError("checkpoint escapes Storage mount")
            artifact = inspect_checkpoint(path, step, reference, base_model)
            if artifact["kind"] == "lora" and base_model and base_model.startswith("/"):
                artifact["base_model"] = storage_path(base_model)
            entries = []
            for file in sorted(path.iterdir()):
                if file.is_file() and (
                    file.suffix in {".json", ".safetensors", ".bin"}
                    and not file.name.startswith(("optimizer", "scheduler"))
                ):
                    stat = file.stat()
                    entries.append([file.name, stat.st_size, stat.st_mtime_ns])
            if entries and artifact["complete"]:
                signature = hashlib.sha256(json.dumps(entries).encode()).hexdigest()
        except (OSError, ValueError, TypeError, KeyError) as error:
            artifact["complete"] = False
            artifact["issues"] = [f"checkpoint_inspection:{type(error).__name__}"]
        results.append({"artifact": artifact, "fingerprint": signature})
    return results


def telemetry(client: PlatformClient, task_id: str, node_id: str) -> dict[str, Any]:
    source = WandbDataSource()
    nodes = _find_nodes(client.task_workflow_result(task_id))
    creds = source.extract_creds(nodes, client, task_id)
    if not creds:
        return {"warning": "W&B credentials unavailable"}
    matches = list(RUN_URL_RE.finditer(client.node_log(node_id, task_id)))
    if not matches:
        return {"warning": "W&B run has not appeared in the training log"}
    entity, project, run_id = matches[-1].groups()
    with tempfile.TemporaryDirectory(prefix="checkpoint-wandb-") as directory:
        source.connect({**creds, "wandb_tmp": directory})
        data = source.get_run_data(f"{entity}/{project}", run_id, keys=[])
    summary = data["summary"]
    step = next(
        (
            int(summary[key])
            for key in TRAINING_STEP_KEYS
            if isinstance(summary.get(key), (int, float))
            and not isinstance(summary[key], bool)
            and math.isfinite(summary[key])
            and summary[key] >= 0
        ),
        None,
    )
    metrics = {
        key: value
        for key, value in summary.items()
        if isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and any(
            word in key.lower()
            for word in ("loss", "accuracy", "reward", "f1", "recall")
        )
    }
    return {
        "training_step": step,
        "run_id": run_id,
        "state": data["state"],
        "metrics": metrics,
    }


def execute(payload: dict[str, Any]) -> dict[str, Any]:
    headers = payload["headers"]
    client = PlatformClient(
        base_url=payload.get("api_base") or _default_api_base(),
        cookie=headers.get("cookie", ""),
        authorization=headers.get("authorization", ""),
        cluster=headers.get("x-cluster", ""),
    )
    if payload["operation"] == "verify":
        workflow = client.task_workflow_result(payload["task_id"])
        if "/workspace" + payload["output_path"] not in json.dumps(workflow):
            raise ValueError("task workflow does not reference this evaluation output")
        return {"verified": True}
    if payload["operation"] == "resolve":
        resolved = resolve_artifacts(
            client, payload["task_id"], payload["training_node_id"]
        )
        nodes = resolved["training_nodes"]
        if len(nodes) != 1 or not nodes[0]["output_path"]:
            raise ValueError("training output is not resolved yet")
        return nodes[0]
    if not payload.get("output_path"):
        nodes = resolve_artifacts(
            client, payload["task_id"], payload["training_node_id"]
        )["training_nodes"]
        if len(nodes) != 1 or not nodes[0]["output_path"]:
            raise ValueError("training output is not resolved yet")
        payload["output_path"] = nodes[0]["output_path"]
        payload["base_model"] = nodes[0]["inputs"].get("base_model") or nodes[0][
            "inputs"
        ].get("model_path")
    result = {
        "checkpoints": observe_files(
            Path(payload["mount_root"]),
            payload["output_path"],
            payload.get("base_model"),
        )
    }
    try:
        result["telemetry"] = telemetry(
            client, payload["task_id"], payload["training_node_id"]
        )
    except Exception as error:
        result["telemetry"] = {"warning": f"W&B unavailable: {type(error).__name__}"}
    return result


if __name__ == "__main__":
    try:
        print(json.dumps(execute(json.load(sys.stdin)), allow_nan=False))
    except Exception as error:
        print(json.dumps({"error": type(error).__name__}))
        sys.exit(1)
