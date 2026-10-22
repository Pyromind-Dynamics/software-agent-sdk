"""Executed workflow provenance and read-only discovery on the Storage mount."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path, PurePosixPath
from typing import Any


def storage_path(value: str) -> str:
    if any(marker in value for marker in ("${", "{{", "}}")):
        raise ValueError(f"Unresolved Storage path: {value!r}")
    if value.startswith("/workspace/"):
        value = value[len("/workspace") :]
    path = PurePosixPath(value)
    if not path.is_absolute() or ".." in path.parts or str(path) == "/":
        raise ValueError(f"Unresolved Storage path: {value!r}")
    return str(path)


def resolve_artifacts(
    client: Any, task_id: str, training_node_id: str | None = None
) -> dict[str, Any]:
    import yaml
    from platform_output import parse_node_output
    from train_analysis import _find_nodes, _node_identifier, _node_type

    payload = client.task_workflow_result(task_id)

    def graph_from(value):
        if any(
            isinstance(value.get(key), list)
            for key in ("nodes", "node_list", "node_infos")
        ):
            return value
        for key in ("workflow", "data"):
            if isinstance(value.get(key), dict):
                nested = graph_from(value[key])
                if nested:
                    return nested
        return {}

    graph = graph_from(payload) or payload
    nodes = _find_nodes(graph)
    by_id = {_node_identifier(node): node for node in nodes}
    train_nodes = [
        node
        for node in nodes
        if ("Train" in _node_type(node) or "train" in _node_type(node))
        and "Config" not in _node_type(node)
    ]
    if training_node_id:
        train_nodes = [
            node for node in train_nodes if _node_identifier(node) == training_node_id
        ]
    if not train_nodes:
        raise ValueError("No matching training node in the executed task")
    cache = {}

    def output(node_id: str) -> dict[str, Any]:
        if node_id not in cache:
            cache[node_id] = parse_node_output(client.node_output(node_id, task_id))
        return cache[node_id]

    def resolve(value: Any, seen: set[str]) -> Any:
        if isinstance(value, dict):
            source = value.get("node_id") or value.get("nodeId")
            key = value.get("output") or value.get("output_name")
            if source is not None and key:
                source = str(source)
                if source in seen:
                    return None
                return resolve(output(source).get(str(key)), seen | {source})
            return {k: resolve(v, seen) for k, v in value.items()}
        return value

    def parameters(node: dict[str, Any]) -> dict[str, Any]:
        data = node.get("data", {})
        combined = {}
        for owner in (data, node):
            if isinstance(owner, dict):
                for key in ("config", "inputs", "parameters", "resolved_inputs"):
                    raw = owner.get(key)
                    if isinstance(raw, str):
                        try:
                            raw = json.loads(raw)
                        except ValueError:
                            continue
                    if isinstance(raw, dict):
                        combined.update(raw)
        return resolve(combined, set())

    # Resolve graph-bound inputs from the executed graph, never the current DSL.
    edges = graph.get("edges", [])
    results = []
    for node in train_nodes:
        node_id = _node_identifier(node)
        config = parameters(node)
        for edge in edges:
            if str(edge.get("target")) == node_id:
                source = str(edge.get("source"))
                field = edge.get("targetHandle")
                source_field = edge.get("sourceHandle")
                if field and source_field:
                    resolved = output(source).get(source_field)
                    if resolved is not None:
                        config[field] = resolved
                    elif source in by_id:
                        upstream = parameters(by_id[source])
                        if source_field in upstream:
                            config[field] = upstream[source_field]
        issues = []
        raw_path = config.get("output_path")
        if not isinstance(raw_path, str) and node_id:
            raw_path = output(node_id).get("model_output_path")
        path = None
        try:
            if isinstance(raw_path, str):
                path = storage_path(raw_path)
        except ValueError as exc:
            issues.append(str(exc))
        if path is None:
            issues.append(
                "output_path could not be resolved from executed inputs/outputs"
            )
        # Config builders emit JSON/YAML text on their named output ports.
        for key in ("model_config", "dataset_config", "training_config", "lora_config"):
            value = config.get(key)
            if isinstance(value, str):
                try:
                    parsed = yaml.safe_load(value)
                    if isinstance(parsed, dict):
                        config[key] = parsed
                except yaml.YAMLError:
                    issues.append(f"unparsed_{key}")
        model_config = config.get("model_config")
        if isinstance(model_config, dict):
            config.setdefault("model_path", model_config.get("model_path"))
        # Whitelist provenance: training configs may contain credentials.
        allowed = (
            "model_path",
            "base_model",
            "model_config",
            "lora_config",
            "grpo_extra_config",
            "train_data_path",
            "val_data_path",
            "eval_data_path",
            "dataset_config",
            "training_config",
            "save_steps",
            "eval_steps",
        )
        results.append(
            {
                "training_node_id": node_id,
                "training_type": _node_type(node),
                "training_method": "lora"
                if config.get("lora_config")
                else "full_or_unknown",
                "output_path": path,
                "inputs": {key: config[key] for key in allowed if key in config},
                "issues": issues,
            }
        )
    return {
        "schema_version": 1,
        "task_id": task_id,
        "training_nodes": results,
        "requires_node_selection": len(results) > 1,
    }


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_checkpoint(
    path: Path, step: int, reference: str, base_model: str | None = None
) -> dict[str, Any]:
    names = {entry.name for entry in path.iterdir() if entry.is_file()}
    kind = (
        "lora"
        if "adapter_config.json" in names
        else "full"
        if "config.json" in names
        else "unknown"
    )
    issues = []
    if kind == "lora":
        config = json.loads((path / "adapter_config.json").read_text())
        base_model = config.get("base_model_name_or_path") or base_model
        if not base_model:
            issues.append("missing_base_model")
        required = [
            name
            for name in ("adapter_model.safetensors", "adapter_model.bin")
            if name in names
        ]
    else:
        required = [
            name for name in ("model.safetensors", "pytorch_model.bin") if name in names
        ]
        for name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
            if name in names:
                index = json.loads((path / name).read_text())
                required.extend(set(index.get("weight_map", {}).values()))
        if "config.json" not in names:
            issues.append("missing_model_config")
    if not required:
        issues.append("missing_weights")
    for name in required:
        file = path / name
        if (
            PurePosixPath(name).name != name
            or not file.is_file()
            or file.stat().st_size == 0
        ):
            issues.append("missing_or_empty_weight:" + name)
    state_file = path / "trainer_state.json"
    if state_file.exists():
        state = json.loads(state_file.read_text())
        if state.get("global_step") is not None and state["global_step"] != step:
            issues.append("trainer_state_step_mismatch")
    return {
        "step": step,
        "path": reference,
        "kind": kind,
        "base_model": base_model,
        "complete": not issues,
        "issues": issues,
    }


def discover(
    directory: str, mount: str = "storage", base_model: str | None = None
) -> dict[str, Any]:
    reference = storage_path(directory)
    root = Path(mount) / reference.lstrip("/")
    if not root.is_dir():
        raise FileNotFoundError(
            f"Training output directory is unavailable: {reference}"
        )
    checkpoints = []
    evaluations = []
    for path in root.iterdir():
        checkpoint = re.fullmatch(r"checkpoint-(\d+)", path.name)
        evaluation = re.fullmatch(r"eval_step_(\d+)\.jsonl", path.name)
        if checkpoint and path.is_dir():
            step = int(checkpoint[1])
            try:
                checkpoints.append(
                    inspect_checkpoint(
                        path, step, reference + "/" + path.name, base_model
                    )
                )
            except (OSError, ValueError, TypeError) as exc:
                checkpoints.append(
                    {
                        "step": step,
                        "path": reference + "/" + path.name,
                        "kind": "unknown",
                        "complete": False,
                        "issues": [str(exc)],
                    }
                )
        elif evaluation and path.is_file():
            samples, fields, errors, count = [], set(), [], 0
            with path.open() as stream:
                for line_no, line in enumerate(stream, 1):
                    if not line.strip():
                        continue
                    count += 1
                    try:
                        row = json.loads(line)
                        if not isinstance(row, dict):
                            raise ValueError("record must be an object")
                        fields.update(row)
                        if len(samples) < 3:
                            samples.append(row)
                    except ValueError as exc:
                        if len(errors) < 10:
                            errors.append(f"line {line_no}: {exc}")
            evaluations.append(
                {
                    "step": int(evaluation[1]),
                    "path": reference + "/" + path.name,
                    "sha256": file_digest(path),
                    "rows": count,
                    "fields": sorted(fields),
                    "samples": samples,
                    "errors": errors,
                }
            )
    return {
        "output_path": reference,
        "checkpoints": sorted(checkpoints, key=lambda x: x["step"]),
        "evaluations": sorted(evaluations, key=lambda x: x["step"]),
    }
