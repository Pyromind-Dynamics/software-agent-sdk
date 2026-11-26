"""Shared, side-effect-free W&B location and credential extraction."""

import math
import re
from typing import Any
from urllib.parse import quote
from uuid import UUID


WANDB_BUILDER_NODE_TYPE = "PyroMindWandbConfigBuilderNode"
WANDB_BUILDER_NODE_TYPES = (WANDB_BUILDER_NODE_TYPE, "WandbConfigBuilderNode")
PYROMIND_WANDB_BASE_URL = "https://worb-cn-east-1.pyromind-asia.cn"
RUN_URL_RE = re.compile(
    rf"(?:https://wandb\.ai|{re.escape(PYROMIND_WANDB_BASE_URL)})"
    r"/([^/\s]+)/([^/\s]+)/runs/([\w-]+)"
)
RUN_SETUP_RE = re.compile(r"wandb:\s+setting up run\s+(\w+)")
RUN_PATH_RE = re.compile(r"run-\d{8}_\d{6}-(\w+)")
CONFIG_LINE_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*:\s*(.*?)\s*$")


def node_id(node: dict[str, Any]) -> str:
    return next(
        (
            str(node[k])
            for k in ("node_code", "node_id", "id", "nodeId", "code")
            if node.get(k) is not None
        ),
        "",
    )


def node_type(node: dict[str, Any]) -> str:
    data = node.get("data")
    if isinstance(data, dict) and isinstance(data.get("nodeType"), str):
        return data["nodeType"]
    return next(
        (
            node[k]
            for k in ("node_type", "type", "nodeType", "node_class")
            if isinstance(node.get(k), str)
        ),
        "",
    )


def _workflow_edges(payload: dict[str, Any]) -> list[dict[str, Any]]:
    edges = payload.get("edges")
    if isinstance(edges, list):
        return [edge for edge in edges if isinstance(edge, dict)]
    for name in ("workflow", "data"):
        nested = payload.get(name)
        if isinstance(nested, dict) and (edges := _workflow_edges(nested)):
            return edges
    return []


def wandb_builder_for_training(
    nodes: list[dict[str, Any]], payload: dict[str, Any], training_node_id: str | None
) -> dict[str, Any] | None:
    builders = [node for node in nodes if "WandbConfigBuilderNode" in node_type(node)]
    bindings = [
        edge
        for edge in _workflow_edges(payload)
        if edge.get("targetHandle") == "wandb_config"
    ]
    if training_node_id and bindings:
        target_ids = {training_node_id}
        for node in nodes:
            if node_id(node) == training_node_id and node.get("id") is not None:
                target_ids.add(str(node["id"]))
        sources = {
            str(edge.get("source"))
            for edge in bindings
            if str(edge.get("target")) in target_ids
        }
        builders = [
            node
            for node in builders
            if node_id(node) in sources or str(node.get("id")) in sources
        ]
    return builders[0] if len(builders) == 1 else None


def worb_run_lookup_url(entity: str, project: str, run: str) -> str:
    parts = (entity, project, run)
    if any(not part or part in {".", ".."} or "/" in part for part in parts):
        raise ValueError("Invalid W&B run path")
    owner, project_name, run_name = (quote(part, safe="") for part in parts)
    return f"{PYROMIND_WANDB_BASE_URL}/{owner}/{project_name}/runs/{run_name}"


def worb_run_api_url(location: str) -> str:
    path = location.removeprefix(PYROMIND_WANDB_BASE_URL)
    match = re.fullmatch(r"/runs/([a-fA-F0-9-]{36})", path)
    if not match:
        raise ValueError("Invalid W&B run redirect")
    return f"{PYROMIND_WANDB_BASE_URL}/api/runs/{UUID(match[1])}"


def finite_number(value: object) -> float | None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except OverflowError:
        return None
    return number if math.isfinite(number) else None


def worb_history_point(point: object) -> tuple[str, int, float, float]:
    if not isinstance(point, dict):
        raise ValueError("Invalid W&B history point")
    key, index = point.get("k"), point.get("i")
    step, value = finite_number(point.get("s")), finite_number(point.get("v"))
    if (
        not isinstance(key, str)
        or type(index) is not int
        or index < 0
        or step is None
        or value is None
        or "min" in point
        or "max" in point
    ):
        raise ValueError("Invalid W&B history point")
    return key, index, step, value


def config_text(text: str) -> dict[str, str]:
    return {
        match.group(1): match.group(2)
        for line in text.splitlines()
        if (match := CONFIG_LINE_RE.match(line))
    }


def wandb_environment(output: dict[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}

    def merge(text: str) -> None:
        for key, value in config_text(text).items():
            if value:
                result.setdefault(key.upper(), value)

    config = output.get("wandb_config")
    if isinstance(config, dict):
        result.update({str(k).upper(): str(v) for k, v in config.items()})
    elif isinstance(config, str):
        merge(config)
    for key, value in output.items():
        name = key.upper()
        if name.startswith("WANDB_") and name != "WANDB_CONFIG" and value is not None:
            result[name] = str(value)
    entries = output.get("entries")
    if isinstance(entries, list):
        merge(
            "".join(
                e.get("m", "")
                for e in entries
                if isinstance(e, dict) and isinstance(e.get("m"), str)
            )
        )
    if not result:
        for value in output.values():
            if isinstance(value, str) and value.strip():
                merge(value)
                break
    return {key: value for key, value in result.items() if key.startswith("WANDB_")}


def wandb_run_reference(text: str) -> tuple[str, str, str] | None:
    text = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", text)
    if match := RUN_URL_RE.search(text):
        return match.group(1), match.group(2), match.group(3)
    if match := RUN_SETUP_RE.search(text) or RUN_PATH_RE.search(text):
        return "", "", match.group(1)
    return None
