from __future__ import annotations

from typing import Any, Literal

from pyromind_runtime.domain.experiment_plans import (
    ExperimentContext,
    ExperimentStage,
    WorkflowPlan,
)


EXPERIMENT_PLANNING_SKILL = "experiment-planning"
EXPERIMENT_GRAPH_TEMPLATE = "experiment-planning/v1"

ProjectNodeType = Literal["dataset", "inference", "evaluator", "train"]

_TASK_TYPES: dict[ProjectNodeType, str] = {
    "dataset": "dataset.prepare",
    "train": "train.run",
    "inference": "inference.run",
    "evaluator": "evaluator.run",
}

_OUTPUT_PORTS: dict[ProjectNodeType, tuple[str, str, str]] = {
    "dataset": ("dataset_version", "dataset_version", "dataset/v1"),
    "train": ("checkpoint", "train_checkpoint", "checkpoint/v1"),
    "inference": ("predictions", "prediction_set", "prediction-set/v1"),
    "evaluator": ("evaluation", "evaluation_result", "evaluation/v1"),
}

_EXECUTION_ROLES: dict[ProjectNodeType, str] = {
    "dataset": "dataset",
    "train": "train",
    "inference": "inference",
    "evaluator": "eval",
}


def build_graph_payload(
    *,
    plan: WorkflowPlan | ExperimentContext,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    resources = _resources_payload(plan)
    tasks_by_stage = _tasks_by_stage(plan)
    nodes = []
    node_types: dict[str, ProjectNodeType] = {}
    for stage in plan.stages:
        if not stage.enabled:
            continue
        node_type = _stage_node_type(stage)
        if node_type is None:
            continue
        node_types[stage.id] = node_type
        config: dict[str, Any] = {
            "resources": resources,
            "tasks": tasks_by_stage.get(stage.id, []),
        }
        if node_type == "dataset":
            config.update(
                {
                    "capability": "data-processing",
                }
            )
        elif node_type == "evaluator":
            config["artifact_format"] = "atif"
        nodes.append(
            _node(
                node_id=stage.id,
                node_type=node_type,
                title=stage.name,
                launch_mode="agent",
                config=config,
                inputs={},
                output_port=_OUTPUT_PORTS[node_type],
            )
        )
    edges = [
        {"source": edge.source, "target": edge.target}
        for edge in plan.edges
        if edge.source in node_types and edge.target in node_types
    ]
    return nodes, edges


def _stage_node_type(stage: ExperimentStage) -> ProjectNodeType | None:
    if stage.kind == "dataset":
        return "dataset"
    if stage.kind == "inference":
        return "inference"
    if stage.kind == "evaluator":
        return "evaluator"
    if stage.kind == "train":
        return "train"
    if stage.kind == "eval":
        return "evaluator"
    value = f"{stage.id} {stage.name}".lower()
    if "dataset" in value or value.startswith("data"):
        return "dataset"
    if "inference" in value:
        return "inference"
    if "evaluator" in value or "evaluation" in value or "eval" in value:
        return "evaluator"
    if "train" in value:
        return "train"
    return None


def _resources_payload(
    plan: WorkflowPlan | ExperimentContext,
) -> list[dict[str, str | None]]:
    return [
        {
            "id": resource.id,
            "name": resource.name,
            "uri": resource.uri,
            "description": resource.description,
        }
        for resource in plan.resources
    ]


def _tasks_by_stage(
    plan: WorkflowPlan | ExperimentContext,
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for task in plan.tasks:
        grouped.setdefault(task.stage_id, []).append(task.model_dump(mode="json"))
    return grouped


def _node(
    *,
    node_id: str,
    node_type: ProjectNodeType,
    title: str,
    launch_mode: Literal["agent", "executor"] = "executor",
    config: dict[str, Any],
    inputs: dict[str, dict[str, Any]],
    output_port: tuple[str, str, str],
) -> dict[str, Any]:
    port_name, artifact_type, schema_ref = output_port
    return {
        "id": node_id,
        "type": node_type,
        "title": title,
        "task_type": _TASK_TYPES[node_type],
        "status": "idle",
        "launch_mode": launch_mode,
        "execution_role": _EXECUTION_ROLES[node_type],
        "inputs": inputs,
        "outputs": {
            port_name: {
                "artifact_type": artifact_type,
                "schema_ref": schema_ref,
                "required": True,
            }
        },
        "executor": {"kind": "contract_probe"},
        "config": config,
        "policies": {},
    }
