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
_CATALOG_NODE_TYPES: dict[str, ProjectNodeType] = {
    "dataset": "dataset",
    "train": "train",
    "eval": "evaluator",
    "inference": "inference",
}


def build_graph_payload(
    *,
    plan: ExperimentContext | WorkflowPlan,
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
                    "platform_task_kind": "data_preparation",
                }
            )
        elif node_type == "evaluator":
            config["artifact_format"] = "atif"
        nodes.append(
            _node(
                node_id=stage.id,
                node_type=node_type,
                title=stage.name,
                launch_mode="agent" if node_type == "dataset" else "executor",
                config=config,
            )
        )
    stage_links = [(edge.source, edge.target) for edge in plan.edges]
    if not stage_links:
        task_stages = {task.id: task.stage_id for task in plan.tasks}
        stage_links = [
            (task_stages[item.task_id], task.stage_id)
            for task in plan.tasks
            for item in task.inputs
            if item.task_id is not None and task_stages[item.task_id] != task.stage_id
        ]
    edges = [
        {"source": source, "target": target}
        for source, target in dict.fromkeys(stage_links)
        if source in node_types and target in node_types
    ]
    return nodes, edges


def _stage_node_type(stage: ExperimentStage) -> ProjectNodeType | None:
    catalog_type = _CATALOG_NODE_TYPES.get(stage.kind or "")
    if catalog_type is not None:
        return catalog_type
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
    plan: ExperimentContext | WorkflowPlan,
) -> list[dict[str, str | None]]:
    return [
        {
            "id": resource.id,
            "name": resource.name,
            "uri": resource.storage_path,
            "description": resource.description,
        }
        for resource in plan.resources
    ]


def _tasks_by_stage(
    plan: ExperimentContext | WorkflowPlan,
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
) -> dict[str, Any]:
    return {
        "id": node_id,
        "type": node_type,
        "title": title,
        "status": "idle",
        "launch_mode": launch_mode,
        "executor": "contract_probe",
        "config": config,
    }
