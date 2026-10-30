from __future__ import annotations

from typing import Any, Literal

from pyromind_runtime.domain.experiment_plans import ExperimentStage, WorkflowPlan


AVI_SINGLE_TURN_SKILL = "avi-single-turn"
AVI_STAGE_IDS = ("dataset", "inference", "evaluator", "train")

ProjectNodeType = Literal["dataset", "inference", "evaluator", "train"]


def build_graph_payload(
    *,
    skill: str | None,
    plan: WorkflowPlan,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    if skill == AVI_SINGLE_TURN_SKILL:
        return _build_avi_graph(plan)
    return _build_generic_graph(plan)


def _build_avi_graph(
    plan: WorkflowPlan,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    stages = {stage.id: stage for stage in plan.stages if stage.enabled}
    missing = [stage_id for stage_id in AVI_STAGE_IDS if stage_id not in stages]
    if missing:
        raise ValueError(
            "AVI plan must enable Dataset, Inference, Evaluator, and Train stages: "
            + ", ".join(missing)
        )

    resources = _resources_payload(plan)
    tasks_by_stage = _tasks_by_stage(plan)
    nodes = []
    for stage_id in AVI_STAGE_IDS:
        stage = stages[stage_id]
        config: dict[str, Any] = {
            "resources": resources,
            "tasks": tasks_by_stage.get(stage_id, []),
        }
        if stage_id == "dataset":
            config.update(
                {
                    "capability": "data-processing",
                    "platform_task_kind": "data_preparation",
                }
            )
        elif stage_id == "evaluator":
            config["artifact_format"] = "atif"
        nodes.append(
            _node(
                node_id=stage.id,
                node_type=stage_id,
                title=stage.name,
                launch_mode="agent" if stage_id == "dataset" else "executor",
                config=config,
            )
        )

    valid_ids = set(AVI_STAGE_IDS)
    edges = [
        {"source": edge.source, "target": edge.target}
        for edge in plan.edges
        if edge.source in valid_ids and edge.target in valid_ids
    ]
    return nodes, edges


def _build_generic_graph(
    plan: WorkflowPlan,
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
        nodes.append(
            _node(
                node_id=stage.id,
                node_type=node_type,
                title=stage.name,
                launch_mode="agent" if node_type == "dataset" else "executor",
                config={
                    "resources": resources,
                    "tasks": tasks_by_stage.get(stage.id, []),
                    **(
                        {
                            "capability": "data-processing",
                            "platform_task_kind": "data_preparation",
                        }
                        if node_type == "dataset"
                        else {}
                    ),
                },
            )
        )
    edges = [
        {"source": edge.source, "target": edge.target}
        for edge in plan.edges
        if edge.source in node_types and edge.target in node_types
    ]
    return nodes, edges


def _stage_node_type(stage: ExperimentStage) -> ProjectNodeType | None:
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


def _resources_payload(plan: WorkflowPlan) -> list[dict[str, str | None]]:
    return [
        {
            "id": resource.id,
            "name": resource.name,
            "uri": resource.uri,
            "description": resource.description,
        }
        for resource in plan.resources
    ]


def _tasks_by_stage(plan: WorkflowPlan) -> dict[str, list[dict[str, Any]]]:
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
