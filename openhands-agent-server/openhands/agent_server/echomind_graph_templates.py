from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal


AVI_SINGLE_TURN_SKILL = "avi-single-turn"

AVI_REQUIRED_REQUIREMENT_KEYS = (
    "dataset_uri",
    "dataset_format",
    "image_roles",
    "label_schema",
    "split_policy",
    "inference_model",
    "train_mode",
    "evaluation_rule",
    "artifact_goal",
)


def build_graph_payload(
    *,
    skill: str,
    requirements: Mapping[str, str],
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    if skill != AVI_SINGLE_TURN_SKILL:
        return [], []

    train_mode = (
        "dpo" if requirements.get("train_mode", "").strip().lower() == "dpo" else "sft"
    )
    nodes = [
        _node(
            node_id="dataset",
            node_type="dataset",
            title="Dataset",
            launch_mode="agent",
            config={
                "capability": "data-processing",
                "platform_task_kind": "data_preparation",
                "input_hint": requirements["dataset_uri"],
                "validation": {
                    "status": "draft",
                    "validation_run_id": None,
                    "candidate_execution_spec": None,
                },
            },
        ),
        _node(
            node_id="inference",
            node_type="inference",
            title="Inference",
            config={
                "model": requirements["inference_model"],
                "dataset_ref": "dataset.dataset",
            },
        ),
        _node(
            node_id="evaluator",
            node_type="evaluator",
            title="Evaluator",
            config={
                "predictions_ref": "inference.predictions",
                "rule": requirements["evaluation_rule"],
                "artifact_format": "atif",
            },
        ),
        _node(
            node_id="train",
            node_type="train",
            title=f"Train ({train_mode.upper()})",
            config={
                "method": train_mode,
                "dataset_ref": "dataset.dataset",
                "evaluation_ref": "evaluator.report",
                "artifact_goal": requirements["artifact_goal"],
            },
        ),
    ]
    edges = [
        {"source": "dataset", "target": "inference"},
        {"source": "inference", "target": "evaluator"},
        {"source": "evaluator", "target": "train"},
    ]
    return nodes, edges


def _node(
    *,
    node_id: str,
    node_type: Literal["dataset", "inference", "evaluator", "train"],
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
