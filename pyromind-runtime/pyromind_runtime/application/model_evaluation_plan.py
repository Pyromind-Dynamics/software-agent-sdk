"""Match planned model evaluations to their frozen inputs and Studio receipts."""

from __future__ import annotations

import shlex
from pathlib import PurePosixPath

from pyromind_runtime.domain.content import JsonObject
from pyromind_runtime.domain.experiment_execution import ExperimentExecutionView
from pyromind_runtime.domain.experiment_plans import ExperimentTask, PlanInput
from pyromind_runtime.domain.pipeline import PipelineRun
from pyromind_runtime.domain.preparation import PreparationPolicy, ProjectView
from pyromind_runtime.domain.storage_paths import StoragePathError, to_storage_api_path


_MODEL_INPUTS = {"model", "merged_model", "checkpoint"}
MODEL_EVALUATION_DATA_INPUTS = {
    "dataset",
    "val_dataset",
    "test_dataset",
    "validation_dataset",
    "eval_data",
    "evaluation_dataset",
}


def has_model_evaluation_inputs(task: ExperimentTask) -> bool:
    names = {item.name for item in task.inputs}
    return "model" in names and bool(names & MODEL_EVALUATION_DATA_INPUTS)


def model_evaluation_tasks(
    project: ProjectView, policy: PreparationPolicy | None
) -> set[str]:
    if policy is None:
        return set()
    targets = {
        (node.id, task.id): task.evaluation_target
        for node in policy.workflow_nodes
        for task in node.tasks
        if task.evaluation_target
    }
    stages = {stage.id: stage.kind for stage in project.plan.stages}
    result = set()
    for task in project.plan.tasks:
        target = targets.get((stages[task.stage_id] or "", task.kind or ""))
        if target == "model" or (
            target == "model_or_predictions"
            and any(item.name in _MODEL_INPUTS for item in task.inputs)
        ):
            result.add(task.id)
    return result


def inference_paths(pipeline: PipelineRun | None) -> tuple[str, str] | None:
    if pipeline is None:
        return None
    inference = pipeline.definition.get("inference")
    model = inference.get("model_path") if isinstance(inference, dict) else None
    dataset = pipeline.definition.get("input_path")
    if isinstance(model, str) and isinstance(dataset, str):
        return model, dataset
    return None


def verified_input_path(item: PlanInput, outputs: dict) -> str | None:
    value = outputs.get(item.output_name)
    if not isinstance(value, str) or not value:
        return None
    if item.name in _MODEL_INPUTS and item.output_name == "model_output":
        selected = outputs.get("selected_checkpoint")
        if isinstance(selected, str) and selected:
            try:
                root = PurePosixPath(to_storage_api_path(value))
                checkpoint = PurePosixPath(to_storage_api_path(selected))
            except StoragePathError:
                return value
            if root in checkpoint.parents:
                return selected
    return value


def validate_evaluation_inputs(
    project: ProjectView,
    task: ExperimentTask,
    rows: list,
    model: str,
    dataset: str,
    *,
    require_bindings: bool = False,
) -> None:
    if require_bindings:
        names = {item.name for item in task.inputs}
        if not names & _MODEL_INPUTS or not names & MODEL_EVALUATION_DATA_INPUTS:
            raise ValueError("Model evaluation requires both model and dataset inputs")
    resources = {item.id: item.storage_path for item in project.plan.resources}
    outputs = {
        row.get("id"): row.get("verified_outputs")
        for row in rows
        if isinstance(row, dict)
    }
    for item in task.inputs:
        if item.name in _MODEL_INPUTS:
            actual = model
        elif item.name in MODEL_EVALUATION_DATA_INPUTS:
            actual = dataset
        else:
            continue
        expected = resources.get(item.resource_id) if item.resource_id else None
        if item.task_id:
            verified = outputs.get(item.task_id)
            expected = (
                verified_input_path(item, verified)
                if isinstance(verified, dict)
                else None
            )
            declared = (
                verified.get(item.output_name) if isinstance(verified, dict) else None
            )
            if (
                isinstance(declared, str)
                and declared
                and actual.startswith("/")
                and (to_storage_api_path(actual) == to_storage_api_path(declared))
            ):
                continue
            if not isinstance(expected, str) or not expected:
                raise ValueError(
                    f"Verify {item.task_id}.{item.output_name} in workflowPLAN before "
                    f"submitting {task.id}"
                )
        if require_bindings and not expected:
            raise ValueError(f"{task.id} has no verified binding for {item.name}")
        if expected and (
            not actual.startswith("/")
            or to_storage_api_path(actual) != to_storage_api_path(expected)
        ):
            raise ValueError(f"{task.id} requires {item.name} at {expected}")


def submission_paths(
    tool_name: str | None, args: JsonObject, pipelines: dict[str, PipelineRun]
) -> tuple[str, str]:
    if tool_name == "df_submit_evaluation":
        if args.get("operation") == "inference":
            model, data = args.get("model_path"), args.get("input_path")
            if isinstance(model, str) and isinstance(data, str):
                return model, data
        if args.get("operation") == "resume":
            paths = inference_paths(pipelines.get(str(args.get("resume_run_id"))))
            data = args.get("input_path")
            if (
                paths
                and isinstance(data, str)
                and to_storage_api_path(data) == to_storage_api_path(paths[1])
            ):
                return paths
    raise ValueError(
        "This plan task requires df_submit_evaluation(operation='inference') "
        "with the planned model and dataset, or resume of that inference run"
    )


def check_inference_receipt(
    run: ExperimentExecutionView,
    receipts: dict[str, PipelineRun],
    *,
    sdk_receipts: dict[str, tuple[str, str]] | None = None,
) -> ExperimentExecutionView:
    if run.platform_task_id and sdk_receipts and run.platform_task_id in sdk_receipts:
        return run
    receipt = receipts.get(run.platform_task_id) if run.platform_task_id else None
    if not inference_paths(receipt) and (
        receipt is not None or run.state == "succeeded"
    ):
        return run.model_copy(
            update={
                "state": "needs_input",
                "report_state": "not_started",
                "report_uri": None,
                "artifacts": (),
                "waiting_reason": "缺少目标模型在指定数据集上的推理回执。",
            }
        )
    return run


def sdk_inference_paths(workflow: JsonObject) -> tuple[str, str] | None:
    """Recognize the frozen inference topology emitted by the SDK evaluation helper."""
    nodes = workflow.get("nodes")
    edges = workflow.get("edges")
    if not isinstance(nodes, list) or not isinstance(edges, list):
        return None
    data = {
        str(node["id"]): node["data"]
        for node in nodes
        if isinstance(node, dict)
        and isinstance(node.get("id"), str)
        and isinstance(node.get("data"), dict)
    }
    matches = set()
    for edge in edges:
        if not isinstance(edge, dict) or (
            edge.get("sourceHandle"),
            edge.get("targetHandle"),
        ) != ("endpoint", "param"):
            continue
        serving, scoring = (
            data.get(str(edge.get("source"))),
            data.get(str(edge.get("target"))),
        )
        if not isinstance(serving, dict) or not isinstance(scoring, dict):
            continue
        serving_config, scoring_config = serving.get("config"), scoring.get("config")
        if (
            serving.get("nodeType") != "VLLMInference"
            or scoring.get("nodeType") != "CustomCommandCPUNode"
            or not isinstance(serving_config, dict)
            or not isinstance(scoring_config, dict)
        ):
            continue
        command = scoring_config.get("command")
        if not isinstance(command, str):
            continue
        try:
            arguments = shlex.split(command)
        except ValueError:
            continue
        if (
            len(arguments) < 4
            or len(arguments) % 2
            or not PurePosixPath(arguments[0]).name.startswith("python")
            or PurePosixPath(arguments[1]).name != "evaluate_inference.py"
        ):
            continue
        flags = dict(zip(arguments[2::2], arguments[3::2], strict=True))
        model, dataset = flags.get("--model-reference"), flags.get("--dataset-path")
        if not model or not dataset or flags.get("--endpoint") != "$param":
            continue
        actual_model = serving_config.get("model_path")
        if not actual_model:
            sources = [
                data.get(str(source.get("source")))
                for source in edges
                if isinstance(source, dict)
                and source.get("target") == edge.get("source")
                and source.get("targetHandle") == "model_path"
                and source.get("sourceHandle") in {"model_path", "merged_model_path"}
            ]
            if len(sources) == 1 and isinstance(source := sources[0], dict):
                config = source.get("config")
                field = {
                    "ModelMergeLoraNode": "lora_path",
                    "CloneAndCacheModel": "model",
                    "DownloadAndCacheModel": "modelname",
                }.get(str(source.get("nodeType")))
                if isinstance(config, dict) and field:
                    actual_model = config.get(field)
        if not isinstance(actual_model, str):
            continue
        try:
            if model.startswith("/") and actual_model.startswith("/"):
                same = to_storage_api_path(model) == to_storage_api_path(actual_model)
            else:
                same = model == actual_model
            to_storage_api_path(dataset)
        except StoragePathError:
            continue
        if same:
            matches.add((model, dataset))
    return next(iter(matches)) if len(matches) == 1 else None
