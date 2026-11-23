"""Match planned model evaluations to their frozen inputs and Studio receipts."""

from __future__ import annotations

from pyromind_runtime.domain.content import JsonObject
from pyromind_runtime.domain.experiment_execution import ExperimentExecutionView
from pyromind_runtime.domain.experiment_plans import ExperimentTask
from pyromind_runtime.domain.pipeline import PipelineRun
from pyromind_runtime.domain.preparation import PreparationPolicy, ProjectView
from pyromind_runtime.domain.storage_paths import to_storage_api_path


_MODEL_INPUTS = {"model", "merged_model", "checkpoint"}
_DATA_INPUTS = {"dataset", "val_dataset", "test_dataset", "validation_dataset"}


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


def validate_evaluation_inputs(
    project: ProjectView, task: ExperimentTask, rows: list, model: str, dataset: str
) -> None:
    resources = {item.id: item.storage_path for item in project.plan.resources}
    outputs = {
        row.get("id"): row.get("verified_outputs")
        for row in rows
        if isinstance(row, dict)
    }
    for item in task.inputs:
        if item.name in _MODEL_INPUTS:
            actual = model
        elif item.name in _DATA_INPUTS:
            actual = dataset
        else:
            continue
        expected = resources.get(item.resource_id) if item.resource_id else None
        if item.task_id:
            verified = outputs.get(item.task_id)
            expected = (
                verified.get(item.output_name) if isinstance(verified, dict) else None
            )
            if not isinstance(expected, str) or not expected:
                raise ValueError(
                    f"Verify {item.task_id}.{item.output_name} in workflowPLAN before "
                    f"submitting {task.id}"
                )
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
    run: ExperimentExecutionView, receipts: dict[str, PipelineRun]
) -> ExperimentExecutionView:
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
