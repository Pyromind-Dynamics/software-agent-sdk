from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Mapping
from hashlib import sha256
from typing import Literal

from pyromind_runtime.application.model_evaluation_plan import (
    model_evaluation_tasks,
    validate_evaluation_inputs,
)
from pyromind_runtime.application.plan_execution import recover_plan_delivery_rows
from pyromind_runtime.application.sdk_execution_metadata import (
    verified_sdk_inference_paths,
)
from pyromind_runtime.domain.base import ContractModel
from pyromind_runtime.domain.content import JsonObject
from pyromind_runtime.domain.experiment_execution import ExperimentExecutionView
from pyromind_runtime.domain.pipeline import PipelineRun
from pyromind_runtime.domain.preparation import ProjectView
from pyromind_runtime.domain.snapshot import ConversationSnapshot, ExternalTaskState
from pyromind_runtime.domain.storage_paths import StoragePathError, to_storage_api_path
from pyromind_runtime.domain.workflow_execution import (
    WorkflowExecutionRequest,
    workflow_contains_training_node,
)
from pyromind_runtime.infrastructure.file_product_store import FileProductStore
from pyromind_runtime.ports.product_store import ProductStore


type Stage = Literal["dataset", "train", "eval"]
type EvidenceSource = Literal["workflow_execution_request", "pipeline_run"]

_STAGES: tuple[Stage, ...] = ("dataset", "train", "eval")
_TASK_ID = re.compile(r"[0-9]{1,64}")


class ConsolidationEvidenceError(ValueError):
    pass


class ConsolidationStageEvidence(ContractModel):
    task_id: str
    source: EvidenceSource
    definition_sha256: str
    revision: int


class ConsolidationEvidence(ContractModel):
    conversation_id: str
    dataset: ConsolidationStageEvidence
    train: ConsolidationStageEvidence
    eval: ConsolidationStageEvidence


def select_consolidation_evidence(
    store: FileProductStore,
    *,
    approved_project: ProjectView | None = None,
    task_ids: Mapping[str, str] | None = None,
) -> ConsolidationEvidence:
    selected = dict(task_ids or {})
    if selected.keys() - set(_STAGES) or any(
        not _TASK_ID.fullmatch(task_id) for task_id in selected.values()
    ):
        raise ConsolidationEvidenceError("Invalid stage or platform task ID")
    if len(set(selected.values())) != len(selected):
        raise ConsolidationEvidenceError("Each stage needs a distinct platform task")

    snapshot = store.load_snapshot()
    tasks = _successful_development_tasks(
        snapshot.external_tasks, conversation_id=snapshot.conversation_id
    )
    candidates: dict[Stage, list[ConsolidationStageEvidence]] = {
        stage: [] for stage in _STAGES
    }
    scoped, eval_ids = _approved_sdk_tasks(store, snapshot, tasks, approved_project)
    pipelines = store.load_pipeline_runs().values()
    pipeline_ids: set[str] = set()
    for pipeline in pipelines:
        task = tasks.get(pipeline.task_id or "")
        if (
            task is not None
            and task.kind == "data_preparation"
            and (
                approved_project is None
                or not task.correlation
                or not task.correlation.execution
                or scoped.get(task.task_id) == "eval"
            )
            and pipeline.conversation_id == snapshot.conversation_id
            and pipeline.submission_status == "submitted"
            and _matches_evaluation_pipeline(pipeline, task)
        ):
            pipeline_ids.add(task.task_id)
            candidates["eval"].append(
                _evidence(
                    task=task,
                    source="pipeline_run",
                    definition=pipeline.definition,
                    revision=pipeline.execution_revision,
                )
            )
    for receipt in store.load_workflow_execution_requests().values():
        task_id = receipt.task_id or ""
        task = tasks.get(task_id)
        if (
            task is None
            or (
                receipt.dataset_recipe is not None
                and receipt.dataset_recipe.replay_of_task_id is not None
            )
            or task_id in pipeline_ids
            or receipt.conversation_id != snapshot.conversation_id
            or receipt.status != "submitted"
            or receipt.correlation != task.correlation
            or not receipt.workflow.get("nodes")
        ):
            continue
        stage: Stage
        if receipt.request_id.startswith("sdk:"):
            if receipt.request_id != "sdk:" + task_id or task_id not in scoped:
                continue
            if task_id in eval_ids:
                stage = "eval"
            elif (
                scoped[task_id] == "train"
                and task.kind == "workflow_run"
                and workflow_contains_training_node(receipt.workflow)
            ):
                stage = "train"
            elif scoped[task_id] == "dataset" and task.kind == "data_preparation":
                stage = "dataset"
            else:
                continue
        elif task.kind == "workflow_run" and (
            receipt.summary.get("has_training_node") is True
            or workflow_contains_training_node(receipt.workflow)
        ):
            stage = "train"
        elif task.kind == "data_preparation":
            stage = "dataset"
        else:
            continue
        if (
            task.correlation
            and task.correlation.execution
            and approved_project
            and scoped.get(task_id) != stage
        ):
            continue
        candidates[stage].append(
            _evidence(
                task=task,
                source="workflow_execution_request",
                definition=receipt.workflow,
                revision=receipt.revision,
            )
        )

    return ConsolidationEvidence(
        conversation_id=snapshot.conversation_id,
        dataset=_select_stage(
            "dataset", candidates["dataset"], selected.get("dataset")
        ),
        train=_select_stage("train", candidates["train"], selected.get("train")),
        eval=_select_stage("eval", candidates["eval"], selected.get("eval")),
    )


def _approved_sdk_tasks(
    store: FileProductStore,
    snapshot: ConversationSnapshot,
    tasks: dict[str, ExternalTaskState],
    approved_project: ProjectView | None,
) -> tuple[dict[str, str], set[str]]:
    scoped: dict[str, str] = {}
    eval_ids: set[str] = set()
    if approved_project is None:
        return scoped, eval_ids
    if approved_project.conversation_id != snapshot.conversation_id:
        raise ConsolidationEvidenceError(
            "Approved plan belongs to another conversation"
        )
    stages = {
        stage.id: stage.kind
        for stage in approved_project.plan.stages
        if stage.enabled and not stage.placeholder
    }
    planned = {task.id: task for task in approved_project.plan.tasks}
    latest: dict[str, ExperimentExecutionView] = {}
    for run in snapshot.experiment_executions or ():
        task = planned.get(run.task_id)
        if (
            run.role_id == "main"
            and run.project_id == approved_project.project_id
            and run.plan_sequence == approved_project.plan_sequence
            and task
            and run.plan_attempt is not None
            and run.stage_id == task.stage_id
            and run.stage_id in stages
        ):
            previous = latest.get(task.id)
            if previous is None or run.plan_attempt > (previous.plan_attempt or 0):
                latest[task.id] = run
    for run in latest.values():
        task = tasks.get(run.platform_task_id or "")
        correlation = task.correlation if task else None
        reference = correlation.execution if correlation else None
        if (
            task
            and correlation
            and correlation.type == "main"
            and (
                not approved_project.cluster
                or not correlation.cluster
                or correlation.cluster == approved_project.cluster
            )
            and reference
            and reference.project_id == run.project_id
            and reference.plan_sequence == run.plan_sequence
            and reference.task_id == run.task_id
            and reference.attempt == run.plan_attempt
            and store.verify_task_correlation(correlation)
        ):
            scoped[task.task_id] = stages[planned[run.task_id].stage_id] or ""
    required = model_evaluation_tasks(approved_project, store.preparation_policy())
    if required:
        rows = recover_plan_delivery_rows(snapshot, approved_project)
        for identifier, paths in verified_sdk_inference_paths(
            store, approved_project, required
        ).items():
            if scoped.get(identifier) != "eval":
                continue
            correlation = tasks[identifier].correlation
            assert correlation is not None and correlation.execution is not None
            task = planned[correlation.execution.task_id]
            if task.id not in required:
                continue
            try:
                validate_evaluation_inputs(
                    approved_project, task, rows, *paths, require_bindings=True
                )
            except ValueError:
                continue
            eval_ids.add(identifier)
    return scoped, eval_ids


def successful_dataset_recipe_runs(
    store: ProductStore, evidence: ConsolidationEvidence
) -> list[WorkflowExecutionRequest]:
    snapshot = store.load_snapshot()
    if snapshot.conversation_id != evidence.conversation_id:
        raise ConsolidationEvidenceError("Recipe belongs to another conversation")
    tasks = _successful_development_tasks(
        snapshot.external_tasks, conversation_id=evidence.conversation_id
    )
    recipes = []
    for receipt in store.load_workflow_execution_requests().values():
        recipe = receipt.dataset_recipe
        task = tasks.get(receipt.task_id or "")
        if (
            recipe is None
            or task is None
            or task.kind != "data_preparation"
            or receipt.status != "submitted"
            or receipt.conversation_id != evidence.conversation_id
            or task.correlation != receipt.correlation
            or not task.output_dir
            or to_storage_api_path(task.output_dir)
            != to_storage_api_path(recipe.output_directory)
        ):
            continue
        if recipe.replay_of_task_id == evidence.dataset.task_id or (
            receipt.task_id == evidence.dataset.task_id
            and receipt.revision == evidence.dataset.revision
            and consolidation_definition_sha256(receipt.workflow)
            == evidence.dataset.definition_sha256
        ):
            recipes.append(receipt)
    counts = Counter(receipt.task_id for receipt in recipes)
    return sorted(
        (receipt for receipt in recipes if counts[receipt.task_id] == 1),
        key=lambda receipt: int(receipt.task_id or "0"),
    )


def _successful_development_tasks(
    tasks: tuple[ExternalTaskState, ...], *, conversation_id: str
) -> dict[str, ExternalTaskState]:
    counts = Counter(task.task_id for task in tasks)
    return {
        task.task_id: task
        for task in tasks
        if counts[task.task_id] == 1
        and _TASK_ID.fullmatch(task.task_id)
        and task.status == "succeeded"
        and task.graph_run_id is None
        and task.correlation is not None
        and task.correlation.conversation_id == conversation_id
        and task.correlation.task_kind == task.kind
    }


def _matches_evaluation_pipeline(
    pipeline: PipelineRun, task: ExternalTaskState
) -> bool:
    definition = pipeline.definition
    if (
        not isinstance(definition.get("input_path"), str)
        or not isinstance(definition.get("dataset_config"), dict)
        or not isinstance(definition.get("evaluation_config"), dict)
        or not isinstance(definition.get("inference"), dict)
        or task.output_dir is None
    ):
        return False
    try:
        return to_storage_api_path(task.output_dir) == to_storage_api_path(
            pipeline.output_dir
        )
    except StoragePathError:
        return False


def _evidence(
    *,
    task: ExternalTaskState,
    source: EvidenceSource,
    definition: JsonObject,
    revision: int,
) -> ConsolidationStageEvidence:
    return ConsolidationStageEvidence(
        task_id=task.task_id,
        source=source,
        definition_sha256=consolidation_definition_sha256(definition),
        revision=revision,
    )


def consolidation_definition_sha256(definition: JsonObject) -> str:
    content = json.dumps(
        definition, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return f"sha256:{sha256(content.encode()).hexdigest()}"


def _select_stage(
    stage: Stage, candidates: list[ConsolidationStageEvidence], task_id: str | None
) -> ConsolidationStageEvidence:
    if task_id is not None:
        candidates = [
            candidate for candidate in candidates if candidate.task_id == task_id
        ]
    if not candidates:
        raise ConsolidationEvidenceError(
            f"No verified successful {stage} task was found"
        )
    if len(candidates) == 1:
        return candidates[0]
    ids = sorted({candidate.task_id for candidate in candidates})
    if task_id is None and len(ids) == len(candidates):
        raise ConsolidationEvidenceError(
            f"Multiple successful {stage} tasks were found. "
            f"Select one of {', '.join(ids)}"
        )
    raise ConsolidationEvidenceError(f"Conflicting receipt evidence for {stage}")
