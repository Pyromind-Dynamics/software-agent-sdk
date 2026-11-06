from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, Literal

from harness_adapter.pi_adapter.adapter import ToolExtensionSpec
from pydantic import BaseModel, ConfigDict, Field
from pydantic_core import to_jsonable_python
from pyromind_runtime.domain.content import JsonObject
from pyromind_runtime.domain.context import RequestContext
from pyromind_runtime.domain.experiment_execution import (
    DelegatedTask,
    RequestExperimentTask,
)
from pyromind_runtime.domain.project_graphs import canonical_execution_role

from openhands.agent_server.echomind_dataset_contract import (
    DatasetSubmissionSpec,
    require_dataset_submission_spec,
)
from openhands.agent_server.echomind_projects import ProjectStore
from openhands.agent_server.echomind_stage_runs import (
    ProjectArtifact,
    ProjectRunStore,
    StageRun,
    require_sample_validation,
)
from pyromind_agent_server.echomind_paths import storage_path_contract


class _StageToolModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class GetProjectStageRun(_StageToolModel):
    project_id: str = Field(min_length=1, max_length=128)
    run_id: str = Field(min_length=1, max_length=128)


class OpenProjectStage(_StageToolModel):
    project_id: str = Field(min_length=1, max_length=128)
    node_id: str = Field(min_length=1, max_length=128)
    request_id: str = Field(min_length=1, max_length=128)


class ApproveProjectStagePlan(_StageToolModel):
    project_id: str = Field(min_length=1, max_length=128)
    run_id: str = Field(min_length=1, max_length=128)
    request_id: str = Field(min_length=1, max_length=128)
    summary: str = Field(min_length=1, max_length=4000)
    decisions: dict[str, str] = Field(default_factory=dict)


class ConfirmProjectStageRun(_StageToolModel):
    project_id: str = Field(min_length=1, max_length=128)
    run_id: str = Field(min_length=1, max_length=128)
    request_id: str = Field(min_length=1, max_length=128)


class RequestProjectStageTask(_StageToolModel):
    project_id: str = Field(min_length=1, max_length=128)
    run_id: str = Field(min_length=1, max_length=128)
    role_id: str = Field(min_length=1)
    request_id: str = Field(min_length=1, max_length=128)
    task: DelegatedTask
    intent: Literal["execute", "prepare"] = "execute"


class RecordProjectStageResult(_StageToolModel):
    project_id: str = Field(min_length=1, max_length=128)
    run_id: str = Field(min_length=1, max_length=128)
    request_id: str = Field(min_length=1, max_length=128)
    outcome: Literal["waiting_user", "waiting_external", "succeeded", "failed"]
    message: str = Field(min_length=1, max_length=4000)
    input_refs: list[str] = Field(default_factory=list)
    details: dict[str, Any] = Field(default_factory=dict)
    external_task_id: str | None = Field(default=None, max_length=128)
    task_run_id: str | None = Field(default=None, max_length=64)
    output_uri: str | None = Field(default=None, max_length=4096)


ECHOMIND_STAGE_TOOL_SPECS = (
    ToolExtensionSpec(
        name="open_project_stage",
        description=(
            "Create or reuse the persistent Agent development StageRun for one "
            "Project node. Pass a stable request_id so retries return the same run."
        ),
        input_schema=OpenProjectStage.model_json_schema(),
    ),
    ToolExtensionSpec(
        name="get_project_stage_run",
        description=(
            "Read one persistent EchoMind Project stage run by project_id and run_id."
        ),
        input_schema=GetProjectStageRun.model_json_schema(),
    ),
    ToolExtensionSpec(
        name="request_project_stage_task",
        description=(
            "Create one deterministic TaskRun for an EchoMind StageRun and bind "
            "both records with the same project_id. Reuse request_id for retries; "
            "use a new request_id when the user asks to redo the Node. Redoing "
            "cancels unfinished TaskRuns in the same StageRun before creating the "
            "replacement. Dataset uses phase='sample' first and phase='submit' "
            "after confirmation; phase='sample' may repair a rejected sample in "
            "the same StageRun. The submit contract is injected from the accepted "
            "sample."
        ),
        input_schema=RequestProjectStageTask.model_json_schema(),
    ),
    ToolExtensionSpec(
        name="approve_project_stage_plan",
        description=(
            "Move one EchoMind Agent development StageRun from clarifying to "
            "running after the user explicitly confirms the Node plan. Pass a "
            "concise summary and the agreed decisions. Do not call before "
            "confirmation."
        ),
        input_schema=ApproveProjectStagePlan.model_json_schema(),
    ),
    ToolExtensionSpec(
        name="confirm_project_stage_run",
        description=(
            "Confirm the completed sample for a waiting_user EchoMind Agent "
            "development StageRun and move it back to running. Call only after "
            "the user explicitly confirms the sample. Do not use "
            "control_experiment_task to resume a StageRun."
        ),
        input_schema=ConfirmProjectStageRun.model_json_schema(),
    ),
    ToolExtensionSpec(
        name="record_project_stage_result",
        description=(
            "Record verified EchoMind stage evidence for one StageRun. Use a new "
            "request_id for each state transition; reuse one only to replay the "
            "same outcome. Record waiting_user after a real sample confirmation, "
            "waiting_external after a platform task is submitted, and succeeded "
            "only after the child verifies its result. Dataset stages require a "
            "submitted TaskRun in waiting_external before succeeding. A child-only "
            "non-Dataset task may succeed without a platform task. Pass task_run_id "
            "when more than one TaskRun exists."
        ),
        input_schema=RecordProjectStageResult.model_json_schema(),
    ),
)


def echomind_stage_tool_handler(
    conversations_dir: Path | str,
    request_stage_task: (
        Callable[[str, RequestExperimentTask, RequestContext], Awaitable[dict]] | None
    ) = None,
    cancel_stage_task: (
        Callable[[str, str, str, RequestContext], Awaitable[dict]] | None
    ) = None,
):
    project_store = ProjectStore.for_conversations_dir(conversations_dir)
    run_store = ProjectRunStore(project_store.root)

    async def handle(
        conversation_id: str,
        tool_name: str,
        arguments: dict[str, Any],
        context: RequestContext,
    ) -> dict[str, Any]:
        if tool_name == "open_project_stage":
            request = OpenProjectStage.model_validate(arguments)
            project = await asyncio.to_thread(
                project_store.load,
                request.project_id,
                owner_user_id=context.user_id,
            )
            _require_project_conversation(
                project.project.echomind_conversation_id, conversation_id
            )
            if project.graph.lifecycle != "development":
                raise ValueError("frozen Projects cannot start Agent development")
            node = next(
                (item for item in project.graph.nodes if item.id == request.node_id),
                None,
            )
            if node is None:
                raise ValueError(f"Project node not found: {request.node_id}")
            if not node.execution_role:
                raise ValueError(
                    f"Project node has no development execution role: {node.id}"
                )
            run, _ = await asyncio.to_thread(
                run_store.start_agent_development,
                project=project,
                node=node,
                request_id=request.request_id,
            )
            return run.model_dump(mode="json")

        if tool_name == "get_project_stage_run":
            request = GetProjectStageRun.model_validate(arguments)
            project = await asyncio.to_thread(
                project_store.load,
                request.project_id,
                owner_user_id=context.user_id,
            )
            _require_project_conversation(
                project.project.echomind_conversation_id, conversation_id
            )
            run = await asyncio.to_thread(
                run_store.get,
                project_id=request.project_id,
                run_id=request.run_id,
            )
            return run.model_dump(mode="json")

        if tool_name == "approve_project_stage_plan":
            request = ApproveProjectStagePlan.model_validate(arguments)
            project = await asyncio.to_thread(
                project_store.load,
                request.project_id,
                owner_user_id=context.user_id,
            )
            _require_project_conversation(
                project.project.echomind_conversation_id, conversation_id
            )
            run = await asyncio.to_thread(
                run_store.approve_plan,
                project=project,
                run_id=request.run_id,
                request_id=request.request_id,
                summary=request.summary,
                decisions=request.decisions,
            )
            return run.model_dump(mode="json")

        if tool_name == "confirm_project_stage_run":
            request = ConfirmProjectStageRun.model_validate(arguments)
            project = await asyncio.to_thread(
                project_store.load,
                request.project_id,
                owner_user_id=context.user_id,
            )
            _require_project_conversation(
                project.project.echomind_conversation_id, conversation_id
            )
            if project.graph.lifecycle != "development":
                raise ValueError("frozen Projects cannot continue stage development")
            run = await asyncio.to_thread(
                run_store.get,
                project_id=request.project_id,
                run_id=request.run_id,
            )
            node = next(
                (item for item in project.graph.nodes if item.id == run.node_id),
                None,
            )
            if node is None:
                raise ValueError(f"Project node not found: {run.node_id}")
            if run.mode != "agent_development" or not node.execution_role:
                raise ValueError(
                    "Only Agent development runs with an execution role can be "
                    "confirmed"
                )
            sample_kind = node.config.get("sample_report_kind", "stage_sample_report")
            sample_artifact = next(
                (
                    artifact
                    for artifact in reversed(run.artifacts)
                    if artifact.kind == sample_kind
                ),
                None,
            )
            if sample_artifact is None:
                raise ValueError(
                    "The Dataset run has no completed sample report to confirm"
                )
            sample_document = await asyncio.to_thread(
                run_store.load_artifact,
                project_id=request.project_id,
                artifact_id=sample_artifact.artifact_id,
            )
            artifact = sample_document.artifact
            if artifact.run_id != run.run_id or artifact.node_id != run.node_id:
                raise ValueError("The sample report does not belong to this StageRun")
            require_sample_validation(
                node,
                sample_document.content.get("details", {}),
            )
            if node.type == "dataset":
                sample_details = sample_document.content.get("details", {})
                if not isinstance(sample_details, dict):
                    raise ValueError("The sample report details are invalid")
                sample_task_run_id = sample_document.content.get("task_run_id")
                if not isinstance(sample_task_run_id, str) or not sample_task_run_id:
                    raise ValueError("The sample report has no TaskRun identity")
                require_dataset_submission_spec(
                    sample_details,
                    task_run_id=sample_task_run_id,
                )
            run, _ = await asyncio.to_thread(
                run_store.begin_continuation,
                project=project,
                run_id=request.run_id,
                request_id=request.request_id,
                command_id=f"agent-confirm:{request.request_id}",
            )
            return run.model_dump(mode="json")

        if tool_name == "request_project_stage_task":
            if request_stage_task is None:
                raise ValueError("Project stage task execution is unavailable")
            request = RequestProjectStageTask.model_validate(arguments)
            project = await asyncio.to_thread(
                project_store.load,
                request.project_id,
                owner_user_id=context.user_id,
            )
            _require_project_conversation(
                project.project.echomind_conversation_id, conversation_id
            )
            run = await asyncio.to_thread(
                run_store.get,
                project_id=request.project_id,
                run_id=request.run_id,
            )
            project_node = next(
                (node for node in project.graph.nodes if node.id == run.node_id),
                None,
            )
            if project_node is None:
                raise ValueError(f"Project node not found: {run.node_id}")
            expected_role = canonical_execution_role(project_node.execution_role)
            requested_role = canonical_execution_role(request.role_id)
            if requested_role != expected_role:
                raise ValueError(
                    f"TaskRun role {request.role_id!r} does not match Project node "
                    f"execution_role {project_node.execution_role!r}"
                )
            existing = next(
                (
                    task_run
                    for task_run in run.task_runs
                    if task_run.request_id == request.request_id
                ),
                None,
            )
            if existing is not None:
                return {
                    "id": existing.task_run_id,
                    "project_id": request.project_id,
                    "stage_run_id": run.run_id,
                    "task_run_id": existing.task_run_id,
                    "idempotent": True,
                }
            unfinished = [
                task_run
                for task_run in run.task_runs
                if task_run.status in {"running", "waiting_external"}
            ]
            cancel = cancel_stage_task
            if unfinished and cancel is None:
                raise ValueError("redoing a TaskRun requires cancellation support")
            for task_run in unfinished:
                assert cancel is not None
                await cancel(
                    conversation_id,
                    task_run.task_run_id,
                    f"supersede:{request.request_id}:{task_run.task_run_id}",
                    context,
                )
            if unfinished:
                run = await asyncio.to_thread(
                    run_store.cancel_task_runs,
                    run,
                    task_run_ids={task_run.task_run_id for task_run in unfinished},
                    message=(
                        f"Cancelled before replacement TaskRun {request.request_id}."
                    ),
                )
            task = request.task
            task_inputs = {
                **task.inputs,
                "path_contract": storage_path_contract(dict(project_node.config)),
            }
            phase = task_inputs.get("phase")
            repair_dataset_sample = (
                project_node.type == "dataset"
                and phase == "sample"
                and run.status == "waiting_user"
            )
            if run.status not in {"running", "failed"} and not repair_dataset_sample:
                raise ValueError(
                    "the StageRun must be running or failed before requesting a TaskRun"
                )
            if project_node.type == "dataset":
                if phase == "sample":
                    task_inputs.pop("submission_spec", None)
                    task = task.model_copy(update={"inputs": task_inputs})
                elif phase == "submit":
                    sample_artifact = _latest_accepted_sample_artifact(run)
                    if sample_artifact is None:
                        raise ValueError(
                            "Dataset submit requires an accepted sample artifact"
                        )
                    sample_document = await asyncio.to_thread(
                        run_store.load_artifact,
                        project_id=request.project_id,
                        artifact_id=sample_artifact.artifact_id,
                    )
                    sample_task_run_id = sample_document.content.get("task_run_id")
                    if (
                        not isinstance(sample_task_run_id, str)
                        or not sample_task_run_id
                    ):
                        raise ValueError(
                            "Dataset sample artifact has no TaskRun identity"
                        )
                    sample_details = sample_document.content.get("details", {})
                    if not isinstance(sample_details, dict):
                        raise ValueError("Dataset sample artifact has invalid details")
                    spec = require_dataset_submission_spec(
                        sample_details,
                        task_run_id=sample_task_run_id,
                    )
                    task = _dataset_submit_task(
                        spec=spec,
                        path_contract=task_inputs["path_contract"],
                        sample_artifact_id=sample_artifact.artifact_id,
                        sample_task_run_id=sample_task_run_id,
                    )
                else:
                    raise ValueError(
                        "Dataset StageRun tasks must use phase='sample' or "
                        "phase='submit'"
                    )
            execution = await request_stage_task(
                conversation_id,
                RequestExperimentTask(
                    role_id=requested_role,
                    request_id=request.request_id,
                    task=task,
                    intent=request.intent,
                    stage_id=run.node_id,
                    stage_run_id=run.run_id,
                ),
                context,
            )
            task_run_id = str(execution["id"])
            bound_run, bound_task, _ = await asyncio.to_thread(
                run_store.bind_task_run,
                run,
                request_id=request.request_id,
                task_run_id=task_run_id,
                role_id=requested_role,
                intent=request.intent,
                allow_waiting_user=repair_dataset_sample,
            )
            return {
                **to_jsonable_python(execution),
                "project_id": request.project_id,
                "stage_run_id": bound_run.run_id,
                "task_run_id": bound_task.task_run_id,
            }

        if tool_name == "record_project_stage_result":
            request = RecordProjectStageResult.model_validate(arguments)
            project = await asyncio.to_thread(
                project_store.load,
                request.project_id,
                owner_user_id=context.user_id,
            )
            _require_project_conversation(
                project.project.echomind_conversation_id, conversation_id
            )
            run = await asyncio.to_thread(
                run_store.record_result,
                project=project,
                run_id=request.run_id,
                request_id=request.request_id,
                outcome=request.outcome,
                message=request.message,
                input_refs=request.input_refs,
                details=request.details,
                external_task_id=request.external_task_id,
                task_run_id=request.task_run_id,
                output_uri=request.output_uri,
            )
            return run.model_dump(mode="json")

        raise ValueError(f"Unsupported EchoMind stage tool: {tool_name}")

    return handle


def _require_project_conversation(
    project_conversation_id: str, conversation_id: str
) -> None:
    if project_conversation_id != conversation_id:
        raise PermissionError("Project belongs to another conversation")


def _latest_accepted_sample_artifact(run: StageRun) -> ProjectArtifact | None:
    accepted_task = next(
        (
            task_run
            for task_run in reversed(run.task_runs)
            if task_run.status == "waiting_user" and task_run.artifact_ids
        ),
        None,
    )
    if accepted_task is None:
        return None
    artifact_id = accepted_task.artifact_ids[-1]
    return next(
        (artifact for artifact in run.artifacts if artifact.artifact_id == artifact_id),
        None,
    )


def _dataset_submit_task(
    *,
    spec: DatasetSubmissionSpec,
    path_contract: JsonObject,
    sample_artifact_id: str,
    sample_task_run_id: str,
) -> DelegatedTask:
    return DelegatedTask(
        name="Submit the accepted Dataset pipeline",
        instruction=(
            "Call df_submit_pipeline exactly once with mode='full' and the supplied "
            "submission_spec. Map pipeline_path to script_path. Do not read a skill, "
            "scan directories, inspect source data, rewrite the pipeline, or choose "
            "different arguments. Return the real task_id, output_dir and status, "
            "then end this turn."
        ),
        inputs={
            "phase": "submit",
            "submission_spec": spec.model_dump(mode="json"),
            "path_contract": path_contract,
            "accepted_sample": {
                "artifact_id": sample_artifact_id,
                "task_run_id": sample_task_run_id,
            },
        },
        constraints=(
            "Use only task.inputs.submission_spec for the platform submission.",
            "Do not call df_submit_pipeline more than once.",
            "Do not submit from Manager text or reconstruct the task from prose.",
        ),
        expected_outputs=(
            "The real platform task_id.",
            "The real platform output_dir.",
            "The submission status returned by df_submit_pipeline.",
        ),
    )
