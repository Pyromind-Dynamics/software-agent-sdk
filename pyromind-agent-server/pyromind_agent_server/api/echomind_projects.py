from __future__ import annotations

import json
import os
from typing import Literal
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field
from pyromind_runtime.domain.commands import UserMessageCommand
from pyromind_runtime.domain.content import TextContent
from pyromind_runtime.domain.project_graphs import (
    GraphPatch,
    ProjectGraphView,
    canonical_execution_role,
)

from openhands.agent_server.dependencies import check_session_api_key
from openhands.agent_server.echomind_dataset_contract import (
    require_dataset_submission_spec,
)
from openhands.agent_server.echomind_projects import (
    ProjectInfo,
    ProjectNode,
    ProjectNotFoundError,
    ProjectStore,
)
from openhands.agent_server.echomind_stage_runs import (
    ProjectArtifactDocument,
    ProjectRunStore,
    StageRun,
    require_sample_validation,
    run_project_node,
)
from openhands.agent_server.pyromind_auth import (
    parse_auth_token_from_cookie_header,
)
from pyromind_agent_server.api.router import (
    _profile_matches,
    context_for_profile,
    get_conversation_service,
    get_product_runtime,
)
from pyromind_agent_server.echomind_paths import (
    storage_path_contract as _storage_path_contract,
)
from pyromind_agent_server.profiles import (
    ECHOMIND_PRODUCT_PROFILE,
    session_policy_for_profile,
)


router = APIRouter(
    prefix="/api/echomind",
    tags=["EchoMind Projects"],
    dependencies=[Depends(check_session_api_key)],
)


class StartEchoMindProjectNodeRequest(BaseModel):
    request_id: str | None = Field(default=None, min_length=1, max_length=128)


class StartEchoMindProjectNodeResponse(BaseModel):
    schema_version: Literal[1] = 1
    project_id: str
    graph_id: str
    node_id: str
    conversation_id: str
    event_id: str
    run: StageRun
    status: Literal["started"] = "started"


class RunEchoMindProjectNodeRequest(BaseModel):
    request_id: str | None = Field(default=None, min_length=1, max_length=128)


class ContinueEchoMindProjectRunRequest(BaseModel):
    request_id: str = Field(min_length=1, max_length=128)


class ContinueEchoMindProjectRunResponse(BaseModel):
    schema_version: Literal[1] = 1
    project_id: str
    graph_id: str
    node_id: str
    conversation_id: str
    event_id: str
    run: StageRun
    status: Literal["started"] = "started"


def _sample_artifact_kind(node: ProjectNode) -> str:
    configured = node.config.get("sample_report_kind")
    return (
        configured
        if isinstance(configured, str) and configured
        else ("stage_sample_report")
    )


def _build_agent_stage_start_context(
    project: ProjectInfo,
    node_id: str,
    run: StageRun,
) -> str:
    node = next(item for item in project.graph.nodes if item.id == node_id)
    execution_role = canonical_execution_role(node.execution_role)
    if execution_role == "dataset":
        execution_instruction = (
            "After approval, delegate the first TaskRun with "
            'task.inputs.phase="sample". The Dataset child must validate a real '
            "sample before the Manager records waiting_user for user confirmation."
        )
    else:
        execution_instruction = (
            "After approval, delegate one execution TaskRun with "
            'task.inputs.phase="execute". This node does not require the Dataset '
            "sample-confirmation cycle."
        )
    return (
        "The user opened Agent-led development for one Project Node. This starts "
        "only the clarification phase; implementation and execution are not yet "
        "authorized.\n\n"
        f"project_id: {project.project.project_id}\n"
        f"node_id: {node.id}\n"
        f"run_id: {run.run_id}\n"
        f"execution_role: {execution_role}\n\n"
        "Read this Project and StageRun with get_project_graph and "
        "get_project_stage_run. Use the Project Skill and scenario reference only "
        "to identify what this Node needs clarified. Then ask the user a small "
        "batch of essential questions about the Node goal, inputs, outputs, "
        "constraints, and acceptance criteria. Do not inspect Storage, create or "
        "modify files, run a sample, submit a platform task, or call "
        "record_project_stage_result yet.\n\n"
        "Only after the user explicitly confirms the Node plan, call "
        "approve_project_stage_plan with a concise summary and the agreed "
        "decisions. Then delegate the real Node work with "
        "request_project_stage_task. Use execution_role from this context and "
        "pass the node inputs, configuration and acceptance criteria. For the "
        f"{execution_instruction} Do not execute the Node implementation, data "
        "processing, validator or platform submission in the Manager conversation."
    )


def _build_agent_stage_continuation_context(
    project: ProjectInfo,
    run: StageRun,
    sample_artifact: ProjectArtifactDocument,
) -> str:
    node = next(item for item in project.graph.nodes if item.id == run.node_id)
    sample_details = sample_artifact.content.get("details")
    if not isinstance(sample_details, dict):
        sample_details = {}
    context = {
        "project_id": project.project.project_id,
        "graph_id": project.graph.graph_id,
        "run_id": run.run_id,
        "node_id": run.node_id,
        "task_type": run.task_type,
        "executor": node.executor.model_dump(mode="json"),
        "node_config": node.config,
        "execution_role": canonical_execution_role(node.execution_role),
        "sample_artifact": sample_artifact.model_dump(mode="json"),
        "sample_submission_spec": sample_details.get("submission_spec"),
        "path_contract": _storage_path_contract(dict(node.config)),
    }
    return (
        "The user explicitly confirmed the sample for this Project Node.\n\n"
        "Server-generated continuation context:\n"
        f"{json.dumps(context, ensure_ascii=False, indent=2)}\n\n"
        "Do not rerun the sample and do not execute the full data-processing "
        "task in the Manager conversation. Delegate one new TaskRun with "
        "request_project_stage_task using execution_role from the context and "
        'task.inputs.phase="submit". The Stage tool injects the frozen '
        "submission_spec from the accepted sample; do not restate or replace it "
        "in Manager prose.\n\n"
        "Use the Dataset child's real result to update the StageRun. Only record "
        "waiting_external from the child's actual platform receipt, and only "
        "record succeeded or failed after the child verifies the final output. "
        "Never submit or resubmit a platform task from the Manager turn."
    )


def _require_echomind_conversation(request: Request, conversation_id: str) -> None:
    if not _profile_matches(
        get_conversation_service(request),
        conversation_id,
        ECHOMIND_PRODUCT_PROFILE,
    ):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="EchoMind conversation not found.",
        )


def _require_sandbox_credentials(request: Request) -> None:
    if os.getenv("PYROMIND_HARNESS_BACKEND", "").strip().lower() != "pi":
        return
    if os.getenv("PYROMIND_PI_TERMINAL_BACKEND", "").strip().lower() != "sandbox":
        return

    auth_token = parse_auth_token_from_cookie_header(request.headers.get("cookie"))
    authorization = request.headers.get("authorization", "")
    scheme, _, value = authorization.partition(" ")
    if not auth_token and not (scheme.lower() == "bearer" and value.strip()):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                "Agent-led Node development runs in the conversation sandbox. "
                "Enter a Cookie containing auth_token=... or a Bearer "
                "Authorization, and set x-cluster, then click Apply."
            ),
        )


@router.get(
    "/projects/{project_id}",
    response_model=ProjectInfo,
    responses={404: {"description": "Project not found"}},
)
async def get_echomind_project(project_id: str, request: Request) -> ProjectInfo:
    service = get_conversation_service(request)
    context = context_for_profile(request, ECHOMIND_PRODUCT_PROFILE)
    try:
        return ProjectStore.for_conversations_dir(service.conversations_dir).load(
            project_id,
            owner_user_id=context.user_id,
        )
    except ProjectNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project not found: {project_id}",
        ) from exc


@router.get(
    "/projects/{project_id}/graph",
    response_model=ProjectGraphView,
    responses={404: {"description": "Project not found"}},
)
async def get_echomind_project_graph(
    project_id: str,
    request: Request,
) -> ProjectGraphView:
    context = context_for_profile(request, ECHOMIND_PRODUCT_PROFILE)
    service = get_conversation_service(request)
    try:
        return ProjectStore.for_conversations_dir(service.conversations_dir).get_graph(
            project_id,
            owner_user_id=context.user_id,
        )
    except ProjectNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project not found: {project_id}",
        ) from exc


@router.patch(
    "/projects/{project_id}/graph",
    response_model=ProjectGraphView,
    responses={404: {"description": "Project not found"}},
)
async def patch_echomind_project_graph(
    project_id: str,
    patch: GraphPatch,
    request: Request,
) -> ProjectGraphView:
    context = context_for_profile(request, ECHOMIND_PRODUCT_PROFILE)
    service = get_conversation_service(request)
    try:
        project = ProjectStore.for_conversations_dir(service.conversations_dir).load(
            project_id,
            owner_user_id=context.user_id,
        )
    except ProjectNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project not found: {project_id}",
        ) from exc
    _require_echomind_conversation(request, project.project.echomind_conversation_id)
    try:
        return await get_product_runtime(request).apply_project_graph_patch(
            project.project.echomind_conversation_id,
            context,
            project_id,
            patch,
        )
    except PermissionError as exc:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(exc),
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc


@router.post(
    "/projects/{project_id}/nodes/{node_id}/start",
    response_model=StartEchoMindProjectNodeResponse,
    status_code=status.HTTP_202_ACCEPTED,
    responses={404: {"description": "Project or node not found"}},
)
async def start_echomind_project_node(
    project_id: str,
    node_id: str,
    request: Request,
    payload: StartEchoMindProjectNodeRequest | None = None,
) -> StartEchoMindProjectNodeResponse:
    _require_sandbox_credentials(request)
    context = context_for_profile(request, ECHOMIND_PRODUCT_PROFILE)
    service = get_conversation_service(request)
    store = ProjectStore.for_conversations_dir(service.conversations_dir)
    try:
        project = store.load(project_id, owner_user_id=context.user_id)
    except ProjectNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project not found: {project_id}",
        ) from exc

    node = next((item for item in project.graph.nodes if item.id == node_id), None)
    if node is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project node not found: {node_id}",
        )
    if project.graph.lifecycle != "development":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Frozen Projects cannot start Agent-led stage development.",
        )
    if not node.execution_role:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Project node has no development execution role: {node_id}",
        )

    conversation_id = project.project.echomind_conversation_id
    _require_echomind_conversation(request, conversation_id)
    runtime = get_product_runtime(request)
    if not await runtime.apply_session_configuration(
        conversation_id,
        session_policy_for_profile(ECHOMIND_PRODUCT_PROFILE),
        context,
        activate_development=True,
    ):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="EchoMind conversation is busy; retry when it becomes idle.",
        )
    run_store = ProjectRunStore(store.root)
    try:
        run, created = run_store.start_agent_development(
            project=project,
            node=node,
            request_id=payload.request_id if payload else None,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc
    if not created:
        if run.command_id is None:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Stage startup is still in progress; retry shortly.",
            )
        return StartEchoMindProjectNodeResponse(
            project_id=project.project.project_id,
            graph_id=project.graph.graph_id,
            node_id=node.id,
            conversation_id=conversation_id,
            event_id=run.command_id,
            run=run,
        )

    command_id = uuid4().hex
    run = run_store.attach_command(run, command_id=command_id)
    command = UserMessageCommand(
        command_id=command_id,
        content=(
            TextContent(
                text=_build_agent_stage_start_context(project, node_id, run),
            ),
        ),
    )
    try:
        receipt = await runtime.submit_command(
            conversation_id,
            command,
            context,
        )
    except Exception as exc:
        run_store.fail(run, exc)
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Project stage context could not be delivered.",
        ) from exc
    if receipt.status == "failed":
        run_store.fail(
            run,
            RuntimeError("Project stage context could not be delivered."),
        )
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Project stage context could not be delivered.",
        )
    return StartEchoMindProjectNodeResponse(
        project_id=project.project.project_id,
        graph_id=project.graph.graph_id,
        node_id=node.id,
        conversation_id=conversation_id,
        event_id=receipt.command_id,
        run=run,
    )


@router.post(
    "/projects/{project_id}/runs/{run_id}/confirm",
    response_model=ContinueEchoMindProjectRunResponse,
    status_code=status.HTTP_202_ACCEPTED,
    responses={404: {"description": "Project, node, or run not found"}},
)
async def confirm_echomind_project_run(
    project_id: str,
    run_id: str,
    payload: ContinueEchoMindProjectRunRequest,
    request: Request,
) -> ContinueEchoMindProjectRunResponse:
    _require_sandbox_credentials(request)
    context = context_for_profile(request, ECHOMIND_PRODUCT_PROFILE)
    service = get_conversation_service(request)
    store = ProjectStore.for_conversations_dir(service.conversations_dir)
    try:
        project = store.load(project_id, owner_user_id=context.user_id)
    except ProjectNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project not found: {project_id}",
        ) from exc
    if project.graph.lifecycle != "development":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Frozen Projects cannot continue stage development.",
        )
    run_store = ProjectRunStore(store.root)
    try:
        run = run_store.get(project_id=project_id, run_id=run_id)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project run not found: {run_id}",
        ) from exc
    node = next((item for item in project.graph.nodes if item.id == run.node_id), None)
    if node is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project node not found: {run.node_id}",
        )
    if run.mode != "agent_development" or not node.execution_role:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                "Only Agent development runs with an execution role can be confirmed."
            ),
        )
    existing_command = next(
        (
            update.command_id
            for update in run.updates
            if update.request_id == payload.request_id
        ),
        None,
    )
    if existing_command is not None:
        return ContinueEchoMindProjectRunResponse(
            project_id=project.project.project_id,
            graph_id=project.graph.graph_id,
            node_id=run.node_id,
            conversation_id=project.project.echomind_conversation_id,
            event_id=existing_command,
            run=run,
        )
    sample_artifact = next(
        (
            artifact
            for artifact in reversed(run.artifacts)
            if artifact.kind == _sample_artifact_kind(node)
        ),
        None,
    )
    if sample_artifact is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="The Dataset run has no completed sample report to confirm.",
        )
    try:
        sample_document = run_store.load_artifact(
            project_id=project_id,
            artifact_id=sample_artifact.artifact_id,
        )
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="The Dataset sample report is missing.",
        ) from exc
    try:
        require_sample_validation(
            node,
            sample_document.content.get("details", {}),
        )
        if node.type == "dataset":
            sample_details = sample_document.content.get("details", {})
            if not isinstance(sample_details, dict):
                raise ValueError("sample details must be an object")
            sample_task_run_id = sample_document.content.get("task_run_id")
            if not isinstance(sample_task_run_id, str) or not sample_task_run_id:
                raise ValueError("sample report has no TaskRun identity")
            require_dataset_submission_spec(
                sample_details,
                task_run_id=sample_task_run_id,
            )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"The Node sample report is not valid for confirmation: {exc}",
        ) from exc

    conversation_id = project.project.echomind_conversation_id
    _require_echomind_conversation(request, conversation_id)
    runtime = get_product_runtime(request)
    if not await runtime.apply_session_configuration(
        conversation_id,
        session_policy_for_profile(ECHOMIND_PRODUCT_PROFILE),
        context,
        activate_development=True,
    ):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="EchoMind conversation is busy; retry when it becomes idle.",
        )

    command_id = uuid4().hex
    try:
        run, created = run_store.begin_continuation(
            project=project,
            run_id=run_id,
            request_id=payload.request_id,
            command_id=command_id,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc
    if not created:
        existing_command = next(
            (
                update.command_id
                for update in run.updates
                if update.request_id == payload.request_id
            ),
            None,
        )
        if existing_command is None:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Stage continuation is still being prepared; retry shortly.",
            )
        return ContinueEchoMindProjectRunResponse(
            project_id=project.project.project_id,
            graph_id=project.graph.graph_id,
            node_id=run.node_id,
            conversation_id=conversation_id,
            event_id=existing_command,
            run=run,
        )

    command = UserMessageCommand(
        command_id=command_id,
        content=(
            TextContent(
                text=_build_agent_stage_continuation_context(
                    project,
                    run,
                    sample_document,
                ),
            ),
        ),
    )
    try:
        receipt = await runtime.submit_command(conversation_id, command, context)
    except Exception as exc:
        run_store.fail(run, exc)
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Node continuation could not be delivered.",
        ) from exc
    if receipt.status == "failed":
        run_store.fail(
            run,
            RuntimeError("Node continuation could not be delivered."),
        )
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Node continuation could not be delivered.",
        )
    return ContinueEchoMindProjectRunResponse(
        project_id=project.project.project_id,
        graph_id=project.graph.graph_id,
        node_id=run.node_id,
        conversation_id=conversation_id,
        event_id=receipt.command_id,
        run=run,
    )


@router.post(
    "/projects/{project_id}/nodes/{node_id}/run",
    response_model=StageRun,
    status_code=status.HTTP_201_CREATED,
    responses={404: {"description": "Project or node not found"}},
)
async def run_echomind_project_node(
    project_id: str,
    node_id: str,
    request: Request,
    payload: RunEchoMindProjectNodeRequest | None = None,
) -> StageRun:
    context = context_for_profile(request, ECHOMIND_PRODUCT_PROFILE)
    service = get_conversation_service(request)
    store = ProjectStore.for_conversations_dir(service.conversations_dir)
    try:
        project = store.load(
            project_id,
            owner_user_id=context.user_id,
        )
    except ProjectNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project not found: {project_id}",
        ) from exc
    node = next((item for item in project.graph.nodes if item.id == node_id), None)
    if node is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project node not found: {node_id}",
        )
    if project.graph.lifecycle != "development":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Frozen Projects cannot run node validation.",
        )
    try:
        return run_project_node(
            project=project,
            node=node,
            run_store=ProjectRunStore(store.root),
            request_id=payload.request_id if payload else None,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc


@router.get(
    "/projects/{project_id}/runs",
    response_model=list[StageRun],
    responses={404: {"description": "Project not found"}},
)
async def list_echomind_project_runs(
    project_id: str,
    request: Request,
) -> list[StageRun]:
    context = context_for_profile(request, ECHOMIND_PRODUCT_PROFILE)
    service = get_conversation_service(request)
    store = ProjectStore.for_conversations_dir(service.conversations_dir)
    try:
        store.load(
            project_id,
            owner_user_id=context.user_id,
        )
    except ProjectNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project not found: {project_id}",
        ) from exc
    return ProjectRunStore(store.root).list(project_id)


@router.get(
    "/projects/{project_id}/artifacts/{artifact_id}",
    response_model=ProjectArtifactDocument,
    responses={404: {"description": "Project or artifact not found"}},
)
async def get_echomind_project_artifact(
    project_id: str,
    artifact_id: str,
    request: Request,
) -> ProjectArtifactDocument:
    context = context_for_profile(request, ECHOMIND_PRODUCT_PROFILE)
    service = get_conversation_service(request)
    store = ProjectStore.for_conversations_dir(service.conversations_dir)
    try:
        store.load(
            project_id,
            owner_user_id=context.user_id,
        )
    except ProjectNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project not found: {project_id}",
        ) from exc
    try:
        return ProjectRunStore(store.root).load_artifact(
            project_id=project_id,
            artifact_id=artifact_id,
        )
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project artifact not found: {artifact_id}",
        ) from exc
