from __future__ import annotations

import json
from typing import Literal
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel
from pyromind_runtime.domain.commands import UserMessageCommand
from pyromind_runtime.domain.content import TextContent
from pyromind_runtime.domain.errors import ProductRuntimeError

from openhands.agent_server.dependencies import check_session_api_key
from openhands.agent_server.echomind_projects import (
    PROJECT_REQUEST_RELATIVE_PATH,
    ProjectInfo,
    ProjectNotFoundError,
    ProjectRequest,
    ProjectRequestInvalidError,
    ProjectRequestNotFoundError,
    ProjectStore,
    read_project_request_text,
)
from openhands.agent_server.echomind_stage_executors import (
    get_project_node_executor,
)
from openhands.agent_server.echomind_stage_runs import (
    ProjectArtifactDocument,
    ProjectRunStore,
    StageRun,
    run_project_node,
)
from pyromind_agent_server.api.router import (
    _product_error,
    _profile_matches,
    context_for_profile,
    get_conversation_service,
    get_product_runtime,
)
from pyromind_agent_server.profiles import ECHOMIND_CHAT_PROFILE


router = APIRouter(
    prefix="/api/echomind",
    tags=["EchoMind Projects"],
    dependencies=[Depends(check_session_api_key)],
)


class CreateEchoMindProjectRequest(BaseModel):
    conversation_id: str
    confirmed: Literal[True]

    model_config = {"extra": "forbid"}


class StartEchoMindProjectNodeResponse(BaseModel):
    schema_version: Literal[1] = 1
    project_id: str
    graph_id: str
    node_id: str
    conversation_id: str
    event_id: str
    status: Literal["started"] = "started"


def _build_dataset_start_context(project: ProjectInfo, node_id: str) -> str:
    node = next(item for item in project.graph.nodes if item.id == node_id)
    context = {
        "project": {
            "project_id": project.project.project_id,
            "graph_id": project.graph.graph_id,
            "name": project.project.name,
            "description": project.project.description,
            "skill": project.project.skill,
            "lifecycle": project.graph.lifecycle,
            "requirements": {
                requirement.key: requirement.value
                for requirement in project.project.requirements
            },
        },
        "stage": {
            "node_id": node.id,
            "node_type": node.type,
            "title": node.title,
            "launch_mode": node.launch_mode,
            "config": node.config,
        },
    }
    return (
        "A user explicitly started this Project stage from the EchoMind UI.\n\n"
        "Server-generated Project and Stage context:\n"
        f"{json.dumps(context, ensure_ascii=False, indent=2)}\n\n"
        "Continue in this existing EchoMind conversation and develop only the "
        "Dataset stage. Use the `data-processing` Skill as the source of truth.\n"
        "Required development sequence:\n"
        "1. Inspect the referenced dataset in Storage and determine its actual "
        "structure. Treat `input_hint` as a hint, not verified metadata.\n"
        "2. Select the matching data-processing paradigm, create the Pipeline, "
        "and run a small sample in the conversation sandbox.\n"
        "3. Show the sample result and wait for explicit user confirmation before "
        "submitting a full validation task.\n"
        "4. If the user confirms, submit the validation task through the "
        "data-processing platform flow and report its task/run/output identifiers.\n\n"
        "Do not create another Project, do not write `graph.json`, do not invent "
        "server identifiers, and do not start Inference, Evaluator, or Train. "
        "Those stages are outside this Dataset development action."
    )


def _require_echomind_conversation(request: Request, conversation_id: str) -> None:
    if not _profile_matches(
        get_conversation_service(request),
        conversation_id,
        ECHOMIND_CHAT_PROFILE,
    ):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="EchoMind conversation not found.",
        )


async def _read_agent_project_request(
    request: Request,
    conversation_id: str,
) -> ProjectRequest:
    try:
        content = await get_product_runtime(request).read_workspace_text(
            conversation_id,
            PROJECT_REQUEST_RELATIVE_PATH.as_posix(),
            context_for_profile(request, ECHOMIND_CHAT_PROFILE),
        )
    except ProductRuntimeError as exc:
        raise _product_error(exc) from exc
    except OSError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="EchoMind workspace file could not be read.",
        ) from exc
    if content is None:
        raise ProjectRequestNotFoundError(PROJECT_REQUEST_RELATIVE_PATH.as_posix())
    return read_project_request_text(content)


@router.get(
    "/conversations/{conversation_id}/project-request",
    responses={404: {"description": "EchoMind project request not found"}},
)
async def get_echomind_project_request(
    conversation_id: str,
    request: Request,
) -> ProjectRequest:
    _require_echomind_conversation(request, conversation_id)
    try:
        return await _read_agent_project_request(request, conversation_id)
    except ProjectRequestNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="EchoMind project request does not exist yet.",
        ) from exc
    except ProjectRequestInvalidError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"EchoMind project request is invalid: {exc}",
        ) from exc


@router.post(
    "/projects",
    response_model=ProjectInfo,
    status_code=status.HTTP_201_CREATED,
)
async def create_echomind_project(
    request: Request,
    body: CreateEchoMindProjectRequest,
) -> ProjectInfo:
    _require_echomind_conversation(request, body.conversation_id)
    context = context_for_profile(request, ECHOMIND_CHAT_PROFILE)
    service = get_conversation_service(request)
    try:
        project_request = await _read_agent_project_request(
            request, body.conversation_id
        )
    except ProjectRequestNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="EchoMind Agent has not produced a project request yet.",
        ) from exc
    except ProjectRequestInvalidError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"EchoMind project request is invalid: {exc}",
        ) from exc
    return ProjectStore.for_conversations_dir(service.conversations_dir).create(
        project_request,
        echomind_conversation_id=body.conversation_id,
        owner_user_id=context.user_id,
    )


@router.get(
    "/projects/{project_id}",
    response_model=ProjectInfo,
    responses={404: {"description": "Project not found"}},
)
async def get_echomind_project(project_id: str, request: Request) -> ProjectInfo:
    service = get_conversation_service(request)
    context = context_for_profile(request, ECHOMIND_CHAT_PROFILE)
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
) -> StartEchoMindProjectNodeResponse:
    context = context_for_profile(request, ECHOMIND_CHAT_PROFILE)
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
    if node.type != "dataset" or node.launch_mode != "agent":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Project node is not Agent-led Dataset development: {node_id}",
        )
    if node.config.get("capability") != "data-processing":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Dataset node is not bound to the data-processing capability.",
        )

    conversation_id = project.project.echomind_conversation_id
    _require_echomind_conversation(request, conversation_id)
    command = UserMessageCommand(
        command_id=uuid4().hex,
        content=(TextContent(text=_build_dataset_start_context(project, node_id)),),
    )
    receipt = await get_product_runtime(request).submit_command(
        conversation_id,
        command,
        context,
    )
    if receipt.status == "failed":
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
) -> StageRun:
    context = context_for_profile(request, ECHOMIND_CHAT_PROFILE)
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
    if node.launch_mode == "agent":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Agent-led Project nodes must use the start endpoint.",
        )
    try:
        get_project_node_executor(node.executor)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc
    return run_project_node(
        project=project,
        node=node,
        run_store=ProjectRunStore(store.root),
    )


@router.get(
    "/projects/{project_id}/runs",
    response_model=list[StageRun],
    responses={404: {"description": "Project not found"}},
)
async def list_echomind_project_runs(
    project_id: str,
    request: Request,
) -> list[StageRun]:
    context = context_for_profile(request, ECHOMIND_CHAT_PROFILE)
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
    context = context_for_profile(request, ECHOMIND_CHAT_PROFILE)
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
