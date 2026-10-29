from __future__ import annotations

from pathlib import Path
from typing import Literal
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel

from openhands.agent_server.agent_profiles import ECHOMIND_PROFILE
from openhands.agent_server.conversation_service import ConversationService
from openhands.agent_server.dependencies import (
    get_conversation_service,
    get_current_user_id,
)
from openhands.agent_server.echomind_projects import (
    ProjectInfo,
    ProjectNotFoundError,
    ProjectRequestInvalidError,
    ProjectRequestNotFoundError,
    ProjectStore,
    read_project_request,
)
from openhands.agent_server.echomind_stage_executors import (
    ProjectNodeExecutionResult,
    execute_project_node,
)
from openhands.agent_server.pyromind_constants import PYROMIND_AGENT_PROFILE_TAG_KEY


projects_router = APIRouter(prefix="/echomind", tags=["EchoMind Projects"])


class CreateEchoMindProjectRequest(BaseModel):
    conversation_id: UUID
    confirmed: Literal[True]

    model_config = {"extra": "forbid"}


@projects_router.post(
    "/projects",
    response_model=ProjectInfo,
    status_code=status.HTTP_201_CREATED,
)
async def create_echomind_project(
    http_request: Request,
    request: CreateEchoMindProjectRequest,
    conversation_service: ConversationService = Depends(get_conversation_service),
) -> ProjectInfo:
    """Create a Project only after explicit user confirmation."""
    user_id = get_current_user_id(http_request)
    event_service = await conversation_service.get_event_service(
        request.conversation_id,
        user_id=user_id,
    )
    if event_service is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="EchoMind conversation not found.",
        )
    if (
        event_service.stored.tags.get(PYROMIND_AGENT_PROFILE_TAG_KEY)
        != ECHOMIND_PROFILE.tag_value
    ):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="EchoMind conversation not found.",
        )

    conversation = event_service.get_conversation()
    try:
        project_request = read_project_request(Path(conversation.workspace.working_dir))
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

    store = ProjectStore.for_conversations_dir(conversation_service.conversations_dir)
    return store.create(
        project_request,
        echomind_conversation_id=str(request.conversation_id),
        owner_user_id=user_id,
    )


@projects_router.get(
    "/projects/{project_id}",
    response_model=ProjectInfo,
    responses={404: {"description": "Project not found"}},
)
async def get_echomind_project(
    project_id: str,
    http_request: Request,
    conversation_service: ConversationService = Depends(get_conversation_service),
) -> ProjectInfo:
    """Read a persisted Project and its Project-owned Graph."""
    store = ProjectStore.for_conversations_dir(conversation_service.conversations_dir)
    try:
        return store.load(
            project_id,
            owner_user_id=get_current_user_id(http_request),
        )
    except ProjectNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project not found: {project_id}",
        ) from exc


@projects_router.post(
    "/projects/{project_id}/nodes/{node_id}/run",
    response_model=ProjectNodeExecutionResult,
    responses={404: {"description": "Project or node not found"}},
)
async def run_echomind_project_node(
    project_id: str,
    node_id: str,
    http_request: Request,
    conversation_service: ConversationService = Depends(get_conversation_service),
) -> ProjectNodeExecutionResult:
    """Resolve one stage executor; current implementations are explicit shells."""
    store = ProjectStore.for_conversations_dir(conversation_service.conversations_dir)
    try:
        project = store.load(
            project_id,
            owner_user_id=get_current_user_id(http_request),
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
    try:
        return execute_project_node(project=project, node=node)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc
