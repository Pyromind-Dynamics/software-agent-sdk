from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Path, status

from openhands.agent_server.dependencies import get_event_service
from openhands.agent_server.event_service import EventService
from openhands.agent_server.workflow_canvas_models import (
    SaveWorkflowCanvasEventSnapshotRequest,
    WorkflowCanvasEventSnapshot,
    WorkflowCanvasEventSnapshotBatchRequest,
    WorkflowCanvasEventSnapshotBatchResponse,
    WorkflowCanvasVersion,
    WorkflowCanvasVersionListItem,
)
from openhands.agent_server.workflow_canvas_store import (
    DuplicateWorkflowCanvasEventSnapshotError,
    FileWorkflowCanvasStore,
    WorkflowCanvasEventSnapshotNotFoundError,
    WorkflowCanvasStoreError,
    WorkflowCanvasVersionNotFoundError,
)


workflow_canvas_router = APIRouter(
    prefix="/conversations/{conversation_id}/workflow-canvas",
    tags=["Workflow Canvas"],
)


def _store(event_service: EventService) -> FileWorkflowCanvasStore:
    return FileWorkflowCanvasStore(
        conversation_dir=event_service.conversation_dir,
        session_id=event_service.stored.id.hex,
    )


def _validate_session_id(
    conversation_id: UUID,
    payload: SaveWorkflowCanvasEventSnapshotRequest,
) -> None:
    if payload.session_id is None:
        return
    if payload.session_id not in {conversation_id.hex, str(conversation_id)}:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="sessionId must match the conversation_id path parameter",
        )


def _not_found(exc: Exception) -> HTTPException:
    return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))


@workflow_canvas_router.post(
    "/event-snapshots",
    response_model=WorkflowCanvasEventSnapshot,
    response_model_exclude_none=True,
)
async def save_workflow_canvas_event_snapshot(
    conversation_id: UUID,
    payload: SaveWorkflowCanvasEventSnapshotRequest,
    event_service: EventService = Depends(get_event_service),
) -> WorkflowCanvasEventSnapshot:
    _validate_session_id(conversation_id, payload)
    try:
        return _store(event_service).save_event_snapshot(payload)
    except DuplicateWorkflowCanvasEventSnapshotError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="DUPLICATE_WORKFLOW_CANVAS_EVENT_SNAPSHOT",
        ) from exc
    except WorkflowCanvasStoreError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc


@workflow_canvas_router.get(
    "/events/{event_id}/snapshot",
    response_model=WorkflowCanvasEventSnapshot,
    response_model_exclude_none=True,
)
async def get_workflow_canvas_event_snapshot(
    event_id: Annotated[str, Path(min_length=1)],
    event_service: EventService = Depends(get_event_service),
) -> WorkflowCanvasEventSnapshot:
    try:
        return _store(event_service).get_event_snapshot(event_id)
    except (
        WorkflowCanvasEventSnapshotNotFoundError,
        WorkflowCanvasVersionNotFoundError,
    ) as exc:
        raise _not_found(exc) from exc


@workflow_canvas_router.post(
    "/event-snapshots/batch",
    response_model=WorkflowCanvasEventSnapshotBatchResponse,
    response_model_exclude_none=True,
)
async def batch_get_workflow_canvas_event_snapshots(
    payload: WorkflowCanvasEventSnapshotBatchRequest,
    event_service: EventService = Depends(get_event_service),
) -> WorkflowCanvasEventSnapshotBatchResponse:
    try:
        snapshots = _store(event_service).batch_get_event_snapshots(payload.event_ids)
    except WorkflowCanvasVersionNotFoundError as exc:
        raise _not_found(exc) from exc
    return WorkflowCanvasEventSnapshotBatchResponse(snapshots=snapshots)


@workflow_canvas_router.get(
    "/versions",
    response_model=list[WorkflowCanvasVersionListItem],
    response_model_exclude_none=True,
)
async def list_workflow_canvas_versions(
    event_service: EventService = Depends(get_event_service),
) -> list[WorkflowCanvasVersionListItem]:
    return _store(event_service).list_versions()


@workflow_canvas_router.get(
    "/versions/{version_id}",
    response_model=WorkflowCanvasVersion,
    response_model_exclude_none=True,
)
async def get_workflow_canvas_version(
    version_id: Annotated[str, Path(min_length=1)],
    event_service: EventService = Depends(get_event_service),
) -> WorkflowCanvasVersion:
    try:
        return _store(event_service).get_version(version_id)
    except WorkflowCanvasVersionNotFoundError as exc:
        raise _not_found(exc) from exc
