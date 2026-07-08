from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast
from uuid import UUID, uuid4

import pytest
from fastapi import HTTPException

from openhands.agent_server.event_service import EventService
from openhands.agent_server.workflow_canvas_models import (
    SaveWorkflowCanvasEventSnapshotRequest,
)
from openhands.agent_server.workflow_canvas_router import (
    save_workflow_canvas_event_snapshot,
)
from openhands.agent_server.workflow_canvas_store import (
    DuplicateWorkflowCanvasEventSnapshotError,
    FileWorkflowCanvasStore,
)


def _request(
    event_id: str = "event-1",
    *,
    role: str = "in",
    dsl: str = "# workflow: demo\n",
    **overrides,
) -> SaveWorkflowCanvasEventSnapshotRequest:
    payload = {
        "eventId": event_id,
        "snapshotRole": role,
        "workflowDslData": dsl,
        "summary": "snapshot",
        "createdBy": "test",
    }
    payload.update(overrides)
    return SaveWorkflowCanvasEventSnapshotRequest.model_validate(payload)


def test_save_event_snapshot_is_idempotent_for_same_event_and_dsl(tmp_path):
    store = FileWorkflowCanvasStore(tmp_path, session_id="s1")
    request = _request()

    first = store.save_event_snapshot(request)
    retry = store.save_event_snapshot(request)

    assert first.event_id == "event-1"
    assert first.snapshot_role == "in"
    assert first.version_id == "v000001"
    assert first.workflow_dsl_data == "# workflow: demo\n"
    assert retry.version_id == first.version_id
    assert [item.version_id for item in store.list_versions()] == ["v000001"]


def test_save_event_snapshot_rejects_same_event_with_different_dsl(tmp_path):
    store = FileWorkflowCanvasStore(tmp_path, session_id="s1")
    store.save_event_snapshot(_request())

    with pytest.raises(DuplicateWorkflowCanvasEventSnapshotError):
        store.save_event_snapshot(_request(dsl="# workflow: changed\n"))


def test_save_event_snapshot_rejects_same_event_with_different_role(tmp_path):
    store = FileWorkflowCanvasStore(tmp_path, session_id="s1")
    store.save_event_snapshot(_request(role="in"))

    with pytest.raises(DuplicateWorkflowCanvasEventSnapshotError):
        store.save_event_snapshot(_request(role="out"))


def test_batch_get_event_snapshots_omits_missing_events(tmp_path):
    store = FileWorkflowCanvasStore(tmp_path, session_id="s1")
    store.save_event_snapshot(_request("event-1"))
    store.save_event_snapshot(_request("event-2", role="out", dsl="# workflow: out\n"))

    result = store.batch_get_event_snapshots(["event-1", "missing", "event-2"])

    assert list(result) == ["event-1", "event-2"]
    assert result["event-1"].snapshot_role == "in"
    assert result["event-2"].snapshot_role == "out"


def test_get_version_returns_dsl_snapshot(tmp_path):
    store = FileWorkflowCanvasStore(tmp_path, session_id="s1")
    snapshot = store.save_event_snapshot(_request())

    version = store.get_version(snapshot.version_id)

    assert version.workflow_dsl_data == "# workflow: demo\n"


@dataclass
class _StoredConversation:
    id: UUID


@dataclass
class _FakeEventService:
    conversation_dir: Path
    stored: _StoredConversation


@pytest.mark.asyncio
async def test_router_saves_event_snapshot_with_matching_session_id(tmp_path):
    conversation_id = uuid4()
    event_service = _FakeEventService(
        conversation_dir=tmp_path,
        stored=_StoredConversation(id=conversation_id),
    )

    result = await save_workflow_canvas_event_snapshot(
        conversation_id,
        _request(sessionId=conversation_id.hex),
        event_service=cast(EventService, event_service),
    )

    assert result.event_id == "event-1"
    assert result.snapshot_role == "in"
    assert result.version_id == "v000001"


@pytest.mark.asyncio
async def test_router_rejects_mismatched_session_id(tmp_path):
    conversation_id = uuid4()
    event_service = _FakeEventService(
        conversation_dir=tmp_path,
        stored=_StoredConversation(id=conversation_id),
    )

    with pytest.raises(HTTPException) as exc_info:
        await save_workflow_canvas_event_snapshot(
            conversation_id,
            _request(sessionId="different-session"),
            event_service=cast(EventService, event_service),
        )

    assert exc_info.value.status_code == 400
