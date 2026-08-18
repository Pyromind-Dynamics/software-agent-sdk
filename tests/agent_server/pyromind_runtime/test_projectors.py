from __future__ import annotations

import json

import pytest
from pyromind_runtime.contracts import (
    HarnessCapabilities,
    HarnessEvent,
    ProductEvent,
    WorkspaceRef,
)
from pyromind_runtime.product import ConversationSnapshot
from pyromind_runtime.projectors import (
    ConversationSnapshotProjector,
    ProductEventProjector,
    WorkflowProductProjector,
    WorkflowProjectionError,
)


def test_product_projector_normalizes_tool_events() -> None:
    source = HarnessEvent(
        session_id="session-1",
        type="tool.started",
        payload={
            "tool_call_id": "call-1",
            "tool_name": "preview_dataset",
            "arguments": {"dataset": "train.csv"},
        },
        provider_metadata={"piToolCallId": "provider-call-1"},
    )

    (event,) = ProductEventProjector().project("conversation-1", source)

    assert event.type == "operation.started"
    assert event.payload == {
        "operation_id": "call-1",
        "name": "preview_dataset",
        "arguments": {"dataset": "train.csv"},
    }
    assert "provider_metadata" not in event.model_dump(mode="json")


def test_snapshot_projector_builds_streaming_message() -> None:
    projector = ConversationSnapshotProjector()
    snapshot = ConversationSnapshot(
        conversation_id="conversation-1",
        capabilities=HarnessCapabilities(partial_message=True),
    )
    events = (
        ProductEvent(
            conversation_id="conversation-1",
            seq=1,
            type="message.started",
            payload={"message_id": "message-1", "role": "assistant"},
        ),
        ProductEvent(
            conversation_id="conversation-1",
            seq=2,
            type="message.delta",
            payload={"message_id": "message-1", "text": "hello "},
        ),
        ProductEvent(
            conversation_id="conversation-1",
            seq=3,
            type="message.delta",
            payload={"message_id": "message-1", "text": "world"},
        ),
        ProductEvent(
            conversation_id="conversation-1",
            seq=4,
            type="message.completed",
            payload={"message_id": "message-1"},
        ),
    )

    for event in events:
        snapshot = projector.reduce(snapshot, event)

    assert snapshot.through_seq == 4
    assert snapshot.messages[0].status == "completed"
    assert snapshot.messages[0].content[0].type == "text"
    assert snapshot.messages[0].content[0].text == "hello world"


def test_workflow_projector_reads_authoritative_canvas_version(tmp_path) -> None:
    workflow_dir = tmp_path / "public_data" / "workflow_canvas"
    workflow_dir.mkdir(parents=True)
    (workflow_dir / "workflow.py").write_text("stale dsl", encoding="utf-8")
    (workflow_dir / "state.json").write_text(
        json.dumps(
            {
                "versions": {
                    "v000002": {
                        "versionId": "v000002",
                        "workflowDslData": "authoritative dsl",
                        "workflowXyflowData": {"nodes": [{"id": "node-1"}]},
                    }
                },
                "eventSnapshots": {"resource-event-1": {"versionId": "v000002"}},
            }
        ),
        encoding="utf-8",
    )
    source = HarnessEvent(
        event_id="resource-event-1",
        session_id="session-1",
        type="resource.updated",
        payload={
            "resource_type": "workflow",
            "resource_id": "workflow-1",
            "version": "resource-event-1",
        },
    )

    event = WorkflowProductProjector().project(
        "conversation-1",
        WorkspaceRef(workspace_id="workspace-1", root=str(tmp_path)),
        source,
    )

    assert event is not None
    assert event.type == "workflow.updated"
    assert event.payload == {
        "resource_id": "workflow-1",
        "version": "v000002",
        "dsl": "authoritative dsl",
        "canvas": {"nodes": [{"id": "node-1"}]},
    }


def test_workflow_projector_rejects_symlink_escape(tmp_path) -> None:
    outside = tmp_path.parent / f"{tmp_path.name}-outside-workflow.py"
    outside.write_text("secret", encoding="utf-8")
    workflow_dir = tmp_path / "public_data" / "workflow_canvas"
    workflow_dir.mkdir(parents=True)
    (workflow_dir / "workflow.py").symlink_to(outside)
    source = HarnessEvent(
        session_id="session-1",
        type="resource.updated",
        payload={
            "resource_type": "workflow",
            "resource_id": "workflow-1",
            "version": "resource-event-1",
        },
    )

    with pytest.raises(WorkflowProjectionError, match="escapes workspace"):
        WorkflowProductProjector().project(
            "conversation-1",
            WorkspaceRef(workspace_id="workspace-1", root=str(tmp_path)),
            source,
        )
