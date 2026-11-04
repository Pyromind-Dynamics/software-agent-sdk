from __future__ import annotations

import json

import pytest
from harness_adapter.pi_adapter.event_translator import translate_runner_event
from pydantic import ValidationError
from pyromind_runtime.application.conversation_runtime import ConversationRuntime
from pyromind_runtime.application.snapshot_projector import SnapshotProjector
from pyromind_runtime.domain.capabilities import HarnessCapabilities
from pyromind_runtime.domain.content import TextContent
from pyromind_runtime.domain.context import RequestContext
from pyromind_runtime.domain.events import ProductEvent
from pyromind_runtime.domain.snapshot import (
    ConversationSnapshot,
    OperationActivity,
    TimelineOperation,
)
from pyromind_runtime.infrastructure.file_product_store import FileProductStore

from .fake_adapter import FakeAdapter


def activity():
    return {
        "items": [
            {
                "kind": "message",
                "item_id": "child-message",
                "started_seq": 1,
                "role": "assistant",
                "content": [{"type": "text", "text": "**Reading samples**"}],
                "status": "completed",
            },
            {
                "kind": "operation",
                "item_id": "child-tool",
                "started_seq": 2,
                "name": "read",
                "status": "completed",
                "arguments": {"path": "/eval.jsonl"},
                "output": [{"type": "text", "text": "Four samples"}],
            },
        ],
        "truncated": False,
    }


def test_adapter_translates_native_activity_into_shared_models():
    event = translate_runner_event(
        {
            "eventId": "native",
            "sessionId": "chat",
            "runId": "native-child",
            "kind": "subagent.progress",
            "payload": {"activity": activity()},
        }
    )[0]
    assert event.type == "operation.progress"
    assert event.payload["activity"] == OperationActivity.model_validate(
        activity()
    ).model_dump(mode="json")
    assert "sessionFile" not in json.dumps(event.payload)
    operation = TimelineOperation.model_validate(
        {"item_id": "old", "started_seq": 1, "name": "read"}
    )
    assert "activity" not in operation.model_dump()
    for items in (
        [{**activity()["items"][0], "role": "system"}],
        [{**activity()["items"][1], "category": "subagent"}],
        [{**activity()["items"][1], "activity": {"items": []}}],
    ):
        with pytest.raises(ValidationError):
            OperationActivity.model_validate({"items": items})


def test_projection_persists_child_activity_without_reopening_completed_work():
    projector = SnapshotProjector()
    snapshot = ConversationSnapshot(
        conversation_id="chat", capabilities=HarnessCapabilities()
    )
    for seq, kind, values in (
        (1, "operation.started", {"name": "eval", "category": "subagent"}),
        (2, "operation.completed", {"output": [{"type": "text", "text": "Done"}]}),
        (3, "operation.progress", {"activity": activity()}),
    ):
        snapshot = projector.reduce(
            snapshot,
            ProductEvent.model_validate(
                {
                    "conversation_id": "chat",
                    "seq": seq,
                    "type": kind,
                    "payload": {"operation_id": "child", **values},
                }
            ),
        )
    operation = snapshot.timeline[0]
    assert isinstance(operation, TimelineOperation)
    assert operation.status == "completed"
    assert operation.completed_seq == 2
    assert isinstance(operation.output[0], TextContent)
    assert operation.output[0].text == "Done"
    assert operation.activity is not None
    restored = ConversationSnapshot.model_validate_json(snapshot.model_dump_json())
    assert restored == snapshot


async def test_read_recovers_activity_once_without_attaching_or_starting_work(
    tmp_path,
):
    class Reader(FakeAdapter):
        def __init__(self):
            super().__init__()
            self.reads = 0

        async def read_operation_activities(self, handle, operation_ids, context):
            self.reads += 1
            assert operation_ids == ("child",)
            return {
                "child": OperationActivity.model_validate(activity()),
                "foreign": OperationActivity(items=()),
            }

    directory = tmp_path / "chat"
    directory.mkdir()
    store = FileProductStore(directory)
    store.create(
        ConversationSnapshot(
            conversation_id="chat", capabilities=HarnessCapabilities()
        ),
        user_id="42",
        profile_id="profile",
    )
    store.append(
        ProductEvent(
            conversation_id="chat",
            type="operation.started",
            payload={"operation_id": "child", "name": "eval", "category": "subagent"},
        )
    )
    store.append(
        ProductEvent(
            conversation_id="chat",
            type="operation.completed",
            payload={"operation_id": "child", "output": []},
        )
    )
    adapter = Reader()
    runtime = ConversationRuntime(tmp_path, adapter)
    context = RequestContext(user_id="42", profile_id="profile")
    for denied in (
        RequestContext(user_id="other", profile_id="profile"),
        RequestContext(user_id="42", profile_id="other"),
    ):
        with pytest.raises(PermissionError):
            await runtime.get_snapshot("chat", denied)
    assert adapter.reads == 0
    first = await runtime.get_snapshot("chat", context)
    second = await runtime.get_snapshot("chat", context)
    assert first == second
    assert adapter.reads == 1 and adapter.attached == []
    assert len(first.timeline) == 1
    operation = first.timeline[0]
    assert isinstance(operation, TimelineOperation)
    assert operation.status == "completed" and operation.activity is not None
