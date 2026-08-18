from __future__ import annotations

import json

import pytest
from pyromind_runtime.contracts import (
    HarnessCapabilities,
    ProductEvent,
    SandboxRef,
    WorkspaceRef,
)
from pyromind_runtime.product import (
    CommandReceipt,
    CommandReceiptConflictError,
    ConversationMetadata,
    ConversationSnapshot,
    EventStoreCorruptionError,
    FileConversationEventStore,
)


def _metadata(conversation_id: str) -> ConversationMetadata:
    capabilities = HarnessCapabilities(cancel=True, partial_message=True)
    return ConversationMetadata(
        conversation_id=conversation_id,
        user_id="user-1",
        harness_id="openhands",
        adapter_session_ref=f"adapter-{conversation_id}",
        capabilities=capabilities,
        workspace=WorkspaceRef(workspace_id="workspace-1", root="/workspace"),
        sandbox=SandboxRef(sandbox_id="sandbox-1", backend="local"),
    )


def _snapshot(metadata: ConversationMetadata) -> ConversationSnapshot:
    return ConversationSnapshot(
        conversation_id=metadata.conversation_id,
        capabilities=metadata.capabilities,
    )


def _reduce(
    snapshot: ConversationSnapshot,
    event: ProductEvent,
) -> ConversationSnapshot:
    status = snapshot.status
    if event.type == "run.started":
        status = "running"
    elif event.type == "run.completed":
        status = "idle"
    elif event.type == "run.failed":
        status = "failed"
    return snapshot.model_copy(update={"through_seq": event.seq, "status": status})


def _event(conversation_id: str, event_type: str) -> ProductEvent:
    return ProductEvent.model_validate(
        {
            "conversation_id": conversation_id,
            "type": event_type,
        }
    )


def test_append_assigns_sequence_and_replays_after_cursor(tmp_path) -> None:
    metadata = _metadata("conversation-1")
    store = FileConversationEventStore(tmp_path, metadata.conversation_id)
    store.create(metadata, _snapshot(metadata))

    first, first_snapshot = store.append(
        _event(metadata.conversation_id, "run.started"),
        _reduce,
    )
    second, second_snapshot = store.append(
        _event(metadata.conversation_id, "run.completed"),
        _reduce,
    )

    assert first.seq == 1
    assert second.seq == 2
    assert first_snapshot.status == "running"
    assert second_snapshot.status == "idle"
    assert second_snapshot.through_seq == 2
    assert store.replay(after_seq=1) == (second,)
    assert store.load_metadata().last_sequence == 2


def test_load_snapshot_rebuilds_stale_snapshot_and_metadata(tmp_path) -> None:
    metadata = _metadata("conversation-2")
    initial_snapshot = _snapshot(metadata)
    store = FileConversationEventStore(tmp_path, metadata.conversation_id)
    store.create(metadata, initial_snapshot)
    persisted, _ = store.append(
        _event(metadata.conversation_id, "run.started"),
        _reduce,
    )
    store.snapshot_path.write_text(initial_snapshot.model_dump_json(), encoding="utf-8")
    store.metadata_path.write_text(metadata.model_dump_json(), encoding="utf-8")

    recovered = store.load_snapshot(_reduce)

    assert recovered.through_seq == persisted.seq
    assert recovered.status == "running"
    assert store.load_metadata().last_sequence == persisted.seq


def test_replay_repairs_an_incomplete_trailing_event(tmp_path) -> None:
    metadata = _metadata("conversation-3")
    store = FileConversationEventStore(tmp_path, metadata.conversation_id)
    store.create(metadata, _snapshot(metadata))
    first, _ = store.append(
        _event(metadata.conversation_id, "run.started"),
        _reduce,
    )
    with store.events_path.open("ab") as event_log:
        event_log.write(b'{"schema_version":1,"event_id":')

    assert store.replay() == (first,)

    second, _ = store.append(
        _event(metadata.conversation_id, "run.completed"),
        _reduce,
    )
    assert [event.seq for event in store.replay()] == [1, 2]
    assert second.seq == 2


def test_replay_rejects_corruption_before_the_last_line(tmp_path) -> None:
    metadata = _metadata("conversation-4")
    store = FileConversationEventStore(tmp_path, metadata.conversation_id)
    store.create(metadata, _snapshot(metadata))
    first = _event(metadata.conversation_id, "run.started").model_copy(
        update={"seq": 1}
    )
    second = _event(metadata.conversation_id, "run.completed").model_copy(
        update={"seq": 2}
    )
    store.events_path.write_text(
        f"{first.model_dump_json()}\nnot-json\n{second.model_dump_json()}\n",
        encoding="utf-8",
    )

    with pytest.raises(EventStoreCorruptionError):
        store.replay()


def test_command_receipts_are_idempotent_and_terminal(tmp_path) -> None:
    metadata = _metadata("conversation-5")
    store = FileConversationEventStore(tmp_path, metadata.conversation_id)
    store.create(metadata, _snapshot(metadata))

    accepted, created = store.claim_command("command-1")
    retried, retried_created = store.claim_command("command-1")
    completed = CommandReceipt(
        command_id="command-1",
        status="completed",
        response={"accepted": True},
    )

    assert created is True
    assert retried_created is False
    assert retried == accepted
    assert store.complete_command(completed) == completed
    assert store.complete_command(completed) == completed

    with pytest.raises(CommandReceiptConflictError):
        store.complete_command(
            completed.model_copy(update={"status": "failed", "response": {}})
        )


def test_conversation_id_cannot_escape_store_root(tmp_path) -> None:
    with pytest.raises(ValueError):
        FileConversationEventStore(tmp_path, "../outside")


def test_persisted_files_use_the_stable_public_shape(tmp_path) -> None:
    metadata = _metadata("conversation-6")
    store = FileConversationEventStore(tmp_path, metadata.conversation_id)
    store.create(metadata, _snapshot(metadata))

    persisted_metadata = json.loads(store.metadata_path.read_text(encoding="utf-8"))
    persisted_snapshot = json.loads(store.snapshot_path.read_text(encoding="utf-8"))

    assert persisted_metadata["harness_id"] == "openhands"
    assert persisted_metadata["last_sequence"] == 0
    assert persisted_snapshot["through_seq"] == 0
    assert "provider_metadata" not in persisted_snapshot
