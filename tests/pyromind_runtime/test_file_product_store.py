from pathlib import Path

import pytest
from pydantic import ValidationError
from pyromind_runtime.domain.capabilities import HarnessCapabilities
from pyromind_runtime.domain.commands import CommandReceipt, UserMessageCommand
from pyromind_runtime.domain.content import TextContent
from pyromind_runtime.domain.events import ProductEvent
from pyromind_runtime.domain.snapshot import ConversationSnapshot
from pyromind_runtime.infrastructure.file_product_store import (
    CommandConflictError,
    FileProductStore,
)


def _store(tmp_path: Path) -> FileProductStore:
    conversation = tmp_path / "workspace" / "conversations" / "conversation-1"
    conversation.mkdir(parents=True)
    store = FileProductStore(conversation)
    store.create(
        ConversationSnapshot(
            conversation_id="conversation-1",
            capabilities=HarnessCapabilities(cancel=True),
        ),
        user_id="user-1",
    )
    return store


def test_store_lives_inside_conversation_and_rebuilds_snapshot(tmp_path: Path) -> None:
    store = _store(tmp_path)
    persisted, snapshot = store.append(
        ProductEvent(
            event_id="event-1",
            conversation_id="conversation-1",
            type="status.changed",
            payload={"status": "running"},
        )
    )

    assert persisted.seq == 1
    assert snapshot.status == "running"
    assert store.directory == (
        tmp_path / "workspace" / "conversations" / "conversation-1" / "product"
    )
    store.snapshot_path.write_text("broken", encoding="utf-8")
    assert store.load_snapshot().status == "running"


def test_store_deduplicates_events(tmp_path: Path) -> None:
    store = _store(tmp_path)
    event = ProductEvent(
        event_id="stable-event",
        conversation_id="conversation-1",
        type="status.changed",
        payload={"status": "running"},
    )
    first, _ = store.append(event)
    second, snapshot = store.append(event)

    assert first == second
    assert snapshot.through_seq == 1
    assert len(store.replay()) == 1


def test_store_deduplicates_translations_from_same_source_event(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    first, _ = store.append(
        ProductEvent(
            event_id="streamed-completion",
            source_event_id="openhands-message",
            conversation_id="conversation-1",
            type="message.completed",
            run_id="command-1",
            payload={
                "message_id": "command-1:assistant",
                "role": "assistant",
                "content": [{"type": "text", "text": "done"}],
            },
        )
    )
    replayed, snapshot = store.append(
        ProductEvent(
            event_id="openhands-message:completed",
            source_event_id="openhands-message",
            conversation_id="conversation-1",
            type="message.completed",
            payload={
                "message_id": "openhands-message",
                "role": "assistant",
                "content": [{"type": "text", "text": "done"}],
            },
        )
    )

    assert replayed == first
    assert snapshot.through_seq == 1
    assert len(snapshot.timeline) == 1
    assert len(store.replay()) == 1


def test_store_repairs_legacy_duplicate_source_events(tmp_path: Path) -> None:
    store = _store(tmp_path)
    original, snapshot = store.append(
        ProductEvent(
            event_id="streamed-completion",
            source_event_id="openhands-message",
            conversation_id="conversation-1",
            type="message.completed",
            run_id="command-1",
            payload={
                "message_id": "command-1:assistant",
                "role": "assistant",
                "content": [{"type": "text", "text": "done"}],
            },
        )
    )
    duplicate = original.model_copy(
        update={
            "seq": 2,
            "event_id": "openhands-message:completed",
            "run_id": None,
            "payload": {
                "message_id": "openhands-message",
                "role": "assistant",
                "content": [{"type": "text", "text": "done"}],
            },
        }
    )
    with store.events_path.open("a", encoding="utf-8") as stream:
        stream.write(duplicate.model_dump_json())
        stream.write("\n")
    store.snapshot_path.write_text(
        snapshot.model_copy(
            update={
                "through_seq": 2,
                "timeline": (
                    *snapshot.timeline,
                    snapshot.timeline[0].model_copy(
                        update={"item_id": "openhands-message", "run_id": None}
                    ),
                ),
            }
        ).model_dump_json(),
        encoding="utf-8",
    )

    repaired = store.load_snapshot()
    assert repaired.through_seq == 2
    assert len(repaired.timeline) == 1
    assert repaired.timeline[0].item_id == "command-1:assistant"


def test_store_does_not_persist_event_that_cannot_be_projected(tmp_path: Path) -> None:
    store = _store(tmp_path)

    with pytest.raises(ValidationError):
        store.append(
            ProductEvent(
                event_id="invalid-plan",
                conversation_id="conversation-1",
                type="plan.updated",
                payload={"steps": ["not-a-plan-step"]},
            )
        )

    assert store.replay() == ()
    assert store.load_snapshot().through_seq == 0


def test_command_idempotency_checks_payload(tmp_path: Path) -> None:
    store = _store(tmp_path)
    command = UserMessageCommand(
        command_id="command-1",
        content=(TextContent(text="hello"),),
    )
    receipt, claimed = store.claim_command(command)
    repeated, claimed_again = store.claim_command(command)

    assert receipt == repeated
    assert claimed
    assert not claimed_again

    with pytest.raises(CommandConflictError):
        store.claim_command(
            UserMessageCommand(
                command_id="command-1",
                content=(TextContent(text="different"),),
            )
        )

    completed = receipt.model_copy(update={"status": "completed"})
    assert store.complete_command(completed) == completed
    assert isinstance(completed, CommandReceipt)


def test_missing_harness_metadata_defaults_to_openhands(tmp_path: Path) -> None:
    store = _store(tmp_path)
    metadata = store.metadata_path.read_text()
    store.metadata_path.write_text(metadata.replace('"harness_id":"openhands",', ""))
    assert store.harness_id() == "openhands"
