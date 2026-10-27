from __future__ import annotations

import pytest
from harness_adapter.pi_adapter.adapter import PiAdapter, _PiSession
from harness_adapter.pi_adapter.event_translator import translate_runner_event
from harness_adapter.pi_adapter.persistence import PiSessionFiles
from pyromind_runtime.application import SnapshotProjector
from pyromind_runtime.application.event_projection import ProductEventProjector
from pyromind_runtime.domain.capabilities import HarnessCapabilities
from pyromind_runtime.domain.context import RequestContext
from pyromind_runtime.domain.snapshot import ConversationSnapshot


async def test_generation_protection_uses_generic_policy_and_failure_events(tmp_path):
    files = PiSessionFiles(tmp_path / "conversation")
    files.initialize({"model": {"provider": "openai", "id": "test"}})
    session = _PiSession(
        session_id="conversation",
        workspace_root=tmp_path / "conversation",
        files=files,
        config={},
        context=RequestContext(user_id="test"),
        running=True,
    )
    other = _PiSession(
        session_id="other",
        workspace_root=tmp_path / "other",
        files=PiSessionFiles(tmp_path / "other"),
        config={},
        context=RequestContext(user_id="test"),
    )
    adapter = PiAdapter(tmp_path, terminal_backend="os-sandbox")
    common = {
        "protocolVersion": 2,
        "type": "pi.event",
        "sessionId": session.session_id,
        "runId": "run-1",
        "occurredAt": "2026-10-27T00:00:00Z",
    }
    await adapter._runner_event(
        session,
        {
            **common,
            "eventId": "policy",
            "kind": "execution.policy.updated",
            "payload": {
                "automatic_turns_suspended": True,
                "reason": "model_repetition_detected",
            },
        },
    )
    assert session.attach_recovery is not None
    assert session.attach_recovery.automatic_turns_suspended
    assert other.attach_recovery is None

    # The public event and snapshot use existing, harness-independent contracts.
    harness_event = session.queue.get_nowait()
    assert harness_event is not None
    event = ProductEventProjector().project("conversation", harness_event)
    assert event is not None
    assert event.type == "execution.policy.updated"
    snapshot = SnapshotProjector().reduce(
        ConversationSnapshot(
            conversation_id="conversation", capabilities=HarnessCapabilities()
        ),
        event.model_copy(update={"seq": 1}),
    )
    restored = ConversationSnapshot.model_validate_json(snapshot.model_dump_json())
    assert restored.automatic_turns_suspended
    assert "pinospin" not in event.model_dump_json()

    frame = {
        **common,
        "eventId": "finished",
        "kind": "run.finished",
        "payload": {
            "outcome": {
                "status": "failed",
                "stop_reason": "error",
                "error_code": "model_repetition_detected",
                "message": "Generation interrupted; send a message to continue.",
            }
        },
    }
    await adapter._runner_event(session, frame)
    await adapter._runner_event(session, frame)
    assert session.queue.qsize() == 1
    finished = session.queue.get_nowait()
    assert finished is not None
    assert finished.type == "run.finished"
    assert finished.payload["status"] == "error"
    failure = finished.payload["error"]
    assert isinstance(failure, dict)
    assert failure["code"] == "model_repetition_detected"


def test_execution_policy_translation_rejects_invalid_suspension():
    with pytest.raises(ValueError, match="must be a boolean"):
        translate_runner_event(
            {
                "kind": "execution.policy.updated",
                "payload": {"automatic_turns_suspended": "false"},
            }
        )
