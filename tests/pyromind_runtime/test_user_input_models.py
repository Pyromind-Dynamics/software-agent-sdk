from __future__ import annotations

import json

import pytest
from pydantic import ValidationError
from pyromind_runtime.application import SnapshotProjector
from pyromind_runtime.domain.capabilities import HarnessCapabilities
from pyromind_runtime.domain.commands import InputResponseCommand
from pyromind_runtime.domain.events import ProductEvent
from pyromind_runtime.domain.snapshot import ConversationSnapshot
from pyromind_runtime.domain.user_input import (
    MAX_USER_INPUT_BYTES,
    PendingUserInput,
    UserInputAnswer,
    UserInputOption,
    UserInputQuestion,
    UserInputRequest,
    UserInputResolution,
)


def _question(**overrides: object) -> dict:
    question: dict = {
        "id": "duplicate_policy",
        "prompt": "如何处理重复行？",
        "options": [
            {"id": "keep_first", "label": "保留第一条"},
            {"id": "drop_all", "label": "全部删除"},
        ],
    }
    question.update(overrides)
    return question


def _request(questions: list[dict]) -> UserInputRequest:
    return UserInputRequest.model_validate({"questions": questions})


def test_waiting_keeps_run_identity_and_rejects_old_run_status() -> None:
    snapshot = ConversationSnapshot(
        conversation_id="c", capabilities=HarnessCapabilities(user_input_reply=True)
    )
    events = [
        ("status.changed", "run-1", {"status": "running"}),
        (
            "input.requested",
            "run-1",
            {"request_id": "input-1", "questions": [_question()]},
        ),
        ("status.changed", "run-1", {"status": "running"}),
        ("status.changed", "old-run", {"status": "idle"}),
    ]
    projector = SnapshotProjector()
    for seq, (kind, run_id, payload) in enumerate(events, 1):
        snapshot = projector.reduce(
            snapshot,
            ProductEvent.model_validate(
                {
                    "conversation_id": "c",
                    "seq": seq,
                    "type": kind,
                    "run_id": run_id,
                    "payload": payload,
                }
            ),
        )
    assert snapshot.status == "waiting_for_input"
    assert snapshot.active_run_id == "run-1"


@pytest.mark.parametrize("outcome", ["cancelled", "interrupted"])
def test_resolution_persists_suspension_before_separate_policy_event(outcome) -> None:
    projector = SnapshotProjector()
    snapshot = ConversationSnapshot(
        conversation_id="c", capabilities=HarnessCapabilities(user_input_reply=True)
    )
    snapshot = projector.reduce(
        snapshot,
        ProductEvent(
            conversation_id="c",
            seq=1,
            type="input.requested",
            run_id="run-1",
            payload={"request_id": "input-1", "questions": [_question()]},
        ),
    )
    snapshot = projector.reduce(
        snapshot,
        ProductEvent(
            conversation_id="c",
            seq=2,
            type="input.resolved",
            run_id="run-1",
            payload={"request_id": "input-1", "outcome": outcome},
        ),
    )
    assert snapshot.automatic_turns_suspended
    assert not snapshot.pending_inputs


def test_valid_request_with_options_and_free_text() -> None:
    request = _request(
        [_question(), {"id": "output_name", "prompt": "输出文件名？", "options": []}]
    )
    assert request.questions[0].options[0].id == "keep_first"
    assert request.questions[1].options == ()


def test_request_question_count_bounds() -> None:
    with pytest.raises(ValidationError):
        _request([])
    with pytest.raises(ValidationError):
        _request([_question(id=f"q{i}") for i in range(4)])


def test_request_rejects_duplicate_question_ids() -> None:
    with pytest.raises(ValidationError, match="question ids must be unique"):
        _request([_question(), _question()])


def test_question_rejects_duplicate_option_ids() -> None:
    with pytest.raises(ValidationError, match="option ids must be unique"):
        UserInputQuestion.model_validate(
            {
                "id": "q",
                "prompt": "p",
                "options": [
                    {"id": "same", "label": "a"},
                    {"id": "same", "label": "b"},
                ],
            }
        )


def test_request_rejects_oversized_payload() -> None:
    with pytest.raises(ValidationError, match="size limit"):
        _request(
            [
                {
                    "id": "q",
                    "prompt": "x" * (MAX_USER_INPUT_BYTES + 1),
                    "options": [],
                }
            ]
        )


def test_answer_requires_exactly_one_choice() -> None:
    with pytest.raises(ValidationError, match="exactly one"):
        UserInputAnswer.model_validate({"question_id": "q"})
    with pytest.raises(ValidationError, match="exactly one"):
        UserInputAnswer.model_validate(
            {"question_id": "q", "option_id": "a", "text": "b"}
        )
    with pytest.raises(ValidationError, match="non-whitespace"):
        UserInputAnswer.model_validate({"question_id": "q", "text": "   "})


def test_input_response_command_payload_rules() -> None:
    with pytest.raises(ValidationError, match="must not carry answers"):
        InputResponseCommand.model_validate(
            {
                "command_id": "c1",
                "request_id": "r1",
                "status": "cancelled",
                "answers": [{"question_id": "q", "option_id": "a"}],
            }
        )
    with pytest.raises(ValidationError, match="requires answers"):
        InputResponseCommand.model_validate(
            {"command_id": "c1", "request_id": "r1", "status": "answered"}
        )
    with pytest.raises(ValidationError, match="at most once"):
        InputResponseCommand.model_validate(
            {
                "command_id": "c1",
                "request_id": "r1",
                "status": "answered",
                "answers": [
                    {"question_id": "q", "option_id": "a"},
                    {"question_id": "q", "text": "custom"},
                ],
            }
        )
    with pytest.raises(ValidationError):
        InputResponseCommand.model_validate(
            {
                "command_id": "c1",
                "request_id": "r1",
                "status": "interrupted",
            }
        )


def test_tool_json_schema_is_array_with_bounds() -> None:
    schema = json.loads(json.dumps(UserInputRequest.tool_json_schema()))
    questions = schema["properties"]["questions"]
    assert questions["type"] == "array"
    assert questions["minItems"] == 1
    assert questions["maxItems"] == 3
    encoded = json.dumps(schema, ensure_ascii=False)
    assert "duplicate_policy" not in encoded


def test_resolution_omits_empty_optional_fields() -> None:
    payload = json.loads(
        UserInputResolution(request_id="r1", outcome="interrupted").model_dump_json()
    )
    assert payload == {"request_id": "r1", "outcome": "interrupted"}
    payload = json.loads(
        UserInputResolution(
            request_id="r1",
            outcome="answered",
            answers=(UserInputAnswer(question_id="q", option_id="a"),),
            command_id="c1",
        ).model_dump_json()
    )
    assert payload["answers"] == [{"question_id": "q", "option_id": "a"}]
    assert payload["command_id"] == "c1"
    assert "reason" not in payload


def test_capabilities_omit_default_user_input_reply() -> None:
    payload = json.loads(HarnessCapabilities(cancel=True).model_dump_json())
    assert "user_input_reply" not in payload
    payload = json.loads(
        HarnessCapabilities(cancel=True, user_input_reply=True).model_dump_json()
    )
    assert payload["user_input_reply"] is True
    assert HarnessCapabilities.model_validate(payload).user_input_reply is True


def test_old_snapshot_json_still_validates() -> None:
    snapshot = ConversationSnapshot(
        conversation_id="c1", capabilities=HarnessCapabilities()
    )
    legacy = json.loads(snapshot.model_dump_json())
    assert "pending_inputs" not in legacy
    assert "automatic_turns_suspended" not in legacy
    restored = ConversationSnapshot.model_validate(legacy)
    assert restored.pending_inputs == ()
    assert restored.automatic_turns_suspended is False


def test_snapshot_includes_input_state_when_set() -> None:
    snapshot = ConversationSnapshot(
        conversation_id="c1",
        capabilities=HarnessCapabilities(),
        pending_inputs=(
            PendingUserInput(
                request_id="r1",
                run_id="run-1",
                operation_id="call-1",
                questions=(
                    UserInputQuestion(
                        id="q",
                        prompt="p",
                        options=(UserInputOption(id="a", label="A"),),
                    ),
                ),
            ),
        ),
        automatic_turns_suspended=True,
    )
    payload = json.loads(snapshot.model_dump_json())
    assert payload["pending_inputs"][0]["request_id"] == "r1"
    assert payload["automatic_turns_suspended"] is True


def _snapshot() -> ConversationSnapshot:
    return ConversationSnapshot(
        conversation_id="conversation-1",
        capabilities=HarnessCapabilities(cancel=True),
    )


def _event(
    seq: int, event_type: str, payload: dict, **overrides: object
) -> ProductEvent:
    frame: dict = {
        "event_id": f"event-{seq}",
        "conversation_id": "conversation-1",
        "seq": seq,
        "type": event_type,
        "payload": payload,
    }
    frame.update(overrides)
    return ProductEvent.model_validate(frame)


def _requested_event(seq: int, request_id: str = "input-1", run_id: str = "run-1"):
    return _event(
        seq,
        "input.requested",
        {
            "request_id": request_id,
            "operation_id": f"call-{request_id}",
            "questions": [_question()],
        },
        run_id=run_id,
    )


def _resolved_event(
    seq: int,
    request_id: str = "input-1",
    outcome: str = "answered",
    run_id: str = "run-1",
):
    payload: dict = {"request_id": request_id, "outcome": outcome}
    if outcome == "answered":
        payload["answers"] = [
            {"question_id": "duplicate_policy", "option_id": "keep_first"}
        ]
    return _event(seq, "input.resolved", payload, run_id=run_id)


def test_input_requested_sets_waiting_state_and_keeps_run() -> None:
    projector = SnapshotProjector()
    snapshot = projector.reduce(
        _snapshot(), _event(1, "status.changed", {"status": "running"}, run_id="run-1")
    )
    assert snapshot.active_run_id == "run-1"
    snapshot = projector.reduce(snapshot, _requested_event(2))
    assert snapshot.status == "waiting_for_input"
    assert snapshot.active_run_id == "run-1"
    assert snapshot.pending_inputs[0].request_id == "input-1"
    assert snapshot.pending_inputs[0].questions[0].options[0].id == "keep_first"


def test_duplicate_input_requested_is_ignored() -> None:
    projector = SnapshotProjector()
    snapshot = projector.reduce(_snapshot(), _requested_event(1))
    duplicate = projector.reduce(snapshot, _requested_event(2))
    assert len(duplicate.pending_inputs) == 1
    assert duplicate.status == "waiting_for_input"


def test_input_resolved_answered_returns_to_running() -> None:
    projector = SnapshotProjector()
    snapshot = projector.reduce(_snapshot(), _requested_event(1))
    snapshot = projector.reduce(snapshot, _resolved_event(2))
    assert snapshot.pending_inputs == ()
    assert snapshot.status == "running"
    assert snapshot.active_run_id == "run-1"


def test_input_resolved_cancelled_and_interrupted_pause() -> None:
    projector = SnapshotProjector()
    for outcome in ("cancelled", "interrupted"):
        snapshot = projector.reduce(_snapshot(), _requested_event(1))
        snapshot = projector.reduce(snapshot, _resolved_event(2, outcome=outcome))
        assert snapshot.status == "paused"
        assert snapshot.active_run_id is None


def test_input_resolved_keeps_waiting_while_requests_remain() -> None:
    projector = SnapshotProjector()
    snapshot = projector.reduce(_snapshot(), _requested_event(1))
    snapshot = projector.reduce(
        snapshot, _requested_event(2, request_id="input-2", run_id="run-1")
    )
    assert len(snapshot.pending_inputs) == 2
    snapshot = projector.reduce(snapshot, _resolved_event(3))
    assert snapshot.status == "waiting_for_input"
    assert snapshot.pending_inputs[0].request_id == "input-2"


def test_permission_events_defer_to_pending_input() -> None:
    projector = SnapshotProjector()
    snapshot = projector.reduce(_snapshot(), _requested_event(1))
    snapshot = projector.reduce(
        snapshot,
        _event(
            2,
            "permission.requested",
            {"permission_id": "p1", "operation_ids": ["op"]},
        ),
    )
    assert snapshot.status == "waiting_for_input"
    assert snapshot.pending_permissions[0].permission_id == "p1"
    snapshot = projector.reduce(
        snapshot,
        _event(3, "permission.resolved", {"permission_id": "p1", "decision": "deny"}),
    )
    assert snapshot.pending_permissions == ()
    assert snapshot.status == "waiting_for_input"
    # Return to running after the answer when no permission request remains.
    snapshot = projector.reduce(snapshot, _resolved_event(4))
    assert snapshot.status == "running"


def test_same_run_status_change_preserves_waiting_for_input() -> None:
    projector = SnapshotProjector()
    snapshot = projector.reduce(
        _snapshot(), _event(1, "status.changed", {"status": "running"}, run_id="run-1")
    )
    snapshot = projector.reduce(snapshot, _requested_event(2))
    snapshot = projector.reduce(
        snapshot,
        _event(3, "status.changed", {"status": "idle"}, run_id="run-1"),
    )
    assert snapshot.status == "waiting_for_input"


def test_background_task_events_do_not_mask_settled_status() -> None:
    projector = SnapshotProjector()
    snapshot = projector.reduce(_snapshot(), _requested_event(1))
    snapshot = projector.reduce(snapshot, _resolved_event(2, outcome="cancelled"))
    assert snapshot.status == "paused"
    task_payload = {
        "task_id": "task-1",
        "kind": "workflow_run",
        "status": "succeeded",
        "submitted_at": "2026-01-01T00:00:00Z",
        "updated_at": "2026-01-01T00:00:00Z",
    }
    snapshot = projector.reduce(
        snapshot, _event(3, "external_task.submitted", task_payload)
    )
    assert snapshot.status == "paused"
    assert snapshot.external_tasks[0].task_id == "task-1"
    snapshot = projector.reduce(
        snapshot, _event(4, "external_task.completed", task_payload)
    )
    assert snapshot.status == "paused"
    assert snapshot.external_tasks[0].status == "succeeded"


def test_execution_policy_updates_suspension_flag() -> None:
    projector = SnapshotProjector()
    snapshot = projector.reduce(
        _snapshot(),
        _event(1, "execution.policy.updated", {"automatic_turns_suspended": True}),
    )
    assert snapshot.automatic_turns_suspended is True
    snapshot = projector.reduce(
        snapshot,
        _event(
            2,
            "execution.policy.updated",
            {"automatic_turns_suspended": "yes"},
        ),
    )
    assert snapshot.automatic_turns_suspended is True
    notice = snapshot.timeline[-1]
    assert notice.kind == "notice"
