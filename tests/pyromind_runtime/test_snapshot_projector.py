from pyromind_runtime.application import SnapshotProjector
from pyromind_runtime.domain.capabilities import HarnessCapabilities
from pyromind_runtime.domain.events import ProductEvent
from pyromind_runtime.domain.snapshot import ConversationSnapshot, TimelineOperation


def _snapshot() -> ConversationSnapshot:
    return ConversationSnapshot(
        conversation_id="conversation-1",
        capabilities=HarnessCapabilities(cancel=True),
    )


def _event(seq: int, event_type: str, payload: dict) -> ProductEvent:
    return ProductEvent.model_validate(
        {
            "event_id": f"event-{seq}",
            "conversation_id": "conversation-1",
            "seq": seq,
            "type": event_type,
            "payload": payload,
        }
    )


def test_projector_accepts_non_object_operation_details() -> None:
    projector = SnapshotProjector()
    snapshot = projector.reduce(
        _snapshot(),
        _event(
            1,
            "operation.started",
            {"operation_id": "op-1", "name": "validate_workflow_dsl"},
        ),
    )
    snapshot = projector.reduce(
        snapshot,
        _event(
            2,
            "operation.completed",
            {
                "operation_id": "op-1",
                "details": ["warning", {"valid": True}],
            },
        ),
    )

    operation = snapshot.timeline[0]
    assert isinstance(operation, TimelineOperation)
    assert operation.details == ["warning", {"valid": True}]
    assert operation.status == "completed"


def test_projector_preserves_timeline_and_current_workflow() -> None:
    projector = SnapshotProjector()
    snapshot = projector.reduce(
        _snapshot(),
        _event(
            1,
            "workflow.updated",
            {
                "resource_id": "workflow",
                "version": "version-1",
                "dsl": "node = InputNode()",
                "canvas": None,
            },
        ),
    )

    assert snapshot.current_workflow is not None
    assert snapshot.current_workflow.version == "version-1"
    assert snapshot.timeline[0].kind == "workflow"


def test_projector_tolerates_completion_without_start() -> None:
    snapshot = SnapshotProjector().reduce(
        _snapshot(),
        _event(
            1,
            "operation.completed",
            {
                "operation_id": "late-operation",
                "name": "preview_dataset",
                "content": [{"type": "text", "text": "done"}],
            },
        ),
    )

    operation = snapshot.timeline[0]
    assert isinstance(operation, TimelineOperation)
    assert operation.category == "observation"
    assert operation.status == "completed"


def test_projector_reads_legacy_stringified_plan_steps() -> None:
    snapshot = SnapshotProjector().reduce(
        _snapshot(),
        _event(
            1,
            "plan.updated",
            {
                "steps": [
                    "step='Inspect input' status='completed'",
                    "step='Build output' status='in_progress'",
                ],
                "explanation": "Continue",
            },
        ),
    )

    assert snapshot.plan is not None
    assert [step.model_dump() for step in snapshot.plan.steps] == [
        {"step": "Inspect input", "status": "completed"},
        {"step": "Build output", "status": "in_progress"},
    ]


def test_projector_tracks_external_task_lifecycle() -> None:
    projector = SnapshotProjector()
    submitted = {
        "task_id": "task-1",
        "kind": "data_cleaning",
        "run_id": "run-1",
        "status": "running",
        "output_dir": "/agent/conv/data_cleaning/run-1",
        "submitted_at": "2026-08-24T00:00:00+00:00",
        "updated_at": "2026-08-24T00:00:00+00:00",
        "resume_pending": False,
    }
    snapshot = projector.reduce(
        _snapshot(), _event(1, "external_task.submitted", submitted)
    )
    assert snapshot.status == "waiting_for_external_task"
    snapshot = projector.reduce(
        snapshot,
        _event(
            2,
            "external_task.completed",
            {**submitted, "status": "succeeded", "resume_pending": True},
        ),
    )
    assert snapshot.status == "idle"
    assert snapshot.external_tasks[0].status == "succeeded"
    assert snapshot.external_tasks[0].resume_pending is True

    snapshot = projector.reduce(
        snapshot,
        _event(3, "external_task.submitted", submitted),
    )
    snapshot = projector.reduce(
        snapshot,
        _event(4, "external_task.updated", {**submitted, "status": "running"}),
    )

    assert snapshot.status == "idle"
    assert snapshot.external_tasks[0].status == "succeeded"
    assert snapshot.external_tasks[0].resume_pending is True
