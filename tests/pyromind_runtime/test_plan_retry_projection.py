from pyromind_runtime.application.conversation_runtime import ConversationRuntime
from pyromind_runtime.application.plan_execution import (
    plan_executions,
    with_platform_states,
)
from pyromind_runtime.domain.capabilities import HarnessCapabilities
from pyromind_runtime.domain.events import ProductEvent
from pyromind_runtime.domain.preparation import ProjectView
from pyromind_runtime.domain.snapshot import ConversationSnapshot, ExternalTaskState

from .fake_adapter import FakeAdapter


def test_two_failed_attempts_then_accepted_training_result(tmp_path):
    task = {"id": "sft_train", "stage_id": "train", "name": "SFT"}
    project = ProjectView.model_validate(
        {
            "project_id": "p",
            "conversation_id": "c",
            "plan_sequence": 1,
            "creation_command_id": "create",
            "plan": {"stages": [{"id": "train", "name": "Train"}], "tasks": [task]},
        }
    )
    runtime = ConversationRuntime(tmp_path, FakeAdapter())
    (tmp_path / "c").mkdir()
    store = runtime._store("c")
    store.create(
        ConversationSnapshot(conversation_id="c", capabilities=HarnessCapabilities()),
        user_id="owner",
    )

    for attempt, (task_id, status) in enumerate(
        [("1269", "failed"), ("1270", "failed"), ("1271", "succeeded")], start=1
    ):
        (run,) = plan_executions(
            project,
            [{**task, "plan_attempt": attempt, "business_status": "in_progress"}],
        )
        runtime._record_execution(
            store, run.model_copy(update={"platform_task_id": task_id})
        )
        external = ExternalTaskState(
            task_id=task_id,
            kind="workflow_run",
            status=status,
            submitted_at="2026-11-23T04:00:00Z",
            updated_at="2026-11-23T04:00:01Z",
        )
        store.append(
            ProductEvent(
                conversation_id="c",
                type="external_task.submitted",
                payload=external.model_copy(update={"status": "pending"}).model_dump(
                    mode="json"
                ),
            )
        )
        store.append(
            ProductEvent(
                conversation_id="c",
                type="external_task.completed",
                payload=external.model_dump(mode="json"),
            )
        )
    snapshot = store.load_snapshot()
    runs = snapshot.experiment_executions
    assert (
        with_platform_states(snapshot).experiment_executions[-1].state
        == "waiting_analysis"
    )
    completed = {**task, "business_status": "completed"}
    for identity in [
        {"plan_attempt": 3, "actual_platform_task_id": "1271"},
        {"attempt": 3},
        {"platform_task_id": "1271"},
    ]:
        (update,) = plan_executions(
            project, [{**completed, **identity}], registered=runs
        )
        assert update.run_id == runs[-1].run_id
        assert update.platform_task_id is None
        runtime._record_execution(store, update)
        after = with_platform_states(store.load_snapshot())
        assert [(r.platform_task_id, r.state) for r in after.experiment_executions] == [
            ("1269", "failed"),
            ("1270", "failed"),
            ("1271", "succeeded"),
        ]
    for ambiguous in [
        {},
        {"attempt": 1, "plan_attempt": 3},
        {"attempt": 1, "platform_task_id": "1271"},
        {"actual_platform_task_id": "unregistered"},
    ]:
        assert not list(
            plan_executions(project, [{**completed, **ambiguous}], registered=runs)
        )
