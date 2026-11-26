import json
from typing import Literal

from pyromind_runtime.application.conversation_runtime import ConversationRuntime
from pyromind_runtime.application.plan_execution import (
    PLAN_PATH,
    plan_executions,
    read_plan_tasks,
    restore_saved_data_status,
    with_platform_states,
)
from pyromind_runtime.domain.capabilities import HarnessCapabilities
from pyromind_runtime.domain.events import ProductEvent
from pyromind_runtime.domain.preparation import ProjectView
from pyromind_runtime.domain.snapshot import (
    ConversationSnapshot,
    ExternalTaskState,
    TimelineOperation,
)

from .fake_adapter import FakeAdapter


def test_wrapped_plan_joins_status_by_task_id_and_named_outputs_by_filename():
    tasks = [{"id": "data", "outputs": [{"name": "val"}, {"name": "train"}]}]
    states = {
        "data": {
            "business_status": "completed",
            "verified_outputs": [
                "/workspace/a/train.jsonl",
                "/workspace/a/val.jsonl",
                "/workspace/b/val.jsonl",
                "/workspace/unrelated.txt",
            ],
        }
    }
    (row,) = read_plan_tasks(
        json.dumps({"plan": {"tasks": tasks}, "task_states": states})
    )
    assert row["business_status"] == "completed"
    assert row["verified_outputs"] == {"train": "/workspace/a/train.jsonl"}


def test_verified_baseline_report_uses_the_single_canonical_platform_report():
    task = {"id": "baseline", "outputs": [{"name": "baseline_report"}]}
    state = {
        "verified_outputs": [
            "/workspace/metrics.json",
            "/workspace/report.json",
            "/workspace/evaluation_report.html",
        ]
    }
    rows = read_plan_tasks(
        json.dumps({"plan": {"tasks": [task]}, "task_states": {"baseline": state}})
    )
    assert rows[0]["verified_outputs"] == {"baseline_report": "/workspace/report.json"}


def test_nested_plan_restores_saved_local_data_completion():
    task = {"id": "data_task", "stage_id": "data", "name": "Prepare data"}
    project = ProjectView.model_validate(
        {
            "project_id": "p",
            "conversation_id": "c",
            "plan_sequence": 1,
            "creation_command_id": "create",
            "plan": {
                "stages": [{"id": "data", "name": "Data", "kind": "dataset"}],
                "tasks": [task],
            },
        }
    )
    record = {
        "plan_task_id": "data_task",
        "plan_attempt": 1,
        "business_status": "completed",
        "verified_outputs": {"dataset": "/workspace/results.jsonl"},
    }
    stored = json.dumps({"tasks": [task], "runtime": {"tasks": {"data_task": record}}})
    (queued,) = plan_executions(project, [{**task, "business_status": "pending"}])
    saved = TimelineOperation(
        item_id="write",
        started_seq=1,
        completed_seq=2,
        status="completed",
        name="write",
        arguments={"path": PLAN_PATH, "content": stored},
    )
    snapshot = ConversationSnapshot(
        conversation_id="c",
        capabilities=HarnessCapabilities(),
        project=project,
        experiment_executions=(queued,),
        timeline=(saved,),
    )
    assert read_plan_tasks(stored)[0]["verified_outputs"] == record["verified_outputs"]
    completed = restore_saved_data_status(snapshot).experiment_executions
    assert completed and completed[0].state == "succeeded"
    assert completed[0].platform_task_id is None
    later_edit = saved.model_copy(
        update={"name": "edit", "item_id": "editing", "completed_seq": 4}
    )
    edited = snapshot.model_copy(update={"timeline": (saved, later_edit)})
    assert restore_saved_data_status(edited) == edited


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

    attempts: list[tuple[str, Literal["failed", "succeeded"]]] = [
        ("1269", "failed"),
        ("1270", "failed"),
        ("1271", "succeeded"),
    ]
    for attempt, (task_id, status) in enumerate(attempts, start=1):
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
    projected = with_platform_states(snapshot).experiment_executions
    assert runs and projected
    assert projected[-1].state == "waiting_analysis"
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
        after = with_platform_states(store.load_snapshot()).experiment_executions
        assert after
        assert [(r.platform_task_id, r.state) for r in after] == [
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


def test_sdk_plan_completion_is_projected_against_the_registered_attempt():
    task = {"id": "ds_prelabel", "stage_id": "ds", "name": "Prelabel"}
    project = ProjectView.model_validate(
        {
            "project_id": "p",
            "conversation_id": "c",
            "plan_sequence": 3,
            "creation_command_id": "create",
            "plan": {
                "stages": [{"id": "ds", "name": "Dataset", "kind": "dataset"}],
                "tasks": [task],
            },
        }
    )
    (queued,) = plan_executions(project, [{**task, "business_status": "pending"}])
    receipt = queued.model_copy(update={"platform_task_id": "1366"})
    saved = {
        "plan_task_id": "ds_prelabel",
        "stage_id": "ds",
        "attempt": 1,
        "platform_task_id": "1366",
        "business_status": "completed",
        "verified_outputs": {"sft_train_messages": "/workspace/dataset/train.jsonl"},
    }

    def project_rows(row):
        rows = read_plan_tasks(json.dumps({"tasks": [row]}), project)
        return list(plan_executions(project, rows, registered=[receipt]))

    (completed,) = project_rows(saved)
    assert completed.run_id == receipt.run_id and completed.state == "succeeded"
    assert not project_rows({**saved, "id": "ds_prelabel", "plan_task_id": "other"})
    assert not project_rows({**saved, "platform_task_id": "unregistered"})


def test_runtime_task_map_uses_approved_tasks_and_bound_receipts():
    task = {"id": "prelabel", "stage_id": "data", "name": "Prepare data"}
    project = ProjectView.model_validate(
        {
            "project_id": "p",
            "conversation_id": "c",
            "plan_sequence": 3,
            "creation_command_id": "create",
            "plan": {
                "goal": "Train on labeled samples",
                "stages": [{"id": "data", "name": "Data", "kind": "dataset"}],
                "tasks": [{**task, "outputs": [{"name": "messages"}]}],
            },
        }
    )
    (queued,) = plan_executions(project, [{**task, "business_status": "pending"}])
    bound = queued.model_copy(update={"platform_task_id": "1379"})
    record = {
        "stage_id": "data",
        "attempt": 1,
        "platform_task_id": "1379",
        "business_status": "completed",
        "verified_outputs": {"messages": "/workspace/train.jsonl"},
    }

    def project_record(**updates):
        records = {"unplanned": record, "prelabel": {**record, **updates}}
        rows = read_plan_tasks(
            json.dumps(
                {
                    "project_plan": {"goal": project.plan.goal},
                    "runtime": {"tasks": records},
                }
            ),
            project,
        )
        return rows, list(plan_executions(project, rows, registered=[bound]))

    (row,), (completed,) = project_record()
    assert row["verified_outputs"] == record["verified_outputs"]
    assert completed.run_id == bound.run_id and completed.state == "succeeded"
    for conflict in [{"id": "other"}, {"stage_id": "train"}, {"plan_task_id": "other"}]:
        assert project_record(**conflict) == ([], [])
    for mismatch in [{"attempt": 2}, {"platform_task_id": "different"}]:
        assert not project_record(**mismatch)[1]
