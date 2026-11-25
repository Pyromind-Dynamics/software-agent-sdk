import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, Mock

import pytest

from pyromind_runtime.application import conversation_runtime
from pyromind_runtime.application.conversation_runtime import ConversationRuntime
from pyromind_runtime.application.plan_artifact_delivery import (
    InvalidPlanDelivery,
    _read_report,
    delivery_report,
    register_plan_delivery,
)
from pyromind_runtime.application.plan_execution import (
    plan_executions,
    recover_plan_delivery_rows,
    with_plan_report,
)
from pyromind_runtime.application.project_observations import (
    observation_report_details,
    project_observations,
)
from pyromind_runtime.application.stage_analysis import ready_stage_analyses
from pyromind_runtime.application.versions import Versions
from pyromind_runtime.domain.context import RequestContext
from pyromind_runtime.domain.experiment_execution import ExperimentExecutionView
from pyromind_runtime.domain.experiment_plans import ExperimentTask
from pyromind_runtime.domain.preparation import ProjectView
from pyromind_runtime.domain.snapshot import ConversationSnapshot, TimelineOperation
from pyromind_runtime.infrastructure.file_artifact_snapshots import (
    FileArtifactSnapshots,
)
from pyromind_runtime.infrastructure.sqlite_version_store import SQLiteVersionStore
from pyromind_runtime.ports.harness import SessionSpec
from pyromind_runtime.ports.storage import (
    EvaluationStorage,
    EvaluationStorageUnavailableError,
)

from .fake_adapter import FakeAdapter


async def test_delivery_recovers_after_sign_in_and_a_storage_gateway_error(
    tmp_path, monkeypatch
):
    context = RequestContext(user_id="42")
    versions = Versions(
        SQLiteVersionStore(tmp_path / "versions.db"),
        FileArtifactSnapshots(tmp_path / "storage"),
    )
    storage = cast(EvaluationStorage, Mock())
    runtime = ConversationRuntime(
        tmp_path / "sessions",
        FakeAdapter(),
        versions=versions,
        observation_storage_factory=lambda _: storage,
    )
    run = ExperimentExecutionView(
        project_id="project",
        plan_sequence=1,
        run_id="project:task:1",
        task_id="task",
        task_name="Prepare",
        role_id="main",
        state="succeeded",
        report_state="not_started",
    )
    task = ExperimentTask.model_validate(
        {
            "id": "task",
            "stage_id": "dataset",
            "name": "Prepare",
            "outputs": [{"name": "dataset_report"}],
        }
    )
    register = Mock(
        side_effect=[
            EvaluationStorageUnavailableError("temporary gateway", retryable=True),
            None,
        ]
    )
    publish = AsyncMock(side_effect=OSError("temporary notification failure"))
    recheck = Mock()
    monkeypatch.setattr(conversation_runtime, "register_plan_delivery", register)
    monkeypatch.setattr(runtime, "_publish_artifacts", publish)
    monkeypatch.setattr(runtime, "_schedule_stage_analysis_recheck", recheck)
    try:
        await runtime.create_conversation(
            SessionSpec(
                conversation_id="session",
                user_id="42",
                workspace_root=str(tmp_path / "sessions/session"),
            ),
            context,
        )
        store = runtime._store("session")
        active = runtime._active["session"]
        failed = runtime._schedule_plan_delivery(
            active, store, task, run, {}, None, retry=False
        )
        assert failed.delivery_error == "credentials_required"

        active.context = replace(context, authorization="Bearer test-only")
        pending = runtime._schedule_plan_delivery(
            active, store, task, run, {}, failed, retry=False
        )
        assert pending.report_state == "registering"
        (worker,) = runtime._stage_analysis_workers.values()

        # Another unauthenticated requester cannot replace the in-flight state.
        active.context = context
        assert (
            runtime._schedule_plan_delivery(
                active, store, task, run, {}, pending, retry=False
            )
            == pending
        )
        with pytest.raises(OSError, match="temporary notification failure"):
            await worker
        assert register.call_count == 2
        publish.assert_awaited_once()
        recheck.assert_called_once_with("session", active, store)
    finally:
        await runtime.close()


def test_dataset_summary_registers_against_the_single_declared_training_file():
    summary = {
        "report_type": "dataset_preparation",
        "n_train_records": 390,
        "label_distribution": {"pass": 130, "defect": 130, "fail": 130},
        "category_distribution": {"no_defect": 130},
    }
    storage = cast(
        EvaluationStorage,
        SimpleNamespace(read=lambda _: json.dumps(summary).encode()),
    )
    outputs = {
        "train_messages": "/run/train_messages.jsonl",
        "dataset_report": "/run/dataset_report.json",
    }
    assert _read_report(storage, "dataset", outputs)["files"] == [
        {
            "name": "train_messages",
            "path": "/workspace/run/train_messages.jsonl",
            "size_bytes": None,
            "sample_count": 390,
            "distributions": {
                "label": summary["label_distribution"],
                "category": summary["category_distribution"],
            },
        }
    ]
    with pytest.raises(InvalidPlanDelivery, match="invalid_report"):
        _read_report(storage, "dataset", {**outputs, "val": "/run/val.jsonl"})


def test_merged_dataset_summary_uses_only_the_declared_train_and_validation_files():
    summary = {
        "output_dir": "/workspace/run/merged",
        "n_train": 390,
        "n_val": 66,
        "train_labels": {"defect": 130, "fail": 130, "pass": 130},
        "val_labels": {"defect": 22, "fail": 22, "pass": 22},
    }
    storage = cast(
        EvaluationStorage,
        SimpleNamespace(read=lambda _: json.dumps(summary).encode()),
    )
    outputs = {
        "train_messages": "/run/merged/train.jsonl",
        "val_messages": "/run/merged/val.jsonl",
        "data_report": "/run/merged/data_report.json",
    }
    assert _read_report(storage, "dataset", outputs)["files"] == [
        {
            "name": f"{split}_messages",
            "path": f"/workspace/run/merged/{split}.jsonl",
            "size_bytes": None,
            "sample_count": summary[f"n_{split}"],
            "distributions": {"label": summary[f"{split}_labels"]},
        }
        for split in ("train", "val")
    ]
    with pytest.raises(InvalidPlanDelivery, match="invalid_report"):
        _read_report(
            storage, "dataset", {**outputs, "val_messages": "/other/val.jsonl"}
        )


def test_evaluation_comparison_preserves_measured_metric_deltas():
    body = {
        "metrics": [
            {"label": "result_accuracy", "value": 0.47, "unit": "ratio"},
            {
                "label": "result_accuracy_delta",
                "baseline": 0.55,
                "candidate": 0.47,
                "delta": -0.08,
                "unit": "ratio",
            },
            {
                "label": "invalid_delta",
                "baseline": 0.5,
                "candidate": 0.4,
                "delta": "unknown",
            },
        ]
    }
    storage = cast(
        EvaluationStorage, SimpleNamespace(read=lambda _: json.dumps(body).encode())
    )
    document = _read_report(storage, "eval", {"final_report": "/run/final_report.json"})
    assert document["metrics"] == [
        {"label": "result_accuracy", "value": 0.47, "unit": "ratio"},
        {"label": "result_accuracy_delta", "value": -0.08, "unit": "ratio"},
    ]


@pytest.mark.parametrize(
    ("stage", "outputs"),
    [
        ("init", {"baseline_metrics": "/workspace/run/baseline.json"}),
        (
            "dataset",
            {
                "studio_dataset": "/workspace/run/dataset",
                "dataset_report": "/workspace/run/dataset_report.json",
            },
        ),
        ("train", {"lora_checkpoint": "/workspace/run/checkpoint"}),
        ("eval", {"eval_report": "/workspace/run/eval.json"}),
    ],
)
def test_completed_plan_outputs_register_once_and_recover(
    stage, outputs, tmp_path, monkeypatch
):
    context = RequestContext(user_id="42")
    snapshots = FileArtifactSnapshots(tmp_path / "storage")
    versions = Versions(SQLiteVersionStore(tmp_path / "versions.db"), snapshots)
    root = snapshots.user_root(context)
    (root / "run/dataset").mkdir(parents=True)
    (root / "run/checkpoint").mkdir()
    (root / "run/dataset/train.jsonl").write_text("{}\n")
    dataset_report = {
        "splits": {
            "train": {
                "file": "/workspace/run/dataset/train.jsonl",
                "n_records": 1,
                "labels": {"pass": 1},
            }
        }
    }
    for name, body in [
        ("dataset_report", dataset_report),
        (
            "baseline",
            {
                "summary": "Baseline accuracy was 0.5.",
                "metrics": [{"label": "accuracy", "value": 0.5, "unit": "ratio"}],
            },
        ),
        ("eval", {"metrics": [{"label": "accuracy", "value": 0.75, "unit": "ratio"}]}),
    ]:
        (root / "run" / f"{name}.json").write_text(json.dumps(body))

    def local_path(reference: str) -> Path:
        return snapshots.path(reference, context)

    storage = cast(
        EvaluationStorage,
        SimpleNamespace(
            read=lambda ref: local_path(ref).read_bytes(),
            list=lambda ref: {file.name for file in local_path(ref).iterdir()},
        ),
    )
    project = ProjectView.model_validate(
        {
            "project_id": "project",
            "conversation_id": "session",
            "plan_sequence": 3,
            "creation_command_id": "create",
            "plan": {
                "stages": [{"id": "stage", "kind": stage, "name": stage}],
                "tasks": [
                    {
                        "id": "task",
                        "stage_id": "stage",
                        "name": "Run stage",
                        "outputs": [{"name": name} for name in outputs],
                    }
                ],
            },
        }
    )
    row = {
        "id": "task",
        "stage_id": "stage",
        "attempt": 1,
        "business_status": "completed",
        "verified_outputs": outputs,
    }
    (run,) = plan_executions(project, [row])
    task = project.plan.tasks[0]
    in_progress = {**row, "business_status": "in_progress"}
    history = (
        TimelineOperation(
            item_id="write",
            started_seq=1,
            completed_seq=2,
            name="write",
            status="completed",
            arguments={
                "path": "public_data/workflowPLAN.json",
                "content": json.dumps(
                    {
                        "plan": project.plan.model_dump(mode="json"),
                        "task_records": {"task": in_progress},
                    }
                ),
            },
        ),
        TimelineOperation(
            item_id="edit",
            started_seq=3,
            completed_seq=4,
            name="edit",
            status="completed",
            arguments={
                "path": "public_data/workflowPLAN.json",
                "edits": [
                    {"oldText": json.dumps(in_progress), "newText": json.dumps(row)}
                ],
            },
        ),
    )
    recovery = ConversationSnapshot.model_validate(
        {
            "conversation_id": "session",
            "capabilities": {},
            "timeline": history,
            "project": project,
            "experiment_executions": (run,),
        }
    )
    (restored_row,) = recover_plan_delivery_rows(recovery, project)
    assert restored_row["verified_outputs"] == outputs
    assert not recover_plan_delivery_rows(
        recovery.model_copy(update={"experiment_executions": ()}), project
    )
    with pytest.raises(InvalidPlanDelivery, match="missing_output"):
        register_plan_delivery(
            versions,
            storage,
            task,
            run,
            {**row, "verified_outputs": {}},
            "session",
            context,
        )
    save = snapshots.save

    def fail_report_once(source, target, request_context):
        if source.text is not None:
            raise OSError("simulated snapshot interruption")
        return save(source, target, request_context)

    with monkeypatch.context() as patch:
        patch.setattr(snapshots, "save", fail_report_once)
        with pytest.raises(ValueError, match="snapshot not completed"):
            register_plan_delivery(
                versions, storage, task, run, restored_row, "session", context
            )
    version_project = versions.project("session", context, create=False)
    assert version_project is not None
    partial = tuple(versions.store.list_versions(version_project.id))
    assert sum(item.state == "reference_only" for item in partial) == len(outputs)
    assert sum(item.state == "failed" for item in partial) == 1

    # Re-open the version store and reconcile before publishing a run update.
    restarted = Versions(SQLiteVersionStore(tmp_path / "versions.db"), snapshots)
    register_plan_delivery(restarted, storage, task, run, row, "session", context)
    saved = tuple(restarted.store.list_versions(version_project.id))
    assert {item.id for item in saved} == {item.id for item in partial}
    register_plan_delivery(restarted, storage, task, run, row, "session", context)
    assert tuple(restarted.store.list_versions(version_project.id)) == saved
    assert len(saved) == len(outputs) + 1
    report = delivery_report(run, row, saved, {})
    assert report is not None
    (new_attempt,) = plan_executions(project, [{**row, "attempt": 2}])
    assert (
        delivery_report(new_attempt, {"report_version_id": report.id}, saved, {})
        is None
    )
    bound_run = with_plan_report(run, report)
    bound_row = {**row, "report_version_id": report.id}
    assert (
        len(ready_stage_analyses(project, [bound_row], (bound_run,), (), saved, []))
        == 1
    )
    snapshot = ConversationSnapshot.model_validate(
        {
            "conversation_id": "session",
            "capabilities": {},
            "project": project,
            "experiment_executions": (bound_run,),
        }
    )
    node = project_observations(snapshot, project.project_id).nodes[0]
    assert node.delivery_status is None
    assert node.report_version_ids == (report.id,)
    details = observation_report_details(
        snapshot, project.project_id, "stage", None, versions, context
    )
    assert details.report_version_ids == (report.id,)
    if stage == "dataset":
        assert details.files[0].sample_count == 1
        assert details.files[0].distributions == {"label": {"pass": 1}}
    if stage in {"init", "eval"}:
        assert details.metrics[0].label == "accuracy"
    if stage == "init":
        assert details.summaries == ("Baseline accuracy was 0.5.",)
