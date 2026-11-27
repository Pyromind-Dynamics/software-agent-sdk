import asyncio
import json
import subprocess
import sys
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
    delivery_key,
    delivery_report,
    delivery_version_key,
    rebuild_legacy_report,
    register_plan_delivery,
)
from pyromind_runtime.application.plan_execution import (
    PLAN_PATH,
    plan_delivery_source,
    plan_executions,
    platform_results_ready,
    recover_plan_delivery_rows,
    with_plan_report,
    with_platform_states,
)
from pyromind_runtime.application.project_observations import (
    observation_analysis_report,
    observation_report_details,
    project_observations,
)
from pyromind_runtime.application.stage_analysis import ready_stage_analyses
from pyromind_runtime.application.versions import Versions, snapshot_result
from pyromind_runtime.domain.content import JsonObject
from pyromind_runtime.domain.context import RequestContext
from pyromind_runtime.domain.experiment_execution import ExperimentExecutionView
from pyromind_runtime.domain.experiment_plans import ExperimentTask, PlanOutput
from pyromind_runtime.domain.preparation import PreparationPolicy, ProjectView
from pyromind_runtime.domain.snapshot import ConversationSnapshot, TimelineOperation
from pyromind_runtime.domain.versions import (
    SaveVersion,
    SnapshotSource,
    VersionParent,
    VersionToolRequest,
)
from pyromind_runtime.infrastructure.file_artifact_snapshots import (
    FileArtifactSnapshots,
)
from pyromind_runtime.infrastructure.sqlite_version_store import SQLiteVersionStore
from pyromind_runtime.ports.harness import SessionSpec
from pyromind_runtime.ports.storage import (
    EvaluationStorage,
    EvaluationStorageTooLargeError,
    EvaluationStorageUnavailableError,
)

from .fake_adapter import FakeAdapter
from .test_preparation import preparation_setup as preparation_setup


def test_primary_report_is_selected_by_type_and_validates_metrics():
    outputs = {"baseline_results": "/run/results.data", "notes": "/run/notes.json"}
    read = Mock(return_value=b'{"metrics":[{"label":"accuracy","value":0.55}]}')
    storage = cast(EvaluationStorage, SimpleNamespace(read=read))
    document = _read_report(storage, "init", outputs, report_name="baseline_results")
    read.assert_called_once_with("/run/results.data")
    assert document["metrics"] == [
        {"label": "accuracy", "value": 0.55, "unit": "number"}
    ]
    read.return_value = b'{"metrics":[{"label":"accuracy","value":"unknown"}]}'
    with pytest.raises(InvalidPlanDelivery, match="invalid_report_metrics"):
        _read_report(storage, "init", outputs, report_name="baseline_results")
    read.return_value = b'{"metrics":[]}'
    with pytest.raises(InvalidPlanDelivery, match="invalid_report_metrics"):
        _read_report(storage, "init", outputs, report_name="baseline_results")


def test_legacy_init_is_converted_only_with_its_verified_platform_receipt():
    body = {
        "report_kind": "init_baseline",
        "source": {"platform_task_id": 1372},
        "metrics": {
            "result_accuracy": 0.55830258,
            "result_total": 542,
            "per_class": {"pass": {"f1": 0.57}},
        },
    }
    storage = cast(
        EvaluationStorage, SimpleNamespace(read=lambda _: json.dumps(body).encode())
    )
    outputs = {"report": "/run/baseline_report.json"}
    report = _read_report(
        storage, "init", outputs, report_name="report", platform_task_id="1372"
    )
    assert report["metrics"] == [
        {"label": "result_accuracy", "value": 0.55830258, "unit": "ratio"},
        {"label": "evaluated_samples", "value": 542.0, "unit": "count"},
        {"label": "pass_f1", "value": 0.57, "unit": "ratio"},
    ]
    with pytest.raises(InvalidPlanDelivery, match="invalid_report_metrics"):
        _read_report(
            storage, "init", outputs, report_name="report", platform_task_id="1373"
        )
    with pytest.raises(InvalidPlanDelivery) as failure:
        _read_report(
            storage,
            "init",
            outputs,
            report_name="report",
            platform_task_id="1372",
            require_schema_version=True,
        )
    assert failure.value.field == "schema_version"


async def test_delivery_receipt_survives_replay_and_stale_workspace_status(
    preparation_setup, tmp_path, monkeypatch
):
    preparation, owner, _ = preparation_setup
    scope = owner.conversation_id
    context = RequestContext(
        user_id=owner.user_id,
        profile_id=owner.profile_id,
        authorization="Bearer test-only",
    )
    snapshots = FileArtifactSnapshots(tmp_path / "storage")
    versions = Versions(SQLiteVersionStore(tmp_path / "versions.db"), snapshots)
    data_dir = snapshots.path("/run", context)
    data_dir.mkdir(parents=True)
    (data_dir / "train.jsonl").write_text('{"label":"pass"}\n')
    policy = PreparationPolicy.model_validate(
        {"workflow_nodes": [{"id": "dataset", "name": "Dataset"}]}
    )
    source = {
        "schema_version": 1,
        "files": [{"path": "/workspace/run/train.jsonl", "sample_count": 42}],
    }
    storage = cast(
        EvaluationStorage,
        SimpleNamespace(
            read=lambda _: json.dumps(source).encode(),
            list=lambda _: {"report.json", "train.jsonl"},
        ),
    )
    project = ProjectView.model_validate(
        {
            "project_id": "project",
            "conversation_id": scope,
            "plan_sequence": 1,
            "creation_command_id": "accept",
            "plan": {
                "stages": [{"id": "dataset", "kind": "dataset", "name": "Dataset"}],
                "tasks": [
                    {
                        "id": "prepare",
                        "stage_id": "dataset",
                        "name": "Prepare",
                        "outputs": [
                            {"name": "report", "kind": "report"},
                            {"name": "train", "kind": "dataset"},
                        ],
                    }
                ],
            },
        }
    )
    row = {
        "id": "prepare",
        "stage_id": "dataset",
        "business_status": "in_progress",
        "attempt": 1,
    }
    (data_dir / "report.json").write_text(json.dumps(source))
    root = tmp_path / "sessions"
    adapter = FakeAdapter()
    runtime = ConversationRuntime(
        root,
        adapter,
        preparation_store=preparation,
        versions=versions,
        observation_storage_factory=lambda _: storage,
    )
    monkeypatch.setattr(runtime, "get_project", AsyncMock(return_value=project))
    workspace_plan = root / scope / PLAN_PATH

    def request(attempt=1, report_path="/workspace/run/report.json"):
        return VersionToolRequest(
            action="deliver_plan_outputs",
            arguments={
                "plan_task_id": "prepare",
                "plan_attempt": attempt,
                "verified_outputs": {
                    "report": report_path,
                    "train": "/workspace/run/train.jsonl",
                },
            },
        )

    try:
        await runtime.create_conversation(
            SessionSpec(
                conversation_id=scope,
                user_id=context.user_id,
                profile_id=context.profile_id,
                workspace_root=str(root / scope),
                preparation_policy=policy,
                tool_allowlist=("product_versions",),
            ),
            context,
        )
        workspace_plan.parent.mkdir(parents=True, exist_ok=True)
        workspace_plan.write_text(json.dumps({"tasks": [row]}))
        store = runtime._store(scope)
        active = runtime._active[scope]
        runtime._queue_task(
            store,
            scope,
            "project-continue:project:prepare",
            "project_continue",
            "prepare",
        )
        await runtime._dispatch_queued_tasks(active, store)
        execution = runtime._plan_attempt(store.load_snapshot(), project, "prepare")
        assert execution and execution.plan_attempt == 1 and execution.state == "queued"
        notification = adapter.external_task_notifications[-1][1]
        assert notification.correlation and notification.correlation.execution
        assert notification.correlation.execution.task_id == "prepare"
        preflight = request()
        preflight.arguments["validate_only"] = True
        result = await runtime.version_call(scope, preflight, "check", context)
        assert result["valid"] is True
        assert not runtime._artifact_versions(scope, context)
        with pytest.raises(ValueError, match="current approved task attempt"):
            await runtime.version_call(scope, request(attempt=2), "future", context)
        source["schema_version"] = 2
        invalid = await runtime.version_call(
            scope, request(), "superseded-command", context
        )
        assert invalid["report_state"] == "registering"
        await asyncio.gather(*runtime._stage_analysis_workers.values())
        source["schema_version"] = 1
        first, second = await asyncio.gather(
            runtime.version_call(scope, request(), "publish", context),
            runtime.version_call(scope, request(), "publish", context),
        )
        assert first["report_state"] in {"registering", "ready"}
        assert second["report_state"] in {"registering", "ready"}
        await asyncio.gather(*runtime._stage_analysis_workers.values())
        replay = await runtime.version_call(scope, request(), "publish", context)
        assert replay["report_state"] == "ready", replay
        report_id = replay["report_version_id"]
        assert isinstance(report_id, str)
        prior = await runtime.version_call(
            scope, request(), "superseded-command", context
        )
        assert (
            prior["report_state"] == "superseded" and prior["report_version_id"] is None
        )
        with pytest.raises(ValueError, match="different outputs"):
            await runtime.version_call(
                scope,
                request(report_path="/workspace/run/other.json"),
                "publish",
                context,
            )
        await runtime._capture_plan_progress(active, store)
        executions = store.load_snapshot().experiment_executions
        assert executions and len(executions) == 1
        run = executions[0]
        assert (run.state, run.report_state) == ("succeeded", "ready")
        assert run.artifacts[0]["artifact_id"] == report_id
        reports = [
            item
            for item in runtime._artifact_versions(scope, context)
            if item.registration_key == delivery_version_key(run, "report")
        ]
        assert [item.id for item in reports] == [report_id]
        assert json.loads(workspace_plan.read_text()) == {"tasks": [row]}
    finally:
        await runtime.close()


def test_train_report_export_normalizes_training_and_merge_task_ids(tmp_path):
    exporter = (
        Path(__file__).resolve().parents[2]
        / ".agents/skills/workflow-execution/scripts/export_plan_report.py"
    )
    result = tmp_path / "training_result.json"
    result.write_text(
        json.dumps(
            {
                "platform_task_ids": {"sft_training": "1398", "lora_merge": "1400"},
                "steps": 180,
            }
        )
    )
    mapping = tmp_path / "report_mapping.json"
    mapping.write_text(
        json.dumps(
            {"metrics": [{"label": "steps", "unit": "count", "path": ["steps"]}]}
        )
    )
    command = [sys.executable, str(exporter), str(result), str(mapping)]
    exported = subprocess.run(
        [*command, "--platform-task-id", "1398", "--platform-task-id", "1400"],
        check=True,
        capture_output=True,
    )
    report = json.loads(exported.stdout)
    assert report["source"] == {"platform_task_ids": ["1398", "1400"]}
    assert "platform_task_ids" not in report
    storage = cast(EvaluationStorage, SimpleNamespace(read=lambda _: exported.stdout))
    assert _read_report(
        storage,
        "train",
        {"report": "/train/report.json"},
        report_name="report",
        platform_task_id="1398",
        platform_task_ids=("1398", "1400"),
        require_platform_source=True,
        require_schema_version=True,
    )["metrics"] == [{"label": "steps", "value": 180, "unit": "count"}]

    mismatched = subprocess.run(
        [*command, "--platform-task-id", "1398"], check=False, capture_output=True
    )
    assert mismatched.returncode == 2
    assert not mismatched.stdout
    assert b"outside --platform-task-id" in mismatched.stderr


def test_v1_platform_report_must_match_its_trusted_task():
    body = {
        "schema_version": 1,
        "source": {"platform_task_id": "other"},
        "metrics": [{"label": "accuracy", "value": 0.72}],
    }
    storage = cast(
        EvaluationStorage, SimpleNamespace(read=lambda _: json.dumps(body).encode())
    )
    outputs = {"report": "/bench/report.json"}
    with pytest.raises(InvalidPlanDelivery) as failure:
        _read_report(
            storage,
            "init",
            outputs,
            report_name="report",
            platform_task_id="bound",
            require_platform_source=True,
            require_schema_version=True,
        )
    assert failure.value.field == "source.platform_task_id"
    body.pop("source")
    with pytest.raises(InvalidPlanDelivery):
        _read_report(
            storage,
            "init",
            outputs,
            report_name="report",
            platform_task_id="bound",
            require_platform_source=True,
            require_schema_version=True,
        )
    body["source"] = {"platform_task_id": "bound"}
    assert _read_report(
        storage,
        "init",
        outputs,
        report_name="report",
        platform_task_id="bound",
        require_platform_source=True,
        require_schema_version=True,
    )["metrics"]
    with pytest.raises(InvalidPlanDelivery) as narrative:
        _read_report(
            storage,
            "init",
            outputs,
            report_name="report",
            platform_task_id="bound",
            allow_narrative=True,
            require_schema_version=True,
        )
    assert narrative.value.field == "summary"


@pytest.mark.parametrize(
    ("stage", "invalid_fields", "field"),
    [
        (
            "init",
            {
                "metrics": [
                    {"label": "accuracy", "value": 0.72},
                    {"label": "bad", "value": "bad"},
                ]
            },
            "metrics[1].value",
        ),
        (
            "dataset",
            {"files": {}, "splits": {"train": {"file": "/workspace/run/train.jsonl"}}},
            "files",
        ),
        ("train", {"schema_version": None}, "schema_version"),
        (
            "eval",
            {"source": {"platform_task_ids": {"scoring": "1398"}}},
            "source.platform_task_ids",
        ),
    ],
)
def test_primary_report_controls_node_completion_even_without_explicit_delivery(
    tmp_path, stage, invalid_fields, field
):
    context = RequestContext(user_id="42")
    snapshots = FileArtifactSnapshots(tmp_path / "Storage")
    versions = Versions(SQLiteVersionStore(tmp_path / "versions.db"), snapshots)
    neighbors = tuple(
        kind for kind in ("init", "dataset", "train", "eval") if kind != stage
    )
    project = ProjectView.model_validate(
        {
            "project_id": "p",
            "conversation_id": "session",
            "plan_sequence": 1,
            "creation_command_id": "create",
            "plan": {
                "stages": [
                    {"id": "s", "kind": stage, "name": stage},
                    *({"id": kind, "kind": kind, "name": kind} for kind in neighbors),
                ],
                "tasks": [
                    {
                        "id": "task",
                        "stage_id": "s",
                        "name": "Report",
                        "outputs": [{"name": "result", "kind": "report"}],
                    },
                    *(
                        {"id": kind, "stage_id": kind, "name": kind}
                        for kind in neighbors
                    ),
                ],
            },
        }
    )
    row = {
        "id": "task",
        "stage_id": "s",
        "business_status": "completed",
        "verified_outputs": {"result": "/workspace/run/report.json"},
    }
    run, *other_runs = plan_executions(
        project,
        [
            row,
            *(
                {"id": kind, "stage_id": kind, "business_status": "in_progress"}
                for kind in neighbors
            ),
        ],
    )
    run = run.model_copy(update={"platform_task_id": "own-result"})
    other_runs = [
        item.model_copy(update={"platform_task_id": f"{item.task_id}-running"})
        for item in other_runs
    ]
    task = project.plan.tasks[0]
    scope = "session"
    valid = {
        "schema_version": 1,
        "metrics": [{"label": "result", "value": 0.72}],
        "source": {"platform_task_id": "own-result"},
        **(
            {"files": [{"name": "train", "path": "/workspace/run/train.jsonl"}]}
            if stage == "dataset"
            else {}
        ),
    }
    read = Mock(return_value=json.dumps({**valid, **invalid_fields}).encode())
    storage = cast(
        EvaluationStorage,
        SimpleNamespace(read=read, list=lambda _: {"report.json", "train.jsonl"}),
    )
    snapshot = ConversationSnapshot.model_validate(
        {
            "conversation_id": scope,
            "capabilities": {},
            "project": project,
            "external_tasks": [
                {
                    "task_id": identity,
                    "kind": "workflow_run",
                    "status": status,
                    "run_id": "same-main-turn",
                    "submitted_at": "2026-11-25T06:00:00Z",
                    "updated_at": "2026-11-25T06:00:00Z",
                }
                for identity, status in [
                    ("own-result", "succeeded"),
                    *((f"{kind}-running", "running") for kind in neighbors),
                ]
            ],
        }
    )

    def node(execution):
        current = snapshot.model_copy(
            update={"experiment_executions": (execution, *other_runs)}
        )
        target, *other_nodes = project_observations(
            with_platform_states(current), "p"
        ).nodes
        assert all(item.business_status == "running" for item in other_nodes)
        return target

    assert node(run).business_status == "waiting_analysis"
    assert node(run).delivery_status == "pending"
    assert (
        node(run.model_copy(update={"report_state": "ready"})).business_status
        == "waiting_analysis"
    )
    with pytest.raises(InvalidPlanDelivery) as failure:
        register_plan_delivery(versions, storage, task, run, row, scope, context)
    assert failure.value.field == field
    version_project = versions.project(scope, context, create=False)
    assert version_project and not versions.store.list_versions(version_project.id)
    failed = run.model_copy(
        update={
            "report_state": "registration_failed",
            "delivery_error": failure.value.code,
        }
    )
    assert node(failed).business_status == "waiting_analysis"
    assert node(failed).delivery_status == "failed"

    read.return_value = json.dumps(valid).encode()
    directory = snapshots.path("/run", context)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "report.json").write_bytes(read.return_value)
    ready = register_plan_delivery(versions, storage, task, run, row, scope, context)
    completed = node(with_plan_report(run, ready))
    assert completed.business_status == "succeeded"
    assert completed.report_version_ids == (ready.id,)


def test_baseline_retry_uses_the_latest_bound_receipt_after_failed_runs():
    project = ProjectView.model_validate(
        {
            "project_id": "p",
            "conversation_id": "session",
            "plan_sequence": 3,
            "creation_command_id": "create",
            "plan": {
                "stages": [{"id": "init", "kind": "init", "name": "Init"}],
                "tasks": [
                    {
                        "id": "baseline_bench",
                        "stage_id": "init",
                        "kind": "bench",
                        "name": "Baseline",
                        "outputs": [{"name": "baseline_report", "kind": "report"}],
                    }
                ],
            },
        }
    )
    (run,) = plan_executions(
        project,
        [{"id": "baseline_bench", "stage_id": "init", "business_status": "completed"}],
    )
    run = run.model_copy(
        update={"platform_task_id": "1395", "platform_task_ids": ("1395", "1397")}
    )
    statuses = {"1395": "failed", "1397": "succeeded"}
    task = project.plan.tasks[0]
    source = plan_delivery_source(run, None, task=task, statuses=statuses)
    assert source.platform_task_id == "1397"
    assert source.platform_task_ids == ("1397",)
    assert platform_results_ready(source, statuses)
    assert (
        plan_delivery_source(
            run, None, task=task, statuses={**statuses, "1395": "succeeded"}
        )
        == run
    )
    assert plan_delivery_source(
        run.model_copy(update={"stage_snapshot": {"kind": "train"}}),
        None,
        task=task,
        statuses=statuses,
    ).platform_task_ids == ("1395", "1397")

    snapshot = ConversationSnapshot.model_validate(
        {
            "conversation_id": "session",
            "capabilities": {},
            "project": project,
            "experiment_executions": (run,),
            "external_tasks": [
                {
                    "task_id": identity,
                    "kind": "workflow_run",
                    "status": status,
                    "submitted_at": "2026-11-27T02:07:00Z",
                    "updated_at": "2026-11-27T02:37:00Z",
                }
                for identity, status in statuses.items()
            ],
        }
    )
    projected = with_platform_states(snapshot)
    assert projected.experiment_executions
    (current,) = projected.experiment_executions
    assert current.platform_task_id == "1397"
    assert current.platform_task_ids == ("1395", "1397")
    assert current.state == "succeeded"
    node = project_observations(projected, "p").nodes[0]
    assert (
        node.business_status == "waiting_analysis" and node.delivery_status == "pending"
    )


def test_failed_snapshot_is_never_reused_for_a_corrected_submission(
    tmp_path, monkeypatch
):
    context = RequestContext(user_id="42")
    snapshots = FileArtifactSnapshots(tmp_path / "Storage")
    versions = Versions(SQLiteVersionStore(tmp_path / "versions.db"), snapshots)
    directory = snapshots.path("/run", context)
    directory.mkdir(parents=True)
    for name, score in (("first.json", 0.51), ("corrected.json", 0.92)):
        (directory / name).write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "metrics": [{"label": "accuracy", "value": score}],
                }
            )
        )
    read = Mock(side_effect=lambda path: snapshots.path(path, context).read_bytes())
    storage = cast(
        EvaluationStorage,
        SimpleNamespace(read=read, list=lambda _: {"first.json", "corrected.json"}),
    )
    task = ExperimentTask(
        id="bench",
        stage_id="init",
        name="Bench",
        kind="bench",
        outputs=(PlanOutput(name="result", kind="report"),),
    )
    run = ExperimentExecutionView(
        run_id="project:bench:1",
        project_id="project",
        task_id=task.id,
        task_name=task.name,
        role_id="main",
        plan_sequence=1,
        stage_id="init",
        stage_snapshot={"kind": "init"},
        state="succeeded",
        report_state="not_started",
        delivery_command_id="first",
    )
    original_save = snapshots.save

    def fail_snapshot(source, target, request_context):
        if source.text is not None:
            raise OSError("simulated report write failure")
        return original_save(source, target, request_context)

    with monkeypatch.context() as patch:
        patch.setattr(snapshots, "save", fail_snapshot)
        with pytest.raises(ValueError, match="snapshot not completed"):
            register_plan_delivery(
                versions,
                storage,
                task,
                run,
                {"verified_outputs": {"result": "/workspace/run/first.json"}},
                "session",
                context,
            )
    version_project = versions.project("session", context)
    assert version_project
    failed = [
        item
        for item in versions.store.list_versions(version_project.id)
        if item.state == "failed"
    ]
    assert len(failed) == 1
    corrected = run.model_copy(update={"delivery_command_id": "corrected"})
    row = {"verified_outputs": {"result": "/workspace/run/corrected.json"}}
    ready = register_plan_delivery(
        versions, storage, task, corrected, row, "session", context
    )
    assert ready.id != failed[0].id
    read.assert_called_with("/run/corrected.json")
    assert (
        json.loads(snapshots.read_text(snapshot_result(ready), context))["metrics"][0][
            "value"
        ]
        == 0.92
    )
    assert (
        register_plan_delivery(
            versions, storage, task, corrected, row, "session", context
        )
        == ready
    )


def test_primary_dataset_report_registers_files_despite_bad_auxiliary_values():
    outputs = {"prepare_report": "/run/report.json", "train": "/run/train.jsonl"}
    body = {
        "files": [
            {
                "path": "/workspace/run/train.jsonl",
                "sample_count": "390",
                "size_bytes": 1024.0,
                "distributions": {
                    "label": {"pass": "270", "fail": 120.0, "invalid": -1}
                },
            }
        ],
        "metrics": [
            {"label": "train_records", "value": 390, "unit": "count"},
            {"label": "reread_validation", "value": 1, "unit": "bool"},
            {"label": "malformed", "value": "unavailable"},
        ],
    }
    storage = cast(
        EvaluationStorage, SimpleNamespace(read=lambda _: json.dumps(body).encode())
    )
    report = _read_report(storage, "dataset", outputs, report_name="prepare_report")
    assert report["metrics"] == [
        {"label": "train_records", "value": 390, "unit": "count"}
    ]
    assert report["checks"] == {"reread_validation": True}
    assert report["files"] == [
        {
            "name": "train.jsonl",
            "path": "/workspace/run/train.jsonl",
            "sample_count": 390,
            "size_bytes": 1024,
            "distributions": {"label": {"pass": 270, "fail": 120}},
        }
    ]
    assert report["issues"] == [
        {"field": "metrics[2].value", "code": "invalid_metric"},
        {"field": "files[0].distributions.label.invalid", "code": "invalid_count"},
    ]
    body.pop("metrics")
    assert _read_report(storage, "dataset", outputs, report_name="prepare_report")[
        "files"
    ]
    body["files"] = []
    with pytest.raises(InvalidPlanDelivery, match="invalid_report") as exc:
        _read_report(storage, "dataset", outputs, report_name="prepare_report")
    assert exc.value.field == "files"


def test_legacy_detection_preserves_summary_reports_and_skips_oversized_non_reports():
    def read(path):
        if path == "/run/config.json":
            raise EvaluationStorageTooLargeError("file exceeds download limit")
        if path == "/run/baseline.json":
            return b'{"metrics":[{"label":"accuracy","value":0.55}]}'
        return b'{"summary":"The original verification result."}'

    storage = cast(EvaluationStorage, SimpleNamespace(read=read))
    outputs = {"model_config": "/run/config.json", "report": "/run/report.json"}
    assert _read_report(storage, "init", outputs) == {
        "summary": "The original verification result.",
        "metrics": [],
    }
    assert _read_report(
        storage, "init", {**outputs, "baseline_results": "/run/baseline.json"}
    )["metrics"] == [{"label": "accuracy", "value": 0.55, "unit": "number"}]
    with pytest.raises(InvalidPlanDelivery, match="invalid_report"):
        _read_report(storage, "init", outputs, report_name="model_config")


def test_baseline_and_attribution_complete_together_but_have_separate_reports(tmp_path):
    context = RequestContext(user_id="42")
    snapshots = FileArtifactSnapshots(tmp_path / "storage")
    versions = Versions(SQLiteVersionStore(tmp_path / "versions.db"), snapshots)
    project = ProjectView.model_validate(
        {
            "project_id": "p",
            "conversation_id": "session",
            "plan_sequence": 2,
            "creation_command_id": "accept",
            "plan": {
                "stages": [{"id": "init", "kind": "init", "name": "Init"}],
                "tasks": [
                    {
                        "id": "bench",
                        "kind": "bench",
                        "stage_id": "init",
                        "name": "Measure",
                        "outputs": [{"name": "result", "kind": "report"}],
                    },
                    {
                        "id": "explain",
                        "kind": "analysis",
                        "stage_id": "init",
                        "name": "Explain",
                        "outputs": [{"name": "analysis", "kind": "report"}],
                    },
                ],
            },
        }
    )
    baseline_body = {
        "schema_version": 1,
        "metrics": [{"label": "result_accuracy", "value": 0.72, "unit": "ratio"}],
    }
    analysis_body = {
        "schema_version": 1,
        "findings": ["Low-confidence samples cause most errors."],
        "badcases": [{"category": "glare", "count": 9}],
    }
    directory = snapshots.path("/run", context)
    directory.mkdir(parents=True)
    (directory / "bench.json").write_text(json.dumps(baseline_body))
    (directory / "explain.json").write_text(json.dumps(analysis_body))
    storage = cast(
        EvaluationStorage,
        SimpleNamespace(
            read=lambda path: snapshots.path(path, context).read_bytes(),
            list=lambda path: {
                file.name for file in snapshots.path(path, context).iterdir()
            },
        ),
    )
    runs = {}
    reports = {}
    for task in project.plan.tasks:
        ready = ExperimentExecutionView(
            project_id="p",
            plan_sequence=2,
            stage_id="init",
            stage_snapshot={"kind": "init"},
            run_id=f"p:{task.id}:1",
            task_id=task.id,
            task_name=task.name,
            role_id="main",
            state="succeeded",
            report_state="not_started",
            delivery_command_id=task.id,
        )
        report = register_plan_delivery(
            versions,
            storage,
            task,
            ready,
            {
                "verified_outputs": {
                    task.outputs[0].name: f"/workspace/run/{task.id}.json"
                }
            },
            "session",
            context,
        )
        reports[task.id] = report.id
        runs[task.id] = with_plan_report(ready, report)

    snapshot = ConversationSnapshot.model_validate(
        {
            "conversation_id": "session",
            "capabilities": {},
            "project": project,
            "experiment_executions": [runs["bench"]],
        }
    )
    waiting = project_observations(snapshot, "p").nodes[0]
    assert waiting.business_status == "waiting_analysis"
    assert waiting.report_version_ids == (reports["bench"],)
    assert waiting.analysis_report_version_ids == ()

    complete = snapshot.model_copy(
        update={"experiment_executions": tuple(runs.values())}
    )
    node = project_observations(complete, "p").nodes[0]
    assert node.business_status == "succeeded"
    assert node.report_version_ids == (reports["bench"],)
    assert node.analysis_report_version_ids == (reports["explain"],)
    metrics = observation_report_details(complete, "p", "init", None, versions, context)
    assert [metric.value for metric in metrics.metrics] == [0.72]
    with pytest.raises(FileNotFoundError):
        observation_report_details(
            complete,
            "p",
            "init",
            None,
            versions,
            context,
            report_version_id=reports["explain"],
        )
    # A later export at the same path cannot change a published report.
    (directory / "explain.json").write_text('{"summary":"later export"}')
    original = observation_analysis_report(
        complete,
        "p",
        "init",
        reports["explain"],
        versions,
        context,
    )
    assert original["filename"] == "explain.json"
    assert json.loads(original["text"]) == analysis_body
    for state, identity in (
        (complete, reports["bench"]),
        (snapshot, reports["explain"]),
    ):
        with pytest.raises(FileNotFoundError):
            observation_analysis_report(state, "p", "init", identity, versions, context)


def test_legacy_report_rebuild_reuses_owned_source_and_keeps_the_previous_version(
    tmp_path,
):
    context = RequestContext(user_id="42")
    snapshots = FileArtifactSnapshots(tmp_path / "storage")
    versions = Versions(SQLiteVersionStore(tmp_path / "versions.db"), snapshots)
    project = versions.project("session", context)
    assert project is not None
    task = ExperimentTask(
        id="baseline",
        stage_id="init",
        name="Evaluate",
        outputs=(PlanOutput(name="baseline_results"),),
    )
    initial_run = ExperimentExecutionView(
        project_id="project",
        plan_sequence=1,
        run_id="project:baseline:1",
        task_id=task.id,
        task_name=task.name,
        stage_id="init",
        role_id="main",
        state="succeeded",
        report_state="not_started",
        stage_snapshot={"kind": "init"},
    )
    bound: JsonObject = {"plan_execution": [1, initial_run.run_id]}
    path = "/run/baseline.json"
    source = snapshots.path(path, context)
    source.parent.mkdir(parents=True)
    source.write_text('{"metrics":[{"label":"accuracy","value":0.55}]}')
    original = versions.save(
        project,
        delivery_key(initial_run, "output:baseline_results"),
        SaveVersion(
            kind="config",
            logical_key="baseline/baseline_results",
            source=SnapshotSource(storage_ref=path, reference_only=True),
            summary=bound,
        ),
        context,
    )
    previous = versions.save(
        project,
        delivery_key(initial_run, "report"),
        SaveVersion(
            kind="report",
            logical_key="baseline/delivery-report",
            source=SnapshotSource(text='{"metrics": []}', filename="report.json"),
            parents=[
                VersionParent(version_id=original.id, relation="baseline_results")
            ],
            summary=bound,
        ),
        context,
    )
    run = with_plan_report(initial_run, previous)
    read = Mock(return_value=b'{"metrics":[{"label":"accuracy","value":0.55}]}')
    storage = cast(EvaluationStorage, SimpleNamespace(read=read))
    _, report = rebuild_legacy_report(
        versions, storage, {task.id: task}, (run,), previous.id, "session", context
    )
    read.assert_called_once_with(path)
    assert report.id != previous.id and report.version_no == previous.version_no + 1
    assert report.manifest.parents[-1] == VersionParent(
        version_id=previous.id, relation="supersedes"
    )
    assert versions.store.get_version(project.id, previous.id) == previous
    with pytest.raises(InvalidPlanDelivery, match="invalid_output"):
        rebuild_legacy_report(
            versions, storage, {}, (run,), previous.id, "session", context
        )
    assert (
        rebuild_legacy_report(
            versions,
            storage,
            {task.id: task},
            (with_plan_report(run, report),),
            previous.id,
            "session",
            context,
        )[1]
        == report
    )
    saved = tuple(versions.store.list_versions(project.id))
    assert delivery_report(run, {"report_version_id": previous.id}, saved, {}) == report
    with pytest.raises(InvalidPlanDelivery, match="ambiguous_report"):
        _read_report(
            storage, "init", {"baseline_results": path, "other": "/run/other.json"}
        )


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


@pytest.mark.parametrize("report_name", [None, "final_report"])
def test_evaluation_comparison_preserves_measured_metric_deltas(report_name):
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
    document = _read_report(
        storage,
        "eval",
        {"final_report": "/run/final_report.json"},
        report_name=report_name,
    )
    assert document["metrics"] == [
        {"label": "result_accuracy", "value": 0.47, "unit": "ratio"},
        {"label": "result_accuracy_delta", "value": -0.08, "unit": "ratio"},
    ]
    assert document["issues"] == [
        {"field": "metrics[2].value", "code": "invalid_metric"}
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
