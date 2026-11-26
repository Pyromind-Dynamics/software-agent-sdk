import json

import pytest

from pyromind_runtime.application.plan_execution import (
    plan_executions,
    read_plan_tasks,
    resolve_plan_output_paths,
    with_platform_states,
)
from pyromind_runtime.application.project_observations import (
    baseline_evaluation_metrics,
    dataset_report_files,
    observation_report_details,
    project_observations,
)
from pyromind_runtime.application.stage_analysis import (
    ready_stage_analyses,
    validate_stage_analysis,
)
from pyromind_runtime.application.versions import Versions, snapshot_result
from pyromind_runtime.domain.content import JsonObject
from pyromind_runtime.domain.context import RequestContext
from pyromind_runtime.domain.observations import ObservationsConfig
from pyromind_runtime.domain.preparation import ProjectView
from pyromind_runtime.domain.snapshot import ConversationSnapshot
from pyromind_runtime.domain.stage_analysis import StageAnalysisJob
from pyromind_runtime.domain.versions import (
    ArtifactVersion,
    SaveVersion,
    SnapshotSource,
)
from pyromind_runtime.infrastructure.file_artifact_snapshots import (
    FileArtifactSnapshots,
)
from pyromind_runtime.infrastructure.sqlite_version_store import SQLiteVersionStore


def test_evaluation_waits_for_analysis_after_platform_results_return():
    project = ProjectView.model_validate(
        {
            "project_id": "p",
            "conversation_id": "session",
            "plan_sequence": 4,
            "creation_command_id": "create",
            "plan": {
                "stages": [{"kind": "eval", "id": "eval", "name": "Evaluation"}],
                "tasks": [
                    {
                        "id": "candidate",
                        "stage_id": "eval",
                        "name": "Candidate evaluation",
                        "kind": "checkpoint_eval",
                    },
                    {
                        "id": "review",
                        "stage_id": "eval",
                        "name": "Compare results",
                        "kind": "analysis",
                    },
                ],
            },
        }
    )
    (candidate,) = plan_executions(
        project,
        [{"id": "candidate", "stage_id": "eval", "business_status": "in_progress"}],
    )
    candidate = candidate.model_copy(update={"platform_task_id": "1280"})
    snapshot = ConversationSnapshot.model_validate(
        {
            "conversation_id": "session",
            "capabilities": {},
            "project": project,
            "experiment_executions": (candidate,),
            "external_tasks": [
                {
                    "task_id": "1280",
                    "kind": "workflow_run",
                    "status": "running",
                    "submitted_at": "2026-11-23T07:59:00Z",
                    "updated_at": "2026-11-23T08:00:00Z",
                }
            ],
        }
    )
    assert (
        project_observations(with_platform_states(snapshot), "p")
        .nodes[0]
        .business_status
        == "running"
    )
    succeeded = snapshot.external_tasks[0].model_copy(update={"status": "succeeded"})
    returned = with_platform_states(
        snapshot.model_copy(update={"external_tasks": (succeeded,)})
    )
    node = project_observations(returned, "p").nodes[0]
    assert node.business_status == "waiting_analysis"
    assert node.current_run_ids == ("p:candidate:1",)

    (review,) = plan_executions(
        project, [{"id": "review", "stage_id": "eval", "business_status": "completed"}]
    )
    delivered = returned.model_copy(
        update={
            "experiment_executions": (
                candidate.model_copy(update={"state": "succeeded"}),
                review,
            )
        }
    )
    assert project_observations(delivered, "p").nodes[0].business_status == "succeeded"


def test_dataset_analysis_uses_this_attempt_and_requires_every_artifact():
    project = ProjectView.model_validate(
        {
            "project_id": "p",
            "conversation_id": "session",
            "plan_sequence": 7,
            "creation_command_id": "create",
            "plan": {
                "stages": [{"kind": "dataset", "id": "cleaning", "name": "Clean"}],
                "tasks": [
                    {
                        "id": "train-split",
                        "stage_id": "cleaning",
                        "name": "Train split",
                        "outputs": [{"name": "train"}],
                    },
                    {
                        "id": "val-split",
                        "stage_id": "cleaning",
                        "name": "Validate split",
                        "outputs": [{"name": "val"}],
                    },
                ],
            },
        }
    )
    rows = read_plan_tasks(
        json.dumps(
            {
                "tasks": [{"id": "train-split", "business_status": "failed"}],
                "task_records": {
                    "train-split": {
                        "stage_id": "untrusted-stage",
                        "platform_task_id": "untrusted-platform-task",
                        "plan_attempt": 2,
                        "business_status": "completed",
                        "verified_outputs": {"train": "train-v2"},
                    },
                    "val-split": {
                        "plan_attempt": 1,
                        "business_status": "completed",
                        "verified_outputs": {"val": "val-v1"},
                        "report_version_id": "summary",
                    },
                    "unplanned": {"business_status": "completed"},
                },
            }
        ),
        project,
    )
    assert [(row["id"], row["stage_id"], row["attempt"]) for row in rows] == [
        ("train-split", "cleaning", 2),
        ("val-split", "cleaning", 1),
    ]
    assert all("platform_task_id" not in row for row in rows)

    def artifact(identity, run, kind="dataset"):
        return ArtifactVersion.model_validate(
            {
                "id": identity,
                "project_id": "versions",
                "kind": kind,
                "logical_key": identity,
                "version_no": 1,
                "registration_key": identity,
                "state": "ready",
                "manifest": {
                    "summary": {"plan_execution": [7, run]},
                    "source": {"text": "evidence"}
                    if kind == "report"
                    else {"storage_ref": f"/output/{identity}.jsonl"},
                    "target": "/snapshot",
                    "request_digest": "digest",
                },
            }
        )

    train = artifact("train-v2", "p:train-split:2")
    val = artifact("val-v1", "p:val-split:1")
    report = artifact("summary", "p:val-split:1", "report")
    runs = tuple(plan_executions(project, rows, (train, val, report)))
    assert not ready_stage_analyses(project, rows, runs, (), (train, report), [])
    old_train = artifact("train-v2", "p:train-split:1")
    assert not ready_stage_analyses(
        project, rows, runs, (), (old_train, val, report), []
    )
    (job,) = ready_stage_analyses(project, rows, runs, (), (train, val, report), [])
    assert job.role == "dataset_analysis"
    assert job.run_ids == ("p:train-split:2", "p:val-split:1")
    assert job.artifact_ids == ("summary", "train-v2", "val-v1")
    assert ready_stage_analyses(project, rows, runs, (), (val, report, train), []) == [
        job
    ]
    paths = resolve_plan_output_paths(project, rows, runs, (train, val), {})
    assert paths[0]["verified_outputs"] == {"train": "/workspace/output/train-v2.jsonl"}
    assert ready_stage_analyses(project, paths, runs, (), (train, val, report), []) == [
        job
    ]
    assert not ready_stage_analyses(
        project, paths, runs, (), (old_train, val, report), []
    )
    rows[0]["business_status"] = "in_progress"
    assert not ready_stage_analyses(project, rows, runs, (), (train, val, report), [])


def test_analysis_reads_metrics_from_evidence_and_rejects_new_numerals():
    evidence: JsonObject = {
        "reports": [
            {
                "text": (
                    '{"metrics":[{"label":"train","value":42,"unit":"number"},'
                    '{"label":"bad","value":"12","unit":"number"}]}'
                )
            }
        ]
    }
    with pytest.raises(ValueError, match="numbers absent"):
        validate_stage_analysis({"summary": "样本 43", "findings": []}, evidence)
    result = validate_stage_analysis({"summary": "样本齐备", "findings": []}, evidence)
    assert [(metric.label, metric.value) for metric in result.metrics] == [
        ("train", 42)
    ]


def test_observation_details_read_bound_business_reports_before_analysis(tmp_path):
    context = RequestContext(user_id="owner")
    snapshots = FileArtifactSnapshots(tmp_path / "storage")
    versions = Versions(
        SQLiteVersionStore(tmp_path / "versions.db"),
        snapshots,
    )
    scope = versions.project("conversation", context)
    assert scope is not None
    project = ProjectView.model_validate(
        {
            "project_id": "project",
            "conversation_id": "conversation",
            "plan_sequence": 5,
            "creation_command_id": "create",
            "plan": {
                "stages": [
                    {"id": "base", "kind": "init", "name": "Baseline"},
                    {"id": "data", "kind": "dataset", "name": "Dataset"},
                    {"id": "eval", "kind": "eval", "name": "Evaluation"},
                ],
                "tasks": [
                    {"id": "bench", "stage_id": "base", "name": "Bench"},
                    {"id": "clean", "stage_id": "data", "name": "Clean"},
                    {"id": "score", "stage_id": "eval", "name": "Score"},
                ],
            },
        }
    )
    baseline = versions.save(
        scope,
        "baseline",
        SaveVersion(
            kind="report",
            logical_key="baseline",
            summary={"plan_execution": [5, "project:bench:2"]},
            source=SnapshotSource(
                text=json.dumps(
                    {
                        "summary": "Registered baseline",
                        "metrics": [
                            {"label": "macro_f1", "value": 0.8, "unit": "ratio"}
                        ],
                    }
                ),
                filename="baseline.json",
            ),
        ),
        context,
    )
    previous_baseline = versions.save(
        scope,
        "previous-baseline",
        SaveVersion(
            kind="report",
            logical_key="baseline",
            summary={"plan_execution": [4, "project:bench:1"]},
            source=SnapshotSource(
                text='{"metrics":[{"label":"macro_f1","value":0.5,"unit":"ratio"}]}',
                filename="baseline.json",
            ),
        ),
        context,
    )
    dataset = versions.save(
        scope,
        "dataset",
        SaveVersion(
            kind="report",
            logical_key="dataset",
            summary={"plan_execution": [5, "project:clean:1"]},
            source=SnapshotSource(
                text=json.dumps(
                    {
                        "train_data": {
                            "train_jsonl": "/workspace/train.jsonl",
                            "val_jsonl": "/workspace/val.jsonl",
                        },
                        "metrics": [
                            {"label": key, "value": value, "unit": "number"}
                            for key, value in (
                                ("train_records", 36),
                                ("train_pass", 12),
                                ("train_defect", 24),
                                ("val_records", 6),
                            )
                        ],
                    }
                ),
                filename="dataset.json",
            ),
        ),
        context,
    )
    evaluation = versions.save(
        scope,
        "evaluation",
        SaveVersion(
            kind="evaluation",
            logical_key="evaluation",
            summary={"plan_execution": [5, "project:score:1"]},
            source=SnapshotSource(
                text='{"metrics":[{"label":"score","value":0.9,"unit":"ratio"}]}',
                filename="evaluation.json",
            ),
        ),
        context,
    )
    train_file = snapshots.path("/train.jsonl", context)
    train_file.parent.mkdir(parents=True, exist_ok=True)
    train_file.write_bytes(b"abc")
    rows = [
        {
            "id": "bench",
            "stage_id": "base",
            "attempt": 2,
            "business_status": "completed",
            "report_version_id": baseline.id,
        },
        {
            "id": "clean",
            "stage_id": "data",
            "business_status": "completed",
            "report_version_id": dataset.id,
        },
        {
            "id": "score",
            "stage_id": "eval",
            "business_status": "completed",
            "report_version_id": evaluation.id,
        },
    ]
    snapshot = ConversationSnapshot.model_validate(
        {
            "conversation_id": "conversation",
            "capabilities": {},
            "project": project,
            "experiment_executions": (
                *(
                    run.model_copy(update={"platform_task_id": "1271"})
                    for run in plan_executions(
                        project.model_copy(update={"plan_sequence": 4}),
                        [
                            {
                                **rows[0],
                                "attempt": 1,
                                "report_version_id": previous_baseline.id,
                            }
                        ],
                        (previous_baseline,),
                    )
                ),
                *(
                    run.model_copy(update={"platform_task_id": "1273"})
                    if run.task_id == "bench"
                    else run
                    for run in plan_executions(
                        project, rows, (baseline, dataset, evaluation)
                    )
                ),
            ),
            "stage_analyses": (
                StageAnalysisJob(
                    job_id="history",
                    project_id="project",
                    plan_sequence=4,
                    stage_id="data",
                    role="dataset_analysis",
                    run_ids=("project:clean:1",),
                    artifact_ids=(dataset.id,),
                    report_ids=(dataset.id,),
                    status="ready",
                ),
            ),
        }
    )

    def evaluation_metrics(task_id: str):
        assert task_id == "1273"
        return baseline_evaluation_metrics(
            {
                "total": 10,
                "rubric_metrics": {
                    "box_f1": {"localization": {"f1": 0.6}},
                    "label_accuracy": {
                        "classification": {
                            "macro_f1": 0.81,
                            "per_class": {"pass": {"recall": 0.9}},
                        }
                    },
                },
            }
        )

    base = observation_report_details(
        snapshot, "project", "base", None, versions, context, evaluation_metrics
    )
    assert base.report_version_ids == (baseline.id,)
    assert base.summaries == ("Registered baseline",)
    assert [(metric.label, metric.value) for metric in base.metrics] == [
        ("macro_f1", 0.8),
        ("evaluated_samples", 10),
        ("box_f1", 0.6),
        ("pass_recall", 0.9),
    ]
    overview = project_observations(snapshot, "project")
    node = overview.nodes[0]
    config = ObservationsConfig.model_validate_json(overview.config.model_dump_json())
    assert config.detail_forms is not None and config.detail_forms.schema_version == 1
    layout = {
        form.project_node_id: form.sections[0] for form in config.detail_forms.forms
    }
    assert layout["base"].selection == "report_version"
    assert [block.type for block in layout["base"].blocks] == [
        "select",
        "status",
        "metrics",
        "findings",
        "artifacts",
    ]
    assert layout["data"].selection == "analysis_version"
    assert [block.type for block in layout["data"].blocks] == [
        "select",
        "status",
        "dataset_files",
    ]
    assert layout["eval"].selection == "analysis_version"
    assert [block.type for block in layout["eval"].blocks] == [
        "select",
        "status",
        "metrics",
    ]
    assert (
        ObservationsConfig.model_validate(
            {"schema_version": 1, "panels": []}
        ).detail_forms
        is None
    )
    assert node.report_version_ids == (baseline.id,)
    assert node.historical_report_version_ids == (previous_baseline.id,)
    assert (
        observation_report_details(
            snapshot,
            "project",
            "base",
            None,
            versions,
            context,
            evaluation_metrics,
            report_version_id=baseline.id,
        )
        == base
    )
    history = observation_report_details(
        snapshot,
        "project",
        "base",
        None,
        versions,
        context,
        report_version_id=previous_baseline.id,
    )
    assert history.report_version_ids == (previous_baseline.id,)
    assert [(metric.label, metric.value) for metric in history.metrics] == [
        ("macro_f1", 0.5)
    ]
    with pytest.raises(FileNotFoundError):
        observation_report_details(
            snapshot,
            "project",
            "base",
            None,
            versions,
            context,
            report_version_id=dataset.id,
        )
    with pytest.raises(FileNotFoundError):
        observation_report_details(
            snapshot, "project", "base", "history", versions, context
        )
    detail = observation_report_details(
        snapshot, "project", "data", "history", versions, context
    )
    assert [
        (file.name, file.path, file.size_bytes, file.sample_count)
        for file in detail.files
    ] == [
        ("train", "/workspace/train.jsonl", 3, 36),
        ("val", "/workspace/val.jsonl", None, 6),
    ]
    assert detail.files[0].distributions == {"label": {"pass": 12, "defect": 24}}
    scored = observation_report_details(
        snapshot,
        "project",
        "eval",
        None,
        versions,
        context,
        report_version_id=evaluation.id,
    )
    assert [(metric.label, metric.value) for metric in scored.metrics] == [
        ("score", 0.9)
    ]
    rows[0]["report_version_id"] = dataset.id
    wrong = snapshot.model_copy(
        update={
            "experiment_executions": tuple(
                plan_executions(project, rows, (baseline, dataset))
            )
        }
    )
    assert not observation_report_details(
        wrong, "project", "base", None, versions, context
    ).report_version_ids


def test_report_details_preserve_other_reports_when_one_is_unreadable(
    tmp_path, monkeypatch
):
    context = RequestContext(user_id="owner")
    snapshots = FileArtifactSnapshots(tmp_path / "storage")
    versions = Versions(SQLiteVersionStore(tmp_path / "versions.db"), snapshots)
    scope = versions.project("conversation", context)
    assert scope is not None
    project = ProjectView.model_validate(
        {
            "project_id": "project",
            "conversation_id": "conversation",
            "plan_sequence": 1,
            "creation_command_id": "create",
            "plan": {
                "stages": [{"id": "eval", "kind": "eval", "name": "Eval"}],
                "tasks": [
                    {"id": task, "stage_id": "eval", "name": task}
                    for task in ("a", "b")
                ],
            },
        }
    )
    artifacts = tuple(
        versions.save(
            scope,
            task,
            SaveVersion(
                kind="report",
                logical_key=task,
                summary={"plan_execution": [1, f"project:{task}:1"]},
                source=SnapshotSource(
                    text='{"metrics":[{"label":"items","value":390,"unit":"count"}]}',
                    filename="report.json",
                ),
            ),
            context,
        )
        for task in ("a", "b")
    )
    rows = [
        {
            "id": task,
            "stage_id": "eval",
            "attempt": 1,
            "business_status": "completed",
            "report_version_id": artifact.id,
        }
        for task, artifact in zip(("a", "b"), artifacts, strict=True)
    ]
    snapshot = ConversationSnapshot.model_validate(
        {
            "conversation_id": "conversation",
            "capabilities": {},
            "project": project,
            "experiment_executions": plan_executions(project, rows, artifacts),
        }
    )
    good, bad = artifacts
    original_read = snapshots.read_text

    def read(snapshot, context):
        if snapshot.storage_ref == snapshot_result(bad).storage_ref:
            raise OSError("unreadable fixture")
        return original_read(snapshot, context)

    monkeypatch.setattr(snapshots, "read_text", read)
    detail = observation_report_details(
        snapshot, "project", "eval", None, versions, context
    )
    assert detail.report_version_ids == (good.id,)
    assert [(metric.value, metric.unit) for metric in detail.metrics] == [
        (390, "count")
    ]
    assert [(issue.field, issue.code) for issue in detail.issues] == [
        (f"reports.{bad.id}", "unreadable_report")
    ]
    with pytest.raises(OSError, match="unreadable fixture"):
        observation_report_details(
            snapshot,
            "project",
            "eval",
            None,
            versions,
            context,
            report_version_id=bad.id,
        )


def test_legacy_dataset_requires_original_entry_save_receipt(tmp_path):
    context = RequestContext(user_id="owner")
    versions = Versions(
        SQLiteVersionStore(tmp_path / "db"), FileArtifactSnapshots(tmp_path / "storage")
    )
    scope = versions.project("conversation", context)
    assert scope
    text = json.dumps(
        {
            "stage": "data",
            "task_id": "clean",
            "plan_attempt": 1,
            "files": [{"name": "train", "path": "/workspace/train", "size_bytes": 12}],
        }
    )
    artifact = versions.save(
        scope,
        "manual:save",
        SaveVersion(
            kind="report",
            logical_key="clean/evidence",
            source=SnapshotSource(text=text, filename="data.json"),
        ),
        context,
    )
    project = ProjectView.model_validate(
        {
            "project_id": "project",
            "conversation_id": "conversation",
            "plan_sequence": 2,
            "creation_command_id": "accept",
            "plan": {
                "stages": [{"id": "data", "kind": "dataset", "name": "Dataset"}],
                "tasks": [{"id": "clean", "stage_id": "data", "name": "Clean"}],
            },
        }
    )
    row = {
        "id": "clean",
        "stage_id": "data",
        "business_status": "completed",
        "report_version_id": artifact.id,
    }
    receipt = {
        "kind": "operation",
        "item_id": "op",
        "started_seq": 3,
        "name": "product_versions",
        "task_id": "project-start:project",
        "status": "completed",
        "arguments": {
            "action": "save",
            "command_id": "save",
            "arguments": {"source": {"text": text}},
        },
        "details": {"id": artifact.id, "project_id": artifact.project_id},
    }
    preparation = {
        "revision": 2,
        "project_id": "project",
        "phase": "active",
        "policy": {"workflow_nodes": [{"id": "data", "name": "Dataset"}]},
        "entry_task": {
            "role_id": "main",
            "request_id": "clean",
            "stage_id": "data",
            "task": {"name": "clean", "instruction": "prepare files"},
        },
    }

    def read(*, op=receipt, attempt=1, revision=2):
        snapshot = ConversationSnapshot.model_validate(
            {
                "conversation_id": "conversation",
                "capabilities": {},
                "project": project,
                "preparation": {**preparation, "revision": revision},
                "timeline": [op],
                "experiment_executions": list(
                    plan_executions(project, [{**row, "attempt": attempt}], (artifact,))
                ),
            }
        )
        return observation_report_details(
            snapshot, "project", "data", None, versions, context
        )

    assert read().report_version_ids == (artifact.id,)
    assert read().files[0].name == "train"
    assert not read(op={**receipt, "task_id": "project-start:other"}).report_version_ids
    assert not read(attempt=2).report_version_ids
    assert not read(revision=1).report_version_ids
    altered = {**receipt["arguments"], "arguments": {"source": {"text": text + " "}}}
    assert not read(op={**receipt, "arguments": altered}).report_version_ids


def test_dataset_report_accepts_explicit_files_and_older_sources():
    explicit = dataset_report_files(
        {
            "files": [
                {
                    "name": "train",
                    "path": "/workspace/train.jsonl",
                    "size_bytes": 0,
                    "sample_count": 0,
                },
                {"name": "unsafe", "path": "/workspace/../outside.jsonl"},
            ]
        }
    )
    assert [(file.name, file.size_bytes) for file in explicit] == [("train", 0)]
    legacy = dataset_report_files(
        {
            "summary": "prepared",
            "sources": {
                "validation": {"path": "/workspace/val.jsonl", "size_bytes": 43}
            },
            "metrics": [{"label": "val_records", "value": 3, "unit": "number"}],
        }
    )
    assert [(file.name, file.size_bytes, file.sample_count) for file in legacy] == [
        ("val", 43, 3)
    ]
