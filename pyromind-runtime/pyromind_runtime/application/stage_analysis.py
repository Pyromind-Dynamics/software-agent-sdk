from __future__ import annotations

import json
import re
from collections.abc import Mapping

from pydantic import ValidationError

from pyromind_runtime.application.versions import Versions, snapshot_result
from pyromind_runtime.domain.content import JsonObject
from pyromind_runtime.domain.context import RequestContext
from pyromind_runtime.domain.experiment_execution import ExperimentExecutionView
from pyromind_runtime.domain.preparation import ProjectView
from pyromind_runtime.domain.snapshot import ExternalTaskState
from pyromind_runtime.domain.stage_analysis import (
    StageAnalysisArtifact,
    StageAnalysisJob,
    StageAnalysisMetric,
    StageAnalysisResult,
)
from pyromind_runtime.domain.storage_paths import StoragePathError, to_storage_api_path
from pyromind_runtime.domain.versions import (
    ArtifactVersion,
    SaveVersion,
    SnapshotSource,
    VersionParent,
    VersionProject,
    VersionRun,
    fingerprint,
)


ROLES = {
    "init": "baseline_analysis",
    "dataset": "dataset_analysis",
    "train": "train_analysis",
    "eval": "evaluation_analysis",
}


def artifact_belongs_to_execution(
    artifact: ArtifactVersion,
    run: ExperimentExecutionView,
    producers: Mapping[str, str | None],
) -> bool:
    if artifact.producer_run_id is not None:
        return bool(
            run.platform_task_id
            and producers.get(artifact.producer_run_id) == run.platform_task_id
        )
    return artifact.manifest.summary.get("plan_execution") == [
        run.plan_sequence,
        run.run_id,
    ]


def _storage_path(value: str | None) -> str | None:
    if not value or not value.startswith("/"):
        return None
    try:
        return to_storage_api_path(value)
    except StoragePathError:
        return None


def ready_stage_analyses(
    project: ProjectView,
    rows: list,
    runs: tuple[ExperimentExecutionView, ...],
    external_tasks: tuple[ExternalTaskState, ...],
    artifacts: tuple[ArtifactVersion, ...],
    producer_runs: list[VersionRun],
) -> list[StageAnalysisJob]:
    expected = {task.id: task for task in project.plan.tasks}
    accepted: dict[str, tuple[int, dict]] = {}
    duplicates = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        task_id = row.get("id")
        task = expected.get(task_id) if isinstance(task_id, str) else None
        if task is None or row.get("stage_id") != task.stage_id:
            continue
        attempt = row.get("plan_attempt", row.get("attempt", 1))
        if type(attempt) is not int or attempt < 1:
            continue
        if task.id in accepted:
            duplicates.add(task.id)
        accepted[task.id] = (attempt, row)
    by_run = {
        run.run_id: run for run in runs if run.plan_sequence == project.plan_sequence
    }
    latest_runs: dict[str, int] = {}
    for run in by_run.values():
        if run.role_id == "main" and run.project_id == project.project_id:
            attempt = run.run_id.rsplit(":", 1)[-1]
            if attempt.isdecimal():
                latest_runs[run.task_id] = max(
                    latest_runs.get(run.task_id, 0), int(attempt)
                )
    platform = {task.task_id: task.status for task in external_tasks}
    versions = {
        item.id: item for item in artifacts if item.state in {"ready", "reference_only"}
    }
    versions_by_path: dict[str, list[ArtifactVersion]] = {}
    for item in versions.values():
        path = _storage_path(item.manifest.source.storage_ref)
        if path is not None:
            versions_by_path.setdefault(path, []).append(item)
    producers = {run.id: run.source.task_id for run in producer_runs}

    jobs = []
    for stage in project.plan.stages:
        role = ROLES.get(stage.kind or "")
        tasks = [task for task in project.plan.tasks if task.stage_id == stage.id]
        if not role or not stage.enabled or stage.placeholder or not tasks:
            continue
        evidence = []
        for task in tasks:
            attempt, row = accepted.get(task.id, (0, {}))
            run_id = f"{project.project_id}:{task.id}:{attempt}"
            run = by_run.get(run_id)
            if (
                task.id in duplicates
                or attempt < latest_runs.get(task.id, 0)
                or row.get("business_status") != "completed"
                or run is None
                or run.role_id != "main"
                or run.state != "succeeded"
                or (
                    run.platform_task_id
                    and platform.get(run.platform_task_id) != "succeeded"
                )
            ):
                break
            outputs = row.get("verified_outputs", row.get("outputs", {}))
            if not isinstance(outputs, dict):
                break
            output_ids = []
            for output in task.outputs:
                value = outputs.get(output.name)
                if not isinstance(value, str):
                    break
                candidates = (
                    [versions[value]]
                    if value in versions
                    else versions_by_path.get(_storage_path(value) or "", [])
                )
                owned = [
                    item
                    for item in candidates
                    if artifact_belongs_to_execution(item, run, producers)
                ]
                if len(owned) != 1:
                    break
                output_ids.append(owned[0].id)
            if len(output_ids) != len(task.outputs):
                break
            report_id = row.get("report_version_id")
            report = versions.get(report_id) if isinstance(report_id, str) else None
            if report and (
                not artifact_belongs_to_execution(report, run, producers)
                or report.state != "ready"
                or report.kind not in {"report", "evaluation"}
            ):
                break
            if not output_ids and report is None and task.outputs:
                break
            evidence.append((run_id, output_ids, report.id if report else None))
        if len(evidence) != len(tasks):
            continue
        report_ids = sorted({report for _, _, report in evidence if report})
        if not report_ids:
            continue
        artifact_ids = sorted(
            set(report_ids) | {item for _, outputs, _ in evidence for item in outputs}
        )
        run_ids = tuple(run for run, _, _ in evidence)
        job_id = fingerprint(
            [1, project.project_id, project.plan_sequence, stage.id, role, evidence]
        )
        jobs.append(
            StageAnalysisJob(
                job_id=job_id,
                project_id=project.project_id,
                plan_sequence=project.plan_sequence,
                stage_id=stage.id,
                role=role,
                run_ids=run_ids,
                artifact_ids=tuple(artifact_ids),
                report_ids=tuple(report_ids),
            )
        )
    return jobs


def stage_analysis_evidence(
    job: StageAnalysisJob,
    versions: Versions,
    project: VersionProject,
    context: RequestContext,
) -> JsonObject:
    artifacts = []
    reports = []
    for identity in job.artifact_ids:
        artifact = versions.store.get_version(project.id, identity)
        snapshot = snapshot_result(artifact)
        if identity in job.report_ids:
            text = versions.snapshots.read_text(snapshot, context)
            if not text.strip() or len(text.encode()) > 48 * 1024:
                raise ValueError(
                    "Analysis evidence must be nonempty and at most 48 KiB"
                )
            reports.append({"artifact_id": identity, "text": text})
        elif snapshot.reference_only:
            versions.snapshots.verify_reference(snapshot.storage_ref, context)
        else:
            versions.snapshots.verify(snapshot, context)
        artifacts.append(
            {
                "artifact_id": identity,
                "name": artifact.logical_key,
                "kind": artifact.kind,
                "storage_ref": snapshot.storage_ref,
            }
        )
    if sum(len(report["text"].encode()) for report in reports) > 64 * 1024:
        raise ValueError("Stage report evidence exceeds 64 KiB")
    return {
        "stage_id": job.stage_id,
        "run_ids": list(job.run_ids),
        "artifacts": artifacts,
        "reports": reports,
        "language": context.accept_language or "zh-CN",
    }


def validate_stage_analysis(
    response: JsonObject, evidence: JsonObject
) -> StageAnalysisResult:
    result = StageAnalysisResult.model_validate(response)
    reports = evidence["reports"]
    assert isinstance(reports, list)
    texts = [
        text
        for report in reports
        if isinstance(report, dict) and isinstance(text := report.get("text"), str)
    ]
    quantities = re.compile(r"\d+(?:\.\d+)?", re.ASCII)
    source_numbers = {value for text in texts for value in quantities.findall(text)}
    for sentence in (result.summary, *result.findings):
        if set(quantities.findall(sentence)) - source_numbers:
            raise ValueError("Stage analysis contains numbers absent from the evidence")

    metrics: list[StageAnalysisMetric] = []
    for text in texts:
        try:
            body = json.loads(text)
        except ValueError:
            continue
        for metric in report_metrics(body):
            if metric not in metrics:
                metrics.append(metric)
    artifacts = evidence.get("artifacts", [])
    assert isinstance(artifacts, list)
    return result.model_copy(
        update={
            "metrics": tuple(metrics[:24]),
            "artifacts": tuple(
                StageAnalysisArtifact.model_validate(item) for item in artifacts
            ),
        }
    )


def report_metrics(body: object) -> list[StageAnalysisMetric]:
    values = body.get("metrics") if isinstance(body, dict) else None
    metrics = []
    if isinstance(values, list):
        for value in values[:24]:
            try:
                metrics.append(StageAnalysisMetric.model_validate(value))
            except ValidationError:
                continue
    return metrics


def existing_stage_analysis(
    job: StageAnalysisJob,
    versions: Versions,
    project: VersionProject,
) -> ArtifactVersion | None:
    return next(
        (
            v
            for v in versions.store.list_versions(project.id)
            if v.registration_key == f"stage-analysis:{job.job_id}"
        ),
        None,
    )


def save_stage_analysis(
    job: StageAnalysisJob,
    result: StageAnalysisResult,
    versions: Versions,
    project: VersionProject,
    context: RequestContext,
) -> ArtifactVersion:
    return versions.save(
        project,
        f"stage-analysis:{job.job_id}",
        SaveVersion(
            kind="report",
            logical_key=f"stage-analysis/{job.stage_id}",
            source=SnapshotSource(
                text=result.model_dump_json(), filename="analysis.json"
            ),
            parents=[
                VersionParent(version_id=identity) for identity in job.artifact_ids
            ],
            summary=result.model_dump(mode="json"),
        ),
        context,
    )
