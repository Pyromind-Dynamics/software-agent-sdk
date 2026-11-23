"""Project the main session's business decisions alongside registered platform facts."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import PurePosixPath

from pyromind_runtime.domain.events import ProductEvent
from pyromind_runtime.domain.experiment_execution import ExperimentExecutionView
from pyromind_runtime.domain.persisted_payload import validate_persisted_payload
from pyromind_runtime.domain.preparation import ProjectView
from pyromind_runtime.domain.snapshot import ConversationSnapshot, TimelineOperation
from pyromind_runtime.domain.versions import ArtifactVersion


PLAN_PATH = "public_data/workflowPLAN.json"
_BUSINESS_STATES = {
    "pending": "queued",
    "in_progress": "preparing",
    "blocked": "needs_input",
    "completed": "succeeded",
    "failed": "failed",
}


def plan_executions(
    project: ProjectView,
    rows: list,
    versions: Iterable[ArtifactVersion] = (),
) -> Iterable[ExperimentExecutionView]:
    tasks = {task.id: task for task in project.plan.tasks}
    stages = {stage.id: stage for stage in project.plan.stages}
    reports = {v.id: v for v in versions if v.kind == "report" and v.state == "ready"}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("id"), str):
            continue
        task = tasks.get(row["id"])
        if not task or row.get("stage_id") != task.stage_id:
            continue
        attempt = row.get("attempt", 1)
        status = row.get("business_status")
        if (
            type(attempt) is not int
            or attempt < 1
            or not isinstance(status, str)
            or status not in _BUSINESS_STATES
        ):
            continue
        platform = row.get("platform_task_id")
        report_id = row.get("report_version_id")
        report = reports.get(report_id) if isinstance(report_id, str) else None
        stage = stages[task.stage_id]
        if not stage.enabled or stage.placeholder:
            continue
        run = ExperimentExecutionView(
            run_id=f"{project.project_id}:{task.id}:{attempt}",
            project_id=project.project_id,
            plan_sequence=project.plan_sequence,
            stage_id=task.stage_id,
            stage_snapshot=stage.model_dump(mode="json"),
            task_id=task.id,
            task_name=task.name,
            role_id="main",
            intent="execute",
            state=_BUSINESS_STATES[status],
            report_state="ready" if report else "not_started",
            report_uri=report.manifest.storage_ref if report else None,
            artifacts=(
                {
                    "artifact_id": report.id,
                    "name": report.logical_key,
                    "uri": report.manifest.storage_ref,
                    "sha256": report.manifest.digest,
                },
            )
            if report
            else (),
            waiting_reason=row.get("next_action")
            if status == "blocked" and isinstance(row.get("next_action"), str)
            else None,
            platform_task_id=platform
            if isinstance(platform, str) and platform
            else None,
        )
        validate_persisted_payload(run.model_dump(mode="json"))
        yield run


def completed_plan_write(event: ProductEvent, snapshot: ConversationSnapshot) -> bool:
    if event.type != "operation.completed" or event.payload.get("name") not in {
        "write",
        "edit",
    }:
        return False
    operation_id = event.payload.get("operation_id")
    for item in reversed(snapshot.timeline):
        if isinstance(item, TimelineOperation) and item.item_id == operation_id:
            path = (
                item.arguments.get("path") if isinstance(item.arguments, dict) else None
            )
            return (
                item.status == "completed"
                and isinstance(path, str)
                and PurePosixPath(path) == PurePosixPath(PLAN_PATH)
            )
    return False


def read_plan_tasks(text: str) -> list:
    plan = json.loads(text)
    if not isinstance(plan, dict) or not isinstance(plan.get("tasks"), list):
        raise ValueError("workflow plan requires a tasks array")
    return plan["tasks"]


def with_platform_states(snapshot: ConversationSnapshot) -> ConversationSnapshot:
    tasks = {item.task_id: item for item in snapshot.external_tasks}
    runs = []
    for run in snapshot.experiment_executions or ():
        task = tasks.get(run.platform_task_id) if run.platform_task_id else None
        if task and run.role_id == "main":
            if task.status in {"running", "pending", "waiting"}:
                run = run.model_copy(update={"state": "waiting_platform"})
            elif task.status != "succeeded":
                run = run.model_copy(
                    update={
                        "state": "failed" if task.status == "failed" else "cancelled",
                        "waiting_reason": task.error_summary,
                    }
                )
            elif run.state in {"preparing", "queued"}:
                run = run.model_copy(update={"state": "waiting_analysis"})
        runs.append(run)
    return (
        snapshot.model_copy(update={"experiment_executions": tuple(runs)})
        if runs
        else snapshot
    )
