"""Project the main session's business decisions alongside registered platform facts."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import PurePosixPath

from pyromind_runtime.application.model_evaluation_plan import check_inference_receipt
from pyromind_runtime.domain.events import ProductEvent
from pyromind_runtime.domain.experiment_execution import ExperimentExecutionView
from pyromind_runtime.domain.persisted_payload import validate_persisted_payload
from pyromind_runtime.domain.pipeline import PipelineRun
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


def _plan_run_id(
    prefix: str, row: dict, runs: list[ExperimentExecutionView]
) -> str | None:
    attempts = [row[key] for key in ("plan_attempt", "attempt") if key in row]
    if any(type(n) is not int or n < 1 or n != attempts[0] for n in attempts):
        return None
    explicit = f"{prefix}:{attempts[0]}" if attempts else None

    # A plan can cite an existing receipt, but only a submission can bind one.
    receipts = [
        row[key]
        for key in ("platform_task_id", "actual_platform_task_id")
        if row.get(key) is not None
    ]
    if receipts:
        if any(not isinstance(ref, str) or ref != receipts[0] for ref in receipts):
            return None
        matches = [run.run_id for run in runs if run.platform_task_id == receipts[0]]
        if explicit is not None:
            return explicit if explicit in matches else None
        return matches[0] if len(matches) == 1 else None
    if explicit is not None:
        return explicit
    first = f"{prefix}:1"
    return first if all(run.run_id == first for run in runs) else None


def plan_executions(
    project: ProjectView,
    rows: list,
    versions: Iterable[ArtifactVersion] = (),
    *,
    registered: Iterable[ExperimentExecutionView] = (),
) -> Iterator[ExperimentExecutionView]:
    tasks = {task.id: task for task in project.plan.tasks}
    stages = {stage.id: stage for stage in project.plan.stages}
    reports = {v.id: v for v in versions if v.kind == "report" and v.state == "ready"}
    history: dict[str, list[ExperimentExecutionView]] = {}
    for run in registered:
        if run.project_id == project.project_id and run.role_id == "main":
            history.setdefault(run.task_id, []).append(run)
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("id"), str):
            continue
        task = tasks.get(row["id"])
        if not task or row.get("stage_id") != task.stage_id:
            continue
        run_id = _plan_run_id(
            f"{project.project_id}:{task.id}", row, history.get(task.id, [])
        )
        status = row.get("business_status")
        if (
            run_id is None
            or not isinstance(status, str)
            or status not in _BUSINESS_STATES
        ):
            continue
        report_id = row.get("report_version_id")
        report = reports.get(report_id) if isinstance(report_id, str) else None
        stage = stages[task.stage_id]
        if not stage.enabled or stage.placeholder:
            continue
        run = ExperimentExecutionView(
            run_id=run_id,
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


def with_platform_states(
    snapshot: ConversationSnapshot,
    model_evals: set[str] | None = None,
    pipelines: Iterable[PipelineRun] = (),
) -> ConversationSnapshot:
    receipts = {
        pipeline.task_id: pipeline for pipeline in pipelines if pipeline.task_id
    }
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
        if run.role_id == "main" and model_evals and run.task_id in model_evals:
            run = check_inference_receipt(run, receipts)
        runs.append(run)
    return (
        snapshot.model_copy(update={"experiment_executions": tuple(runs)})
        if runs
        else snapshot
    )
