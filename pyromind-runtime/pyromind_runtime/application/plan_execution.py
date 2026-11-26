"""Project the main session's business decisions alongside registered platform facts."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import PurePosixPath

from pyromind_runtime.application.model_evaluation_plan import (
    check_inference_receipt,
    verified_input_path,
)
from pyromind_runtime.application.stage_analysis import artifact_belongs_to_execution
from pyromind_runtime.domain.events import ProductEvent
from pyromind_runtime.domain.experiment_execution import ExperimentExecutionView
from pyromind_runtime.domain.experiment_plans import ExperimentTask
from pyromind_runtime.domain.persisted_payload import validate_persisted_payload
from pyromind_runtime.domain.pipeline import PipelineRun
from pyromind_runtime.domain.preparation import ProjectView
from pyromind_runtime.domain.snapshot import (
    ConversationSnapshot,
    ExternalTaskState,
    TimelineOperation,
)
from pyromind_runtime.domain.storage_paths import (
    StoragePathError,
    from_storage_api_path,
    normalize_workspace_path,
)
from pyromind_runtime.domain.versions import ArtifactVersion


PLAN_PATH = "public_data/workflowPLAN.json"
_BUSINESS_STATES = {
    "pending": "queued",
    "in_progress": "preparing",
    "blocked": "needs_input",
    "completed": "succeeded",
    "failed": "failed",
}
_RUNTIME_TASK_FIELDS = (
    "business_status",
    "plan_attempt",
    "attempt",
    "platform_task_id",
    "actual_platform_task_id",
    "report_version_id",
    "next_action",
    "verified_outputs",
)


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
        matches = [
            run.run_id
            for run in runs
            if receipts[0] in (run.platform_task_id, *run.platform_task_ids)
        ]
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
    require_canonical_delivery: bool = False,
) -> Iterator[ExperimentExecutionView]:
    tasks = {task.id: task for task in project.plan.tasks}
    stages = {stage.id: stage for stage in project.plan.stages}
    reports = {
        v.id: v
        for v in versions
        if v.kind in {"report", "evaluation"} and v.state == "ready"
    }
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
            plan_attempt=int(run_id.rsplit(":", 1)[1]),
            stage_id=task.stage_id,
            stage_snapshot=stage.model_dump(mode="json"),
            task_id=task.id,
            task_name=task.name,
            role_id="main",
            intent="execute",
            state=_BUSINESS_STATES[status],
            report_state="not_started",
            report_required=task.report_output is not None,
            waiting_reason=row.get("next_action")
            if status == "blocked" and isinstance(row.get("next_action"), str)
            else None,
        )
        if report and (task.report_output is None or not require_canonical_delivery):
            run = with_plan_report(run, report)
        validate_persisted_payload(run.model_dump(mode="json"))
        yield run


def with_plan_report(
    run: ExperimentExecutionView, report: ArtifactVersion
) -> ExperimentExecutionView:
    return run.model_copy(
        update={
            "report_state": "ready",
            "delivery_error": None,
            "delivery_error_field": None,
            "report_uri": report.manifest.storage_ref,
            "artifacts": (
                {
                    "artifact_id": report.id,
                    "name": report.logical_key,
                    "uri": report.manifest.storage_ref,
                    "sha256": report.manifest.digest,
                },
            ),
        }
    )


def delivery_rows(
    project: ProjectView, rows: list, runs: tuple[ExperimentExecutionView, ...]
) -> list:
    """Apply accepted delivery inputs for the current attempts to workspace rows."""
    current = {}
    for run in (*runs, *plan_executions(project, rows, registered=runs)):
        if (
            run.project_id != project.project_id
            or run.plan_sequence != project.plan_sequence
            or run.role_id != "main"
        ):
            continue
        previous = current.get(run.task_id)
        if previous is None or (run.plan_attempt or 1) > (previous.plan_attempt or 1):
            current[run.task_id] = run
    indexes: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        if isinstance(row, dict) and isinstance(row.get("id"), str):
            indexes.setdefault(row["id"], []).append(index)
    result = list(rows)
    for task_id, run in current.items():
        positions = indexes.get(task_id, [])
        if len(positions) > 1:
            continue
        row = rows[positions[0]] if positions else {}
        history = [
            item
            for item in runs
            if item.task_id == task_id
            and item.project_id == project.project_id
            and item.plan_sequence == project.plan_sequence
            and item.role_id == "main"
        ]
        if (
            row
            and _plan_run_id(f"{project.project_id}:{task_id}", row, history)
            != run.run_id
        ):
            continue
        if (
            run.delivery_outputs
            and (run.delivery_command_id or run.report_state == "ready")
            and run.state not in {"failed", "cancelled", "needs_input"}
            and row.get("business_status") not in {"blocked", "failed"}
        ):
            updated = {
                **row,
                "id": task_id,
                "stage_id": run.stage_id,
                "attempt": run.plan_attempt or 1,
                "business_status": "completed",
                "verified_outputs": run.delivery_outputs,
            }
            if positions:
                result[positions[0]] = updated
            else:
                result.append(updated)
    return result


def _platform_task_ids(run: ExperimentExecutionView) -> set[str]:
    ids = set(run.platform_task_ids)
    if run.platform_task_id:
        ids.add(run.platform_task_id)
    return ids


def platform_results_ready(
    run: ExperimentExecutionView, statuses: dict[str, str]
) -> bool:
    return all(
        statuses.get(task_id) == "succeeded" for task_id in _platform_task_ids(run)
    )


def platform_output_directories(
    run: ExperimentExecutionView, tasks: tuple[ExternalTaskState, ...]
) -> tuple[str, ...]:
    ids = _platform_task_ids(run)
    return tuple(
        task.output_dir
        for task in tasks
        if task.task_id in ids and task.status == "succeeded" and task.output_dir
    )


def completed_plan_write(event: ProductEvent, snapshot: ConversationSnapshot) -> bool:
    if event.type != "operation.completed" or event.payload.get("name") not in {
        "write",
        "edit",
    }:
        return False
    operation_id = event.payload.get("operation_id")
    for item in reversed(snapshot.timeline):
        if isinstance(item, TimelineOperation) and item.item_id == operation_id:
            return _is_plan_write(item)
    return False


def _is_plan_write(item: TimelineOperation) -> bool:
    path = item.arguments.get("path") if isinstance(item.arguments, dict) else None
    return (
        item.name in {"write", "edit"}
        and item.status == "completed"
        and isinstance(path, str)
        and PurePosixPath(path) == PurePosixPath(PLAN_PATH)
    )


def recover_plan_delivery_rows(
    snapshot: ConversationSnapshot, project: ProjectView
) -> list:
    """Recover paths only for already recorded successful attempts."""
    completed = {
        run.run_id: run
        for run in snapshot.experiment_executions or ()
        if run.project_id == project.project_id
        and run.plan_sequence == project.plan_sequence
        and run.role_id == "main"
        and run.state == "succeeded"
    }
    if not completed:
        return []
    text: str | None = None
    for item in snapshot.timeline:
        if not isinstance(item, TimelineOperation) or not _is_plan_write(item):
            continue
        arguments = item.arguments
        if not isinstance(arguments, dict):
            continue
        if item.name == "write":
            content = arguments.get("content")
            text = content if isinstance(content, str) else None
            continue
        edits = arguments.get("edits")
        if text is None or not isinstance(edits, list) or not edits:
            text = None
            continue
        for edit in edits:
            old = edit.get("oldText") if isinstance(edit, dict) else None
            new = edit.get("newText") if isinstance(edit, dict) else None
            if not isinstance(old, str) or not old or not isinstance(new, str):
                text = None
                break
            if text.count(old) != 1:
                text = None
                break
            text = text.replace(old, new, 1)
    if text is None:
        return []
    try:
        document = json.loads(text)
        if not isinstance(document, dict):
            return []
        plan = document.get("plan", document)
        if not isinstance(plan, dict) or plan.get("goal") != project.plan.goal:
            return []
        declarations = {
            entry["id"]: entry
            for entry in plan.get("tasks", [])
            if isinstance(entry, dict) and isinstance(entry.get("id"), str)
        }
        rows = read_plan_tasks(text, project)
    except (TypeError, ValueError):
        return []
    tasks = {task.id: task for task in project.plan.tasks}
    recovered = []
    for row in rows:
        if not isinstance(row, dict) or row.get("business_status") != "completed":
            continue
        task_id = row.get("id")
        task = tasks.get(task_id) if isinstance(task_id, str) else None
        if task is None:
            continue
        declaration = declarations.get(task.id, {})
        outputs = declaration.get("outputs")
        if (
            declaration.get("stage_id") != task.stage_id
            or not isinstance(outputs, list)
            or [out.get("name") for out in outputs if isinstance(out, dict)]
            != [out.name for out in task.outputs]
        ):
            continue
        run_id = _plan_run_id(
            f"{project.project_id}:{task.id}",
            row,
            [run for run in completed.values() if run.task_id == task.id],
        )
        if run_id in completed:
            recovered.append(row)
    return recovered


def read_plan_tasks(text: str, project: ProjectView | None = None) -> list:
    plan = json.loads(text)
    if not isinstance(plan, dict):
        raise ValueError("workflow plan requires a tasks array")
    definition = plan if "tasks" in plan else plan.get("plan")
    if "task_records" in plan:
        records = plan["task_records"]
        if not isinstance(records, dict):
            raise ValueError("workflow plan task_records must be an object")
        if project is not None:
            return [
                {
                    "id": task.id,
                    "stage_id": task.stage_id,
                    "attempt": record.get("plan_attempt", 1),
                    "business_status": record.get("business_status"),
                    "verified_outputs": record.get("verified_outputs", {}),
                    "report_version_id": record.get("report_version_id"),
                    "next_action": record.get("next_action"),
                }
                for task in project.plan.tasks
                if isinstance(record := records.get(task.id), dict)
            ]
    runtime = plan.get("runtime")
    runtime_records = runtime.get("tasks") if isinstance(runtime, dict) else None
    if isinstance(definition, dict) and isinstance(definition.get("tasks"), list):
        tasks = definition["tasks"]
    elif (
        definition is None
        and project is not None
        and plan.get("project_id") == project.project_id
        and isinstance(runtime_records, dict)
    ):
        # Use the approved task definitions when the workspace has only records.
        tasks = [
            task.model_dump(mode="json")
            for task in project.plan.tasks
            if task.id in runtime_records
        ]
    else:
        raise ValueError("workflow plan requires a tasks array")
    records = (
        runtime_records
        if isinstance(runtime_records, dict)
        else plan.get("task_states", plan.get("task_records", {}))
    )
    merged = []
    for task in tasks:
        if not isinstance(task, dict):
            merged.append(task)
            continue
        if "plan_task_id" in task:
            if "id" in task and task["id"] != task["plan_task_id"]:
                continue
            task = {**task, "id": task["plan_task_id"]}
        task_id = task.get("id")
        record = (
            records.get(task_id)
            if isinstance(records, dict) and isinstance(task_id, str)
            else None
        )
        if isinstance(record, dict) and record.get("plan_task_id", task_id) == task_id:
            task = {
                **{key: record[key] for key in _RUNTIME_TASK_FIELDS if key in record},
                **task,
            }
        outputs = task.get("verified_outputs")
        if isinstance(outputs, list):
            declared = task.get("outputs")
            names = (
                {
                    v["name"]
                    for v in declared
                    if isinstance(v, dict) and isinstance(v.get("name"), str)
                }
                if isinstance(declared, list)
                else set()
            )
            paths: dict[str, list[str]] = {}
            for value in outputs:
                if isinstance(value, str) and value.strip():
                    file = PurePosixPath(value)
                    name = file.stem
                    if name not in names and file.name == "report.json":
                        name = f"{task_id}_report"
                    if name in names:
                        paths.setdefault(name, []).append(value)
            task = {
                **task,
                "verified_outputs": {
                    key: values[0] for key, values in paths.items() if len(values) == 1
                },
            }
        merged.append(task)
    return merged


def resolve_plan_output_paths(
    project: ProjectView,
    rows: list,
    registered: tuple[ExperimentExecutionView, ...],
    artifacts: tuple[ArtifactVersion, ...],
    producers: dict[str, str | None],
) -> list:
    """Resolve legacy version IDs only when they belong to the selected attempt."""
    versions = {
        version.id: version
        for version in artifacts
        if version.state in {"ready", "reference_only"}
    }
    by_run = {
        run.run_id: run
        for run in registered
        if run.project_id == project.project_id
        and run.plan_sequence == project.plan_sequence
        and run.role_id == "main"
    }
    resolved = {}
    for projected in plan_executions(project, rows, registered=registered):
        registered_run = by_run.get(projected.run_id)
        run = registered_run or projected
        resolved[run.task_id] = run
    result = []
    for row in rows:
        outputs = row.get("verified_outputs") if isinstance(row, dict) else None
        task_id = row.get("id") if isinstance(row, dict) else None
        run = resolved.get(task_id) if isinstance(task_id, str) else None
        if run is None or not isinstance(outputs, dict):
            result.append(row)
            continue
        paths = dict(outputs)
        for key, value in outputs.items():
            artifact = versions.get(value) if isinstance(value, str) else None
            if artifact is None or not artifact_belongs_to_execution(
                artifact, run, producers
            ):
                continue
            path = artifact.manifest.source.storage_ref or artifact.manifest.storage_ref
            if path and path.startswith("/"):
                try:
                    paths[key] = from_storage_api_path(path)
                except StoragePathError:
                    continue
        result.append({**row, "verified_outputs": paths})
    return result


def completed_callback_task(
    snapshot: ConversationSnapshot,
    platform_id: str,
    project_id: str | None = None,
    *,
    allow_pending_analysis: bool = False,
) -> str | None:
    """Match a successful platform receipt to its unique approved project step."""
    external = next(
        (task for task in snapshot.external_tasks if task.task_id == platform_id), None
    )
    if (
        not external
        or external.status != "succeeded"
        or external.resume_auto_run is False
        or (external.correlation and external.correlation.type != "main")
    ):
        return None
    # external.run_id can be a platform pipeline ID. The submission receipt on
    # the approved project step identifies its owner across both ID schemes.
    matches = [
        run.task_id
        for run in snapshot.experiment_executions or ()
        if platform_id in (run.platform_task_id, *run.platform_task_ids)
        and (project_id is None or run.project_id == project_id)
        and run.role_id == "main"
        and (
            run.state in {"succeeded", "reported"}
            or (
                allow_pending_analysis
                and run.state
                in {"preparing", "waiting_analysis", "waiting_platform", "queued"}
            )
        )
        and run.report_state != "failed"
    ]
    return matches[0] if len(matches) == 1 else None


def next_approved_task(
    project: ProjectView,
    rows: list,
    completed_task_id: str,
    *,
    completed_platform_id: str | None = None,
    ready_deliveries: set[str] | None = None,
) -> tuple[ExperimentTask, dict[str, str]] | None:
    records = {
        row["id"]: row
        for row in rows
        if isinstance(row, dict) and isinstance(row.get("id"), str)
    }
    completed = records.get(completed_task_id, {})
    primary_reports = {
        task.id for task in project.plan.tasks if task.report_output is not None
    }
    if (
        completed.get("business_status") != "completed"
        or (
            ready_deliveries is not None
            and completed_task_id in primary_reports
            and completed_task_id not in ready_deliveries
        )
        or not isinstance(completed.get("verified_outputs"), dict)
        or not completed["verified_outputs"]
        or any(
            ref and ref != completed_platform_id
            for ref in (
                completed.get("platform_task_id"),
                completed.get("actual_platform_task_id"),
            )
        )
    ):
        return None
    enabled = {
        stage.id
        for stage in project.plan.stages
        if stage.enabled and not stage.placeholder
    }
    resources = {item.id: item.storage_path for item in project.plan.resources}
    for task in project.plan.tasks:
        if task.stage_id not in enabled:
            continue
        row = records.get(task.id, {})
        status = row.get("business_status", "pending")
        if status == "completed":
            if (
                ready_deliveries is not None
                and task.id in primary_reports
                and task.id not in ready_deliveries
            ):
                return None
            continue
        if (
            status != "pending"
            or row.get("platform_task_id")
            or row.get("actual_platform_task_id")
        ):
            return None
        inputs = {}
        for item in task.inputs:
            path = resources.get(item.resource_id or "")
            if item.task_id:
                upstream = records.get(item.task_id, {})
                outputs = upstream.get("verified_outputs")
                if upstream.get("business_status") != "completed" or not isinstance(
                    outputs, dict
                ):
                    return None
                if (
                    ready_deliveries is not None
                    and item.task_id in primary_reports
                    and item.task_id not in ready_deliveries
                ):
                    return None
                path = verified_input_path(item, outputs)
            if not isinstance(path, str) or not path.strip():
                return None
            try:
                inputs[item.name] = normalize_workspace_path(
                    path, field=f"{task.id}.{item.name}"
                )
            except StoragePathError:
                return None
        return (task, inputs) if inputs else None
    return None


def restore_saved_data_status(snapshot: ConversationSnapshot) -> ConversationSnapshot:
    project = snapshot.project
    runs = snapshot.experiment_executions or ()
    if project is None:
        return snapshot
    data_stages = {stage.id for stage in project.plan.stages if stage.kind == "dataset"}
    queued = {
        run.run_id
        for run in runs
        if run.state == "queued"
        and run.stage_id in data_stages
        and not run.platform_task_id
        and run.plan_sequence == project.plan_sequence
    }
    if not queued:
        return snapshot
    last_write = max(
        (
            item
            for item in snapshot.timeline
            if isinstance(item, TimelineOperation) and _is_plan_write(item)
        ),
        key=lambda item: item.completed_seq or item.started_seq,
        default=None,
    )
    if (
        last_write is None
        or last_write.name != "write"
        or not isinstance(last_write.arguments, dict)
    ):
        return snapshot
    text = last_write.arguments.get("content")
    if not isinstance(text, str):
        return snapshot
    try:
        rows = [
            row
            for row in read_plan_tasks(text, project)
            if isinstance(row, dict)
            and row.get("stage_id") in data_stages
            and row.get("business_status") == "completed"
            and isinstance(outputs := row.get("verified_outputs"), dict)
            and outputs
            and not (row.get("platform_task_id") or row.get("actual_platform_task_id"))
        ]
        completed = {
            run.run_id: run
            for run in plan_executions(project, rows, registered=runs)
            if run.run_id in queued
        }
    except ValueError:
        return snapshot
    if not completed:
        return snapshot
    return snapshot.model_copy(
        update={
            "experiment_executions": tuple(
                completed.get(run.run_id, run) for run in runs
            )
        }
    )


def with_platform_states(
    snapshot: ConversationSnapshot,
    model_evals: set[str] | None = None,
    pipelines: Iterable[PipelineRun] = (),
    *,
    sdk_receipts: dict[str, tuple[str, str]] | None = None,
) -> ConversationSnapshot:
    receipts = {
        pipeline.task_id: pipeline for pipeline in pipelines if pipeline.task_id
    }
    tasks = {item.task_id: item for item in snapshot.external_tasks}
    runs = []
    for run in snapshot.experiment_executions or ():
        identities = run.platform_task_ids or (
            (run.platform_task_id,) if run.platform_task_id else ()
        )
        platform = [tasks[identity] for identity in identities if identity in tasks]
        if platform and run.role_id == "main":
            if any(
                task.status in {"running", "pending", "waiting"} for task in platform
            ):
                run = run.model_copy(update={"state": "waiting_platform"})
            elif any(task.status != "succeeded" for task in platform):
                failed = next(task for task in platform if task.status != "succeeded")
                run = run.model_copy(
                    update={
                        "state": "failed" if failed.status == "failed" else "cancelled",
                        "waiting_reason": failed.error_summary,
                    }
                )
            elif len(platform) == len(identities) and run.state in {
                "preparing",
                "queued",
            }:
                run = run.model_copy(update={"state": "waiting_analysis"})
        if run.role_id == "main" and model_evals and run.task_id in model_evals:
            run = check_inference_receipt(run, receipts, sdk_receipts=sdk_receipts)
        runs.append(run)
    return (
        snapshot.model_copy(update={"experiment_executions": tuple(runs)})
        if runs
        else snapshot
    )
