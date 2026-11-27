"""Read right-panel data from accepted Project plans and host-bound Studio runs."""

import hashlib
import json
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import PurePosixPath

from pyromind_runtime.application.observation_forms import observation_detail_form
from pyromind_runtime.application.stage_analysis import (
    ROLES,
    artifact_belongs_to_execution,
    report_metrics,
)
from pyromind_runtime.application.train_observer import observed_at, record_version
from pyromind_runtime.application.versions import Versions, snapshot_result
from pyromind_runtime.domain.context import RequestContext
from pyromind_runtime.domain.experiment_execution import ExperimentExecutionView
from pyromind_runtime.domain.observations import (
    MetricConfig,
    ObservationDatasetFile,
    ObservationDetailForms,
    ObservationNode,
    ObservationOverview,
    ObservationReportDetails,
    ObservationsConfig,
    PanelConfig,
    RunBinding,
)
from pyromind_runtime.domain.snapshot import ConversationSnapshot, TimelineOperation
from pyromind_runtime.domain.stage_analysis import ReportIssue, StageAnalysisMetric
from pyromind_runtime.domain.storage_paths import (
    normalize_workspace_path,
    to_storage_api_path,
)
from pyromind_runtime.domain.versions import ArtifactVersion, canonical, fingerprint
from pyromind_runtime.ports.storage import (
    EvaluationStorage,
    EvaluationStorageUnavailableError,
)


_TRAINING_METRICS = tuple(
    MetricConfig(key=key, title=key, axis="training_step", axis_key="step")
    for key in ("loss", "eval_loss", "learning_rate", "reward")
)


def _baseline_task_ids(snapshot: ConversationSnapshot, stage_id: str) -> set[str]:
    if snapshot.project is None:
        return set()
    tasks = [task for task in snapshot.project.plan.tasks if task.stage_id == stage_id]
    return {task.id for task in tasks if task.kind == "bench"} or {
        task.id for task in tasks if task.kind != "analysis"
    }


def stage_report_runs(
    snapshot: ConversationSnapshot, project_id: str, node_id: str, kind: str
) -> dict[str, list[ExperimentExecutionView]]:
    result: dict[str, list[ExperimentExecutionView]] = {}
    baseline_tasks = _baseline_task_ids(snapshot, node_id) if kind == "init" else None
    for run in reversed(snapshot.experiment_executions or ()):
        if (
            run.project_id != project_id
            or run.stage_id != node_id
            or run.role_id != "main"
            or run.report_state != "ready"
            or run.plan_sequence is None
            or (baseline_tasks is not None and run.task_id not in baseline_tasks)
            or (run.stage_snapshot or {}).get("kind") != kind
        ):
            continue
        for artifact in run.artifacts:
            if isinstance(identity := artifact.get("artifact_id"), str):
                result.setdefault(identity, []).append(run)
    return result


def project_observations(
    snapshot: ConversationSnapshot, project_id: str
) -> ObservationOverview:
    project = snapshot.project
    if project is None or project.project_id != project_id:
        return ObservationOverview(
            project_id=project_id,
            config_version="unconfigured",
            record_version=str(snapshot.through_seq),
            config=ObservationsConfig(),
            nodes=(),
            bindings=(),
            available_runs=(),
            observed_at=observed_at(),
        )
    stages = {
        stage.id: stage
        for stage in project.plan.stages
        if stage.enabled and not stage.placeholder and stage.kind
    }
    current_runs: dict[str, ExperimentExecutionView] = {}
    tasks = {task.id: task.stage_id for task in project.plan.tasks}
    runs = [
        run
        for run in snapshot.experiment_executions or ()
        if run.project_id == project_id
        and run.role_id == "main"
        and run.plan_sequence == project.plan_sequence
        and run.stage_id in stages
        and tasks.get(run.task_id) == run.stage_id
    ]

    def attempt(run: ExperimentExecutionView) -> int:
        suffix = run.run_id.rsplit(":", 1)[-1]
        return int(suffix) if suffix.isdecimal() else 0

    for run in runs:
        previous = current_runs.get(run.task_id)
        if previous is None or attempt(run) > attempt(previous):
            current_runs[run.task_id] = run
    external = {task.task_id: task for task in snapshot.external_tasks}
    panels = tuple(
        PanelConfig(project_node_id=stage.id, metrics=_TRAINING_METRICS)
        for stage in stages.values()
        if stage.kind == "train"
    )
    bindings = tuple(
        RunBinding(
            project_node_id=run.stage_id,
            run_id=run.run_id,
            task_id=task.task_id,
            business_status=task.status,
        )
        for run in sorted(runs, key=attempt)
        if run.stage_id is not None
        and stages[run.stage_id].kind == "train"
        and (task := external.get(run.platform_task_id or "")) is not None
        and task.kind == "workflow_run"
    )
    nodes = []
    for stage in stages.values():
        stage_runs = [
            (task, current_runs.get(task.id))
            for task in project.plan.tasks
            if task.stage_id == stage.id
        ]
        states = {run.state if run else "queued" for _, run in stage_runs}
        execution_states = {
            run.state if run else "queued"
            for task, run in stage_runs
            if task.kind != "analysis"
        } or states
        pending_report = any(
            run
            and (task.report_output is not None or run.report_required)
            and run.state in {"succeeded", "waiting_analysis"}
            and not run.registered_report_ids
            for task, run in stage_runs
        )
        if states & {"failed", "cancelled", "needs_input"}:
            status = next(
                state
                for state in ("failed", "cancelled", "needs_input")
                if state in states
            )
        elif states == {"succeeded"} and not pending_report:
            status = "succeeded"
        elif execution_states and execution_states <= {"succeeded", "waiting_analysis"}:
            status = "waiting_analysis"
        elif states - {"queued"}:
            status = "running"
        else:
            status = "queued"
        baseline_tasks = (
            _baseline_task_ids(snapshot, stage.id) if stage.kind == "init" else None
        )
        report_ids = tuple(
            dict.fromkeys(
                artifact_id
                for _, run in stage_runs
                if run and (baseline_tasks is None or run.task_id in baseline_tasks)
                for artifact_id in run.registered_report_ids
            )
        )
        analysis_report_ids: tuple[str, ...] = (
            tuple(
                dict.fromkeys(
                    analysis_id
                    for task, run in stage_runs
                    if task.kind == "analysis" and run and run.state == "succeeded"
                    for analysis_id in run.registered_report_ids
                )
            )
            if stage.kind == "init"
            else ()
        )
        history = (
            stage_report_runs(snapshot, project_id, stage.id, stage.kind)
            if stage.kind in ROLES
            else {}
        )
        failed_delivery = next(
            (
                run
                for _, run in stage_runs
                if run
                and run.report_state == "registration_failed"
                and run.delivery_error
            ),
            None,
        )
        delivery_status = (
            "failed"
            if failed_delivery
            else "registering"
            if any(run and run.report_state == "registering" for _, run in stage_runs)
            else "pending"
            if pending_report
            else None
        )
        nodes.append(
            ObservationNode(
                id=stage.id,
                title=stage.name,
                type=stage.kind or "",
                business_status=status,
                execution_complete=execution_states == {"succeeded"}
                and all(
                    run and run.state == "succeeded" and run.registered_report_ids
                    for task, run in stage_runs
                    if task.kind == "analysis" and task.report_output
                )
                and any(run and run.report_evidence_key for _, run in stage_runs),
                delivery_status=delivery_status,
                delivery_error=failed_delivery.delivery_error
                if failed_delivery
                else None,
                delivery_error_field=failed_delivery.delivery_error_field
                if failed_delivery
                else None,
                current_run_ids=tuple(run.run_id for _, run in stage_runs if run),
                report_version_ids=report_ids,
                analysis_report_version_ids=analysis_report_ids,
                historical_report_version_ids=tuple(
                    identity for identity in history if identity not in report_ids
                ),
            )
        )
    config = ObservationsConfig(
        panels=panels,
        detail_forms=ObservationDetailForms(
            forms=tuple(
                observation_detail_form(stage.id, stage.kind or "", stage.name)
                for stage in stages.values()
            )
        ),
    )
    return ObservationOverview(
        project_id=project_id,
        config_version=record_version([project.plan_sequence, config.model_dump()]),
        plan_sequence=project.plan_sequence,
        record_version=str(snapshot.through_seq),
        config=config,
        nodes=tuple(nodes),
        bindings=bindings,
        available_runs=bindings,
        stage_analyses=tuple(
            job
            for job in snapshot.stage_analyses
            if job.project_id == project_id
            and job.stage_id in stages
            and (job.plan_sequence == project.plan_sequence or job.status == "ready")
        ),
        observed_at=observed_at(),
    )


def _legacy_entry_report(
    snapshot: ConversationSnapshot,
    run: ExperimentExecutionView,
    artifact: ArtifactVersion,
) -> bool:
    preparation = snapshot.preparation
    entry = preparation.entry_task if preparation else None
    if not (
        preparation
        and not preparation.policy.allowed_roles
        and preparation.project_id == run.project_id
        and preparation.revision == run.plan_sequence
        and entry
        and entry.request_id == run.task_id
        and entry.stage_id == run.stage_id
        and run.run_id == f"{run.project_id}:{run.task_id}:1"
        and run.platform_task_id is None
        and artifact.kind == "report"
        and artifact.producer_run_id is None
        and "plan_execution" not in artifact.manifest.summary
        and artifact.logical_key.startswith(f"{run.task_id}/")
    ):
        return False

    # The host dispatches project-start to the accepted entry task's first attempt.
    for item in snapshot.timeline:
        if not (
            isinstance(item, TimelineOperation)
            and item.name == "product_versions"
            and item.task_id == f"project-start:{run.project_id}"
            and item.status == "completed"
            and isinstance(item.arguments, dict)
            and item.arguments.get("action") == "save"
            and artifact.registration_key
            == f"manual:{item.arguments.get('command_id')}"
            and isinstance(item.details, dict)
            and item.details.get("id") == artifact.id
            and item.details.get("project_id") == artifact.project_id
        ):
            continue
        arguments = item.arguments.get("arguments")
        source = arguments.get("source") if isinstance(arguments, dict) else None
        text = source.get("text") if isinstance(source, dict) else None
        if not isinstance(text, str) or len(text.encode()) > 48 * 1024:
            continue
        digest = hashlib.sha256(text.encode()).hexdigest()
        if artifact.manifest.digest != fingerprint(
            {artifact.manifest.source.filename: digest}
        ):
            continue
        try:
            body = json.loads(text)
        except (ValueError, RecursionError):
            continue
        if (
            isinstance(body, dict)
            and body.get("stage") == run.stage_id
            and body.get("task_id") == run.task_id
            and type(body.get("plan_attempt")) is int
            and body["plan_attempt"] == 1
        ):
            return True
    return False


def summary_only_attribution(text: str) -> bool:
    try:
        body = json.loads(text)
    except ValueError:
        return False
    return (
        isinstance(body, dict)
        and isinstance(body.get("summary"), str)
        and not body.get("findings")
        and not body.get("metrics")
        and not (
            body.keys()
            - {
                "schema_version",
                "kind",
                "name",
                "summary",
                "source",
                "findings",
                "metrics",
            }
        )
    )


def attribution_task_source(
    snapshot: ConversationSnapshot,
    node_id: str,
    report_id: str,
    rows: list[dict],
    storage: EvaluationStorage,
) -> dict[str, str] | None:
    """Read a richer task output only when its producer and parent receipt match."""
    project = snapshot.project
    if project is None:
        return None
    runs = [
        r
        for r in snapshot.experiment_executions or ()
        if r.project_id == project.project_id
        and r.plan_sequence == project.plan_sequence
        and r.stage_id == node_id
        and r.role_id == "main"
    ]
    run = next((r for r in runs if report_id in r.registered_report_ids), None)
    if run is None:
        return None
    task = next(
        (t for t in project.plan.tasks if t.id == run.task_id and t.kind == "analysis"),
        None,
    )
    if not task or not task.report_output:
        return None
    matching = [
        row
        for row in rows
        if row.get("id") == task.id and row.get("stage_id") == node_id
    ]
    if len(matching) != 1:
        return None
    row = matching[0]
    attempt = run.plan_attempt or 1
    attempts = [row[k] for k in ("attempt", "plan_attempt") if k in row] or [1]
    if any(type(n) is not int or n != attempt for n in attempts):
        return None
    outputs = row.get("verified_outputs")
    path = outputs.get(task.report_output.name) if isinstance(outputs, dict) else None
    if not isinstance(path, str) or not path.endswith(".json"):
        return None
    source = to_storage_api_path(normalize_workspace_path(path))
    limit = 48 * 1024
    payload = storage.read(source)
    if len(payload) > limit:
        return None
    body = json.loads(payload)
    if (
        not isinstance(body, dict)
        or body.get("plan_task_id") != task.id
        or type(body.get("plan_attempt")) is not int
        or body["plan_attempt"] != attempt
        or not isinstance(body.get("summary"), str)
        or not body["summary"].strip()
        or not (body.get("findings") or report_metrics(body))
    ):
        return None
    parents = {i.task_id for i in task.inputs if i.task_id}
    parent = max(
        (
            r
            for r in runs
            if r.task_id == body.get("parent_task_id") and r.task_id in parents
        ),
        key=lambda r: r.plan_attempt or 1,
        default=None,
    )
    if (
        parent is None
        or not body.get("platform_task_id")
        or str(body["platform_task_id"])
        not in (parent.platform_task_id, *parent.platform_task_ids)
    ):
        return None
    result = {"task_source_text": canonical(body)}
    markdown = str(PurePosixPath(path).with_suffix(".md"))
    evidence = row.get("evidence_paths")
    if isinstance(evidence, list) and markdown in evidence:
        try:
            content = storage.read(to_storage_api_path(markdown))
            if len(content) <= limit:
                text = content.decode("utf-8")
                if body["summary"].strip() in text:
                    result["task_source_markdown"] = text
        except (ValueError, LookupError, OSError, EvaluationStorageUnavailableError):
            pass
    return result


def observation_analysis_report(
    snapshot: ConversationSnapshot,
    project_id: str,
    node_id: str,
    report_version_id: str,
    versions: Versions,
    context: RequestContext,
) -> dict[str, str]:
    overview = project_observations(snapshot, project_id)
    node = next((node for node in overview.nodes if node.id == node_id), None)
    if not node or report_version_id not in node.analysis_report_version_ids:
        raise FileNotFoundError("Baseline analysis report is unavailable")
    assert snapshot.project is not None
    project = versions.project(snapshot.conversation_id, context, create=False)
    if project is None:
        raise FileNotFoundError("Baseline analysis report is unavailable")
    artifact = versions.store.get_version(project.id, report_version_id)
    if artifact.kind != "report" or artifact.state != "ready":
        raise FileNotFoundError("Baseline analysis report is unavailable")
    producers = {
        run.id: run.source.task_id for run in versions.store.list_runs(project.id)
    }
    tasks = {
        task.id: task
        for task in snapshot.project.plan.tasks
        if task.stage_id == node_id and task.kind == "analysis"
    }
    run = next(
        (
            run
            for run in snapshot.experiment_executions or ()
            if run.run_id in node.current_run_ids
            and run.project_id == project_id
            and run.stage_id == node.id
            and run.role_id == "main"
            and run.plan_sequence == overview.plan_sequence
            and run.task_id in tasks
            and run.state == "succeeded"
            and run.report_state == "ready"
            and any(item.get("artifact_id") == artifact.id for item in run.artifacts)
            and artifact_belongs_to_execution(artifact, run, producers)
        ),
        None,
    )
    if run is None:
        raise FileNotFoundError("Baseline analysis report is unavailable")
    text = versions.snapshots.read_text(snapshot_result(artifact), context)
    filename = artifact.manifest.source.filename
    if filename.endswith(".json"):
        try:
            document = json.loads(text)
        except json.JSONDecodeError:
            document = None
        if isinstance(document, dict) and isinstance(
            original := document.get("attribution_report"), dict
        ):
            text = canonical(original)
            original_filename = document.get("attribution_report_filename")
            if isinstance(original_filename, str) and original_filename:
                filename = original_filename
    return {"filename": filename, "text": text}


def observation_report_details(
    snapshot: ConversationSnapshot,
    project_id: str,
    node_id: str,
    analysis_job_id: str | None,
    versions: Versions,
    context: RequestContext,
    load_baseline_metrics: Callable[[str], list[StageAnalysisMetric]] | None = None,
    *,
    report_version_id: str | None = None,
    load_dataset_source: Callable[[str], bytes] | None = None,
) -> ObservationReportDetails:
    overview = project_observations(snapshot, project_id)
    node = next((item for item in overview.nodes if item.id == node_id), None)
    if node is None or node.type not in ROLES:
        raise FileNotFoundError("Observation stage report is unavailable")
    project = versions.project(snapshot.conversation_id, context, create=False)
    if project is None:
        raise FileNotFoundError("Observation stage report is unavailable")

    current_runs = {}
    job = None
    historical_runs = (
        stage_report_runs(snapshot, project_id, node.id, node.type)
        if report_version_id is not None
        or (node.type in {"init", "dataset"} and analysis_job_id)
        else {}
    )
    if analysis_job_id is not None:
        job = next(
            (
                item
                for item in overview.stage_analyses
                if item.job_id == analysis_job_id
                and item.stage_id == node.id
                and item.role == ROLES[node.type]
                and item.status == "ready"
            ),
            None,
        )
        if job is None:
            raise FileNotFoundError("Observation stage report is unavailable")
        report_ids = (
            tuple(
                identity for identity in job.report_ids if identity in historical_runs
            )
            if node.type == "init"
            else job.report_ids
        )
    else:
        current_runs = {
            artifact_id: run
            for run in snapshot.experiment_executions or ()
            if run.plan_sequence == overview.plan_sequence
            and run.project_id == project_id
            and run.stage_id == node.id
            and run.role_id == "main"
            and run.run_id in node.current_run_ids
            and run.report_state == "ready"
            for artifact in run.artifacts
            if isinstance(artifact_id := artifact.get("artifact_id"), str)
        }
        report_ids = node.report_version_ids

    if report_version_id is not None:
        allowed_ids = report_ids
        if analysis_job_id is None:
            allowed_ids += node.historical_report_version_ids
        if report_version_id not in allowed_ids:
            raise FileNotFoundError("Observation stage report is unavailable")
        report_ids = (report_version_id,)

    producers = (
        {run.id: run.source.task_id for run in versions.store.list_runs(project.id)}
        if current_runs or historical_runs
        else {}
    )
    read_ids = []
    metrics: list[StageAnalysisMetric] = []
    summaries: list[str] = []
    files: dict[str, ObservationDatasetFile] = {}
    checks: dict[str, bool] = {}
    issues: list[ReportIssue] = []
    read_errors: list[Exception] = []
    baseline_task_ids: set[str] = set()
    readable = []
    source_runs = {}
    for identity in report_ids:
        artifact = versions.store.get_version(project.id, identity)
        if artifact.state != "ready" or artifact.kind not in {"report", "evaluation"}:
            continue
        candidates = (
            [current_runs[identity]]
            if identity in current_runs
            else historical_runs.get(identity, [])
        )
        run = next(
            (
                run
                for run in candidates
                if (
                    job is None
                    or (
                        run.plan_sequence == job.plan_sequence
                        and run.run_id in job.run_ids
                    )
                )
                and (
                    artifact_belongs_to_execution(artifact, run, producers)
                    or (
                        node.type == "dataset"
                        and _legacy_entry_report(snapshot, run, artifact)
                    )
                )
            ),
            None,
        )
        if (analysis_job_id is None or node.type == "init") and run is None:
            continue
        readable.append(artifact)
        if run is not None:
            source_runs[artifact.id] = run
        if node.type == "init" and run is not None and run.platform_task_id:
            baseline_task_ids.add(run.platform_task_id)

    def read_report(artifact: ArtifactVersion) -> str | Exception:
        try:
            return versions.snapshots.read_text(snapshot_result(artifact), context)
        except (OSError, ValueError) as exc:
            return exc

    with ThreadPoolExecutor(max_workers=4) as pool:
        baseline = (
            pool.submit(load_baseline_metrics, next(iter(baseline_task_ids)))
            if load_baseline_metrics and len(baseline_task_ids) == 1
            else None
        )
        contents = pool.map(read_report, readable)
        for artifact, text in zip(readable, contents, strict=True):
            field = f"reports.{artifact.id}"
            if isinstance(text, Exception):
                read_errors.append(text)
                issues.append(ReportIssue(field=field, code="unreadable_report"))
                continue
            if len(text.encode()) > 48 * 1024:
                issues.append(ReportIssue(field=field, code="invalid_report"))
                continue
            try:
                body = json.loads(text)
            except ValueError:
                body = None
            if not isinstance(body, dict):
                issues.append(ReportIssue(field=field, code="invalid_report"))
                continue
            read_ids.append(artifact.id)
            collected: list[ReportIssue] = []
            report_checks = (
                dataset_report_checks(body) if node.type == "dataset" else None
            )
            for metric in report_metrics(
                body,
                issues=collected,
                checks=report_checks,
                allow_comparisons=node.type == "eval",
            ):
                if metric not in metrics:
                    metrics.append(metric)
            if node.type == "dataset":
                listed = dataset_report_files(body, issues=collected)
                source_run = source_runs.get(artifact.id)
                if (
                    load_dataset_source
                    and source_run
                    and source_run.delivery_outputs
                    and any(not f.distributions.get("label") for f in listed)
                ):
                    attempt = fingerprint([source_run.plan_sequence, source_run.run_id])
                    root = (
                        PurePosixPath("/workspace/.pyromind-agent")
                        / snapshot.conversation_id
                        / "reports"
                        / attempt
                    )
                    for value in source_run.delivery_outputs.values():
                        path = PurePosixPath(value)
                        if (
                            path.parent != root
                            or path.suffix != ".json"
                            or len(path.stem) != 64
                        ):
                            continue
                        try:
                            source = json.loads(
                                load_dataset_source(to_storage_api_path(value))
                            )
                            if (
                                isinstance(source, dict)
                                and fingerprint(source) == path.stem
                            ):
                                listed = restore_dataset_file_statistics(
                                    listed, source, source_run.delivery_outputs
                                )
                        except (OSError, ValueError, LookupError, RuntimeError):
                            pass
                files.update((file.path, file) for file in listed)
                if report_checks:
                    checks.update(report_checks)
            elif isinstance(summary := body.get("summary"), str):
                summary = summary.strip()
                if 0 < len(summary) <= 900 and summary not in summaries:
                    summaries.append(summary)
            saved_issues = body.get("issues")
            if isinstance(saved_issues, list):
                for issue in saved_issues[:32]:
                    try:
                        collected.append(ReportIssue.model_validate(issue))
                    except ValueError:
                        continue
            issues.extend(collected)
        if baseline is not None and read_ids:
            registered_labels = {metric.label for metric in metrics}
            metrics.extend(
                metric
                for metric in baseline.result()
                if metric.label not in registered_labels
            )
    if not read_ids and read_errors:
        raise read_errors[0]
    if report_version_id is not None and not read_ids:
        if issues:
            raise ValueError("Selected observation report is invalid")
        raise FileNotFoundError("Observation stage report is unavailable")
    file_entries = tuple(files.values())[:32]
    missing_sizes = {
        to_storage_api_path(file.path)
        for file in file_entries
        if file.size_bytes is None
    }
    if missing_sizes:
        sizes = versions.snapshots.file_sizes(tuple(sorted(missing_sizes)), context)
        file_entries = tuple(
            file.model_copy(
                update={"size_bytes": sizes.get(to_storage_api_path(file.path))}
            )
            if file.size_bytes is None
            else file
            for file in file_entries
        )
    return ObservationReportDetails(
        project_id=project_id,
        project_node_id=node_id,
        report_version_ids=tuple(read_ids),
        metrics=tuple(metrics[:24]),
        files=file_entries,
        summaries=tuple(summaries[:8]),
        checks=dict(list(checks.items())[:32]),
        issues=tuple(dict.fromkeys(issues))[:32],
    )


def baseline_evaluation_metrics(body: object) -> list[StageAnalysisMetric]:
    def fields(value: object) -> dict[str, object]:
        return value if isinstance(value, dict) else {}

    report = fields(body)
    rubrics = fields(report.get("rubric_metrics"))
    scores = fields(fields(rubrics.get("label_accuracy")).get("classification"))
    localization = fields(fields(rubrics.get("box_f1")).get("localization"))
    results: list[dict[str, object]] = [
        {"label": label, "value": report.get(key), "unit": unit}
        for label, key, unit in (
            ("evaluated_samples", "total", "number"),
            ("api_or_evaluation_errors", "api_or_evaluation_errors", "number"),
            ("mean_latency_seconds", "mean_latency_seconds", "seconds"),
            ("mean_overall_score", "mean_overall_score", "ratio"),
        )
    ]
    results.extend(
        {"label": label, "value": source.get(key), "unit": "ratio"}
        for label, source, key in (
            ("label_accuracy", scores, "accuracy"),
            ("macro_f1", scores, "macro_f1"),
            ("box_f1", localization, "f1"),
            ("box_precision", localization, "precision"),
            ("box_recall", localization, "recall"),
        )
    )
    classes = fields(scores.get("per_class"))
    results.extend(
        {
            "label": f"{label}_{key}",
            "value": fields(classes.get(label)).get(key),
            "unit": "ratio",
        }
        for label in ("pass", "defect", "fail")
        for key in ("precision", "recall")
    )
    return report_metrics({"metrics": results})


def restore_dataset_file_statistics(
    files: list[ObservationDatasetFile], raw: dict, outputs: dict[str, str]
) -> list[ObservationDatasetFile]:
    """Recover file-level counts from the older split-indexed report shape."""
    by_split = raw.get("distribution")
    if not isinstance(by_split, dict):
        return files
    raw_files = {
        item["path"]: item
        for item in raw.get("files", [])
        if isinstance(item, dict) and isinstance(item.get("path"), str)
    }
    for index, file in enumerate(files):
        if file.distributions.get("label"):
            continue
        item = raw_files.get(file.path)
        if (
            not item
            or item.get("sample_count") != file.sample_count
            or file.sample_count is None
        ):
            continue
        splits = {
            key.removesuffix("_by_label")
            for key in by_split
            if key.endswith("_by_label")
        }
        candidates = [
            split
            for split in splits
            if {
                path
                for name, path in outputs.items()
                if name == split or name.startswith(split + "_")
            }
            == {file.path}
        ]
        if len(candidates) != 1:
            continue
        labels = by_split[candidates[0] + "_by_label"]
        if (
            not isinstance(labels, dict)
            or not labels
            or not all(
                isinstance(k, str) and k and type(v) is int and v >= 0
                for k, v in labels.items()
            )
            or sum(labels.values()) != file.sample_count
        ):
            continue
        size = item.get("size_bytes", item.get("bytes"))
        files[index] = file.model_copy(
            update={
                "distributions": {**file.distributions, "label": labels},
                "size_bytes": file.size_bytes
                if file.size_bytes is not None
                else size
                if type(size) is int and size >= 0
                else None,
            }
        )
    return files


def dataset_report_checks(body: dict) -> dict[str, bool]:
    raw = body.get("checks")
    if not isinstance(raw, dict):
        return {}
    return {
        key: value
        for key, value in list(raw.items())[:32]
        if isinstance(key, str) and 0 < len(key) <= 100 and isinstance(value, bool)
    }


def dataset_report_files(
    body: dict, *, issues: list[ReportIssue] | None = None, explicit_only: bool = False
) -> list[ObservationDatasetFile]:
    def count(value: object, field: str) -> int | None:
        if type(value) is int and value >= 0:
            return value
        if type(value) is float and value >= 0 and value.is_integer():
            return int(value)
        if (
            isinstance(value, str)
            and len(value) <= 20
            and value.isascii()
            and value.isdecimal()
        ):
            return int(value)
        if issues is not None and value is not None:
            issues.append(ReportIssue(field=field[:240], code="invalid_count"))
        return None

    def make_file(item: object, field: str) -> ObservationDatasetFile | None:
        item = item if isinstance(item, dict) else {}
        path = item.get("path")
        if not isinstance(path, str) or len(path) > 2048:
            if issues is not None:
                issues.append(ReportIssue(field=f"{field}.path", code="invalid_file"))
            return None
        try:
            path = normalize_workspace_path(path)
            to_storage_api_path(path)
        except ValueError:
            if issues is not None:
                issues.append(ReportIssue(field=f"{field}.path", code="invalid_file"))
            return None
        name = item.get("name")
        raw = item.get("distributions")
        distributions = {}
        if isinstance(raw, dict):
            for group, labels in list(raw.items())[:24]:
                if isinstance(labels, dict):
                    counts = {
                        label: hits
                        for label, value in labels.items()
                        if (
                            hits := count(
                                value, f"{field}.distributions.{group}.{label}"
                            )
                        )
                        is not None
                    }
                    if counts:
                        distributions[group] = counts
        return ObservationDatasetFile(
            name=name[:128]
            if isinstance(name, str) and name
            else PurePosixPath(path).name,
            path=path,
            size_bytes=count(item.get("size_bytes"), f"{field}.size_bytes"),
            sample_count=count(item.get("sample_count"), f"{field}.sample_count"),
            distributions=distributions,
        )

    entries = body.get("files")
    if isinstance(entries, list):
        explicit_files = [
            file
            for index, item in enumerate(entries[:32])
            if (file := make_file(item, f"files[{index}]"))
        ]
        if explicit_files or explicit_only:
            return explicit_files
    if explicit_only:
        return []
    splits = body.get("splits")
    if isinstance(splits, dict):
        split_files = [
            file
            for index, (name, item) in enumerate(list(splits.items())[:32])
            if isinstance(item, dict)
            and (
                file := make_file(
                    {
                        "name": name,
                        "path": item.get("file"),
                        "sample_count": item.get("n_records"),
                        "size_bytes": item.get("size_bytes"),
                        "distributions": {
                            "label": item.get("labels"),
                            "category": item.get("categories"),
                        },
                    },
                    f"splits[{index}]",
                )
            )
        ]
        if split_files:
            return split_files
    train_data = body.get("train_data")
    train_data = train_data if isinstance(train_data, dict) else {}
    sources = body.get("sources")
    sources = sources if isinstance(sources, dict) else {}
    metrics = {
        metric.label: int(metric.value)
        for metric in report_metrics(body)
        if metric.unit in {"number", "count"}
        and metric.value >= 0
        and metric.value.is_integer()
    }
    files = []
    for split in ("train", "val"):
        source = sources.get(split) or sources.get(f"{split}_jsonl")
        if source is None and split == "val":
            source = sources.get("validation")
        metadata = source if isinstance(source, dict) else {"path": source}
        path = train_data.get(f"{split}_jsonl") or metadata.get("path")
        if path is None:
            continue
        labels = {
            label: metrics[key]
            for label in ("pass", "defect", "fail")
            if (key := f"{split}_{label}") in metrics
        }
        file = make_file(
            {
                "name": split,
                "path": path,
                "size_bytes": train_data.get(
                    f"{split}_size_bytes", metadata.get("size_bytes")
                ),
                "sample_count": metadata.get(
                    "sample_count", metrics.get(f"{split}_records")
                ),
                "distributions": metadata.get("distributions", {"label": labels}),
            },
            split,
        )
        if file:
            files.append(file)
    return files
