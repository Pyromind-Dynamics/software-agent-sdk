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
from pyromind_runtime.domain.stage_analysis import StageAnalysisMetric
from pyromind_runtime.domain.storage_paths import (
    normalize_workspace_path,
    to_storage_api_path,
)
from pyromind_runtime.domain.versions import ArtifactVersion, fingerprint


_TRAINING_METRICS = tuple(
    MetricConfig(key=key, title=key, axis="training_step", axis_key="step")
    for key in ("loss", "eval_loss", "learning_rate", "reward")
)


def stage_report_runs(
    snapshot: ConversationSnapshot, project_id: str, node_id: str, kind: str
) -> dict[str, list[ExperimentExecutionView]]:
    result: dict[str, list[ExperimentExecutionView]] = {}
    for run in reversed(snapshot.experiment_executions or ()):
        if (
            run.project_id != project_id
            or run.stage_id != node_id
            or run.role_id != "main"
            or run.report_state != "ready"
            or run.plan_sequence is None
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
            (task.kind, current_runs.get(task.id))
            for task in project.plan.tasks
            if task.stage_id == stage.id
        ]
        states = {run.state if run else "queued" for _, run in stage_runs}
        execution_states = {
            run.state if run else "queued"
            for kind, run in stage_runs
            if kind != "analysis"
        } or states
        if states & {"failed", "cancelled", "needs_input"}:
            status = next(
                state
                for state in ("failed", "cancelled", "needs_input")
                if state in states
            )
        elif states == {"succeeded"}:
            status = "succeeded"
        elif execution_states and execution_states <= {"succeeded", "waiting_analysis"}:
            status = "waiting_analysis"
        elif states - {"queued"}:
            status = "running"
        else:
            status = "queued"
        report_ids = tuple(
            dict.fromkeys(
                artifact_id
                for _, run in stage_runs
                if run and run.report_state == "ready"
                for artifact in run.artifacts
                if isinstance(artifact_id := artifact.get("artifact_id"), str)
            )
        )
        history = (
            stage_report_runs(snapshot, project_id, stage.id, stage.kind)
            if stage.kind in ROLES
            else {}
        )
        delivery_error = next(
            (
                run.delivery_error
                for _, run in stage_runs
                if run and run.report_state == "registration_failed"
            ),
            None,
        )
        delivery_status = (
            "failed"
            if delivery_error
            else "registering"
            if any(run and run.report_state == "registering" for _, run in stage_runs)
            else None
        )
        nodes.append(
            ObservationNode(
                id=stage.id,
                title=stage.name,
                type=stage.kind or "",
                business_status=status,
                delivery_status=delivery_status,
                delivery_error=delivery_error,
                current_run_ids=tuple(run.run_id for _, run in stage_runs if run),
                report_version_ids=report_ids,
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
        report_ids = job.report_ids
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
    baseline_task_ids: set[str] = set()
    readable = []
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
        if analysis_job_id is None and run is None:
            continue
        readable.append(artifact)
        if node.type == "init" and run is not None and run.platform_task_id:
            baseline_task_ids.add(run.platform_task_id)
    with ThreadPoolExecutor(max_workers=4) as pool:
        baseline = (
            pool.submit(load_baseline_metrics, next(iter(baseline_task_ids)))
            if load_baseline_metrics and len(baseline_task_ids) == 1
            else None
        )
        contents = pool.map(
            lambda artifact: versions.snapshots.read_text(
                snapshot_result(artifact), context
            ),
            readable,
        )
        for artifact, text in zip(readable, contents, strict=True):
            if len(text.encode()) > 48 * 1024:
                continue
            read_ids.append(artifact.id)
            try:
                body = json.loads(text)
            except ValueError:
                continue
            for metric in report_metrics(body):
                if metric not in metrics:
                    metrics.append(metric)
            if isinstance(body, dict):
                if node.type == "dataset":
                    files.update(
                        (file.path, file) for file in dataset_report_files(body)
                    )
                elif isinstance(summary := body.get("summary"), str):
                    summary = summary.strip()
                    if 0 < len(summary) <= 900 and summary not in summaries:
                        summaries.append(summary)
        if baseline is not None and read_ids:
            registered_labels = {metric.label for metric in metrics}
            metrics.extend(
                metric
                for metric in baseline.result()
                if metric.label not in registered_labels
            )
    if report_version_id is not None and not read_ids:
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


def dataset_report_files(body: dict) -> list[ObservationDatasetFile]:
    def count(value: object) -> int | None:
        if type(value) is int and value >= 0:
            return value
        return None

    def make_file(item: dict) -> ObservationDatasetFile | None:
        path = item.get("path")
        if not isinstance(path, str) or len(path) > 2048:
            return None
        try:
            path = normalize_workspace_path(path)
            to_storage_api_path(path)
        except ValueError:
            return None
        name = item.get("name")
        raw = item.get("distributions")
        distributions = {}
        if isinstance(raw, dict):
            for group, labels in list(raw.items())[:24]:
                if isinstance(labels, dict):
                    counts = {
                        label: hits
                        for label, value in list(labels.items())[:100]
                        if (hits := count(value)) is not None
                    }
                    if counts:
                        distributions[group] = counts
        return ObservationDatasetFile(
            name=name[:128]
            if isinstance(name, str) and name
            else PurePosixPath(path).name,
            path=path,
            size_bytes=count(item.get("size_bytes")),
            sample_count=count(item.get("sample_count")),
            distributions=distributions,
        )

    entries = body.get("files")
    if isinstance(entries, list):
        explicit_files = [
            file
            for item in entries[:32]
            if isinstance(item, dict) and (file := make_file(item))
        ]
        if explicit_files:
            return explicit_files
    splits = body.get("splits")
    if isinstance(splits, dict):
        split_files = [
            file
            for name, item in list(splits.items())[:32]
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
                    }
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
        if metric.unit == "number" and metric.value >= 0 and metric.value.is_integer()
    }
    files = []
    for split in ("train", "val"):
        source = sources.get(split) or sources.get(f"{split}_jsonl")
        if source is None and split == "val":
            source = sources.get("validation")
        metadata = source if isinstance(source, dict) else {"path": source}
        labels = {
            label: metrics[key]
            for label in ("pass", "defect", "fail")
            if (key := f"{split}_{label}") in metrics
        }
        file = make_file(
            {
                "name": split,
                "path": train_data.get(f"{split}_jsonl") or metadata.get("path"),
                "size_bytes": train_data.get(
                    f"{split}_size_bytes", metadata.get("size_bytes")
                ),
                "sample_count": metadata.get(
                    "sample_count", metrics.get(f"{split}_records")
                ),
                "distributions": metadata.get("distributions", {"label": labels}),
            }
        )
        if file:
            files.append(file)
    return files
