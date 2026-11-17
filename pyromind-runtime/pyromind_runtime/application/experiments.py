"""Project existing execution evidence without scheduling or selecting models."""

from __future__ import annotations

import math
from collections.abc import Iterator
from typing import cast

from pyromind_runtime.domain.content import JsonObject
from pyromind_runtime.domain.experiments import (
    ExperimentArtifact,
    ExperimentCatalog,
    ExperimentEvaluation,
    ExperimentExecution,
    ExperimentMetric,
    ExperimentVersion,
)
from pyromind_runtime.domain.pipeline import PipelineRun
from pyromind_runtime.domain.snapshot import ExternalTaskState
from pyromind_runtime.domain.storage_paths import to_storage_api_path
from pyromind_runtime.domain.training_evaluation import (
    CheckpointArtifact,
    StepEvaluation,
)
from pyromind_runtime.domain.versions import (
    ArtifactVersion,
    VersionProject,
    VersionRun,
    fingerprint,
)
from pyromind_runtime.domain.workflow_execution import WorkflowExecutionRequest


def _object(value: object) -> JsonObject:
    return cast(JsonObject, value) if isinstance(value, dict) else {}


def _string(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _reference(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    try:
        return to_storage_api_path(value)
    except ValueError:
        return None


def _artifact(version: ArtifactVersion) -> ExperimentArtifact:
    return ExperimentArtifact(
        id=version.id,
        kind=version.kind,
        state=version.state,
        storage_ref=version.manifest.storage_ref,
        digest=version.manifest.digest,
        parents=tuple(version.manifest.parents),
        output_role=version.manifest.output_role,
        producer_run_id=version.producer_run_id,
        prediction=_object(version.manifest.summary.get("prediction"))
        if version.registration_key.startswith("training-artifact:")
        else None,
    )


def experiment_catalog(
    project: VersionProject,
    runs: list[VersionRun],
    artifacts: list[ArtifactVersion],
    requests: dict[str, WorkflowExecutionRequest],
    pipelines: dict[str, PipelineRun],
    tasks: tuple[ExternalTaskState, ...],
    *,
    project_id: str | None = None,
    archives: dict[str, str] | None = None,
) -> ExperimentCatalog:
    """Group workflow attempts by iteration and attach evaluations by provenance."""
    archives = archives or {}
    if any(run.project_id != project.id for run in runs) or any(
        artifact.project_id != project.id for artifact in artifacts
    ):
        raise ValueError("experiment evidence belongs to another project")
    if any(
        run.conversation_id != project.conversation_id for run in pipelines.values()
    ):
        raise ValueError("pipeline belongs to another conversation")
    if any(
        request.conversation_id != project.conversation_id
        for request in requests.values()
    ):
        raise ValueError("workflow belongs to another conversation")
    versions = {version.id: version for version in artifacts}
    task_map = {task.task_id: task for task in tasks}
    workflows = [run for run in runs if run.source.kind == "workflow"]
    attempts_by_iteration: dict[str, list[VersionRun]] = {}
    for run in workflows:
        attempts_by_iteration.setdefault(run.spec.iteration_id, []).append(run)
    experiment_by_run = {
        run.id: attempts[0].id
        for attempts in attempts_by_iteration.values()
        for run in attempts
    }

    def execution(run: VersionRun) -> ExperimentExecution:
        request = requests.get(run.source.source_id)
        pipeline = pipelines.get(run.source.source_id)
        task_id = run.source.task_id or (
            request.task_id if request else pipeline.task_id if pipeline else None
        )
        task = task_map.get(task_id) if task_id else None
        return ExperimentExecution(
            run_id=run.id,
            task_id=task_id,
            status=task.status
            if task
            else request.status
            if request
            else pipeline.submission_status
            if pipeline
            else "unknown",
            output_dir=task.output_dir
            if task
            else pipeline.output_dir
            if pipeline
            else None,
            archive_ref=archives.get(run.id),
        )

    workflow_tasks = {
        identity: run.id
        for run in workflows
        if (identity := execution(run).task_id) is not None
    }
    pipeline_runs = {
        run.source.source_id: run
        for run in runs
        if run.source.kind == "pipeline"
        and (pipeline := pipelines.get(run.source.source_id)) is not None
        and pipeline.execution_revision == run.source.execution_revision
    }
    evaluations = {
        identity: pipeline
        for identity, pipeline in pipelines.items()
        if identity in pipeline_runs
        and (
            pipeline.definition.get("kind") == "prediction_evaluation"
            or isinstance(pipeline.definition.get("inference"), dict)
        )
    }
    assignments: dict[str, str] = {}
    for identity, pipeline in evaluations.items():
        source_task = _string(
            _object(pipeline.definition.get("source_manifest")).get("task_id")
        )
        if source_task in workflow_tasks:
            assignments[identity] = workflow_tasks[source_task]
        checkpoint = versions.get(
            pipeline_runs[identity].spec.inputs.get("source_checkpoint", "")
        )
        if checkpoint and checkpoint.producer_run_id in experiment_by_run:
            assignments[identity] = checkpoint.producer_run_id

    # Selected-model outputs can come from a scoring run rather than training.
    for _ in range(len(evaluations)):
        changed = False
        producers = {run.id: run.id for run in workflows} | {
            pipeline_runs[identity].id: owner for identity, owner in assignments.items()
        }
        models: dict[str, set[str]] = {}
        for artifact in artifacts:
            path = _reference(artifact.manifest.storage_ref)
            owner = producers.get(artifact.producer_run_id or "")
            if artifact.kind == "model" and path and owner:
                models.setdefault(path, set()).add(owner)
        for identity, pipeline in evaluations.items():
            if identity in assignments:
                continue
            inference = _object(pipeline.definition.get("inference"))
            selection_id = _string(inference.get("selection_run_id"))
            candidates = (
                {assignments[selection_id]}
                if selection_id in assignments
                else models.get(_reference(inference.get("model_path")) or "", set())
            )
            if len(candidates) == 1:
                assignments[identity] = next(iter(candidates))
                changed = True
        if not changed:
            break

    def evaluation(identity: str, pipeline: PipelineRun) -> ExperimentEvaluation:
        record = pipeline_runs[identity]
        source = _object(pipeline.definition.get("inference"))
        scoring = pipeline.definition.get("kind") == "prediction_evaluation"
        delivery = _object(pipeline.result.get("inference_delivery"))
        task = task_map.get(pipeline.task_id or "")
        return ExperimentEvaluation(
            **execution(record).model_dump(),
            operation="score" if scoring else "inference",
            training_run_id=assignments.get(identity),
            report_path=_string(delivery.get("report_path")),
            prediction_path=_string(delivery.get("predictions_path")),
            metrics_path=_string(delivery.get("metrics_path")),
            evaluation_status=_string(delivery.get("evaluation_status")),
            error_summary=task.error_summary if task else None,
            model_path=_string(source.get("model_path")),
            dataset_role=str(pipeline.definition.get("dataset_role", "unknown")),
            steps=pipeline.evaluations,
            selection=pipeline.result if scoring and pipeline.evaluations else {},
        )

    delivered_evaluations: dict[str, list[ExperimentEvaluation]] = {}
    for artifact in artifacts:
        owner = artifact.producer_run_id
        if (
            owner not in experiment_by_run
            or not artifact.registration_key.startswith("eval-delivery:")
            or artifact.manifest.output_role != "eval_delivery"
            or artifact.state != "ready"
        ):
            continue
        data = artifact.manifest.summary
        group = fingerprint(
            [data["dataset_role"], data["dataset_sha256"], data["scoring_sha256"]]
        )
        steps = tuple(
            StepEvaluation(
                step=row["step"],
                run_id=f"{artifact.id}:{row['step']}",
                output_dir=data["report_path"].rsplit("/", 1)[0],
                dataset_role=data["dataset_role"],
                metrics=row["metrics"],
                comparison_group=group,
                eligible=True,
                checkpoint=CheckpointArtifact(
                    step=row["step"], path=row["checkpoint_path"], complete=True
                ),
            )
            for row in data["steps"]
        )
        selected = next(
            (step for step in steps if step.step == data.get("selected_step")), None
        )
        delivered_evaluations.setdefault(owner, []).append(
            ExperimentEvaluation(
                run_id=artifact.id,
                status="succeeded",
                operation="score",
                training_run_id=owner,
                report_path=data["report_path"],
                dataset_role=data["dataset_role"],
                steps=steps,
                output_dir=steps[0].output_dir,
                archive_ref=artifact.manifest.storage_ref,
                selection={
                    "selected_step": selected.step,
                    "selected_run_id": selected.run_id,
                }
                if selected
                else {},
            )
        )

    model_producers: dict[str, set[str]] = {}
    producer_experiments = experiment_by_run | {
        pipeline_runs[identity].id: experiment_by_run[owner]
        for identity, owner in assignments.items()
    }
    for artifact in artifacts:
        path = _reference(artifact.manifest.storage_ref)
        owner = producer_experiments.get(artifact.producer_run_id or "")
        if artifact.kind == "model" and path and owner:
            model_producers.setdefault(path, set()).add(owner)
    views = []
    preceding: set[str] = set()
    for number, iteration in enumerate(project.context.iterations, 1):
        attempts = attempts_by_iteration.get(iteration.id)
        if not attempts:
            continue
        experiment_id = attempts[0].id
        run = attempts[-1]
        run_ids = {attempt.id for attempt in attempts}
        inputs = {
            role: _artifact(versions[identity])
            for role, identity in run.spec.inputs.items()
            if identity in versions
        }
        parents: set[str] = set()
        for identity in {
            identity
            for attempt in attempts
            for identity in attempt.spec.inputs.values()
        }:
            artifact = versions.get(identity)
            if artifact is None or artifact.kind != "model":
                continue
            path = _reference(artifact.manifest.storage_ref)
            producers = model_producers.get(path or "", set())
            if len(producers) == 1 and producers <= preceding:
                parents.update(producers)
        related = tuple(
            evaluation(identity, pipeline)
            for identity, pipeline in evaluations.items()
            if assignments.get(identity) in run_ids
        ) + tuple(
            delivery
            for attempt in attempts
            for delivery in delivered_evaluations.get(attempt.id, ())
        )
        owners = run_ids | {
            pipeline_runs[identity].id
            for identity, owner in assignments.items()
            if owner in run_ids
        }
        views.append(
            ExperimentVersion(
                experiment_id=experiment_id,
                version=f"v{number}",
                project_id=project_id,
                artifact_project_id=project.id,
                conversation_id=project.conversation_id,
                iteration_id=iteration.id,
                created_at=attempts[0].created_at,
                directory_ref=(
                    f"/.pyromind-agent/{project.conversation_id}/projects/"
                    f"{project_id or project.id}/experiments/v{number}"
                ),
                parent_experiment_ids=tuple(sorted(parents)),
                execution=execution(run),
                attempts=tuple(execution(attempt) for attempt in attempts),
                inputs=inputs,
                outputs=tuple(
                    _artifact(artifact)
                    for artifact in artifacts
                    if artifact.producer_run_id in owners
                ),
                evaluations=related,
                issues=tuple(
                    "reference_only_input:" + role
                    for role, value in inputs.items()
                    if value.state == "reference_only"
                ),
            )
        )
        preceding.add(experiment_id)
    return ExperimentCatalog(
        conversation_id=project.conversation_id,
        project_id=project_id,
        artifact_project_id=project.id,
        versions=tuple(views),
        unassigned_evaluations=tuple(
            evaluation(identity, pipeline)
            for identity, pipeline in evaluations.items()
            if identity not in assignments
        ),
    )


def _metrics(value: JsonObject, prefix: str = "") -> Iterator[tuple[str, float]]:
    for name, item in value.items():
        key = prefix + name
        if isinstance(item, dict):
            yield from _metrics(item, key + ".")
        elif (
            not isinstance(item, bool)
            and isinstance(item, (float, int))
            and math.isfinite(item)
        ):
            yield key, float(item)


def experiment_metrics(catalog: ExperimentCatalog) -> tuple[ExperimentMetric, ...]:
    rows = []
    for version in catalog.versions:
        for evaluation in version.evaluations:
            if evaluation.status != "succeeded":
                continue
            for step in evaluation.steps:
                if not step.eligible or not step.comparison_group:
                    continue
                for name, value in _metrics(step.metrics):
                    rows.append(
                        ExperimentMetric(
                            experiment_id=version.experiment_id,
                            version=version.version,
                            evaluation_run_id=evaluation.run_id,
                            step_run_id=step.run_id,
                            step=step.step,
                            role=step.role,
                            dataset_role=step.dataset_role,
                            comparison_group=step.comparison_group,
                            metric=name,
                            value=value,
                            selected=(
                                step.dataset_role == "validation"
                                and step.role == "checkpoint"
                                and step.checkpoint is not None
                                and step.checkpoint.complete
                                and step.checkpoint.step == step.step
                                and evaluation.selection.get("selected_step")
                                == step.step
                                and evaluation.selection.get("selected_run_id")
                                == step.run_id
                            ),
                        )
                    )
    return tuple(rows)
