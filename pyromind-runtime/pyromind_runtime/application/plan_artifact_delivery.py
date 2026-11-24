"""Register approved plan outputs and freeze a report for their execution."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import PurePosixPath

from pyromind_runtime.application.project_observations import (
    baseline_evaluation_metrics,
    dataset_report_files,
)
from pyromind_runtime.application.stage_analysis import (
    artifact_belongs_to_execution,
    report_metrics,
)
from pyromind_runtime.application.versions import Versions
from pyromind_runtime.domain.content import JsonObject
from pyromind_runtime.domain.context import RequestContext
from pyromind_runtime.domain.experiment_execution import ExperimentExecutionView
from pyromind_runtime.domain.experiment_plans import ExperimentTask
from pyromind_runtime.domain.observations import ObservationDatasetFile
from pyromind_runtime.domain.storage_paths import (
    normalize_workspace_path,
    to_storage_api_path,
)
from pyromind_runtime.domain.versions import (
    ArtifactKind,
    ArtifactVersion,
    SaveVersion,
    SnapshotSource,
    VersionParent,
    canonical,
    fingerprint,
)
from pyromind_runtime.ports.storage import EvaluationStorage


DELIVERY_STAGE_KINDS = frozenset({"init", "dataset", "train", "eval"})


class InvalidPlanDelivery(ValueError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def delivery_key(run: ExperimentExecutionView, name: str) -> str:
    return "plan-delivery:" + fingerprint([run.plan_sequence, run.run_id, name])


def delivery_report(
    run: ExperimentExecutionView,
    row: Mapping,
    artifacts: tuple[ArtifactVersion, ...],
    producers: Mapping[str, str | None],
) -> ArtifactVersion | None:
    selected = row.get("report_version_id")
    eligible = [
        item
        for item in artifacts
        if item.state == "ready"
        and item.kind in {"report", "evaluation"}
        and artifact_belongs_to_execution(item, run, producers)
    ]
    return next(
        (item for item in eligible if item.id == selected),
        next(
            (
                item
                for item in eligible
                if item.registration_key == delivery_key(run, "report")
            ),
            None,
        ),
    )


def register_plan_delivery(
    versions: Versions,
    storage: EvaluationStorage,
    task: ExperimentTask,
    run: ExperimentExecutionView,
    row: Mapping,
    scope: str,
    context: RequestContext,
) -> None:
    project = versions.project(scope, context)
    assert project is not None
    artifacts = tuple(versions.store.list_versions(project.id))
    producers = {
        item.id: item.source.task_id for item in versions.store.list_runs(project.id)
    }
    if delivery_report(run, row, artifacts, producers):
        return

    saved = {item.registration_key: item for item in artifacts}
    previous = saved.get(delivery_key(run, "report"))
    if previous is not None:
        versions.finish(previous, context)
        return

    outputs = row.get("verified_outputs")
    if not task.outputs or not isinstance(outputs, dict):
        raise InvalidPlanDelivery("missing_output")
    owned = {
        item.id: item
        for item in artifacts
        if item.state in {"ready", "reference_only"}
        and artifact_belongs_to_execution(item, run, producers)
    }
    paths: dict[str, str] = {}
    replacements: dict[str, ArtifactVersion] = {}
    for output in task.outputs:
        value = outputs.get(output.name)
        if not isinstance(value, str) or not value:
            raise InvalidPlanDelivery("missing_output")
        if value in owned:
            artifact = owned[value]
            replacements[output.name] = artifact
            value = (
                artifact.manifest.source.storage_ref or artifact.manifest.storage_ref
            )
        if not isinstance(value, str):
            raise InvalidPlanDelivery("invalid_output")
        try:
            paths[output.name] = (
                to_storage_api_path(value)
                if output.name in replacements
                else to_storage_api_path(normalize_workspace_path(value))
            )
        except ValueError as exc:
            raise InvalidPlanDelivery("invalid_output") from exc

    stage = (run.stage_snapshot or {}).get("kind")
    document = _read_report(storage, stage, paths)
    directories: dict[str, set[str]] = {}
    check_paths = {path for name, path in paths.items() if name not in replacements}
    if stage == "dataset":
        check_paths.update(
            to_storage_api_path(file.path) for file in dataset_report_files(document)
        )
    for path in sorted(check_paths):
        parent = str(PurePosixPath(path).parent)
        if parent not in directories:
            directories[parent] = storage.list(parent)
        if PurePosixPath(path).name not in directories[parent]:
            raise InvalidPlanDelivery("missing_output")

    binding: JsonObject = {"plan_execution": [run.plan_sequence, run.run_id]}
    parents = []
    for name, path in paths.items():
        prior = replacements.get(name) or next(
            (
                item
                for item in owned.values()
                if item.manifest.source.storage_ref in (path, f"/workspace{path}")
            ),
            None,
        )
        if prior is None:
            prior = versions.save(
                project,
                delivery_key(run, f"output:{name}"),
                SaveVersion(
                    kind=_output_kind(stage, name),
                    logical_key=f"{task.id}/{name}",
                    source=SnapshotSource(storage_ref=path, reference_only=True),
                    summary=binding,
                    output_role=name,
                ),
                context,
            )
        parents.append(VersionParent(version_id=prior.id, relation=name))
    versions.save(
        project,
        delivery_key(run, "report"),
        SaveVersion(
            kind="report",
            logical_key=f"{task.id}/delivery-report",
            source=SnapshotSource(text=canonical(document), filename="report.json"),
            parents=parents,
            summary=binding,
        ),
        context,
    )


def _output_kind(stage: object, name: str) -> ArtifactKind:
    if name.endswith(("report", "metrics")):
        return "evaluation" if stage in ("init", "eval") else "report"
    if stage == "dataset":
        return "dataset"
    if stage == "train" and ("model" in name or "checkpoint" in name):
        return "model"
    return "config"


def _read_report(
    storage: EvaluationStorage, stage: object, outputs: dict[str, str]
) -> JsonObject:
    reports = [
        path
        for name, path in outputs.items()
        if name.endswith("report") and path.endswith(".json")
    ]
    candidates = reports or [
        path
        for name, path in outputs.items()
        if name.endswith(("metrics", "result")) and path.endswith(".json")
    ]
    body = {}
    if candidates:
        try:
            source = json.loads(storage.read(candidates[0]))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise InvalidPlanDelivery("invalid_report") from exc
        if not isinstance(source, dict):
            raise InvalidPlanDelivery("invalid_report")
        body = source
    metrics = report_metrics(body)
    if not metrics and stage in ("init", "eval"):
        metrics = baseline_evaluation_metrics(body)
    document: JsonObject = {
        "metrics": [item.model_dump(mode="json") for item in metrics]
    }
    if isinstance(summary := body.get("summary"), str):
        summary = summary.strip()
        if 0 < len(summary) <= 900:
            document["summary"] = summary
    if stage == "dataset":
        files = dataset_report_files(body)
        dataset_outputs = [
            (name, path)
            for name, path in outputs.items()
            if path.endswith((".jsonl", ".csv", ".parquet"))
        ]
        if not files and len(dataset_outputs) == 1:
            name, path = dataset_outputs[0]
            records = body.get("n_train_records")
            if (
                body.get("report_type") == "dataset_preparation"
                and (name == "train" or name.startswith("train_"))
                and type(records) is int
                and records >= 0
            ):
                files = dataset_report_files(
                    {
                        "files": [
                            {
                                "name": name,
                                "path": f"/workspace{path}",
                                "sample_count": records,
                                "distributions": {
                                    "label": body.get("label_distribution"),
                                    "category": body.get("category_distribution"),
                                },
                            }
                        ]
                    }
                )
        if not files and any(name.endswith("report") for name in outputs):
            raise InvalidPlanDelivery("invalid_report")
        if not files:
            files = [
                ObservationDatasetFile(name=name, path=f"/workspace{path}")
                for name, path in dataset_outputs
            ]
        document["files"] = [item.model_dump(mode="json") for item in files]
    return document
