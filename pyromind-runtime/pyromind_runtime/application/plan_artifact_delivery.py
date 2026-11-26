"""Register approved plan outputs and freeze a report for their execution."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import PurePosixPath

from pyromind_runtime.application.project_observations import (
    baseline_evaluation_metrics,
    dataset_report_checks,
    dataset_report_files,
)
from pyromind_runtime.application.stage_analysis import (
    artifact_belongs_to_execution,
    report_metrics,
)
from pyromind_runtime.application.versions import Versions, snapshot_result
from pyromind_runtime.domain.content import JsonObject
from pyromind_runtime.domain.context import RequestContext
from pyromind_runtime.domain.experiment_execution import ExperimentExecutionView
from pyromind_runtime.domain.experiment_plans import ExperimentTask
from pyromind_runtime.domain.observations import ObservationDatasetFile
from pyromind_runtime.domain.stage_analysis import ReportIssue
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
from pyromind_runtime.ports.storage import (
    EvaluationStorage,
    EvaluationStorageTooLargeError,
)


DELIVERY_STAGE_KINDS = frozenset({"init", "dataset", "train", "eval"})
DATASET_REPORT_FORMAT_VERSION = 2
MERGED_DATASET_OUTPUTS = {"train_messages": "train", "val_messages": "val"}
REBUILT_REPORT = "report:rebuild"


class InvalidPlanDelivery(ValueError):
    def __init__(self, code: str, field: str | None = None) -> None:
        super().__init__(code)
        self.code = code
        self.field = field


def delivery_key(run: ExperimentExecutionView, name: str) -> str:
    return "plan-delivery:" + fingerprint([run.plan_sequence, run.run_id, name])


def delivery_report(
    run: ExperimentExecutionView,
    row: Mapping,
    artifacts: tuple[ArtifactVersion, ...],
    producers: Mapping[str, str | None],
    *,
    declared_report: bool = False,
) -> ArtifactVersion | None:
    selected = row.get("report_version_id")
    eligible = [
        item
        for item in artifacts
        if item.state == "ready"
        and item.kind in {"report", "evaluation"}
        and artifact_belongs_to_execution(item, run, producers)
    ]
    by_key = {item.registration_key: item for item in eligible}
    registered = by_key.get(delivery_key(run, "report"))
    if declared_report:
        return registered
    rebuilt = by_key.get(delivery_key(run, REBUILT_REPORT))
    return rebuilt or next(
        (item for item in eligible if item.id == selected), registered
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
    report_output = task.report_output
    if delivery_report(
        run, row, artifacts, producers, declared_report=report_output is not None
    ):
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
            raise InvalidPlanDelivery(
                "missing_output", f"verified_outputs.{output.name}"[:240]
            )
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
    kinds: dict[str, ArtifactKind] = {
        output.name: output.kind for output in task.outputs if output.kind is not None
    }
    document = _read_report(
        storage,
        stage,
        paths,
        report_name=report_output.name if report_output else None,
        typed_outputs=kinds,
    )
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
                    kind=kinds.get(name) or _output_kind(stage, name),
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


def rebuild_legacy_report(
    versions: Versions,
    storage: EvaluationStorage,
    tasks: Mapping[str, ExperimentTask],
    runs: tuple[ExperimentExecutionView, ...],
    report_id: str,
    scope: str,
    context: RequestContext,
) -> tuple[ExperimentExecutionView, ArtifactVersion]:
    project = versions.project(scope, context, create=False)
    if project is None:
        raise FileNotFoundError("No project versions")
    artifacts = {item.id: item for item in versions.store.list_versions(project.id)}
    previous = artifacts.get(report_id)
    if previous is None or previous.state != "ready" or previous.kind != "report":
        raise FileNotFoundError("No registered report")
    run = next(
        (
            item
            for item in runs
            if previous.registration_key == delivery_key(item, "report")
            and artifact_belongs_to_execution(previous, item, {})
        ),
        None,
    )
    task = tasks.get(run.task_id) if run is not None else None
    if run is None or task is None or task.report_output is not None:
        raise InvalidPlanDelivery("invalid_output")
    key = delivery_key(run, REBUILT_REPORT)
    rebuilt = next(
        (item for item in artifacts.values() if item.registration_key == key), None
    )
    if rebuilt is not None:
        return run, versions.finish(rebuilt, context)
    if not any(item.get("artifact_id") == report_id for item in run.artifacts):
        raise InvalidPlanDelivery("invalid_output")
    body = json.loads(versions.snapshots.read_text(snapshot_result(previous), context))
    if not isinstance(body, dict) or body.get("metrics") != []:
        raise InvalidPlanDelivery("invalid_report")

    outputs = {}
    names = {output.name for output in task.outputs}
    for parent in previous.manifest.parents:
        source = artifacts.get(parent.version_id)
        if (
            parent.relation not in names
            or source is None
            or source.state not in {"ready", "reference_only"}
            or not artifact_belongs_to_execution(source, run, {})
        ):
            continue
        path = source.manifest.source.storage_ref or source.manifest.storage_ref
        if path:
            outputs[parent.relation] = to_storage_api_path(path)
    document = _read_report(storage, (run.stage_snapshot or {}).get("kind"), outputs)
    if not document["metrics"]:
        raise InvalidPlanDelivery("invalid_report_metrics")
    rebuilt = versions.save(
        project,
        key,
        SaveVersion(
            kind="report",
            logical_key=previous.logical_key,
            source=SnapshotSource(text=canonical(document), filename="report.json"),
            parents=[
                *previous.manifest.parents,
                VersionParent(version_id=report_id, relation="supersedes"),
            ],
            summary=previous.manifest.summary,
        ),
        context,
    )
    return run, rebuilt


def _output_kind(stage: object, name: str) -> ArtifactKind:
    if name.endswith(("report", "metrics")):
        return "evaluation" if stage in ("init", "eval") else "report"
    if stage == "dataset":
        return "dataset"
    if stage == "train" and ("model" in name or "checkpoint" in name):
        return "model"
    return "config"


def _read_report(
    storage: EvaluationStorage,
    stage: object,
    outputs: dict[str, str],
    *,
    report_name: str | None = None,
    typed_outputs: Mapping[str, ArtifactKind] | None = None,
) -> JsonObject:
    if report_name is not None:
        body = _read_report_body(storage, outputs[report_name])
        document = _report_document(body, stage, outputs)
        if stage == "dataset":
            if not document.get("files"):
                raise InvalidPlanDelivery("invalid_report", "files")
        elif not document["metrics"]:
            issues = document.get("issues")
            first = issues[0] if isinstance(issues, list) and issues else None
            field = first.get("field") if isinstance(first, dict) else None
            raise InvalidPlanDelivery(
                "invalid_report_metrics", field if isinstance(field, str) else "metrics"
            )
        return document

    untyped_outputs = {
        name: path
        for name, path in outputs.items()
        if name not in (typed_outputs or {})
    }
    candidates = {
        name: path for name, path in untyped_outputs.items() if path.endswith(".json")
    }
    needs_dataset_manifest = stage == "dataset" and any(
        name.endswith("report") for name in untyped_outputs
    )
    legacy_path = next(
        (path for name, path in candidates.items() if name.endswith("report")),
        None,
    ) or next(
        (
            path
            for name, path in candidates.items()
            if name.endswith(("metrics", "result"))
        ),
        None,
    )
    found = None
    legacy_document = None
    for path in dict.fromkeys(candidates.values()):
        try:
            body = _read_report_body(storage, path)
        except InvalidPlanDelivery:
            continue
        document = _report_document(body, stage, outputs, legacy=True)
        if path == legacy_path:
            legacy_document = document
        if document.get("files") or (
            document["metrics"] and not needs_dataset_manifest
        ):
            if found is not None:
                raise InvalidPlanDelivery("ambiguous_report")
            found = document
    if found is None and legacy_path is not None and legacy_document is None:
        raise InvalidPlanDelivery("invalid_report")

    document: JsonObject = found or legacy_document or {"metrics": []}
    if stage == "dataset" and not document.get("files"):
        if needs_dataset_manifest:
            raise InvalidPlanDelivery("invalid_report")
        document["files"] = [
            ObservationDatasetFile(name=name, path=f"/workspace{path}").model_dump(
                mode="json"
            )
            for name, path in _dataset_paths(outputs).items()
        ]
    return document


def _read_report_body(storage: EvaluationStorage, path: str) -> dict:
    try:
        data = storage.read(path)
        if len(data) > 48 * 1024:
            raise InvalidPlanDelivery("invalid_report")
        body = json.loads(data)
    except (EvaluationStorageTooLargeError, UnicodeError, json.JSONDecodeError) as exc:
        raise InvalidPlanDelivery("invalid_report") from exc
    if not isinstance(body, dict):
        raise InvalidPlanDelivery("invalid_report")
    return body


def _dataset_paths(outputs: dict[str, str]) -> dict[str, str]:
    return {
        name: path
        for name, path in outputs.items()
        if path.endswith((".jsonl", ".csv", ".parquet"))
    }


def _report_document(
    body: dict,
    stage: object,
    outputs: dict[str, str],
    *,
    legacy: bool = False,
) -> JsonObject:
    issues: list[ReportIssue] = []
    checks = dataset_report_checks(body) if stage == "dataset" else None
    metrics = report_metrics(
        body, issues=issues, checks=checks, allow_comparisons=stage == "eval"
    )
    if legacy and not metrics and stage in ("init", "eval"):
        metrics = baseline_evaluation_metrics(body)
    document: JsonObject = {
        "metrics": [item.model_dump(mode="json") for item in metrics]
    }
    if isinstance(summary := body.get("summary"), str):
        summary = summary.strip()
        if 0 < len(summary) <= 900:
            document["summary"] = summary
    if stage == "dataset":
        files = dataset_report_files(body, issues=issues)
        dataset_outputs = _dataset_paths(outputs)
        if not files:
            files = _merged_dataset_report_files(body, dataset_outputs)
        if not files and len(dataset_outputs) == 1:
            ((name, path),) = dataset_outputs.items()
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
        document["files"] = [item.model_dump(mode="json") for item in files]
    if checks:
        document["checks"] = {name: passed for name, passed in checks.items()}
    if issues:
        document["issues"] = [item.model_dump(mode="json") for item in issues[:32]]
    return document


def _merged_dataset_report_files(
    body: dict, outputs: dict[str, str]
) -> list[ObservationDatasetFile]:
    directory = body.get("output_dir")
    if (
        not isinstance(directory, str)
        or outputs.keys() != MERGED_DATASET_OUTPUTS.keys()
    ):
        return []
    try:
        output_dir = PurePosixPath(
            to_storage_api_path(normalize_workspace_path(directory))
        )
    except ValueError:
        return []
    if len(set(outputs.values())) != len(outputs):
        return []
    files = []
    for name, split in MERGED_DATASET_OUTPUTS.items():
        path = PurePosixPath(outputs[name])
        records = body.get(f"n_{split}")
        if path.parent != output_dir or type(records) is not int or records < 0:
            return []
        files.append(
            {
                "name": name,
                "path": f"/workspace{path}",
                "sample_count": records,
                "distributions": {"label": body.get(f"{split}_labels")},
            }
        )
    return dataset_report_files({"files": files})
