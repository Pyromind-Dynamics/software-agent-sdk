from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import PurePosixPath
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from pyromind_runtime.domain.echomind_model_reports import (
    EvaluationRubric,
    ModelEvaluationRecord,
    ModelReportCatalog,
)
from pyromind_runtime.domain.storage_paths import to_storage_api_path
from pyromind_runtime.ports.model_reports import ModelReportReader, ModelReportStore
from pyromind_runtime.ports.storage import EvaluationStorageMissingError


class ModelReportError(ValueError):
    pass


class _ReportValidation(BaseModel):
    invalid_cases: list[str] = Field(default_factory=list)
    case_errors: dict[str, list[str]] = Field(default_factory=dict)
    report_errors: list[str] = Field(default_factory=list)


class _EvaluationSummary(BaseModel):
    model_config = ConfigDict(strict=True)

    status: Literal["succeeded", "failed"]
    evaluation_status: Literal["validated", "invalid"]
    report_status: Literal["ready", "failed", "not_requested"]
    metrics: dict[str, Any]
    artifacts: dict[str, str]
    validation: _ReportValidation
    badcase_count: int = Field(ge=0)


class ProjectModelReports:
    def __init__(self, store: ModelReportStore) -> None:
        self.store = store

    def catalog(self, project_id: str) -> ModelReportCatalog:
        return ModelReportCatalog(
            project_id=project_id,
            candidates=tuple(self.store.list_candidates(project_id)),
        )

    def get(self, project_id: str, candidate_id: str) -> ModelEvaluationRecord:
        for candidate in self.catalog(project_id).candidates:
            if candidate.candidate_id == candidate_id:
                return candidate
        raise LookupError("Model candidate not found in this Project")

    def refresh(
        self, project_id: str, candidate_id: str, reader: ModelReportReader
    ) -> ModelEvaluationRecord:
        candidate = self.get(project_id, candidate_id)
        report_path = to_storage_api_path(candidate.report_uri)
        report = _object(reader.read(report_path))
        manifest_path = str(PurePosixPath(report_path).parent / "run_manifest.json")
        try:
            manifest = _object(reader.read(manifest_path))
        except EvaluationStorageMissingError:
            manifest = {}
        updated = normalize_report(candidate, report, manifest)
        self.store.save_report(updated)
        return updated

    def rank(
        self, project_id: str, *, metric: str, comparison_group: str, descending: bool
    ) -> tuple[ModelEvaluationRecord, ...]:
        if not comparison_group:
            raise ModelReportError("Ranking requires a verified comparison group")
        return tuple(
            sorted(
                (
                    record
                    for record in self.catalog(project_id).candidates
                    if record.report_status == "ready"
                    and record.comparison_group == comparison_group
                    and metric in record.metrics
                ),
                key=lambda record: record.metrics[metric],
                reverse=descending,
            )
        )


def _object(payload: bytes) -> dict[str, Any]:
    if len(payload) > 8 * 1024 * 1024:
        raise ModelReportError("Evaluation summary exceeds 8 MiB")
    parsed = json.loads(payload)
    if not isinstance(parsed, dict):
        raise ModelReportError("Evaluation summary must be an object")
    return parsed


def _numeric_metrics(value: dict[str, Any], prefix: str = "") -> dict[str, float]:
    result: dict[str, float] = {}
    for name, item in value.items():
        key = f"{prefix}.{name}" if prefix else name
        if isinstance(item, dict):
            result.update(_numeric_metrics(item, key))
        elif isinstance(item, (int, float)) and not isinstance(item, bool):
            if math.isfinite(item):
                result[key] = float(item)
    return result


def _digest(value: object) -> str | None:
    if isinstance(value, str) and re.fullmatch(r"(?:sha256:)?[a-f0-9]{64}", value):
        return value.removeprefix("sha256:")
    return None


def normalize_report(
    candidate: ModelEvaluationRecord,
    report: dict[str, Any],
    manifest: dict[str, Any],
) -> ModelEvaluationRecord:
    summary = _EvaluationSummary.model_validate(report)
    metrics = summary.metrics
    if manifest.get("run_id") not in (None, candidate.run_id):
        raise ModelReportError("Evaluation manifest belongs to another run")
    for model in (metrics.get("model_reference"), manifest.get("model_reference")):
        if model is not None and to_storage_api_path(model) != to_storage_api_path(
            candidate.model_uri
        ):
            raise ModelReportError("Evaluation model does not match the candidate")
    dataset = metrics.get("dataset_path")
    if dataset and candidate.evaluation_dataset_uri:
        if to_storage_api_path(dataset) != to_storage_api_path(
            candidate.evaluation_dataset_uri
        ):
            raise ModelReportError("Evaluation dataset does not match the run")
    dataset_digest = _digest(manifest.get("dataset_sha256"))
    rubric_digest = _digest(manifest.get("evaluation_config_sha256"))
    group = None
    issues: list[str] = []
    if dataset_digest and rubric_digest and manifest.get("runtime_sha256"):
        comparison = {
            key: manifest.get(key)
            for key in (
                "dataset_sha256",
                "media_sha256",
                "dataset_config",
                "evaluation_config_sha256",
                "runtime_sha256",
                "generation",
            )
        }
        group = hashlib.sha256(
            json.dumps(comparison, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    else:
        issues.append("missing_comparison_fingerprints")
    evaluation = manifest.get("evaluation_config")
    rubrics: list[EvaluationRubric] = []
    threshold = None
    if isinstance(evaluation, dict):
        for rubric in evaluation.get("rubrics", []):
            if isinstance(rubric, dict):
                rubrics.append(
                    EvaluationRubric(
                        name=rubric["name"],
                        criterion=rubric["criterion"],
                        weight=rubric.get("weight", 1),
                        required=rubric.get("required", False),
                    )
                )
        threshold = evaluation.get("pass_threshold")
    valid = (
        summary.status == "succeeded"
        and summary.evaluation_status == "validated"
        and not summary.validation.invalid_cases
        and not summary.validation.case_errors
        and not summary.validation.report_errors
    )
    numeric = _numeric_metrics(metrics)
    if not valid or not numeric:
        issues.append("evaluation_not_valid")
    html_uri = None
    if summary.report_status == "ready":
        parent = PurePosixPath(to_storage_api_path(candidate.report_uri)).parent
        for uri in summary.artifacts.values():
            path = PurePosixPath(to_storage_api_path(uri))
            if path.suffix == ".html" and path.parent == parent:
                html_uri = str(path)
                break
    else:
        issues.append("html_report_not_ready")
    return ModelEvaluationRecord.model_validate(
        candidate.model_dump()
        | {
            "report_status": "ready" if valid and numeric else "invalid",
            "evaluation_status": summary.evaluation_status,
            "metrics": numeric,
            "dataset_sha256": dataset_digest,
            "evaluation_config_sha256": rubric_digest,
            "comparison_group": group,
            "rubrics": tuple(rubrics),
            "pass_threshold": threshold,
            "report_html_uri": html_uri,
            "issues": tuple(issues),
        }
    )
