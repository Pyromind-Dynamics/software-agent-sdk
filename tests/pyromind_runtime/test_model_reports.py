import json
from datetime import UTC, datetime
from uuid import uuid4

import pytest
from pydantic import ValidationError

from pyromind_runtime.application.model_reports import (
    ModelReportError,
    ProjectModelReports,
    normalize_report,
)
from pyromind_runtime.domain.echomind_model_reports import ModelEvaluationRecord
from pyromind_runtime.ports.storage import EvaluationStorageMissingError


@pytest.fixture
def candidate():
    run_id = str(uuid4())
    return ModelEvaluationRecord(
        candidate_id=run_id,
        project_id=str(uuid4()),
        run_id=run_id,
        release_id=str(uuid4()),
        release_digest="sha256:" + "0" * 64,
        round_no=1,
        execution_mode="published",
        model_uri="/workspace/models/adapter/checkpoint-15",
        deployment_model_uri="/workspace/models/merged",
        report_uri="/workspace/evaluation/report.json",
        report_artifact_id=str(uuid4()),
        evaluation_dataset_uri="/workspace/fixed-test.jsonl",
        created_at=datetime.now(UTC),
    )


@pytest.fixture
def evaluation_products(candidate):
    report = {
        "status": "succeeded",
        "evaluation_status": "validated",
        "report_status": "ready",
        "validation": {},
        "badcase_count": 0,
        "metrics": {
            "model_reference": candidate.model_uri,
            "dataset_path": candidate.evaluation_dataset_uri,
            "pass_rate": 1,
            "review_only": False,
            "mean_latency_seconds": 0.5,
            "rubric_metrics": {"label_accuracy": {"mean_score": 1}},
        },
        "artifacts": {"report.html": "/evaluation/report.html"},
    }
    manifest = {
        "run_id": candidate.run_id,
        "model_reference": candidate.model_uri,
        "dataset_sha256": "a" * 64,
        "evaluation_config_sha256": "b" * 64,
        "runtime_sha256": "c" * 64,
        "media_sha256": {"image.jpg": "d" * 64},
        "generation": {"temperature": 0},
        "evaluation_config": {
            "pass_threshold": 0.7,
            "rubrics": [
                {
                    "name": "label_accuracy",
                    "criterion": "Label matches ground truth",
                    "required": True,
                    "weight": 4,
                }
            ],
        },
    }
    return report, manifest


def test_report_uses_eval_metrics_and_rule_evidence(candidate, evaluation_products):
    report, manifest = evaluation_products
    normalized = normalize_report(candidate, report, manifest)
    assert normalized.report_status == "ready"
    assert normalized.metrics["pass_rate"] == 1
    assert normalized.metrics["rubric_metrics.label_accuracy.mean_score"] == 1
    assert "fail_recall" not in normalized.metrics
    assert "review_only" not in normalized.metrics
    assert normalized.rubrics[0].required is True
    assert normalized.rubrics[0].weight == 4
    assert normalized.pass_threshold == 0.7
    assert normalized.report_html_uri == "/evaluation/report.html"
    assert normalized.comparison_group


@pytest.mark.parametrize(
    "field,value",
    [
        ("dataset_sha256", "e" * 64),
        ("evaluation_config_sha256", "e" * 64),
        ("media_sha256", {"image.jpg": "e" * 64}),
        ("generation", {"temperature": 1}),
    ],
)
def test_comparison_requires_same_frozen_inputs(
    candidate, evaluation_products, field, value
):
    report, manifest = evaluation_products
    original = normalize_report(candidate, report, manifest)
    changed = normalize_report(candidate, report, {**manifest, field: value})
    assert original.comparison_group != changed.comparison_group
    without_manifest = normalize_report(candidate, report, {})
    assert without_manifest.report_status == "ready"
    assert without_manifest.comparison_group is None
    assert "missing_comparison_fingerprints" in without_manifest.issues


@pytest.mark.parametrize("mismatch", ["run_id", "model_reference"])
def test_report_rejects_wrong_run_or_model(candidate, evaluation_products, mismatch):
    report, manifest = evaluation_products
    with pytest.raises(ModelReportError, match="another run|does not match"):
        normalize_report(candidate, report, {**manifest, mismatch: "/wrong/model"})


def test_invalid_eval_is_not_a_ready_candidate(candidate, evaluation_products):
    report, manifest = evaluation_products
    record = normalize_report(
        candidate, {**report, "evaluation_status": "invalid"}, manifest
    )
    assert record.report_status == "invalid"
    assert "evaluation_not_valid" in record.issues


@pytest.mark.parametrize("status", [None, "valid", "valid_with_case_errors"])
def test_unrecognized_evaluation_status_is_not_deployment_evidence(
    candidate, evaluation_products, status
):
    report, manifest = evaluation_products
    with pytest.raises(ValidationError):
        normalize_report(candidate, {**report, "evaluation_status": status}, manifest)


def test_refresh_reads_existing_products_and_caches_summary(
    candidate, evaluation_products
):
    report, manifest = evaluation_products
    records = [candidate]
    reads = []

    class Store:
        def list_candidates(self, project_id):
            return [record for record in records if record.project_id == project_id]

        def save_report(self, record):
            records[:] = [record]

    class Reader:
        def read(self, path):
            reads.append(path)
            if path == "/evaluation/report.json":
                return json.dumps(report).encode()
            if path == "/evaluation/run_manifest.json":
                return json.dumps(manifest).encode()
            raise EvaluationStorageMissingError(path)

    service = ProjectModelReports(Store())
    assert service.catalog(candidate.project_id).candidates == (candidate,)
    assert reads == []
    updated = service.refresh(candidate.project_id, candidate.candidate_id, Reader())
    assert reads == ["/evaluation/report.json", "/evaluation/run_manifest.json"]
    assert service.get(candidate.project_id, candidate.candidate_id) == updated
    with pytest.raises(LookupError):
        service.get(str(uuid4()), candidate.candidate_id)
    assert updated.comparison_group
    assert service.rank(
        candidate.project_id,
        metric="pass_rate",
        comparison_group=updated.comparison_group,
        descending=True,
    ) == (updated,)
    assert not service.rank(
        candidate.project_id,
        metric="fail_recall",
        comparison_group=updated.comparison_group,
        descending=True,
    )
