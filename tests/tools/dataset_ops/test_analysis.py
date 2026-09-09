from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from openhands.tools.dataset_ops.adapters import DatasetRecord, JsonlTextAdapter
from openhands.tools.dataset_ops.analysis import (
    analyze_existing_labels,
    build_distribution_report,
    stable_sample,
    wilson_interval,
)
from openhands.tools.dataset_ops.contracts import (
    AnalysisSpec,
    AnalyzedSample,
    LabelAssignment,
    SamplingSpec,
    Taxonomy,
    TaxonomyDimension,
    TaxonomyLabel,
)
from openhands.tools.dataset_ops.labeling import parse_label_response


def taxonomy() -> Taxonomy:
    return Taxonomy(
        taxonomy_id="quality-v1",
        modality="text",
        generated_by="user",
        dimensions=(
            TaxonomyDimension(
                id="quality",
                name="Quality",
                assignment="single",
                labels=(
                    TaxonomyLabel(
                        id="good",
                        name="Good",
                        description="usable",
                        boundary="meets the stated requirements",
                    ),
                    TaxonomyLabel(
                        id="bad",
                        name="Bad",
                        description="unusable",
                        boundary="does not meet the requirements",
                    ),
                ),
            ),
        ),
    )


def test_inferred_analysis_cap_is_200() -> None:
    with pytest.raises(ValidationError, match="capped at 200"):
        SamplingSpec(mode="capped_random", max_samples=201)


def test_existing_labels_require_full_deterministic_mode() -> None:
    with pytest.raises(ValidationError, match="sampling.mode='all'"):
        AnalysisSpec(
            source_path="train.jsonl",
            adapter="jsonl_text",
            label_source="existing_labels",
            taxonomy=taxonomy(),
            label_fields={"quality": "quality"},
            sampling=SamplingSpec(mode="capped_random"),
        )


def test_existing_labels_are_counted_exactly_without_model() -> None:
    spec = AnalysisSpec(
        source_path="train.jsonl",
        adapter="jsonl_text",
        label_source="existing_labels",
        taxonomy=taxonomy(),
        label_fields={"quality": "metadata.quality"},
        sampling=SamplingSpec(mode="all"),
    )
    records = [
        DatasetRecord(
            str(index),
            Path("train.jsonl"),
            {"metadata": {"quality": label}},
            split="train",
        )
        for index, label in enumerate(("good", "good", "bad", "unknown"))
    ]
    report, analyzed, failures = analyze_existing_labels(records, spec)
    assert report.analysis_mode == "exact"
    assert report.total_records == 4
    assert report.analyzed_records == 3
    assert report.failed_records == report.unknown_records == 1
    assert len(analyzed) == 3
    assert failures[0]["sample_id"] == "3"
    assert {item.label: item.count for item in report.distributions} == {
        "good": 2,
        "bad": 1,
    }


def test_stable_hash_sample_is_unique_and_reproducible() -> None:
    records = [
        DatasetRecord(str(index), Path("data.jsonl"), {"value": index})
        for index in range(1000)
    ]
    first, total = stable_sample(records, 200, seed=91)
    second, _ = stable_sample(reversed(records), 200, seed=91)
    third, _ = stable_sample(records, 200, seed=92)
    assert total == 1000
    assert len(first) == len({item.sample_id for item in first}) == 200
    assert [item.sample_id for item in first] == [item.sample_id for item in second]
    assert [item.sample_id for item in first] != [item.sample_id for item in third]


def test_duplicate_sample_ids_are_rejected() -> None:
    record = DatasetRecord("same", Path("data.jsonl"), {})
    with pytest.raises(ValueError, match="duplicate sample_id"):
        stable_sample([record, record], 2, seed=1)


def test_sampled_report_has_estimate_and_wilson_interval() -> None:
    samples = [
        AnalyzedSample(
            sample_id=str(index),
            source="model",
            assignments=(
                LabelAssignment(
                    dimension="quality", labels=("good" if index < 8 else "bad",)
                ),
            ),
        )
        for index in range(10)
    ]
    report = build_distribution_report(
        taxonomy=taxonomy(), samples=samples, total_records=1000, sampled=True
    )
    good = next(item for item in report.distributions if item.label == "good")
    assert report.analysis_mode == "sampled_estimate"
    assert good.count == 8
    assert good.ratio == 0.8
    assert good.estimated_total_count == 800
    assert good.confidence_95 is not None
    assert good.confidence_95.low < 0.8 < good.confidence_95.high
    assert wilson_interval(0, 0) == (0.0, 0.0)


def test_invalid_single_assignment_is_not_counted() -> None:
    report = build_distribution_report(
        taxonomy=taxonomy(),
        samples=(
            AnalyzedSample(
                sample_id="1",
                source="model",
                assignments=(
                    LabelAssignment(dimension="quality", labels=("good", "bad")),
                ),
            ),
        ),
        total_records=1,
        sampled=False,
    )
    assert report.valid_records == 0
    assert all(item.count == 0 for item in report.distributions)


def test_model_label_response_is_strict() -> None:
    parsed = parse_label_response(
        {"assignments": [{"dimension": "quality", "labels": ["good"]}]},
        sample_id="sample-1",
        taxonomy=taxonomy(),
    )
    assert parsed.assignments[0].labels == ("good",)
    with pytest.raises(ValueError, match="unknown labels"):
        parse_label_response(
            {"assignments": [{"dimension": "quality", "labels": ["unknown"]}]},
            sample_id="sample-1",
            taxonomy=taxonomy(),
        )


def test_jsonl_adapter_excludes_validation(tmp_path: Path) -> None:
    source = tmp_path / "data.jsonl"
    source.write_text(
        "\n".join(
            json.dumps(item)
            for item in (
                {"id": "a", "split": "train"},
                {"id": "b", "split": "validation"},
            )
        ),
        encoding="utf-8",
    )
    assert [item.sample_id for item in JsonlTextAdapter().training_records(source)] == [
        "a"
    ]
