"""Deterministic sampling and distribution statistics."""

from __future__ import annotations

import hashlib
import heapq
import math
from collections import Counter
from collections.abc import Iterable, Sequence
from itertools import combinations
from typing import Any

from openhands.tools.dataset_ops.adapters import DatasetRecord
from openhands.tools.dataset_ops.contracts import (
    AnalysisSpec,
    AnalyzedSample,
    ConfidenceInterval,
    DistributionReport,
    LabelAssignment,
    LabelDistribution,
    Taxonomy,
)


def analyze_existing_labels(
    records: Iterable[DatasetRecord], spec: AnalysisSpec
) -> tuple[DistributionReport, tuple[AnalyzedSample, ...], tuple[dict[str, str], ...]]:
    """Build an exact report without invoking a model."""

    if spec.label_source != "existing_labels":
        raise ValueError(
            "analyze_existing_labels requires label_source='existing_labels'"
        )
    dimensions = {item.id: item for item in spec.taxonomy.dimensions}
    analyzed: list[AnalyzedSample] = []
    failures: list[dict[str, str]] = []
    total = 0
    unknown = 0
    for record in records:
        if record.is_evaluation:
            continue
        total += 1
        assignments: list[LabelAssignment] = []
        try:
            for dimension_id, field_path in spec.label_fields.items():
                dimension = dimensions[dimension_id]
                value = _field_value(record.value, field_path)
                raw_labels = value if isinstance(value, list) else [value]
                labels = tuple(
                    str(item).strip()
                    for item in raw_labels
                    if item is not None and str(item).strip()
                )
                known = {item.id for item in dimension.labels}
                if not labels or not set(labels).issubset(known):
                    raise ValueError(
                        f"empty or unknown label for dimension {dimension_id!r}"
                    )
                if len(set(labels)) != len(labels):
                    raise ValueError(f"duplicate labels for dimension {dimension_id!r}")
                if dimension.assignment == "single" and len(labels) != 1:
                    raise ValueError(
                        f"dimension {dimension_id!r} requires exactly one label"
                    )
                assignments.append(
                    LabelAssignment(dimension=dimension_id, labels=labels)
                )
            analyzed.append(
                AnalyzedSample(
                    sample_id=record.sample_id,
                    assignments=tuple(assignments),
                    source="existing",
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            unknown += 1
            failures.append(
                {
                    "sample_id": record.sample_id,
                    "stage": "existing_label_mapping",
                    "error": str(exc),
                }
            )
    report = build_distribution_report(
        taxonomy=spec.taxonomy,
        samples=analyzed,
        total_records=total,
        failed_records=len(failures),
        unknown_records=unknown,
        sampled=False,
    )
    return report, tuple(analyzed), tuple(failures)


def _field_value(value: dict[str, Any], path: str) -> Any:
    current: Any = value
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            raise KeyError(f"missing label field {path!r}")
        current = current[part]
    return current


def stable_sample(
    records: Iterable[DatasetRecord], limit: int, seed: int
) -> tuple[list[DatasetRecord], int]:
    """Return a reproducible uniform hash sample and the total record count."""

    if limit < 1:
        raise ValueError("sample limit must be positive")
    ranked: list[tuple[int, str, DatasetRecord]] = []
    sample_ids: set[str] = set()
    total = 0
    for record in records:
        total += 1
        if record.sample_id in sample_ids:
            raise ValueError(f"duplicate sample_id: {record.sample_id!r}")
        sample_ids.add(record.sample_id)
        rank = int.from_bytes(
            hashlib.sha256(f"{seed}:{record.sample_id}".encode()).digest(), "big"
        )
        item = (-rank, record.sample_id, record)
        if len(ranked) < limit:
            heapq.heappush(ranked, item)
        elif item > ranked[0]:
            heapq.heapreplace(ranked, item)
    selected = sorted(ranked, key=lambda item: (-item[0], item[1]))
    return [record for _, _, record in selected], total


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 0.0
    proportion = successes / total
    denominator = 1 + z**2 / total
    center = (proportion + z**2 / (2 * total)) / denominator
    margin = (
        z
        * math.sqrt(proportion * (1 - proportion) / total + z**2 / (4 * total**2))
        / denominator
    )
    return max(0.0, center - margin), min(1.0, center + margin)


def build_distribution_report(
    *,
    taxonomy: Taxonomy,
    samples: Sequence[AnalyzedSample],
    total_records: int,
    failed_records: int = 0,
    unknown_records: int = 0,
    sampled: bool,
) -> DistributionReport:
    known: dict[str, set[str]] = {
        dimension.id: {label.id for label in dimension.labels}
        for dimension in taxonomy.dimensions
    }
    counts: Counter[tuple[str, str]] = Counter()
    cooccurrence: Counter[str] = Counter()
    valid = 0
    taxonomy_dimensions = {dimension.id: dimension for dimension in taxonomy.dimensions}
    for sample in samples:
        sample_labels: list[str] = []
        sample_valid = True
        seen_dimensions: set[str] = set()
        pending: list[tuple[str, str]] = []
        for assignment in sample.assignments:
            if (
                assignment.dimension not in known
                or assignment.dimension in seen_dimensions
            ):
                sample_valid = False
                continue
            seen_dimensions.add(assignment.dimension)
            labels = set(assignment.labels)
            dimension = taxonomy_dimensions[assignment.dimension]
            if not labels.issubset(known[assignment.dimension]):
                sample_valid = False
                continue
            if dimension.assignment == "single" and len(labels) != 1:
                sample_valid = False
                continue
            for label in labels:
                pending.append((assignment.dimension, label))
        if seen_dimensions != set(known):
            sample_valid = False
        if sample_valid:
            valid += 1
            for dimension_id, label in pending:
                counts[(dimension_id, label)] += 1
                sample_labels.append(f"{dimension_id}:{label}")
        for left, right in combinations(sorted(set(sample_labels)), 2):
            cooccurrence[f"{left}|{right}"] += 1

    analyzed = len(samples)
    distributions: list[LabelDistribution] = []
    for dimension in taxonomy.dimensions:
        for label in dimension.labels:
            count = counts[(dimension.id, label.id)]
            ratio = count / valid if valid else 0.0
            low, high = wilson_interval(count, valid)
            distributions.append(
                LabelDistribution(
                    dimension=dimension.id,
                    label=label.id,
                    count=count,
                    ratio=ratio,
                    estimated_total_count=(
                        round(ratio * total_records) if sampled else None
                    ),
                    confidence_95=(
                        ConfidenceInterval(low=low, high=high) if sampled else None
                    ),
                )
            )
    return DistributionReport(
        analysis_mode="sampled_estimate" if sampled else "exact",
        total_records=total_records,
        analyzed_records=analyzed,
        valid_records=valid,
        failed_records=failed_records,
        unknown_records=unknown_records,
        distributions=tuple(distributions),
        cooccurrence=dict(cooccurrence),
    )
