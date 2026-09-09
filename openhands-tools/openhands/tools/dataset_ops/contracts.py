"""Versioned contracts for dataset distribution analysis and synthesis."""

from __future__ import annotations

import hashlib
import json
from typing import Annotated, Literal

from pydantic import BaseModel, Field, model_validator


MAX_INFERRED_ANALYSIS_SAMPLES = 200


class DatasetContract(BaseModel):
    """Strict base model for artifacts exchanged with dataset jobs."""

    model_config = {"extra": "forbid"}


class TaxonomyLabel(DatasetContract):
    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    boundary: str = Field(min_length=1)


class TaxonomyDimension(DatasetContract):
    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    assignment: Literal["single", "multi"]
    labels: tuple[TaxonomyLabel, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_label_ids(self) -> TaxonomyDimension:
        ids = [label.id for label in self.labels]
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate label id in dimension {self.id!r}")
        return self


class Taxonomy(DatasetContract):
    schema_version: Literal[1] = 1
    taxonomy_id: str = Field(min_length=1)
    modality: Literal["text", "image", "mixed"]
    dimensions: tuple[TaxonomyDimension, ...] = Field(min_length=1)
    generated_by: Literal["user", "agent", "imported"] = "agent"
    model: str | None = None
    prompt_fingerprint: str | None = None

    @model_validator(mode="after")
    def unique_dimension_ids(self) -> Taxonomy:
        ids = [dimension.id for dimension in self.dimensions]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate taxonomy dimension id")
        if self.generated_by == "agent" and (
            not self.model or not self.prompt_fingerprint
        ):
            raise ValueError(
                "agent-generated taxonomy requires model and prompt_fingerprint"
            )
        return self


class SamplingSpec(DatasetContract):
    mode: Literal["all", "capped_random"]
    max_samples: int = Field(default=MAX_INFERRED_ANALYSIS_SAMPLES, ge=1)
    seed: int = 42

    @model_validator(mode="after")
    def enforce_cap(self) -> SamplingSpec:
        if (
            self.mode == "capped_random"
            and self.max_samples > MAX_INFERRED_ANALYSIS_SAMPLES
        ):
            raise ValueError(
                "inferred-taxonomy analysis is capped at "
                f"{MAX_INFERRED_ANALYSIS_SAMPLES} samples"
            )
        return self


class AnalysisSpec(DatasetContract):
    schema_version: Literal[1] = 1
    source_path: str = Field(min_length=1)
    source_fingerprint: str | None = Field(default=None, min_length=16)
    adapter: Literal["jsonl_text", "vision_manifest", "avi_pcb"]
    label_source: Literal["existing_labels", "inferred_taxonomy"]
    taxonomy: Taxonomy
    taxonomy_fingerprint: str | None = None
    label_fields: dict[str, str] = Field(default_factory=dict)
    sampling: SamplingSpec
    exclude_splits: tuple[Literal["validation", "test", "eval"], ...] = (
        "validation",
        "test",
        "eval",
    )

    @model_validator(mode="after")
    def validate_sampling_mode(self) -> AnalysisSpec:
        expected = "all" if self.label_source == "existing_labels" else "capped_random"
        if self.sampling.mode != expected:
            raise ValueError(f"{self.label_source} requires sampling.mode={expected!r}")
        if self.label_source == "existing_labels" and not self.label_fields:
            raise ValueError("existing_labels requires label_fields")
        dimension_ids = {item.id for item in self.taxonomy.dimensions}
        if self.label_source == "existing_labels" and set(self.label_fields) != (
            dimension_ids
        ):
            raise ValueError("label_fields must map every taxonomy dimension exactly")
        if self.label_source == "inferred_taxonomy" and self.label_fields:
            raise ValueError("inferred_taxonomy must not declare existing label_fields")
        canonical_taxonomy = json.dumps(
            self.taxonomy.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        expected_fingerprint = hashlib.sha256(canonical_taxonomy).hexdigest()
        if self.taxonomy_fingerprint not in {None, expected_fingerprint}:
            raise ValueError("taxonomy_fingerprint does not match taxonomy")
        self.taxonomy_fingerprint = expected_fingerprint
        return self


class LabelAssignment(DatasetContract):
    dimension: str = Field(min_length=1)
    labels: tuple[str, ...] = Field(min_length=1)
    confidence: float | None = Field(default=None, ge=0, le=1)
    evidence: str | None = None


class AnalyzedSample(DatasetContract):
    sample_id: str = Field(min_length=1)
    assignments: tuple[LabelAssignment, ...]
    source: Literal["existing", "model"]


class ConfidenceInterval(DatasetContract):
    low: float = Field(ge=0, le=1)
    high: float = Field(ge=0, le=1)


class LabelDistribution(DatasetContract):
    dimension: str
    label: str
    count: int = Field(ge=0)
    ratio: float = Field(ge=0, le=1)
    estimated_total_count: int | None = Field(default=None, ge=0)
    confidence_95: ConfidenceInterval | None = None


class DistributionReport(DatasetContract):
    schema_version: Literal[1] = 1
    analysis_mode: Literal["exact", "sampled_estimate"]
    total_records: int = Field(ge=0)
    analyzed_records: int = Field(ge=0)
    valid_records: int = Field(ge=0)
    failed_records: int = Field(ge=0)
    unknown_records: int = Field(ge=0)
    distributions: tuple[LabelDistribution, ...]
    cooccurrence: dict[str, int] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_counts_and_estimates(self) -> DistributionReport:
        if self.valid_records > self.analyzed_records:
            raise ValueError("valid_records cannot exceed analyzed_records")
        if self.analyzed_records + self.failed_records > self.total_records:
            raise ValueError("analyzed plus failed records cannot exceed total_records")
        if self.unknown_records > self.failed_records:
            raise ValueError("unknown_records cannot exceed failed_records")
        for item in self.distributions:
            if self.analysis_mode == "exact" and (
                item.estimated_total_count is not None or item.confidence_95 is not None
            ):
                raise ValueError("exact distributions cannot contain estimates")
            if self.analysis_mode == "sampled_estimate" and (
                item.estimated_total_count is None or item.confidence_95 is None
            ):
                raise ValueError("sampled distributions require estimate and interval")
        return self


class GapItem(DatasetContract):
    dimension: str = Field(min_length=1)
    label: str = Field(min_length=1)
    current_count: int = Field(ge=0)
    current_ratio: float = Field(ge=0, le=1)
    estimated_total_count: int | None = Field(default=None, ge=0)
    target_count: int = Field(ge=1)
    reason: str = Field(min_length=1)
    recommended_strategy: str = Field(min_length=1)

    @model_validator(mode="after")
    def target_exceeds_current(self) -> GapItem:
        current = self.estimated_total_count
        if current is None:
            current = self.current_count
        if self.target_count <= current:
            raise ValueError("target_count must exceed the current count")
        return self


class GapPlan(DatasetContract):
    schema_version: Literal[1] = 1
    status: Literal["draft", "approved"] = "draft"
    analysis_run_id: str = Field(min_length=1)
    gaps: tuple[GapItem, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_gaps(self) -> GapPlan:
        keys = [(item.dimension, item.label) for item in self.gaps]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate dimension/label gap")
        return self


class HostSelector(DatasetContract):
    include_splits: tuple[Literal["train"], ...] = ("train",)
    labels: tuple[str, ...] = ()


class SynthesisRequest(DatasetContract):
    dimension: str = Field(min_length=1)
    label: str = Field(min_length=1)
    strategy_id: str = Field(min_length=1)
    count: int = Field(ge=1)
    parameters: dict[str, object] = Field(default_factory=dict)


class AugmentationPlan(DatasetContract):
    schema_version: Literal[1] = 1
    source_path: str = Field(min_length=1)
    source_fingerprint: str | None = Field(default=None, min_length=16)
    adapter: Literal["jsonl_text", "vision_manifest", "avi_pcb"]
    gap_plan: GapPlan
    host_selector: HostSelector = Field(default_factory=HostSelector)
    requests: tuple[SynthesisRequest, ...] = Field(min_length=1)
    seed: int = 42
    max_children_per_source: int = Field(default=5, ge=1, le=20)
    output_schema: str = Field(min_length=1)

    @model_validator(mode="after")
    def require_approved_gap_plan(self) -> AugmentationPlan:
        if self.gap_plan.status != "approved":
            raise ValueError("augmentation requires an approved gap plan")
        gaps = {(item.dimension, item.label): item for item in self.gap_plan.gaps}
        requested: dict[tuple[str, str], int] = {}
        for item in self.requests:
            key = (item.dimension, item.label)
            if key not in gaps:
                raise ValueError("augmentation request is not present in approved gaps")
            requested[key] = requested.get(key, 0) + item.count
        for key, gap in gaps.items():
            current = gap.estimated_total_count
            if current is None:
                current = gap.current_count
            if requested.get(key, 0) != gap.target_count - current:
                raise ValueError(
                    "augmentation request count must equal the approved target gap"
                )
        return self


class ProvenanceRecord(DatasetContract):
    schema_version: Literal[1] = 1
    sample_id: str = Field(min_length=1)
    parent_sample_id: str = Field(min_length=1)
    strategy_id: str = Field(min_length=1)
    parameters: dict[str, object] = Field(default_factory=dict)
    seed: int
    annotation: dict[str, object]
    validation: dict[str, object]


class DatasetValidationReport(DatasetContract):
    schema_version: Literal[1] = 1
    kind: Literal["dataset_analysis", "data_synthesis"]
    valid: bool
    primary_output: str = Field(min_length=1)
    records: int = Field(ge=0)
    source_fingerprint: str = Field(min_length=16)


class DatasetProgress(DatasetContract):
    schema_version: Literal[1] = 1
    total: int = Field(ge=0)
    processed: int = Field(ge=0)
    succeeded: int = Field(ge=0)
    failed: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_progress(self) -> DatasetProgress:
        if self.processed > self.total:
            raise ValueError("processed cannot exceed total")
        if self.succeeded + self.failed > self.processed:
            raise ValueError("succeeded plus failed cannot exceed processed")
        return self


RuntimeProfile = Annotated[
    Literal["dataflow_text", "dataflow_vision", "avi_pcb_cpu"],
    Field(
        description=(
            "Pinned execution environment; arbitrary dependencies are forbidden."
        )
    ),
]
