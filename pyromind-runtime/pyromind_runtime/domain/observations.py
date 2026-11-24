"""Project-scoped, read-only stage and training observations."""

from typing import Annotated, Literal

from pydantic import Field, model_validator

from pyromind_runtime.domain.base import ContractModel
from pyromind_runtime.domain.stage_analysis import StageAnalysisJob, StageAnalysisMetric


class MetricConfig(ContractModel):
    key: str = Field(min_length=1, max_length=128)
    title: str = Field(min_length=1, max_length=128)
    unit: Literal["number", "ratio", "seconds", "percent"] = "number"
    axis: Literal["log_step", "training_step", "time"] = "log_step"
    axis_key: str | None = Field(default=None, min_length=1, max_length=128)

    @model_validator(mode="after")
    def require_axis_key(self):
        if self.axis == "training_step" and not self.axis_key:
            raise ValueError("training_step requires an explicit axis_key")
        return self


class PanelConfig(ContractModel):
    project_node_id: str = Field(min_length=1, max_length=128)
    training_type: str = Field(default="Train", min_length=1, max_length=64)
    metrics: tuple[MetricConfig, ...] = Field(default=(), max_length=20)
    platform_node_id: str | None = Field(default=None, max_length=128)

    @model_validator(mode="after")
    def unique_metrics(self):
        if len({metric.key for metric in self.metrics}) != len(self.metrics):
            raise ValueError("metric keys must be unique within a panel")
        return self


DisplayLabel = Annotated[str, Field(min_length=1, max_length=160)]


class ObservationSelectBlock(ContractModel):
    type: Literal["select"] = "select"
    label: DisplayLabel
    min_options: Literal[1, 2] = 1


class ObservationMetricGroup(ContractModel):
    title: DisplayLabel
    labels: dict[DisplayLabel, DisplayLabel] = Field(max_length=32)


class ObservationMetricsBlock(ContractModel):
    type: Literal["metrics"] = "metrics"
    source: Literal["report", "analysis"]
    groups: tuple[ObservationMetricGroup, ...] = Field(default=(), max_length=16)
    empty_message: DisplayLabel | None = None


class ObservationStatusBlock(ContractModel):
    type: Literal["status"] = "status"
    source: Literal["report", "analysis"]
    subject: DisplayLabel
    loading_message: DisplayLabel | None = None
    unavailable_message: DisplayLabel | None = None


class ObservationDatasetBlock(ContractModel):
    type: Literal["dataset_files"] = "dataset_files"
    empty_message: DisplayLabel


class ObservationContentBlock(ContractModel):
    type: Literal[
        "summary", "findings", "artifacts", "analysis_version", "training_curves"
    ]


ObservationDetailBlock = Annotated[
    ObservationSelectBlock
    | ObservationMetricsBlock
    | ObservationStatusBlock
    | ObservationDatasetBlock
    | ObservationContentBlock,
    Field(discriminator="type"),
]


class ObservationDetailSection(ContractModel):
    id: Annotated[str, Field(min_length=1, max_length=128)]
    label: DisplayLabel
    selection: Literal["report_version", "analysis_version", "task_id"]
    blocks: tuple[ObservationDetailBlock, ...] = Field(min_length=1, max_length=24)


class ObservationDetailForm(ContractModel):
    project_node_id: Annotated[str, Field(min_length=1, max_length=128)]
    sections: tuple[ObservationDetailSection, ...] = Field(min_length=1, max_length=8)

    @model_validator(mode="after")
    def unique_sections(self):
        if len({section.id for section in self.sections}) != len(self.sections):
            raise ValueError("only one display section per section ID")
        return self


class ObservationDetailForms(ContractModel):
    schema_version: Literal[1] = 1
    forms: tuple[ObservationDetailForm, ...] = ()

    @model_validator(mode="after")
    def unique_nodes(self):
        if len({form.project_node_id for form in self.forms}) != len(self.forms):
            raise ValueError("only one display form per Project node")
        return self


class ObservationsConfig(ContractModel):
    schema_version: Literal[1] = 1
    panels: tuple[PanelConfig, ...] = Field(default=(), max_length=32)
    detail_forms: ObservationDetailForms | None = Field(
        default=None, exclude_if=lambda value: value is None
    )

    @model_validator(mode="after")
    def unique_nodes(self):
        if len({panel.project_node_id for panel in self.panels}) != len(self.panels):
            raise ValueError("only one panel configuration per Project node")
        return self


class ObservationNode(ContractModel):
    id: str
    title: str
    type: str
    business_status: str
    delivery_status: Literal["registering", "failed"] | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    delivery_error: str | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    current_run_ids: tuple[str, ...] = ()
    report_version_ids: tuple[str, ...] = ()
    historical_report_version_ids: tuple[str, ...] = Field(
        default=(), exclude_if=lambda value: not value
    )


class ObservationDatasetFile(ContractModel):
    name: str
    path: str
    size_bytes: int | None = None
    sample_count: int | None = None
    distributions: dict[str, dict[str, int]] = Field(default_factory=dict)


class ObservationReportDetails(ContractModel):
    project_id: str
    project_node_id: str
    report_version_ids: tuple[str, ...]
    metrics: tuple[StageAnalysisMetric, ...] = ()
    files: tuple[ObservationDatasetFile, ...] = ()
    summaries: tuple[str, ...] = ()


class RunBinding(ContractModel):
    project_node_id: str
    run_id: str
    task_id: str
    business_status: str
    platform_node_id: str | None = None
    training_type: str | None = None


class ObservationOverview(ContractModel):
    schema_version: Literal[1] = 1
    project_id: str
    frozen_execution_version: str | None = None
    config_version: str
    plan_sequence: int | None = None
    record_version: str
    config: ObservationsConfig
    nodes: tuple[ObservationNode, ...]
    bindings: tuple[RunBinding, ...]
    available_runs: tuple[RunBinding, ...]
    stage_analyses: tuple[StageAnalysisJob, ...] = ()
    observed_at: str


class MetricPoint(ContractModel):
    key: str
    value: float = Field(allow_inf_nan=False)
    unit: str
    source_point_id: str
    source_run_id: str
    log_step: int = Field(ge=0)
    training_step: float | None = Field(default=None, allow_inf_nan=False)
    epoch: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    time: float | None = Field(default=None, allow_inf_nan=False)


class SourcePage(ContractModel):
    points: tuple[MetricPoint, ...]
    source_run_id: str
    next_step: int
    has_more: bool = False
    gap: bool = False


class SeriesPage(ContractModel):
    project_id: str
    project_node_id: str
    run_id: str
    config_version: str
    points: tuple[MetricPoint, ...] = ()
    next_cursor: str | None = None
    truncated: bool = False
    gap: bool = False
    observation_status: Literal["available", "unavailable", "not_configured"]
    analysis_status: Literal["not_requested"] = "not_requested"
    observed_at: str
    message: str | None = None
