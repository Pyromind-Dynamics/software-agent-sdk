"""Harness-neutral contracts for scoring saved training predictions."""

from typing import Literal

from pydantic import Field

from pyromind_runtime.domain.base import ContractModel
from pyromind_runtime.domain.content import JsonObject


class CheckpointArtifact(ContractModel):
    step: int = Field(ge=0)
    path: str
    kind: Literal["full", "lora", "unknown"] = "unknown"
    complete: bool = False
    base_model: str | None = None
    issues: list[str] = Field(default_factory=list)


class MetricObjective(ContractModel):
    metric: str = "pass_rate"
    direction: Literal["max", "min"] = "max"


class MetricConstraint(ContractModel):
    metric: str
    operator: Literal["ge", "le"]
    value: float = Field(allow_inf_nan=False)


class CheckpointSelectionPolicy(ContractModel):
    primary: MetricObjective = Field(default_factory=MetricObjective)
    secondary: list[MetricObjective] = Field(default_factory=list)
    constraints: list[MetricConstraint] = Field(default_factory=list)
    comparison_group: str | None = None


class StepEvaluation(ContractModel):
    role: Literal["checkpoint", "baseline"] = "checkpoint"
    dataset_role: Literal["validation", "test", "train"] = "validation"
    step: int = Field(ge=0)
    run_id: str
    output_dir: str
    metrics: JsonObject = Field(default_factory=dict)
    comparison_group: str | None = None
    eligible: bool = False
    issues: list[str] = Field(default_factory=list)
    checkpoint: CheckpointArtifact | None = None


class CheckpointSelection(ContractModel):
    schema_version: Literal[1] = 1
    selected_step: int | None = None
    selected_run_id: str | None = None
    model: CheckpointArtifact | None = None
    meets_target: bool | None = None
    ranking: list[int] = Field(default_factory=list)
    excluded: dict[str, list[str]] = Field(default_factory=dict)
    unmet_constraints: list[str] = Field(default_factory=list)
    best_evaluated_step: int | None = None
    reason: str
    telemetry: JsonObject = Field(default_factory=dict)
