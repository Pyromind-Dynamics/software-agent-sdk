"""Durable contracts for evaluating checkpoints while a training job runs."""

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from pyromind_runtime.domain.base import AwareDatetime, ContractModel
from pyromind_runtime.domain.content import JsonObject
from pyromind_runtime.domain.training_evaluation import (
    CheckpointArtifact,
    CheckpointSelection,
    CheckpointSelectionPolicy,
    StepEvaluation,
)


TaskStatus = Literal["pending", "running", "succeeded", "failed", "stopped", "unknown"]


class CheckpointEvaluationRequest(ContractModel):
    training_task_id: str = Field(min_length=1)
    training_node_id: str = Field(min_length=1)
    template: JsonObject
    selection: CheckpointSelectionPolicy = Field(
        default_factory=CheckpointSelectionPolicy
    )


class CheckpointObservation(ContractModel):
    artifact: CheckpointArtifact
    # Digest of required file names, sizes and modification timestamps.
    fingerprint: str | None = None


class TrainingObservation(ContractModel):
    status: TaskStatus
    checkpoints: list[CheckpointObservation] = Field(default_factory=list)
    training_step: int | None = Field(default=None, ge=0)
    telemetry: JsonObject = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class EvaluationExecution(ContractModel):
    status: TaskStatus
    evaluation: StepEvaluation | None = None
    weights_not_ready: bool = False
    error: str | None = None


class CheckpointCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)
    artifact: CheckpointArtifact
    fingerprint: str | None = None
    stable_since: AwareDatetime | None = None
    observations: int = 0
    status: Literal[
        "discovered",
        "ready",
        "submitting",
        "running",
        "uncertain",
        "succeeded",
        "failed",
        "unavailable",
        "cancelled",
    ] = "discovered"
    run_id: UUID | None = None
    task_id: str | None = None
    attempt: int = 0
    evaluation: StepEvaluation | None = None
    result_wait_since: AwareDatetime | None = None
    error: str | None = None


class CheckpointEvaluationState(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)
    schema_version: Literal[1] = 1
    status: Literal[
        "watching",
        "draining",
        "waiting_credentials",
        "needs_attention",
        "cancelling",
        "completed",
        "cancelled",
    ] = "watching"
    candidates: dict[str, CheckpointCandidate] = Field(default_factory=dict)
    training_status: TaskStatus = "pending"
    terminal_since: AwareDatetime | None = None
    discovery_closed: bool = False
    cancel_requested: bool = False
    next_poll_at: AwareDatetime
    last_observed_at: AwareDatetime | None = None
    selection: CheckpointSelection | None = None
    telemetry: JsonObject = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    error: str | None = None


TERMINAL_TASK_STATUSES = frozenset({"succeeded", "failed", "stopped"})
FINISHED_CANDIDATES = frozenset({"succeeded", "failed", "unavailable", "cancelled"})
