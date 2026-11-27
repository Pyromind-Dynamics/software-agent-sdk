from __future__ import annotations

from typing import Literal

from pydantic import Field

from pyromind_runtime.domain.base import ContractModel
from pyromind_runtime.domain.content import JsonObject
from pyromind_runtime.domain.execution_policy import CheckpointView


class DelegatedTask(ContractModel):
    name: str = Field(min_length=1, max_length=255)
    instruction: str = Field(min_length=1, max_length=32000)
    inputs: JsonObject = Field(default_factory=dict)
    constraints: tuple[str, ...] = ()
    expected_outputs: tuple[str, ...] = ()


class DelegateExperimentTask(ContractModel):
    """Prepared entry task passed to the Manager at Project creation."""

    role_id: str = Field(min_length=1)
    request_id: str = Field(min_length=1, max_length=128)
    task: DelegatedTask
    intent: Literal["execute", "prepare"] = "execute"
    stage_id: str | None = None


class ExecutionBinding(ContractModel):
    project_id: str
    task_run_id: str
    stage_id: str
    plan_sequence: int


class ExperimentExecutionView(ContractModel):
    run_id: str
    project_id: str
    plan_revision: int | None = None
    plan_sequence: int | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    stage_snapshot: JsonObject | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    plan_attempt: int | None = Field(
        default=None, ge=1, exclude_if=lambda value: value is None
    )
    task_id: str
    stage_id: str | None = None
    task_name: str
    role_id: str
    state: str
    report_state: str
    report_required: bool = Field(default=False, exclude_if=lambda value: not value)
    report_input_paths: tuple[str, ...] = Field(
        default=(), exclude_if=lambda value: not value
    )
    report_evidence_key: str | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    delivery_command_id: str | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    delivery_requests: dict[str, str] = Field(
        default_factory=dict, exclude_if=lambda value: not value
    )
    execution_outputs: dict[str, str] = Field(
        default_factory=dict, exclude_if=lambda value: not value
    )
    delivery_outputs: dict[str, str] = Field(
        default_factory=dict, exclude_if=lambda value: not value
    )
    delivery_error: str | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    delivery_error_field: str | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    dataset_report_format_version: int | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    fingerprint: str | None = None
    proposal: JsonObject | None = None
    metrics: JsonObject | None = None
    platform_task_id: str | None = None
    platform_task_ids: tuple[str, ...] = Field(
        default=(), exclude_if=lambda value: not value
    )
    artifacts: tuple[JsonObject, ...] = ()
    intent: str = "prepare"
    summary: str | None = None
    report_uri: str | None = None
    blockers: tuple[str, ...] = ()
    waiting_reason: str | None = None
    checkpoint: CheckpointView | None = Field(
        default=None, exclude_if=lambda value: value is None
    )

    @property
    def registered_report_ids(self) -> tuple[str, ...]:
        if self.report_state != "ready":
            return ()
        return tuple(
            identity
            for artifact in self.artifacts
            if isinstance(identity := artifact.get("artifact_id"), str) and identity
        )
