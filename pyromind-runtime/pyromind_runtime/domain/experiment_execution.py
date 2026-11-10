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
    task_id: str
    stage_id: str | None = None
    task_name: str
    role_id: str
    state: str
    report_state: str
    fingerprint: str | None = None
    proposal: JsonObject | None = None
    metrics: JsonObject | None = None
    platform_task_id: str | None = None
    artifacts: tuple[JsonObject, ...] = ()
    intent: str = "prepare"
    summary: str | None = None
    report_uri: str | None = None
    blockers: tuple[str, ...] = ()
    waiting_reason: str | None = None
    checkpoint: CheckpointView | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
