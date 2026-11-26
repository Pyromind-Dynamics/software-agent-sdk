from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from pyromind_runtime.domain.base import AwareDatetime, ContractModel, utc_now


DeploymentStatus = Literal[
    "submitting",
    "submission_unknown",
    "waiting_inference",
    "switching",
    "verifying",
    "rolling_back",
    "rollback_verifying",
    "succeeded",
    "failed",
]
TERMINAL_DEPLOYMENT_STATUSES = frozenset({"succeeded", "failed"})


class DeploymentConflictError(ValueError):
    pass


class DeploymentCandidate(ContractModel):
    candidate_id: str
    model_uri: str = Field(min_length=1)


class DeploymentTarget(ContractModel):
    resource_id: str = Field(min_length=1)
    model_uri: str = Field(min_length=1)
    endpoint: str | None = None


class DeploymentRequest(ContractModel):
    request_id: str = Field(min_length=1, max_length=128)
    candidate_id: str | None = None
    rollback_deployment_id: str | None = None

    @model_validator(mode="after")
    def one_target(self) -> DeploymentRequest:
        if bool(self.candidate_id) == bool(self.rollback_deployment_id):
            raise ValueError("Choose a candidate or a deployment to roll back")
        return self


class DeploymentRecord(ContractModel):
    deployment_id: str
    project_id: str
    request: DeploymentRequest
    candidate_id: str | None = None
    model_uri: str
    previous: DeploymentTarget
    target: DeploymentTarget | None = None
    status: DeploymentStatus
    error: str | None = None
    cleanup_status: Literal["not_required", "pending", "released"] = "not_required"
    cleanup_error: str | None = None
    created_at: AwareDatetime = Field(default_factory=utc_now)
    updated_at: AwareDatetime = Field(default_factory=utc_now)
    phase_started_at: AwareDatetime = Field(default_factory=utc_now)
