"""Project-scoped continuous batch execution, separate from model deployment."""

from datetime import datetime
from typing import Literal

from pydantic import Field, field_validator

from pyromind_runtime.domain.base import ContractModel
from pyromind_runtime.domain.storage_paths import normalize_workspace_path


class BatchScheduleConfig(ContractModel):
    enabled: bool = False
    poll_interval_seconds: int = Field(default=60, ge=30, le=3600)
    source_manifest_uri: str | None = None
    validation_dataset_uri: str | None = None

    @field_validator("source_manifest_uri", "validation_dataset_uri")
    @classmethod
    def validate_storage_uri(cls, value: str | None) -> str | None:
        return normalize_workspace_path(value) if value is not None else None


class ReadyProjectBatch(ContractModel):
    batch_id: str = Field(min_length=1, max_length=256)
    index: int = Field(ge=0)


class BatchRunState(ContractModel):
    run_id: str
    status: Literal["running", "succeeded", "failed", "canceled"]
    report_archived: bool = False


class BatchReceipt(ContractModel):
    batch_id: str
    batch_index: int
    tracer_resource_id: str
    release_id: str
    run_id: str | None = None
    claimed: bool = False
    status: Literal[
        "pending",
        "claiming",
        "claim_uncertain",
        "running",
        "failed",
        "releasing",
        "released",
    ] = "pending"
    created_at: datetime
    released_at: datetime | None = None


class BatchScheduleState(ContractModel):
    project_id: str
    owner_user_id: str
    config: BatchScheduleConfig = Field(default_factory=BatchScheduleConfig)
    status: Literal["disabled", "idle", "running", "blocked"] = "disabled"
    last_checked_at: datetime | None = None
    last_error: str | None = None
    receipts: tuple[BatchReceipt, ...] = ()
