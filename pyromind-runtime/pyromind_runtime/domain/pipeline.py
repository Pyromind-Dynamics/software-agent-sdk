from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import Field

from pyromind_runtime.domain.base import ContractModel
from pyromind_runtime.domain.content import JsonObject


class PipelineRun(ContractModel):
    schema_version: Literal[1] = 1
    run_id: UUID
    conversation_id: str
    definition: JsonObject
    fingerprint: str
    output_dir: str
    revision: int = Field(default=1, ge=1)
    execution_revision: int = Field(default=1, ge=1)
    task_id: str | None = None
    submission_status: Literal["preparing", "submitted", "failed", "uncertain"] = (
        "preparing"
    )
