from __future__ import annotations

from typing import Literal

from pydantic import Field

from pyromind_runtime.domain.base import ContractModel
from pyromind_runtime.domain.content import JsonObject


class WorkflowExecutionRequest(ContractModel):
    request_id: str = Field(min_length=1)
    conversation_id: str = Field(min_length=1)
    permission_id: str = Field(min_length=1)
    workflow_path: str = Field(min_length=1)
    fingerprint: str = Field(min_length=1)
    workflow: JsonObject
    summary: JsonObject
    status: Literal[
        "awaiting_confirmation",
        "cancelled",
        "submitting",
        "submitted",
        "failed",
        "uncertain",
    ] = "awaiting_confirmation"
    task_id: str | None = None
    error: str | None = None
    revision: int = Field(default=1, ge=1)
