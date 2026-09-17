from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal

from pydantic import Field

from pyromind_runtime.domain.base import ContractModel, utc_now
from pyromind_runtime.domain.content import ContentBlock, JsonObject


class UserMessageCommand(ContractModel):
    command_id: str = Field(min_length=1)
    type: Literal["user_message"] = "user_message"
    content: tuple[ContentBlock, ...] = Field(min_length=1)
    workflow_xyflow: JsonObject | None = None


class CancelCommand(ContractModel):
    command_id: str = Field(min_length=1)
    type: Literal["cancel"] = "cancel"


class PermissionResponseCommand(ContractModel):
    command_id: str = Field(min_length=1)
    type: Literal["permission_response"] = "permission_response"
    permission_id: str = Field(min_length=1)
    decision: Literal["allow_once", "deny"]
    reason: str | None = None


class RollbackWorkflowCommand(ContractModel):
    command_id: str = Field(min_length=1)
    type: Literal["rollback_workflow"] = "rollback_workflow"
    event_id: str = Field(min_length=1)


type ProductCommand = Annotated[
    UserMessageCommand
    | CancelCommand
    | PermissionResponseCommand
    | RollbackWorkflowCommand,
    Field(discriminator="type"),
]


class CommandReceipt(ContractModel):
    command_id: str = Field(min_length=1)
    status: Literal["accepted", "completed", "failed"]
    response: JsonObject = Field(default_factory=dict)
    recorded_at: datetime = Field(default_factory=utc_now)
