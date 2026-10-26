from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal

from pydantic import Field, model_validator

from pyromind_runtime.domain.base import ContractModel, utc_now
from pyromind_runtime.domain.content import ContentBlock, JsonObject
from pyromind_runtime.domain.user_input import (
    MAX_USER_INPUT_BYTES,
    UserInputAnswer,
)


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


class InputResponseCommand(ContractModel):
    """Answer or cancel one pending user-input request.

    ``interrupted`` is a server-side outcome only; clients submit ``answered``
    (with the complete answer set) or ``cancelled`` (without answers).
    """

    command_id: str = Field(min_length=1)
    type: Literal["input_response"] = "input_response"
    request_id: str = Field(min_length=1)
    status: Literal["answered", "cancelled"]
    answers: tuple[UserInputAnswer, ...] = ()

    @model_validator(mode="after")
    def _validate_payload(self) -> InputResponseCommand:
        if self.status == "cancelled" and self.answers:
            raise ValueError("cancelled input response must not carry answers")
        if self.status == "answered" and not self.answers:
            raise ValueError("answered input response requires answers")
        question_ids = [answer.question_id for answer in self.answers]
        if len(question_ids) != len(set(question_ids)):
            raise ValueError("answers must address each question at most once")
        if len(self.model_dump_json().encode("utf-8")) > MAX_USER_INPUT_BYTES:
            raise ValueError("input response exceeds the size limit")
        return self


type ProductCommand = Annotated[
    UserMessageCommand
    | CancelCommand
    | PermissionResponseCommand
    | RollbackWorkflowCommand
    | InputResponseCommand,
    Field(discriminator="type"),
]


class CommandReceipt(ContractModel):
    command_id: str = Field(min_length=1)
    status: Literal["accepted", "completed", "failed"]
    response: JsonObject = Field(default_factory=dict)
    recorded_at: datetime = Field(default_factory=utc_now)
