from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, JsonValue, model_validator

from pyromind_runtime.contracts.base import ContractModel
from pyromind_runtime.contracts.content import JsonObject


PROTOCOL_VERSION = 1

type PiRunnerEventKind = Literal[
    "agent.started",
    "agent.completed",
    "agent.failed",
    "agent.cancelled",
    "message.started",
    "message.delta",
    "message.completed",
    "tool.started",
    "tool.progress",
    "tool.completed",
    "tool.failed",
    "usage.updated",
    "resource.updated",
]


class PiRunnerEvent(ContractModel):
    protocolVersion: Literal[1] = 1
    type: Literal["pi.event"] = "pi.event"
    eventId: str = Field(min_length=1)
    sessionId: str = Field(min_length=1)
    runId: str = Field(min_length=1)
    occurredAt: str = Field(min_length=1)
    kind: PiRunnerEventKind
    payload: JsonObject


class PiRunnerRequest(ContractModel):
    protocolVersion: Literal[1] = 1
    type: Literal["request"] = "request"
    requestId: str = Field(min_length=1)
    method: str = Field(min_length=1)
    params: JsonObject


class PiRunnerError(ContractModel):
    code: str = Field(min_length=1)
    message: str


class PiRunnerResponse(ContractModel):
    protocolVersion: Literal[1] = 1
    type: Literal["response"] = "response"
    requestId: str = Field(min_length=1)
    result: JsonValue = None
    error: PiRunnerError | None = None

    @model_validator(mode="after")
    def validate_result(self) -> Self:
        fields_set = self.model_fields_set
        has_result = "result" in fields_set
        has_error = self.error is not None
        if has_result == has_error:
            raise ValueError("response must contain exactly one of result or error")
        return self
