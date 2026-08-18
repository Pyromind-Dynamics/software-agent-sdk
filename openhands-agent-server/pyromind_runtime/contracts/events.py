from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import uuid4

from pydantic import Field

from pyromind_runtime.contracts.base import ContractModel, utc_now
from pyromind_runtime.contracts.content import JsonObject


type HarnessEventType = Literal[
    "run.started",
    "run.completed",
    "run.failed",
    "message.started",
    "message.delta",
    "message.completed",
    "tool.started",
    "tool.progress",
    "tool.completed",
    "tool.failed",
    "permission.requested",
    "permission.resolved",
    "usage.updated",
    "resource.updated",
]

type ProductEventType = Literal[
    "conversation.created",
    "run.started",
    "run.completed",
    "run.failed",
    "message.started",
    "message.delta",
    "message.completed",
    "operation.started",
    "operation.progress",
    "operation.completed",
    "operation.failed",
    "permission.requested",
    "permission.resolved",
    "usage.updated",
    "workflow.updated",
]


class HarnessEvent(ContractModel):
    event_id: str = Field(default_factory=lambda: uuid4().hex)
    session_id: str = Field(min_length=1)
    occurred_at: datetime = Field(default_factory=utc_now)
    type: HarnessEventType
    run_id: str | None = None
    payload: JsonObject = Field(default_factory=dict)
    provider_metadata: JsonObject = Field(default_factory=dict, exclude=True)


class ProductEvent(ContractModel):
    schema_version: int = Field(default=1, ge=1)
    event_id: str = Field(default_factory=lambda: uuid4().hex)
    conversation_id: str = Field(min_length=1)
    seq: int = Field(default=0, ge=0)
    occurred_at: datetime = Field(default_factory=utc_now)
    type: ProductEventType
    run_id: str | None = None
    payload: JsonObject = Field(default_factory=dict)
