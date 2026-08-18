from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import uuid4

from pydantic import Field

from pyromind_runtime.domain.base import ContractModel, utc_now
from pyromind_runtime.domain.content import JsonObject


type HarnessEventType = Literal[
    "history.synced",
    "status.changed",
    "message.started",
    "message.delta",
    "message.completed",
    "operation.started",
    "operation.progress",
    "operation.completed",
    "operation.failed",
    "permission.requested",
    "permission.resolved",
    "plan.updated",
    "context_compaction.updated",
    "workflow.updated",
    "usage.updated",
    "notice.raised",
]

type ProductEventType = Literal[
    "conversation.created",
    "status.changed",
    "message.started",
    "message.delta",
    "message.completed",
    "operation.started",
    "operation.progress",
    "operation.completed",
    "operation.failed",
    "permission.requested",
    "permission.resolved",
    "plan.updated",
    "context_compaction.updated",
    "workflow.updated",
    "usage.updated",
    "notice.raised",
]


class HarnessEvent(ContractModel):
    event_id: str = Field(default_factory=lambda: uuid4().hex)
    session_id: str = Field(min_length=1)
    occurred_at: datetime = Field(default_factory=utc_now)
    type: HarnessEventType
    run_id: str | None = None
    payload: JsonObject = Field(default_factory=dict)
    source_event_id: str | None = None


class ProductEvent(ContractModel):
    schema_version: int = Field(default=1, ge=1)
    event_id: str = Field(default_factory=lambda: uuid4().hex)
    conversation_id: str = Field(min_length=1)
    seq: int = Field(default=0, ge=0)
    occurred_at: datetime = Field(default_factory=utc_now)
    type: ProductEventType
    run_id: str | None = None
    payload: JsonObject = Field(default_factory=dict)
    source_event_id: str | None = None
