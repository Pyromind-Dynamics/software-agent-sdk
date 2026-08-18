from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import Field

from pyromind_runtime.contracts.base import ContractModel, utc_now
from pyromind_runtime.contracts.content import ContentBlock, JsonObject
from pyromind_runtime.contracts.harness import HarnessCapabilities
from pyromind_runtime.contracts.sandbox import SandboxRef, WorkspaceRef


type ConversationStatus = Literal[
    "idle",
    "running",
    "waiting_permission",
    "failed",
    "closed",
]
type MessageRole = Literal["user", "assistant", "system"]
type MessageStatus = Literal["streaming", "completed", "failed"]
type OperationStatus = Literal["running", "completed", "failed"]
type CommandStatus = Literal["accepted", "completed", "failed"]


class SnapshotMessage(ContractModel):
    message_id: str = Field(min_length=1)
    started_seq: int = Field(ge=1)
    role: MessageRole
    content: tuple[ContentBlock, ...] = ()
    status: MessageStatus
    run_id: str | None = None


class SnapshotOperation(ContractModel):
    operation_id: str = Field(min_length=1)
    started_seq: int = Field(ge=1)
    name: str = Field(min_length=1)
    status: OperationStatus
    arguments: JsonObject = Field(default_factory=dict)
    output: tuple[ContentBlock, ...] = ()
    details: JsonObject | None = None
    error_code: str | None = None
    run_id: str | None = None


class SnapshotPermission(ContractModel):
    permission_id: str = Field(min_length=1)
    operation_id: str | None = None
    description: str
    choices: tuple[str, ...]


class SnapshotUsage(ContractModel):
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    cached_tokens: int = Field(default=0, ge=0)
    cost_usd: float | None = Field(default=None, ge=0)


class WorkflowState(ContractModel):
    resource_id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    dsl: str
    canvas: JsonObject | None = None


class ConversationSnapshot(ContractModel):
    schema_version: int = Field(default=1, ge=1)
    conversation_id: str = Field(min_length=1)
    through_seq: int = Field(default=0, ge=0)
    status: ConversationStatus = "idle"
    messages: tuple[SnapshotMessage, ...] = ()
    operations: tuple[SnapshotOperation, ...] = ()
    pending_permissions: tuple[SnapshotPermission, ...] = ()
    workflow: WorkflowState | None = None
    usage: SnapshotUsage = Field(default_factory=SnapshotUsage)
    capabilities: HarnessCapabilities


class CommandReceipt(ContractModel):
    command_id: str = Field(min_length=1)
    status: CommandStatus
    response: JsonObject = Field(default_factory=dict)
    recorded_at: datetime = Field(default_factory=utc_now)


class ConversationMetadata(ContractModel):
    schema_version: int = Field(default=1, ge=1)
    conversation_id: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    harness_id: str = Field(min_length=1)
    adapter_session_ref: str = Field(min_length=1)
    capabilities: HarnessCapabilities
    workspace: WorkspaceRef
    sandbox: SandboxRef
    last_sequence: int = Field(default=0, ge=0)
    command_receipts: dict[str, CommandReceipt] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
