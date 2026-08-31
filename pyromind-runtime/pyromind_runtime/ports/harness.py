from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol

from pydantic import Field

from pyromind_runtime.domain.base import ContractModel
from pyromind_runtime.domain.capabilities import HarnessCapabilities
from pyromind_runtime.domain.commands import ProductCommand
from pyromind_runtime.domain.content import ContentBlock, JsonObject
from pyromind_runtime.domain.context import RequestContext
from pyromind_runtime.domain.events import HarnessEvent
from pyromind_runtime.domain.snapshot import WorkflowState


class SessionSpec(ContractModel):
    conversation_id: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    workspace_root: str = Field(min_length=1)
    initial_message: tuple[ContentBlock, ...] = ()
    workflow_xyflow: JsonObject | None = None
    model_configuration: JsonObject = Field(default_factory=dict)
    extra: JsonObject = Field(default_factory=dict)


class SessionHandle(ContractModel):
    session_id: str = Field(min_length=1)
    adapter_session_ref: str = Field(min_length=1)
    harness_id: str = Field(default="openhands", min_length=1)
    capabilities: HarnessCapabilities


class ProductCheckpoint(ContractModel):
    event_id: str = Field(min_length=1)
    through_seq: int = Field(ge=1)
    workflow: WorkflowState
    adapter_checkpoint_ref: str | None = None


class ForkSpec(ContractModel):
    source_conversation_id: str = Field(min_length=1)
    target_conversation_id: str = Field(min_length=1)
    event_id: str = Field(min_length=1)
    title: str | None = None


class RestoreWorkflowSpec(ContractModel):
    command_id: str = Field(min_length=1)
    checkpoint: ProductCheckpoint
    trigger_turn: bool = False


class RestoreWorkflowResult(ContractModel):
    workflow_file_action: str
    adapter_event_ref: str | None = None


class ExternalTaskNotification(ContractModel):
    task_id: str = Field(min_length=1)
    kind: str = Field(min_length=1)
    run_id: str | None = None
    status: str = Field(min_length=1)
    output_dir: str | None = None
    error_summary: str | None = None
    visible_text: str | None = None
    hidden_text: str
    trigger_turn: bool = True
    reset_attempt_budget: bool = False


class HarnessAdapter(Protocol):
    async def describe(self) -> tuple[str, HarnessCapabilities]: ...

    async def create_session(
        self,
        spec: SessionSpec,
        context: RequestContext,
    ) -> SessionHandle: ...

    async def attach_session(
        self,
        conversation_id: str,
        context: RequestContext,
    ) -> SessionHandle: ...

    async def send(
        self,
        handle: SessionHandle,
        command: ProductCommand,
        context: RequestContext,
    ) -> JsonObject: ...

    async def fork(
        self,
        handle: SessionHandle,
        spec: ForkSpec,
        checkpoint: ProductCheckpoint,
        context: RequestContext,
    ) -> SessionHandle: ...

    async def restore_workflow(
        self,
        handle: SessionHandle,
        spec: RestoreWorkflowSpec,
        context: RequestContext,
    ) -> RestoreWorkflowResult: ...

    async def notify_external_task(
        self,
        handle: SessionHandle,
        notification: ExternalTaskNotification,
        context: RequestContext,
    ) -> JsonObject: ...

    def subscribe(self, handle: SessionHandle) -> AsyncIterator[HarnessEvent]: ...

    async def close(self, handle: SessionHandle) -> None: ...
