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
from pyromind_runtime.domain.snapshot import ConversationSnapshot


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
        snapshot: ConversationSnapshot,
        context: RequestContext,
    ) -> SessionHandle: ...

    def subscribe(self, handle: SessionHandle) -> AsyncIterator[HarnessEvent]: ...

    async def close(self, handle: SessionHandle) -> None: ...
