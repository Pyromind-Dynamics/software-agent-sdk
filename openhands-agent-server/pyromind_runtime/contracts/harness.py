from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Literal, Protocol, runtime_checkable

from pydantic import Field

from pyromind_runtime.contracts.base import ContractModel
from pyromind_runtime.contracts.content import ContentBlock
from pyromind_runtime.contracts.events import HarnessEvent
from pyromind_runtime.contracts.sandbox import ModelProfile, SandboxRef, WorkspaceRef
from pyromind_runtime.contracts.tools import ToolSpec


type CapabilityName = Literal[
    "resume",
    "steer",
    "cancel",
    "permission_reply",
    "partial_message",
    "custom_tools",
    "fork",
]


class HarnessCapabilities(ContractModel):
    resume: bool = False
    steer: bool = False
    cancel: bool = False
    permission_reply: bool = False
    partial_message: bool = False
    custom_tools: bool = False
    fork: bool = False
    native_workspace_tools: frozenset[str] = frozenset()

    def missing(self, required: frozenset[CapabilityName]) -> frozenset[str]:
        supported = {
            "resume": self.resume,
            "steer": self.steer,
            "cancel": self.cancel,
            "permission_reply": self.permission_reply,
            "partial_message": self.partial_message,
            "custom_tools": self.custom_tools,
            "fork": self.fork,
        }
        return frozenset(name for name in required if not supported[name])


class HarnessDescriptor(ContractModel):
    harness_id: str = Field(pattern=r"^[a-z][a-z0-9_-]{0,31}$")
    display_name: str = Field(min_length=1)
    capabilities: HarnessCapabilities


class SessionSpec(ContractModel):
    product_session_id: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    workspace: WorkspaceRef
    sandbox: SandboxRef
    model_profile: ModelProfile
    tools: tuple[ToolSpec, ...] = ()
    required_capabilities: frozenset[CapabilityName] = frozenset()


class SessionHandle(ContractModel):
    session_id: str = Field(min_length=1)
    harness_id: str = Field(min_length=1)
    adapter_session_ref: str = Field(min_length=1)
    capabilities: HarnessCapabilities


class UserMessageCommand(ContractModel):
    command_id: str = Field(min_length=1)
    type: Literal["user_message"] = "user_message"
    content: tuple[ContentBlock, ...] = Field(min_length=1)
    delivery: Literal["auto", "interrupt", "after_current"] = "auto"


type HarnessCommand = UserMessageCommand
type PermissionDecision = Literal["allow_once", "allow_session", "deny"]


class PermissionResponse(ContractModel):
    permission_id: str = Field(min_length=1)
    decision: PermissionDecision
    reason: str | None = None


class HarnessProtocol(Protocol):
    async def describe(self) -> HarnessDescriptor: ...

    async def create_session(self, spec: SessionSpec) -> SessionHandle: ...

    async def send(self, session_id: str, command: HarnessCommand) -> None: ...

    async def cancel(self, session_id: str) -> None: ...

    async def respond_permission(
        self,
        session_id: str,
        response: PermissionResponse,
    ) -> None: ...

    def subscribe(self, session_id: str) -> AsyncIterator[HarnessEvent]: ...

    async def close(self, session_id: str) -> None: ...


@runtime_checkable
class ForkableHarnessProtocol(Protocol):
    async def fork_session(
        self,
        source_session_id: str,
        spec: SessionSpec,
    ) -> SessionHandle: ...
