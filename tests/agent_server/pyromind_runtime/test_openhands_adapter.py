from __future__ import annotations

import sys
from collections.abc import AsyncIterator
from pathlib import Path
from typing import cast
from uuid import UUID, uuid4

import pytest
from litellm.types.utils import ChatCompletionMessageToolCall, Function
from pydantic import SecretStr
from pyromind_runtime.adapters.openhands import OpenHandsAdapter
from pyromind_runtime.adapters.pi import (
    PiAdapter,
    StaticPiRunnerLauncher,
    safe_runner_environment,
)
from pyromind_runtime.contracts import (
    HarnessEvent,
    PermissionResponse,
    SessionHandle,
    SessionSpec,
    TextContentBlock,
    UserMessageCommand,
)
from pyromind_runtime.contracts.sandbox import ModelProfile, SandboxRef, WorkspaceRef
from pyromind_runtime.product import ConversationSnapshot
from pyromind_runtime.projectors import (
    ConversationSnapshotProjector,
    ProductEventProjector,
)

from openhands.agent_server.conversation_service import ConversationService
from openhands.agent_server.models import (
    ConfirmationResponseRequest,
    StartConversationRequest,
)
from openhands.agent_server.pub_sub import Subscriber
from openhands.sdk import LLM, Agent
from openhands.sdk.event import (
    ActionEvent,
    ConversationStateUpdateEvent,
    Event,
    MessageEvent,
    ObservationEvent,
    StreamingDeltaEvent,
)
from openhands.sdk.llm import Message, MessageToolCall, TextContent
from openhands.sdk.workspace import LocalWorkspace
from openhands.tools.terminal.definition import TerminalAction, TerminalObservation


class _FakeEventService:
    def __init__(self, conversation_id: UUID, user_id: str) -> None:
        self.conversation_id = conversation_id
        self.stored = _StoredConversation(user_id)
        self.subscriber: Subscriber[Event] | None = None
        self.subscriber_id = uuid4()
        self.messages: list[Message] = []
        self.confirmations: list[ConfirmationResponseRequest] = []
        self.unsubscribed: list[UUID] = []

    async def subscribe_to_events(self, subscriber: Subscriber[Event]) -> UUID:
        self.subscriber = subscriber
        await subscriber(
            ConversationStateUpdateEvent(
                key="full_state",
                value={"execution_status": "idle", "stats": {}},
            )
        )
        return self.subscriber_id

    async def unsubscribe_from_events(self, subscriber_id: UUID) -> bool:
        self.unsubscribed.append(subscriber_id)
        return True

    async def send_message(self, message: Message, run: bool = False) -> None:
        self.messages.append(message)
        assert run is True
        await self.emit(MessageEvent(source="user", llm_message=message))
        await self.emit(
            ConversationStateUpdateEvent(key="execution_status", value="running")
        )

    async def respond_to_confirmation(
        self,
        request: ConfirmationResponseRequest,
    ) -> None:
        self.confirmations.append(request)

    async def emit(self, event: Event) -> None:
        assert self.subscriber is not None
        await self.subscriber(event)


class _StoredConversation:
    def __init__(self, user_id: str) -> None:
        self.user_id = user_id


class _ConversationInfo:
    def __init__(self, conversation_id: UUID) -> None:
        self.id = conversation_id


class _FakeConversationService:
    def __init__(self, event_service: _FakeEventService) -> None:
        self.event_service = event_service
        self.event_services = {event_service.conversation_id: event_service}
        self.requests: list[StartConversationRequest] = []
        self.interrupts: list[tuple[UUID, str | None]] = []
        self.forks: list[tuple[UUID, UUID, str | None]] = []

    async def start_conversation(
        self,
        request: StartConversationRequest,
    ) -> tuple[_ConversationInfo, bool]:
        self.requests.append(request)
        return _ConversationInfo(self.event_service.conversation_id), True

    async def get_event_service(
        self,
        conversation_id: UUID,
        *,
        user_id: str | None = None,
    ) -> _FakeEventService | None:
        event_service = self.event_services.get(conversation_id)
        if event_service is None or user_id != event_service.stored.user_id:
            return None
        return event_service

    async def interrupt_conversation(
        self,
        conversation_id: UUID,
        *,
        user_id: str | None = None,
    ) -> None:
        self.interrupts.append((conversation_id, user_id))

    async def fork_conversation(
        self,
        source_id: UUID,
        *,
        fork_id: UUID,
        workspace: LocalWorkspace,
        user_id: str | None = None,
    ) -> _ConversationInfo | None:
        source = self.event_services.get(source_id)
        if source is None or user_id != source.stored.user_id:
            return None
        fork = _FakeEventService(fork_id, source.stored.user_id)
        self.event_services[fork_id] = fork
        self.forks.append((source_id, fork_id, str(workspace.working_dir)))
        return _ConversationInfo(fork_id)


def _start_request(spec: SessionSpec) -> StartConversationRequest:
    return StartConversationRequest(
        agent=Agent(
            llm=LLM(
                usage_id="test-llm",
                model="gpt-4o-mini",
                api_key=SecretStr("test-key"),
            ),
            tools=[],
        ),
        workspace=LocalWorkspace(working_dir=spec.workspace.root),
        conversation_id=UUID(spec.product_session_id),
        user_id=spec.user_id,
    )


async def _session_factory(spec: SessionSpec) -> StartConversationRequest:
    return _start_request(spec)


@pytest.fixture
def adapter_setup(tmp_path: Path):
    conversation_id = uuid4()
    event_service = _FakeEventService(conversation_id, "user-1")
    conversation_service = _FakeConversationService(event_service)
    adapter = OpenHandsAdapter(
        lambda: cast(ConversationService, conversation_service),
        _session_factory,
    )
    spec = SessionSpec(
        product_session_id=conversation_id.hex,
        user_id="user-1",
        workspace=WorkspaceRef(
            workspace_id=conversation_id.hex,
            root=str(tmp_path / "workspace"),
        ),
        sandbox=SandboxRef(sandbox_id=conversation_id.hex, backend="managed"),
        model_profile=ModelProfile(profile_id="default"),
    )
    return adapter, spec, event_service, conversation_service


async def _next_types(
    events: AsyncIterator[HarnessEvent],
    count: int,
) -> list[HarnessEvent]:
    return [await anext(events) for _ in range(count)]


@pytest.mark.asyncio
async def test_subscribes_before_accepting_first_command(adapter_setup) -> None:
    adapter, spec, event_service, _ = adapter_setup
    handle = await adapter.create_session(spec)
    assert event_service.subscriber is not None

    events = adapter.subscribe(handle.session_id)
    await adapter.send(
        handle.session_id,
        UserMessageCommand(
            command_id="command-1",
            content=(TextContentBlock(text="hello"),),
        ),
    )

    received = await _next_types(events, 3)
    assert [event.type for event in received] == [
        "message.started",
        "message.completed",
        "run.started",
    ]
    assert all(event.run_id == "command-1" for event in received)
    assert event_service.messages[0].content[0] == TextContent(text="hello")


@pytest.mark.asyncio
async def test_streaming_message_uses_one_neutral_message_id(adapter_setup) -> None:
    adapter, spec, event_service, _ = adapter_setup
    handle = await adapter.create_session(spec)
    events = adapter.subscribe(handle.session_id)

    await adapter.send(
        handle.session_id,
        UserMessageCommand(
            command_id="command-2",
            content=(TextContentBlock(text="stream"),),
        ),
    )
    await _next_types(events, 3)
    await event_service.emit(StreamingDeltaEvent(content="hel"))
    await event_service.emit(StreamingDeltaEvent(content="lo"))
    await event_service.emit(
        MessageEvent(
            source="agent",
            llm_message=Message(
                role="assistant",
                content=[TextContent(text="hello")],
            ),
        )
    )

    received = await _next_types(events, 4)
    assert [event.type for event in received] == [
        "message.started",
        "message.delta",
        "message.delta",
        "message.completed",
    ]
    message_ids = {str(event.payload["message_id"]) for event in received}
    assert message_ids == {"command-2:assistant"}


@pytest.mark.asyncio
async def test_tool_permission_and_result_are_harness_neutral(adapter_setup) -> None:
    adapter, spec, event_service, _ = adapter_setup
    handle = await adapter.create_session(spec)
    events = adapter.subscribe(handle.session_id)
    await adapter.send(
        handle.session_id,
        UserMessageCommand(
            command_id="command-3",
            content=(TextContentBlock(text="run a command"),),
        ),
    )
    await _next_types(events, 3)

    action = _terminal_action()
    await event_service.emit(action)
    await event_service.emit(
        ConversationStateUpdateEvent(
            key="execution_status",
            value="waiting_for_confirmation",
        )
    )
    started, permission = await _next_types(events, 2)
    assert started.type == "tool.started"
    assert started.payload["arguments"] == {
        "command": "echo hi",
        "api_key": "<redacted>",
    }
    assert permission.type == "permission.requested"
    assert permission.payload["operation_id"] == action.tool_call_id

    permission_id = str(permission.payload["permission_id"])
    await adapter.respond_permission(
        handle.session_id,
        PermissionResponse(permission_id=permission_id, decision="allow_once"),
    )
    resolved = await anext(events)
    assert resolved.type == "permission.resolved"
    assert event_service.confirmations == [ConfirmationResponseRequest(accept=True)]

    await event_service.emit(
        ObservationEvent(
            observation=TerminalObservation.from_text(
                "hi",
                command="echo hi",
                exit_code=0,
            ),
            action_id=action.id,
            tool_name=action.tool_name,
            tool_call_id=action.tool_call_id,
        )
    )
    completed = await anext(events)
    assert completed.type == "tool.completed"
    assert "provider_metadata" not in completed.model_dump(mode="json")


@pytest.mark.asyncio
async def test_usage_workflow_completion_cancel_and_close(adapter_setup) -> None:
    adapter, spec, event_service, conversation_service = adapter_setup
    handle = await adapter.create_session(spec)
    events = adapter.subscribe(handle.session_id)
    await adapter.send(
        handle.session_id,
        UserMessageCommand(
            command_id="command-4",
            content=(TextContentBlock(text="finish"),),
        ),
    )
    await _next_types(events, 3)

    await event_service.emit(
        ConversationStateUpdateEvent(
            key="stats",
            value={
                "usage_to_metrics": {
                    "main": {
                        "accumulated_cost": 0.25,
                        "accumulated_token_usage": {
                            "prompt_tokens": 10,
                            "completion_tokens": 4,
                            "cache_read_tokens": 3,
                        },
                    }
                }
            },
        )
    )
    await event_service.emit(
        ConversationStateUpdateEvent(
            key="pyromind_workflow",
            value={"workflow": "ignored by adapter"},
        )
    )
    await event_service.emit(
        ConversationStateUpdateEvent(key="execution_status", value="finished")
    )
    usage, resource, completed = await _next_types(events, 3)
    assert usage.type == "usage.updated"
    assert usage.payload == {
        "input_tokens": 10,
        "output_tokens": 4,
        "cached_tokens": 3,
        "cost_usd": 0.25,
    }
    assert resource.type == "resource.updated"
    assert "workflow" not in resource.payload
    assert completed.type == "run.completed"

    await adapter.cancel(handle.session_id)
    assert conversation_service.interrupts == [(UUID(handle.session_id), "user-1")]
    await adapter.close(handle.session_id)
    assert event_service.unsubscribed == [event_service.subscriber_id]
    with pytest.raises(StopAsyncIteration):
        await anext(events)


@pytest.mark.asyncio
async def test_fork_wraps_existing_openhands_fork_behavior(adapter_setup) -> None:
    adapter, spec, _, conversation_service = adapter_setup
    source = await adapter.create_session(spec)
    fork_id = uuid4()
    fork_spec = spec.model_copy(
        update={
            "product_session_id": fork_id.hex,
            "required_capabilities": frozenset({"fork"}),
        }
    )

    fork = await adapter.fork_session(source.session_id, fork_spec)

    assert fork.session_id == fork_id.hex
    assert fork.adapter_session_ref == fork_id.hex
    assert fork.capabilities.fork is True
    assert conversation_service.forks == [
        (UUID(source.session_id), fork_id, spec.workspace.root)
    ]
    await adapter.close(fork.session_id)
    await adapter.close(source.session_id)


@pytest.mark.asyncio
async def test_openhands_and_pi_share_product_message_semantics(adapter_setup) -> None:
    openhands_adapter, spec, event_service, _ = adapter_setup
    pi_adapter = PiAdapter(
        StaticPiRunnerLauncher(
            (
                sys.executable,
                str(Path(__file__).with_name("fake_pi_runner.py")),
            ),
            environment=safe_runner_environment({"PATH": "/usr/bin:/bin"}),
        )
    )
    openhands_handle = await openhands_adapter.create_session(spec)
    pi_spec = spec.model_copy(update={"product_session_id": uuid4().hex})
    pi_handle = await pi_adapter.create_session(pi_spec)
    openhands_events = openhands_adapter.subscribe(openhands_handle.session_id)
    pi_events = pi_adapter.subscribe(pi_handle.session_id)
    command = UserMessageCommand(
        command_id="contract-run",
        content=(TextContentBlock(text="hello"),),
    )

    await openhands_adapter.send(openhands_handle.session_id, command)
    await event_service.emit(
        MessageEvent(
            source="agent",
            llm_message=Message(
                role="assistant",
                content=[TextContent(text="done")],
            ),
        )
    )
    await event_service.emit(
        ConversationStateUpdateEvent(
            key="stats",
            value={
                "usage_to_metrics": {
                    "main": {
                        "accumulated_cost": 0,
                        "accumulated_token_usage": {
                            "prompt_tokens": 1,
                            "completion_tokens": 1,
                            "cache_read_tokens": 0,
                        },
                    }
                }
            },
        )
    )
    await event_service.emit(
        ConversationStateUpdateEvent(key="execution_status", value="finished")
    )
    await pi_adapter.send(pi_handle.session_id, command)

    openhands_output = await _next_types(openhands_events, 7)
    pi_output = await _next_types(pi_events, 8)
    assert _semantic_snapshot(openhands_handle, openhands_output) == _semantic_snapshot(
        pi_handle,
        pi_output,
    )

    await openhands_adapter.close(openhands_handle.session_id)
    await pi_adapter.close(pi_handle.session_id)


def _terminal_action() -> ActionEvent:
    tool_call = MessageToolCall.from_chat_tool_call(
        ChatCompletionMessageToolCall(
            id="call-1",
            type="function",
            function=Function(
                name="terminal",
                arguments='{"command":"echo hi","api_key":"secret"}',
            ),
        )
    )
    return ActionEvent(
        thought=[TextContent(text="run")],
        action=TerminalAction(command="echo hi"),
        tool_name="terminal",
        tool_call_id=tool_call.id,
        tool_call=tool_call,
        llm_response_id="response-1",
        summary="run echo",
    )


def _semantic_snapshot(
    handle: SessionHandle,
    events: list[HarnessEvent],
) -> tuple[object, ...]:
    product_projector = ProductEventProjector()
    snapshot_projector = ConversationSnapshotProjector()
    snapshot = ConversationSnapshot(
        conversation_id=handle.session_id,
        capabilities=handle.capabilities,
    )
    sequence = 0
    for harness_event in events:
        for projected in product_projector.project(handle.session_id, harness_event):
            sequence += 1
            snapshot = snapshot_projector.reduce(
                snapshot,
                projected.model_copy(update={"seq": sequence}),
            )
    messages = tuple(
        (
            message.role,
            tuple(
                block.text
                for block in message.content
                if isinstance(block, TextContentBlock)
            ),
            message.status,
        )
        for message in snapshot.messages
    )
    return (
        snapshot.status,
        messages,
        snapshot.operations,
        snapshot.usage,
    )
