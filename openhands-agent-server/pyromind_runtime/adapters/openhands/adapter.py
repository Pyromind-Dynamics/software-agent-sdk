from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol
from uuid import UUID, uuid4

from pydantic import JsonValue

from openhands.agent_server.conversation_service import ConversationService
from openhands.agent_server.event_service import EventService
from openhands.agent_server.models import (
    ConfirmationResponseRequest,
    ServerErrorEvent,
    StartConversationRequest,
)
from openhands.agent_server.pub_sub import Subscriber
from openhands.agent_server.pyromind_constants import PYROMIND_WORKFLOW_EVENT_KEY
from openhands.sdk.conversation.state import ConversationExecutionStatus
from openhands.sdk.event import (
    ActionEvent,
    AgentErrorEvent,
    ConversationStateUpdateEvent,
    Event,
    InterruptEvent,
    MessageEvent,
    ObservationEvent,
    PauseEvent,
    StreamingDeltaEvent,
    UserRejectObservation,
)
from openhands.sdk.event.conversation_error import ConversationErrorEvent
from openhands.sdk.llm import ImageContent, Message, TextContent
from openhands.sdk.utils.redact import sanitize_dict
from openhands.sdk.workspace import LocalWorkspace
from pyromind_runtime.contracts.content import (
    ContentBlock,
    JsonObject,
    ResourceContentBlock,
    TextContentBlock,
)
from pyromind_runtime.contracts.events import HarnessEvent
from pyromind_runtime.contracts.harness import (
    HarnessCapabilities,
    HarnessCommand,
    HarnessDescriptor,
    PermissionResponse,
    SessionHandle,
    SessionSpec,
)


logger = logging.getLogger(__name__)


_CAPABILITIES = HarnessCapabilities(
    resume=True,
    steer=False,
    cancel=True,
    permission_reply=True,
    partial_message=True,
    custom_tools=False,
    fork=True,
    native_workspace_tools=frozenset(
        {"terminal", "file_editor", "grep", "apply_patch"}
    ),
)


class OpenHandsSessionFactory(Protocol):
    def __call__(self, spec: SessionSpec) -> Awaitable[StartConversationRequest]: ...


@dataclass(slots=True)
class _TranslationState:
    session_id: str
    run_id: str | None = None
    run_started: bool = False
    last_status: str | None = None
    streaming_message_id: str | None = None
    pending_actions: dict[str, ActionEvent] = field(default_factory=dict)
    pending_permission_id: str | None = None
    last_usage: JsonObject | None = None

    def begin_command(self, command_id: str) -> None:
        if self.run_id is None:
            self.run_id = command_id
            self.last_status = None

    def ensure_run_id(self) -> str:
        if self.run_id is None:
            self.run_id = uuid4().hex
        return self.run_id

    def end_run(self) -> None:
        self.run_id = None
        self.run_started = False
        self.streaming_message_id = None
        self.pending_actions.clear()
        self.pending_permission_id = None


class _QueueSubscriber(Subscriber[Event]):
    def __init__(
        self,
        callback: Callable[[Event], Awaitable[None]],
    ) -> None:
        self._callback = callback

    async def __call__(self, event: Event) -> None:
        await self._callback(event)


@dataclass(slots=True)
class _OpenHandsSession:
    event_service: EventService
    subscriber_id: UUID
    queue: asyncio.Queue[HarnessEvent | None]
    translation: _TranslationState


class OpenHandsAdapter:
    """A transport-only wrapper around the existing OpenHands services."""

    def __init__(
        self,
        conversation_service_provider: Callable[[], ConversationService],
        session_factory: OpenHandsSessionFactory,
    ) -> None:
        self._conversation_service_provider = conversation_service_provider
        self._session_factory = session_factory
        self._sessions: dict[str, _OpenHandsSession] = {}
        self._lock = asyncio.Lock()

    async def describe(self) -> HarnessDescriptor:
        return HarnessDescriptor(
            harness_id="openhands",
            display_name="OpenHands",
            capabilities=_CAPABILITIES,
        )

    async def create_session(self, spec: SessionSpec) -> SessionHandle:
        missing = _CAPABILITIES.missing(spec.required_capabilities)
        if missing:
            raise ValueError(
                "OpenHands does not support required capabilities: "
                + ", ".join(sorted(missing))
            )
        if spec.tools:
            raise ValueError(
                "OpenHandsAdapter does not yet support neutral custom ToolSpec entries"
            )

        request = await self._session_factory(spec)
        conversation_service = self._conversation_service_provider()
        info, _ = await conversation_service.start_conversation(request)
        event_service = await conversation_service.get_event_service(
            info.id,
            user_id=spec.user_id,
        )
        if event_service is None:
            raise RuntimeError(f"OpenHands event service not found: {info.id}")

        return await self._attach_session(
            session_id=spec.product_session_id,
            adapter_session_ref=info.id.hex,
            event_service=event_service,
        )

    async def fork_session(
        self,
        source_session_id: str,
        spec: SessionSpec,
    ) -> SessionHandle:
        missing = _CAPABILITIES.missing(spec.required_capabilities)
        if missing:
            raise ValueError(
                "OpenHands does not support required capabilities: "
                + ", ".join(sorted(missing))
            )
        if spec.tools:
            raise ValueError(
                "OpenHandsAdapter does not yet support neutral custom ToolSpec entries"
            )
        self._session(source_session_id)
        conversation_service = self._conversation_service_provider()
        info = await conversation_service.fork_conversation(
            UUID(source_session_id),
            fork_id=UUID(spec.product_session_id),
            workspace=LocalWorkspace(working_dir=spec.workspace.root),
            user_id=spec.user_id,
        )
        if info is None:
            raise RuntimeError(
                f"OpenHands source session not found: {source_session_id}"
            )
        event_service = await conversation_service.get_event_service(
            info.id,
            user_id=spec.user_id,
        )
        if event_service is None:
            raise RuntimeError(f"OpenHands fork event service not found: {info.id}")
        return await self._attach_session(
            session_id=spec.product_session_id,
            adapter_session_ref=info.id.hex,
            event_service=event_service,
        )

    async def _attach_session(
        self,
        *,
        session_id: str,
        adapter_session_ref: str,
        event_service: EventService,
    ) -> SessionHandle:

        queue: asyncio.Queue[HarnessEvent | None] = asyncio.Queue()
        translation = _TranslationState(session_id=session_id)

        async def on_event(event: Event) -> None:
            try:
                for translated in _translate_event(translation, event):
                    queue.put_nowait(translated)
            except Exception:
                logger.exception(
                    "Failed to translate OpenHands event %s for session %s",
                    type(event).__name__,
                    session_id,
                )

        subscriber_id = await event_service.subscribe_to_events(
            _QueueSubscriber(on_event)
        )
        session = _OpenHandsSession(
            event_service=event_service,
            subscriber_id=subscriber_id,
            queue=queue,
            translation=translation,
        )
        async with self._lock:
            existing = self._sessions.get(session_id)
            if existing is not None:
                await event_service.unsubscribe_from_events(subscriber_id)
                raise RuntimeError(f"OpenHands session is already active: {session_id}")
            self._sessions[session_id] = session

        return SessionHandle(
            session_id=session_id,
            harness_id="openhands",
            adapter_session_ref=adapter_session_ref,
            capabilities=_CAPABILITIES,
        )

    async def send(self, session_id: str, command: HarnessCommand) -> None:
        session = self._session(session_id)
        session.translation.begin_command(command.command_id)
        content = [_to_openhands_content(block) for block in command.content]
        await session.event_service.send_message(
            Message(role="user", content=content),
            run=True,
        )

    async def cancel(self, session_id: str) -> None:
        session = self._session(session_id)
        conversation_service = self._conversation_service_provider()
        await conversation_service.interrupt_conversation(
            UUID(session_id),
            user_id=session.event_service.stored.user_id,
        )

    async def respond_permission(
        self,
        session_id: str,
        response: PermissionResponse,
    ) -> None:
        session = self._session(session_id)
        pending_id = session.translation.pending_permission_id
        if pending_id is None or response.permission_id != pending_id:
            raise ValueError(f"permission is not pending: {response.permission_id}")
        session.queue.put_nowait(
            HarnessEvent(
                session_id=session_id,
                type="permission.resolved",
                run_id=session.translation.run_id,
                payload={
                    "permission_id": response.permission_id,
                    "decision": response.decision,
                },
            )
        )
        session.translation.pending_permission_id = None
        await session.event_service.respond_to_confirmation(
            ConfirmationResponseRequest(
                accept=response.decision in {"allow_once", "allow_session"},
                reason=response.reason or "User rejected the action.",
            )
        )

    def subscribe(self, session_id: str) -> AsyncIterator[HarnessEvent]:
        queue = self._session(session_id).queue

        async def stream() -> AsyncIterator[HarnessEvent]:
            while True:
                event = await queue.get()
                if event is None:
                    return
                yield event

        return stream()

    async def close(self, session_id: str) -> None:
        async with self._lock:
            session = self._sessions.pop(session_id, None)
        if session is None:
            return
        await session.event_service.unsubscribe_from_events(session.subscriber_id)
        session.queue.put_nowait(None)

    def _session(self, session_id: str) -> _OpenHandsSession:
        session = self._sessions.get(session_id)
        if session is None:
            raise ValueError(f"OpenHands session is not active: {session_id}")
        return session


def _translate_event(
    state: _TranslationState,
    event: Event,
) -> tuple[HarnessEvent, ...]:
    if isinstance(event, MessageEvent):
        return _translate_message(state, event)
    if isinstance(event, StreamingDeltaEvent):
        return _translate_delta(state, event)
    if isinstance(event, ActionEvent):
        state.ensure_run_id()
        state.pending_actions[event.tool_call_id] = event
        return (
            _event(
                state,
                event,
                "tool.started",
                {
                    "tool_call_id": event.tool_call_id,
                    "tool_name": event.tool_name,
                    "arguments": _tool_arguments(event),
                    **({"summary": event.summary} if event.summary else {}),
                },
            ),
        )
    if isinstance(event, ObservationEvent):
        state.pending_actions.pop(event.tool_call_id, None)
        event_type = "tool.failed" if event.observation.is_error else "tool.completed"
        payload: JsonObject = {
            "tool_call_id": event.tool_call_id,
            "tool_name": event.tool_name,
            "content": _to_product_content(event.observation.to_llm_content),
        }
        if event.observation.is_error:
            payload["error_code"] = "tool_execution_failed"
        return (_event(state, event, event_type, payload),)
    if isinstance(event, UserRejectObservation):
        state.pending_actions.pop(event.tool_call_id, None)
        return (
            _event(
                state,
                event,
                "tool.failed",
                {
                    "tool_call_id": event.tool_call_id,
                    "tool_name": event.tool_name,
                    "content": [{"type": "text", "text": event.rejection_reason}],
                    "error_code": "permission_denied",
                },
            ),
        )
    if isinstance(event, AgentErrorEvent):
        state.pending_actions.pop(event.tool_call_id, None)
        return (
            _event(
                state,
                event,
                "tool.failed",
                {
                    "tool_call_id": event.tool_call_id,
                    "tool_name": event.tool_name,
                    "content": [{"type": "text", "text": event.error}],
                    "error_code": "agent_tool_error",
                },
            ),
        )
    if isinstance(event, ConversationStateUpdateEvent):
        return _translate_state_update(state, event)
    if isinstance(event, (ConversationErrorEvent, ServerErrorEvent)):
        run_id = state.ensure_run_id()
        state.last_status = ConversationExecutionStatus.ERROR.value
        translated = _event(
            state,
            event,
            "run.failed",
            {"error_code": event.code, "message": event.detail},
            run_id=run_id,
        )
        state.end_run()
        return (translated,)
    if isinstance(event, (PauseEvent, InterruptEvent)):
        run_id = state.ensure_run_id()
        state.last_status = ConversationExecutionStatus.PAUSED.value
        translated = _event(
            state,
            event,
            "run.completed",
            {"outcome": "cancelled"},
            run_id=run_id,
        )
        state.end_run()
        return (translated,)
    return ()


def _translate_message(
    state: _TranslationState,
    event: MessageEvent,
) -> tuple[HarnessEvent, ...]:
    if event.source not in {"user", "agent"}:
        return ()
    role = "user" if event.source == "user" else "assistant"
    message_id = event.id
    if role == "assistant" and state.streaming_message_id is not None:
        message_id = state.streaming_message_id
        state.streaming_message_id = None
        return (
            _event(
                state,
                event,
                "message.completed",
                {
                    "message_id": message_id,
                    "role": role,
                    "content": _to_product_content(event.llm_message.content),
                },
            ),
        )
    content = _to_product_content(event.llm_message.content)
    return (
        _event(
            state,
            event,
            "message.started",
            {"message_id": message_id, "role": role, "content": content},
        ),
        _event(
            state,
            event,
            "message.completed",
            {"message_id": message_id, "role": role, "content": content},
            event_id=f"{event.id}:completed",
        ),
    )


def _translate_delta(
    state: _TranslationState,
    event: StreamingDeltaEvent,
) -> tuple[HarnessEvent, ...]:
    if event.content is None:
        return ()
    run_id = state.ensure_run_id()
    message_id = state.streaming_message_id
    events: list[HarnessEvent] = []
    if message_id is None:
        message_id = f"{run_id}:assistant"
        state.streaming_message_id = message_id
        events.append(
            _event(
                state,
                event,
                "message.started",
                {"message_id": message_id, "role": "assistant", "content": []},
                event_id=f"{event.id}:started",
            )
        )
    events.append(
        _event(
            state,
            event,
            "message.delta",
            {"message_id": message_id, "text": event.content},
        )
    )
    return tuple(events)


def _translate_state_update(
    state: _TranslationState,
    event: ConversationStateUpdateEvent,
) -> tuple[HarnessEvent, ...]:
    output: list[HarnessEvent] = []
    status_value: object | None = None
    stats_value: object | None = None
    if event.key == "full_state" and isinstance(event.value, dict):
        status_value = event.value.get("execution_status")
        stats_value = event.value.get("stats")
    elif event.key == "execution_status":
        status_value = event.value
    elif event.key == "stats":
        stats_value = event.value
    elif event.key == PYROMIND_WORKFLOW_EVENT_KEY:
        output.append(
            _event(
                state,
                event,
                "resource.updated",
                {
                    "resource_type": "workflow",
                    "resource_id": "pyromind_workflow",
                    "version": event.id,
                },
                event_id=f"{event.id}:resource",
            )
        )

    if status_value is not None:
        output.extend(_translate_status(state, event, str(status_value)))
    usage = _usage_payload(stats_value)
    if usage is not None and usage != state.last_usage:
        state.last_usage = usage
        output.append(
            _event(
                state,
                event,
                "usage.updated",
                usage,
                event_id=f"{event.id}:usage",
            )
        )
    return tuple(output)


def _translate_status(
    state: _TranslationState,
    event: Event,
    status: str,
) -> tuple[HarnessEvent, ...]:
    if status == state.last_status:
        return ()
    state.last_status = status
    if status == ConversationExecutionStatus.RUNNING.value:
        run_id = state.ensure_run_id()
        if state.run_started:
            return ()
        state.run_started = True
        return (
            _event(
                state,
                event,
                "run.started",
                {},
                run_id=run_id,
                event_id=f"{event.id}:run-started",
            ),
        )
    if status == ConversationExecutionStatus.WAITING_FOR_CONFIRMATION.value:
        if state.pending_permission_id is not None:
            return ()
        if state.run_id is None and not state.pending_actions:
            return ()
        run_id = state.ensure_run_id()
        permission_id = f"{run_id}:confirmation"
        state.pending_permission_id = permission_id
        actions = tuple(state.pending_actions.values())
        description = "Allow the pending OpenHands operation?"
        operation_id: str | None = None
        if actions:
            operation_id = actions[0].tool_call_id if len(actions) == 1 else None
            description = "Allow pending operations: " + ", ".join(
                action.summary or action.tool_name for action in actions
            )
        return (
            _event(
                state,
                event,
                "permission.requested",
                {
                    "permission_id": permission_id,
                    "operation_id": operation_id,
                    "description": description,
                    "choices": ["allow_once", "deny"],
                },
                run_id=run_id,
                event_id=f"{event.id}:permission",
            ),
        )
    if status == ConversationExecutionStatus.FINISHED.value:
        if state.run_id is None and not state.run_started:
            return ()
        run_id = state.ensure_run_id()
        translated = _event(
            state,
            event,
            "run.completed",
            {"outcome": "completed"},
            run_id=run_id,
            event_id=f"{event.id}:run-completed",
        )
        state.end_run()
        return (translated,)
    if status in {
        ConversationExecutionStatus.ERROR.value,
        ConversationExecutionStatus.STUCK.value,
    }:
        if state.run_id is None and not state.run_started:
            return ()
        run_id = state.ensure_run_id()
        translated = _event(
            state,
            event,
            "run.failed",
            {"error_code": status, "message": f"OpenHands run ended with {status}"},
            run_id=run_id,
            event_id=f"{event.id}:run-failed",
        )
        state.end_run()
        return (translated,)
    if status == ConversationExecutionStatus.PAUSED.value and state.run_started:
        run_id = state.ensure_run_id()
        translated = _event(
            state,
            event,
            "run.completed",
            {"outcome": "cancelled"},
            run_id=run_id,
            event_id=f"{event.id}:run-cancelled",
        )
        state.end_run()
        return (translated,)
    return ()


def _event(
    state: _TranslationState,
    source: Event,
    event_type: str,
    payload: JsonObject,
    *,
    event_id: str | None = None,
    run_id: str | None = None,
) -> HarnessEvent:
    data: dict[str, object] = {
        "event_id": event_id or source.id,
        "session_id": state.session_id,
        "type": event_type,
        "run_id": state.run_id if run_id is None else run_id,
        "payload": payload,
        "provider_metadata": {"event_kind": type(source).__name__},
    }
    try:
        data["occurred_at"] = datetime.fromisoformat(source.timestamp)
    except ValueError:
        pass
    return HarnessEvent.model_validate(data)


def _tool_arguments(event: ActionEvent) -> JsonObject:
    try:
        parsed = json.loads(event.tool_call.arguments)
    except (json.JSONDecodeError, TypeError):
        parsed = {"raw": event.tool_call.arguments}
    if not isinstance(parsed, dict):
        parsed = {"value": parsed}
    sanitized = sanitize_dict(parsed)
    return json.loads(json.dumps(sanitized, default=str))


def _to_openhands_content(block: ContentBlock) -> TextContent | ImageContent:
    if isinstance(block, TextContentBlock):
        return TextContent(text=block.text)
    if isinstance(block, ResourceContentBlock):
        if block.text is not None:
            return TextContent(text=block.text)
        return TextContent(text=block.uri)
    return ImageContent(image_urls=[block.data])


def _to_product_content(
    content: object,
) -> list[JsonValue]:
    output: list[JsonValue] = []
    if not isinstance(content, (list, tuple)):
        return output
    for block in content:
        if isinstance(block, TextContent):
            output.append({"type": "text", "text": block.text})
        elif isinstance(block, ImageContent):
            output.extend(
                {"type": "resource", "uri": image_url} for image_url in block.image_urls
            )
    return output


def _usage_payload(value: object) -> JsonObject | None:
    if not isinstance(value, dict):
        return None
    metrics = value.get("usage_to_metrics")
    if not isinstance(metrics, dict):
        return None
    input_tokens = 0
    output_tokens = 0
    cached_tokens = 0
    cost_usd = 0.0
    has_cost = False
    for metric in metrics.values():
        if not isinstance(metric, dict):
            continue
        cost = metric.get("accumulated_cost")
        if isinstance(cost, (int, float)) and not isinstance(cost, bool):
            cost_usd += float(cost)
            has_cost = True
        token_usage = metric.get("accumulated_token_usage")
        if not isinstance(token_usage, dict):
            continue
        input_tokens += _non_negative_int(token_usage.get("prompt_tokens"))
        output_tokens += _non_negative_int(token_usage.get("completion_tokens"))
        cached_tokens += _non_negative_int(token_usage.get("cache_read_tokens"))
    payload: JsonObject = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cached_tokens": cached_tokens,
    }
    if has_cost:
        payload["cost_usd"] = cost_usd
    return payload


def _non_negative_int(value: object) -> int:
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return 0
