from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from uuid import UUID

from pyromind_runtime.domain.capabilities import HarnessCapabilities
from pyromind_runtime.domain.commands import (
    CancelCommand,
    PermissionResponseCommand,
    ProductCommand,
    RollbackWorkflowCommand,
    UserMessageCommand,
)
from pyromind_runtime.domain.content import JsonObject, TextContent
from pyromind_runtime.domain.context import RequestContext
from pyromind_runtime.domain.events import HarnessEvent
from pyromind_runtime.domain.snapshot import ConversationSnapshot
from pyromind_runtime.ports.harness import SessionHandle, SessionSpec

from harness_adapter.openhands_adapter.event_translator import (
    TranslationState,
    translate_event,
)
from harness_adapter.openhands_adapter.session_factory import (
    LegacyPyromindSessionFactory,
    request_from_context,
)
from openhands.agent_server.conversation_service import ConversationService
from openhands.agent_server.event_service import EventService
from openhands.agent_server.models import ConfirmationResponseRequest, EventSortOrder
from openhands.agent_server.pub_sub import Subscriber
from openhands.agent_server.pyromind_router import (
    PyromindSendMessageRequest,
    PyromindWorkflowRollbackRequest,
    rollback_pyromind_workflow_at_event,
    send_pyromind_message,
)
from openhands.sdk.event import Event


logger = logging.getLogger(__name__)

OPENHANDS_CAPABILITIES = HarnessCapabilities(
    resume=True,
    cancel=True,
    permission_reply=True,
    partial_message=True,
    fork=True,
    workflow_rollback=True,
    native_workspace_tools=frozenset(
        {"terminal", "file_editor", "grep", "apply_patch"}
    ),
)


class _QueueSubscriber(Subscriber[Event]):
    def __init__(self, callback: Callable[[Event], Awaitable[None]]) -> None:
        self._callback = callback

    async def __call__(self, event: Event) -> None:
        await self._callback(event)


@dataclass(slots=True)
class _ActiveSession:
    event_service: EventService
    subscriber_id: UUID
    queue: asyncio.Queue[HarnessEvent | None]
    translation: TranslationState


class OpenHandsAdapter:
    """Thin adapter around the pre-migration OpenHands services."""

    def __init__(
        self,
        conversation_service_provider: Callable[[], ConversationService],
        session_factory: LegacyPyromindSessionFactory | None = None,
    ) -> None:
        self._conversation_service_provider = conversation_service_provider
        self._session_factory = session_factory or LegacyPyromindSessionFactory(
            conversation_service_provider
        )
        self._sessions: dict[str, _ActiveSession] = {}
        self._lock = asyncio.Lock()

    async def describe(self) -> tuple[str, HarnessCapabilities]:
        return "openhands", OPENHANDS_CAPABILITIES

    async def create_session(
        self,
        spec: SessionSpec,
        context: RequestContext,
    ) -> SessionHandle:
        started_at = time.perf_counter()
        conversation_id, event_service = await self._session_factory.create(
            spec, context
        )
        logger.info(
            "openhands.start_event_service_ms=%.3f conversation_id=%s",
            (time.perf_counter() - started_at) * 1000,
            conversation_id,
        )
        return await self._attach(conversation_id, event_service)

    async def attach_session(
        self,
        conversation_id: str,
        context: RequestContext,
    ) -> SessionHandle:
        existing = self._sessions.get(conversation_id)
        if existing is not None:
            return self._handle(conversation_id)
        service = self._conversation_service_provider()
        event_service = await service.get_event_service(
            UUID(conversation_id),
            user_id=None if context.user_id == "anonymous" else context.user_id,
        )
        if event_service is None:
            raise FileNotFoundError(
                f"OpenHands conversation not found: {conversation_id}"
            )
        return await self._attach(conversation_id, event_service)

    async def _attach(
        self,
        conversation_id: str,
        event_service: EventService,
    ) -> SessionHandle:
        backfill_started_at = time.perf_counter()
        queue: asyncio.Queue[HarnessEvent | None] = asyncio.Queue()
        translation = TranslationState(session_id=conversation_id)
        live_buffer: list[Event] = []
        backfilling = True

        async def on_event(event: Event) -> None:
            if backfilling:
                live_buffer.append(event)
                return
            self._translate_into(queue, translation, event)

        subscriber_id = await event_service.subscribe_to_events(
            _QueueSubscriber(on_event)
        )
        try:
            page_id: str | None = None
            while True:
                page = await event_service.search_events(
                    page_id=page_id,
                    limit=100,
                    sort_order=EventSortOrder.TIMESTAMP,
                )
                for source_event in page.items:
                    self._translate_into(queue, translation, source_event)
                page_id = page.next_page_id
                if page_id is None:
                    break
            backfilling = False
            for source_event in live_buffer:
                self._translate_into(queue, translation, source_event)
            queue.put_nowait(
                HarnessEvent(
                    event_id=f"{conversation_id}:history-synced",
                    session_id=conversation_id,
                    type="history.synced",
                )
            )
            logger.info(
                "adapter.history_backfill_ms=%.3f conversation_id=%s "
                "harness_id=openhands",
                (time.perf_counter() - backfill_started_at) * 1000,
                conversation_id,
            )
        except Exception:
            await event_service.unsubscribe_from_events(subscriber_id)
            raise

        session = _ActiveSession(
            event_service=event_service,
            subscriber_id=subscriber_id,
            queue=queue,
            translation=translation,
        )
        async with self._lock:
            existing = self._sessions.get(conversation_id)
            if existing is not None:
                await event_service.unsubscribe_from_events(subscriber_id)
                return self._handle(conversation_id)
            self._sessions[conversation_id] = session
        return self._handle(conversation_id)

    async def send(
        self,
        handle: SessionHandle,
        command: ProductCommand,
        context: RequestContext,
    ) -> JsonObject:
        session = self._session(handle.session_id)
        if isinstance(command, UserMessageCommand):
            text = "\n".join(
                block.text
                for block in command.content
                if isinstance(block, TextContent)
            )
            if not text:
                raise ValueError("the legacy Pyromind message route requires text")
            session.translation.begin_command(command.command_id)
            result = await send_pyromind_message(
                request_from_context(context),
                PyromindSendMessageRequest(
                    text=text,
                    workflow_xyflow=command.workflow_xyflow,
                    run=True,
                ),
                session.event_service,
            )
            return result.model_dump(mode="json")
        if isinstance(command, CancelCommand):
            await self._conversation_service_provider().interrupt_conversation(
                UUID(handle.adapter_session_ref),
                user_id=None if context.user_id == "anonymous" else context.user_id,
            )
            return {"cancelled": True}
        if isinstance(command, PermissionResponseCommand):
            pending_id = session.translation.pending_permission_id
            if pending_id != command.permission_id:
                raise ValueError(f"permission is not pending: {command.permission_id}")
            await session.event_service.respond_to_confirmation(
                ConfirmationResponseRequest(
                    accept=command.decision == "allow_once",
                    reason=(
                        command.reason
                        or (
                            "User approved the pending action."
                            if command.decision == "allow_once"
                            else "User rejected the action."
                        )
                    ),
                )
            )
            session.translation.pending_permission_id = None
            session.queue.put_nowait(
                HarnessEvent(
                    session_id=handle.session_id,
                    type="permission.resolved",
                    run_id=session.translation.run_id,
                    payload={
                        "permission_id": command.permission_id,
                        "decision": command.decision,
                    },
                )
            )
            return {"resolved": True}
        if isinstance(command, RollbackWorkflowCommand):
            result = await rollback_pyromind_workflow_at_event(
                request_from_context(context),
                UUID(handle.adapter_session_ref),
                PyromindWorkflowRollbackRequest(eventId=command.event_id),
                session.event_service,
            )
            return result.model_dump(mode="json", by_alias=True)
        raise TypeError(f"unsupported command: {type(command).__name__}")

    async def fork(
        self,
        handle: SessionHandle,
        snapshot: ConversationSnapshot,
        context: RequestContext,
    ) -> SessionHandle:
        raise NotImplementedError("fork-at-event is exposed by the server façade")

    def subscribe(self, handle: SessionHandle) -> AsyncIterator[HarnessEvent]:
        queue = self._session(handle.session_id).queue

        async def stream() -> AsyncIterator[HarnessEvent]:
            while True:
                event = await queue.get()
                if event is None:
                    return
                yield event

        return stream()

    async def close(self, handle: SessionHandle) -> None:
        async with self._lock:
            session = self._sessions.pop(handle.session_id, None)
        if session is None:
            return
        await session.event_service.unsubscribe_from_events(session.subscriber_id)
        session.queue.put_nowait(None)

    def _session(self, conversation_id: str) -> _ActiveSession:
        session = self._sessions.get(conversation_id)
        if session is None:
            raise ValueError(f"OpenHands session is not active: {conversation_id}")
        return session

    @staticmethod
    def _translate_into(
        queue: asyncio.Queue[HarnessEvent | None],
        state: TranslationState,
        event: Event,
    ) -> None:
        try:
            for translated in translate_event(state, event):
                queue.put_nowait(translated)
        except Exception as exc:
            logger.exception(
                "Failed to translate OpenHands event %s", type(event).__name__
            )
            queue.put_nowait(
                HarnessEvent(
                    event_id=f"{event.id}:projection-error",
                    source_event_id=event.id,
                    session_id=state.session_id,
                    type="notice.raised",
                    payload={
                        "severity": "warning",
                        "code": "openhands_event_projection_failed",
                        "message": (
                            f"Could not project {type(event).__name__}: "
                            f"{type(exc).__name__}"
                        ),
                    },
                )
            )

    @staticmethod
    def _handle(conversation_id: str) -> SessionHandle:
        return SessionHandle(
            session_id=conversation_id,
            adapter_session_ref=conversation_id,
            harness_id="openhands",
            capabilities=OPENHANDS_CAPABILITIES,
        )
