from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

from pyromind_runtime.domain.capabilities import HarnessCapabilities
from pyromind_runtime.domain.commands import ProductCommand
from pyromind_runtime.domain.content import JsonObject
from pyromind_runtime.domain.context import RequestContext
from pyromind_runtime.domain.events import HarnessEvent, HarnessEventType
from pyromind_runtime.ports.harness import (
    ExternalTaskNotification,
    ForkSpec,
    ProductCheckpoint,
    RestoreWorkflowResult,
    RestoreWorkflowSpec,
    SessionHandle,
    SessionSpec,
)


class FakeAdapter:
    capabilities = HarnessCapabilities(
        resume=True,
        cancel=True,
        permission_reply=True,
        partial_message=True,
        fork=True,
        workflow_rollback=True,
        external_task_resume=True,
    )

    def __init__(self, harness_id: str = "openhands") -> None:
        self.harness_id = harness_id
        self.queues: dict[str, asyncio.Queue[HarnessEvent | None]] = {}
        self.sent: list[tuple[str, ProductCommand, RequestContext]] = []
        self.closed: list[str] = []
        self.created_specs: list[SessionSpec] = []
        self.external_task_notifications: list[
            tuple[str, ExternalTaskNotification, RequestContext]
        ] = []
        self.forks: list[tuple[ForkSpec, ProductCheckpoint]] = []
        self.restores: list[RestoreWorkflowSpec] = []

    async def describe(self):
        return "fake", self.capabilities

    async def create_session(
        self, spec: SessionSpec, context: RequestContext
    ) -> SessionHandle:
        self.created_specs.append(spec)
        conversation_id = spec.conversation_id
        conversation_dir = Path(spec.workspace_root)
        if conversation_dir.name != conversation_id:
            conversation_dir /= conversation_id
        (conversation_dir / "public_data").mkdir(parents=True)
        handle = self._open(conversation_id)
        for block in spec.initial_message:
            self.emit(
                conversation_id,
                "message.started",
                {
                    "message_id": "initial-message",
                    "role": "user",
                    "content": [block.model_dump(mode="json")],
                },
                event_id="initial-message:start",
            )
            self.emit(
                conversation_id,
                "message.completed",
                {
                    "message_id": "initial-message",
                    "role": "user",
                    "content": [block.model_dump(mode="json")],
                },
                event_id="initial-message:complete",
            )
        self.synced(conversation_id)
        return handle

    async def attach_session(
        self, conversation_id: str, context: RequestContext
    ) -> SessionHandle:
        handle = self._open(conversation_id)
        self.synced(conversation_id)
        return handle

    async def send(
        self,
        handle: SessionHandle,
        command: ProductCommand,
        context: RequestContext,
    ) -> JsonObject:
        self.sent.append((handle.session_id, command, context))
        return {"accepted": True}

    async def fork(
        self,
        handle: SessionHandle,
        spec: ForkSpec,
        checkpoint: ProductCheckpoint,
        context: RequestContext,
    ) -> SessionHandle:
        self.forks.append((spec, checkpoint))
        source = Path(self.created_specs[0].workspace_root)
        if source.name != spec.source_conversation_id:
            source /= spec.source_conversation_id
        target = source.parent / spec.target_conversation_id
        (target / "public_data").mkdir(parents=True)
        handle = self._open(spec.target_conversation_id)
        self.synced(spec.target_conversation_id)
        return handle

    async def restore_workflow(
        self,
        handle: SessionHandle,
        spec: RestoreWorkflowSpec,
        context: RequestContext,
    ) -> RestoreWorkflowResult:
        self.restores.append(spec)
        return RestoreWorkflowResult(workflow_file_action="updated")

    async def notify_external_task(
        self,
        handle: SessionHandle,
        notification: ExternalTaskNotification,
        context: RequestContext,
    ) -> JsonObject:
        self.external_task_notifications.append(
            (handle.session_id, notification, context)
        )
        return {"accepted": True}

    def subscribe(self, handle: SessionHandle) -> AsyncIterator[HarnessEvent]:
        queue = self.queues[handle.session_id]

        async def stream() -> AsyncIterator[HarnessEvent]:
            while True:
                event = await queue.get()
                if event is None:
                    return
                yield event

        return stream()

    async def close(self, handle: SessionHandle) -> None:
        self.closed.append(handle.session_id)
        self.queues[handle.session_id].put_nowait(None)

    def emit(
        self,
        conversation_id: str,
        event_type: HarnessEventType,
        payload: dict | None = None,
        *,
        event_id: str | None = None,
        run_id: str | None = None,
    ) -> None:
        self.queues[conversation_id].put_nowait(
            HarnessEvent(
                event_id=event_id or f"{conversation_id}:{event_type}",
                session_id=conversation_id,
                type=event_type,
                run_id=run_id,
                payload=payload or {},
            )
        )

    def synced(self, conversation_id: str) -> None:
        self.emit(
            conversation_id,
            "history.synced",
            event_id=f"{conversation_id}:history-synced",
        )

    def _open(self, conversation_id: str) -> SessionHandle:
        self.queues.setdefault(conversation_id, asyncio.Queue())
        return SessionHandle(
            session_id=conversation_id,
            adapter_session_ref=conversation_id,
            harness_id=self.harness_id,
            capabilities=self.capabilities,
        )
