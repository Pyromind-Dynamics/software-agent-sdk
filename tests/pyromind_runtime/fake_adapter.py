from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

from pyromind_runtime.domain.capabilities import HarnessCapabilities
from pyromind_runtime.domain.commands import ProductCommand
from pyromind_runtime.domain.context import RequestContext
from pyromind_runtime.domain.events import HarnessEvent
from pyromind_runtime.domain.snapshot import ConversationSnapshot
from pyromind_runtime.ports.harness import SessionHandle, SessionSpec


class FakeAdapter:
    capabilities = HarnessCapabilities(
        resume=True,
        cancel=True,
        permission_reply=True,
        partial_message=True,
        workflow_rollback=True,
    )

    def __init__(self) -> None:
        self.queues: dict[str, asyncio.Queue[HarnessEvent | None]] = {}
        self.sent: list[tuple[str, ProductCommand, RequestContext]] = []
        self.closed: list[str] = []

    async def describe(self):
        return "fake", self.capabilities

    async def create_session(
        self, spec: SessionSpec, context: RequestContext
    ) -> SessionHandle:
        conversation_id = spec.conversation_id
        conversation_dir = Path(spec.workspace_root) / conversation_id
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
    ):
        self.sent.append((handle.session_id, command, context))
        return {"accepted": True}

    async def fork(
        self,
        handle: SessionHandle,
        snapshot: ConversationSnapshot,
        context: RequestContext,
    ) -> SessionHandle:
        raise NotImplementedError

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
        event_type: str,
        payload: dict | None = None,
        *,
        event_id: str | None = None,
    ) -> None:
        self.queues[conversation_id].put_nowait(
            HarnessEvent(
                event_id=event_id or f"{conversation_id}:{event_type}",
                session_id=conversation_id,
                type=event_type,
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
            capabilities=self.capabilities,
        )
