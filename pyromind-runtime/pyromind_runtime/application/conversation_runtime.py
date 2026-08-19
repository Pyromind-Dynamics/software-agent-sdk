from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncGenerator, Mapping
from dataclasses import dataclass
from pathlib import Path

from pyromind_runtime.application.event_projection import ProductEventProjector
from pyromind_runtime.domain.commands import (
    CommandReceipt,
    ProductCommand,
    RollbackWorkflowCommand,
)
from pyromind_runtime.domain.context import RequestContext
from pyromind_runtime.domain.events import ProductEvent
from pyromind_runtime.domain.snapshot import ConversationSnapshot
from pyromind_runtime.infrastructure.file_product_store import (
    FileProductStore,
    ProductStoreError,
)
from pyromind_runtime.ports.harness import HarnessAdapter, SessionHandle, SessionSpec


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class _ActiveConversation:
    handle: SessionHandle
    adapter: HarnessAdapter
    task: asyncio.Task[None]
    ready: asyncio.Event


class ConversationRuntime:
    def __init__(
        self,
        conversation_root: Path | str,
        adapters: HarnessAdapter | Mapping[str, HarnessAdapter],
        *,
        default_harness_id: str = "openhands",
    ) -> None:
        self.conversation_root = Path(conversation_root)
        self.adapters = (
            dict(adapters)
            if isinstance(adapters, Mapping)
            else {default_harness_id: adapters}
        )
        if default_harness_id not in self.adapters:
            raise ValueError(f"default harness is not registered: {default_harness_id}")
        self.default_harness_id = default_harness_id
        self._projector = ProductEventProjector()
        self._active: dict[str, _ActiveConversation] = {}
        self._activation_locks: dict[str, asyncio.Lock] = {}
        self._subscribers: dict[str, set[asyncio.Queue[ProductEvent]]] = {}

    async def create_conversation(
        self,
        spec: SessionSpec,
        context: RequestContext,
    ) -> ConversationSnapshot:
        adapter = self._adapter(self.default_harness_id)
        handle = await adapter.create_session(spec, context)
        if handle.harness_id != self.default_harness_id:
            await adapter.close(handle)
            raise ValueError(
                "adapter returned a handle for a different harness: "
                f"{handle.harness_id}"
            )
        store = self._store(handle.session_id)
        try:
            store.create(
                ConversationSnapshot(
                    conversation_id=handle.session_id,
                    capabilities=handle.capabilities,
                ),
                user_id=context.user_id,
                harness_id=handle.harness_id,
            )
            store.append(
                ProductEvent(
                    event_id=f"{handle.session_id}:created",
                    conversation_id=handle.session_id,
                    type="conversation.created",
                    payload={},
                )
            )
        except Exception:
            await adapter.close(handle)
            raise
        active = self._start_pump(handle, adapter, store)
        self._active[handle.session_id] = active
        await active.ready.wait()
        return store.load_snapshot()

    async def get_snapshot(
        self,
        conversation_id: str,
        context: RequestContext,
    ) -> ConversationSnapshot:
        store = await self._ensure_active(conversation_id, context)
        store.authorize(context.user_id)
        return store.load_snapshot()

    async def submit_command(
        self,
        conversation_id: str,
        command: ProductCommand,
        context: RequestContext,
    ) -> CommandReceipt:
        store = await self._ensure_active(conversation_id, context)
        store.authorize(context.user_id)
        if (
            isinstance(command, RollbackWorkflowCommand)
            and not store.load_snapshot().capabilities.workflow_rollback
        ):
            raise ValueError("current harness does not support workflow rollback")
        receipt, claimed = store.claim_command(command)
        if not claimed:
            return receipt
        active = self._active[conversation_id]
        try:
            response = await active.adapter.send(active.handle, command, context)
        except Exception as exc:
            failed = receipt.model_copy(
                update={
                    "status": "failed",
                    "response": {
                        "error": type(exc).__name__,
                        "message": str(exc),
                    },
                }
            )
            store.complete_command(failed)
            raise
        completed = receipt.model_copy(
            update={"status": "completed", "response": response}
        )
        return store.complete_command(completed)

    async def stream_events(
        self,
        conversation_id: str,
        after_seq: int,
        context: RequestContext,
    ) -> AsyncGenerator[ProductEvent]:
        store = await self._ensure_active(conversation_id, context)
        store.authorize(context.user_id)
        queue: asyncio.Queue[ProductEvent] = asyncio.Queue()
        self._subscribers.setdefault(conversation_id, set()).add(queue)
        cursor = after_seq
        try:
            for event in store.replay(after_seq):
                cursor = event.seq
                yield event
            while True:
                event = await queue.get()
                if event.seq <= cursor:
                    continue
                cursor = event.seq
                yield event
        finally:
            subscribers = self._subscribers.get(conversation_id)
            if subscribers is not None:
                subscribers.discard(queue)
                if not subscribers:
                    self._subscribers.pop(conversation_id, None)

    async def close(self) -> None:
        active = tuple(self._active.values())
        self._active.clear()
        for conversation in active:
            conversation.task.cancel()
            await conversation.adapter.close(conversation.handle)
        await asyncio.gather(
            *(conversation.task for conversation in active),
            return_exceptions=True,
        )

    def list_snapshots(
        self, context: RequestContext
    ) -> tuple[ConversationSnapshot, ...]:
        snapshots: list[ConversationSnapshot] = []
        if not self.conversation_root.is_dir():
            return ()
        for conversation_dir in self.conversation_root.iterdir():
            if not conversation_dir.is_dir():
                continue
            store = FileProductStore(conversation_dir)
            if not store.metadata_path.is_file():
                continue
            try:
                store.authorize(context.user_id)
                snapshots.append(store.load_snapshot())
            except (PermissionError, OSError, ProductStoreError):
                continue
        snapshots.sort(key=lambda item: item.through_seq, reverse=True)
        return tuple(snapshots)

    async def _ensure_active(
        self,
        conversation_id: str,
        context: RequestContext,
    ) -> FileProductStore:
        existing = self._active.get(conversation_id)
        if existing is not None:
            await existing.ready.wait()
            return self._store(conversation_id)
        lock = self._activation_locks.setdefault(conversation_id, asyncio.Lock())
        async with lock:
            existing = self._active.get(conversation_id)
            if existing is not None:
                await existing.ready.wait()
                return self._store(conversation_id)
            store = self._store(conversation_id)
            if store.metadata_path.is_file():
                store.authorize(context.user_id)
            harness_id = store.harness_id()
            adapter = self._adapter(harness_id)
            handle = await adapter.attach_session(conversation_id, context)
            if handle.harness_id != harness_id:
                await adapter.close(handle)
                raise ValueError(
                    "adapter returned a handle for a different harness: "
                    f"{handle.harness_id}"
                )
            store.create(
                ConversationSnapshot(
                    conversation_id=conversation_id,
                    capabilities=handle.capabilities,
                ),
                user_id=context.user_id,
                harness_id=harness_id,
            )
            active = self._start_pump(handle, adapter, store)
            self._active[conversation_id] = active
            await active.ready.wait()
            return store

    def _start_pump(
        self,
        handle: SessionHandle,
        adapter: HarnessAdapter,
        store: FileProductStore,
    ) -> _ActiveConversation:
        ready = asyncio.Event()
        task = asyncio.create_task(
            self._pump(handle, adapter, store, ready),
            name=f"product-events-{handle.session_id}",
        )
        return _ActiveConversation(
            handle=handle,
            adapter=adapter,
            task=task,
            ready=ready,
        )

    async def _pump(
        self,
        handle: SessionHandle,
        adapter: HarnessAdapter,
        store: FileProductStore,
        ready: asyncio.Event,
    ) -> None:
        try:
            async for harness_event in adapter.subscribe(handle):
                if harness_event.type == "history.synced":
                    ready.set()
                    continue
                product_event = self._projector.project(
                    handle.session_id, harness_event
                )
                if product_event is None:
                    continue
                persisted, _ = store.append(product_event)
                self._publish(persisted)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("Product event pump failed for %s", handle.session_id)
            notice = ProductEvent(
                event_id=f"{handle.session_id}:event-pump-failed",
                conversation_id=handle.session_id,
                type="notice.raised",
                payload={
                    "severity": "error",
                    "code": "event_pump_failed",
                    "message": f"Product event pump stopped: {type(exc).__name__}",
                },
            )
            try:
                persisted, _ = store.append(notice)
                self._publish(persisted)
            except Exception:
                logger.exception("Could not persist Product event pump failure")
        finally:
            ready.set()

    def _publish(self, event: ProductEvent) -> None:
        for queue in tuple(self._subscribers.get(event.conversation_id, ())):
            queue.put_nowait(event)

    def _store(self, conversation_id: str) -> FileProductStore:
        if not conversation_id or "/" in conversation_id or "\\" in conversation_id:
            raise ValueError("unsafe conversation id")
        return FileProductStore(self.conversation_root / conversation_id)

    def _adapter(self, harness_id: str) -> HarnessAdapter:
        try:
            return self.adapters[harness_id]
        except KeyError as exc:
            raise ValueError(f"harness is not registered: {harness_id}") from exc
