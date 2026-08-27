from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncGenerator, Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from pyromind_runtime.application.event_projection import ProductEventProjector
from pyromind_runtime.domain.commands import (
    CommandReceipt,
    ProductCommand,
    RollbackWorkflowCommand,
    UserMessageCommand,
)
from pyromind_runtime.domain.context import RequestContext
from pyromind_runtime.domain.events import ProductEvent
from pyromind_runtime.domain.snapshot import ConversationSnapshot
from pyromind_runtime.infrastructure.file_product_store import (
    FileProductStore,
    ProductStoreError,
)
from pyromind_runtime.ports.external_tasks import ExternalTaskRegistry
from pyromind_runtime.ports.harness import HarnessAdapter, SessionHandle, SessionSpec


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class _ActiveConversation:
    handle: SessionHandle
    adapter: HarnessAdapter
    task: asyncio.Task[None]
    ready: asyncio.Event
    context: RequestContext


class ConversationRuntime:
    def __init__(
        self,
        conversation_root: Path | str,
        adapters: HarnessAdapter | Mapping[str, HarnessAdapter],
        *,
        default_harness_id: str = "openhands",
        external_tasks: ExternalTaskRegistry | None = None,
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
        self._external_tasks = external_tasks
        self._projector = ProductEventProjector()
        self._active: dict[str, _ActiveConversation] = {}
        self._activation_locks: dict[str, asyncio.Lock] = {}
        self._subscribers: dict[str, set[asyncio.Queue[ProductEvent]]] = {}
        self._first_command_pending: set[str] = set()
        self._first_delta_started_at: dict[tuple[str, str], float] = {}

    async def create_conversation(
        self,
        spec: SessionSpec,
        context: RequestContext,
    ) -> ConversationSnapshot:
        total_started_at = time.perf_counter()
        adapter = self._adapter(self.default_harness_id)
        adapter_started_at = time.perf_counter()
        handle = await adapter.create_session(spec, context)
        self._log_timing(
            "adapter.create_session_ms",
            adapter_started_at,
            conversation_id=handle.session_id,
            harness_id=handle.harness_id,
        )
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
        active = self._start_pump(handle, adapter, store, context)
        self._active[handle.session_id] = active
        ready_started_at = time.perf_counter()
        await active.ready.wait()
        self._log_timing(
            "runtime.ready_wait_ms",
            ready_started_at,
            conversation_id=handle.session_id,
            harness_id=handle.harness_id,
        )
        self._first_command_pending.add(handle.session_id)
        snapshot = store.load_snapshot()
        self._log_timing(
            "product.create.total_ms",
            total_started_at,
            conversation_id=handle.session_id,
            harness_id=handle.harness_id,
        )
        return snapshot

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
        active.context = context
        first_command = (
            isinstance(command, UserMessageCommand)
            and conversation_id in self._first_command_pending
        )
        command_started_at = time.perf_counter()
        if isinstance(command, UserMessageCommand):
            self._first_delta_started_at[(conversation_id, command.command_id)] = (
                command_started_at
            )
        try:
            response = await active.adapter.send(active.handle, command, context)
        except Exception as exc:
            if isinstance(command, UserMessageCommand):
                self._first_delta_started_at.pop(
                    (conversation_id, command.command_id), None
                )
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
        if first_command:
            self._first_command_pending.discard(conversation_id)
            self._log_timing(
                "first_command.accept_ms",
                command_started_at,
                conversation_id=conversation_id,
                harness_id=active.handle.harness_id,
            )
        completed = receipt.model_copy(
            update={"status": "completed", "response": response}
        )
        return store.complete_command(completed)

    async def deliver_external_task_status(
        self,
        conversation_id: str,
        *,
        task_id: str,
        status: str,
        error_summary: str | None = None,
        auto_run: bool = True,
        from_workflow_debug: bool = False,
    ) -> ProductEvent:
        """Persist one callback and notify its owning harness at most once."""
        store = self._store(conversation_id)
        snapshot = store.load_snapshot()
        task = next(
            (item for item in snapshot.external_tasks if item.task_id == task_id),
            None,
        )
        if task is None and self._external_tasks is not None:
            payload = self._external_tasks.resolve(conversation_id, task_id)
            if payload is not None:
                self.register_external_task(conversation_id, dict(payload))
                snapshot = store.load_snapshot()
                task = next(
                    (
                        item
                        for item in snapshot.external_tasks
                        if item.task_id == task_id
                    ),
                    None,
                )
        if task is None:
            raise ValueError(f"unknown external task: {task_id}")
        normalized = _normalize_external_status(status)
        terminal = normalized in {"succeeded", "failed", "terminated", "stopped"}
        active = self._active.get(conversation_id)
        notification: dict[str, Any] = {
            **task.model_dump(mode="json"),
            "status": normalized,
            "updated_at": datetime.now().astimezone().isoformat(),
            "resume_pending": active is None and normalized != "stopped",
            "error_summary": _controlled_error(error_summary),
        }
        event = ProductEvent(
            event_id=(
                f"external-task:{task_id}:terminal"
                if terminal
                else f"external-task:{task_id}:{normalized}"
            ),
            conversation_id=conversation_id,
            type="external_task.completed" if terminal else "external_task.updated",
            payload=notification,
        )
        before_seq = snapshot.through_seq
        persisted, updated_snapshot = store.append(event)
        if self._external_tasks is not None:
            try:
                persisted_status = persisted.payload.get("status")
                if not isinstance(persisted_status, str):
                    raise ValueError("persisted external task status is missing")
                self._external_tasks.update_status(
                    conversation_id, task_id, persisted_status
                )
            except Exception:
                logger.exception(
                    "Could not update external task registry for %s", task_id
                )
        if updated_snapshot.through_seq == before_seq:
            logger.info(
                "external_task.callback_duplicate conversation_id=%s task_id=%s "
                "status=%s",
                conversation_id,
                task_id,
                normalized,
            )
            return persisted
        self._publish(persisted)
        if active is not None and terminal and normalized != "stopped":
            try:
                await active.adapter.notify_external_task(
                    active.handle,
                    {
                        **notification,
                        "auto_run": auto_run,
                        "from_workflow_debug": from_workflow_debug,
                    },
                    active.context,
                )
            except Exception:
                logger.exception(
                    "Could not notify %s for external task %s; deferring",
                    active.handle.harness_id,
                    task_id,
                )
                pending = {**notification, "resume_pending": True}
                persisted, _ = store.append(
                    ProductEvent(
                        event_id=f"external-task:{task_id}:{normalized}:pending",
                        conversation_id=conversation_id,
                        type="external_task.updated",
                        payload=pending,
                    )
                )
                self._publish(persisted)
        return persisted

    def resolve_external_task_owner(self, task_id: str) -> str | None:
        if self._external_tasks is None:
            return None
        return self._external_tasks.owner(task_id)

    def register_external_task(
        self, conversation_id: str, payload: dict[str, Any]
    ) -> ProductEvent:
        store = self._store(conversation_id)
        snapshot = store.load_snapshot()
        task_id = str(payload.get("task_id") or "")
        if not task_id:
            raise ValueError("external task_id is required")
        event = ProductEvent(
            event_id=f"external-task:{task_id}:submitted",
            conversation_id=conversation_id,
            type="external_task.submitted",
            payload=payload,
        )
        persisted, updated = store.append(event)
        if updated.through_seq != snapshot.through_seq:
            self._publish(persisted)
        return persisted

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
        self._first_command_pending.clear()
        self._first_delta_started_at.clear()
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
            active = self._start_pump(handle, adapter, store, context)
            self._active[conversation_id] = active
            await active.ready.wait()
            await self._resume_pending_external_tasks(active, store)
            return store

    async def _resume_pending_external_tasks(
        self, active: _ActiveConversation, store: FileProductStore
    ) -> None:
        for task in store.load_snapshot().external_tasks:
            if not task.resume_pending or task.status == "stopped":
                continue
            notification = task.model_dump(mode="json")
            await active.adapter.notify_external_task(
                active.handle, notification, active.context
            )
            notification["resume_pending"] = False
            persisted, _ = store.append(
                ProductEvent(
                    event_id=f"external-task:{task.task_id}:{task.status}:resumed",
                    conversation_id=active.handle.session_id,
                    type="external_task.updated",
                    payload=notification,
                )
            )
            self._publish(persisted)

    def _start_pump(
        self,
        handle: SessionHandle,
        adapter: HarnessAdapter,
        store: FileProductStore,
        context: RequestContext,
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
            context=context,
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
                if harness_event.type == "message.delta" and harness_event.run_id:
                    started_at = self._first_delta_started_at.pop(
                        (handle.session_id, harness_event.run_id), None
                    )
                    if started_at is not None:
                        self._log_timing(
                            "first_delta_latency_ms",
                            started_at,
                            conversation_id=handle.session_id,
                            harness_id=handle.harness_id,
                            run_id=harness_event.run_id,
                        )
                elif (
                    harness_event.type == "message.completed"
                    and harness_event.run_id
                    and harness_event.payload.get("role") == "assistant"
                ):
                    self._first_delta_started_at.pop(
                        (handle.session_id, harness_event.run_id), None
                    )
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

    @staticmethod
    def _log_timing(
        metric: str,
        started_at: float,
        **dimensions: str,
    ) -> None:
        elapsed_ms = (time.perf_counter() - started_at) * 1000
        suffix = " ".join(f"{key}={value}" for key, value in dimensions.items())
        logger.info("%s=%.3f %s", metric, elapsed_ms, suffix)


def _normalize_external_status(value: str) -> str:
    normalized = value.strip().lower()
    aliases = {
        "success": "succeeded",
        "succeeded": "succeeded",
        "error": "failed",
        "failed": "failed",
        "terminated": "terminated",
        "stopped": "stopped",
        "pending": "pending",
        "running": "running",
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        raise ValueError(f"unsupported external task status: {value!r}") from exc


def _controlled_error(value: str | None) -> str | None:
    if not value:
        return None
    compact = " ".join(value.split())
    return compact[:2000]
