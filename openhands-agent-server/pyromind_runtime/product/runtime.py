from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncGenerator, AsyncIterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from pyromind_runtime.contracts.events import HarnessEvent, ProductEvent
from pyromind_runtime.contracts.harness import (
    CapabilityName,
    ForkableHarnessProtocol,
    HarnessProtocol,
    PermissionResponse,
    SessionHandle,
    SessionSpec,
)
from pyromind_runtime.contracts.sandbox import ModelProfile, SandboxRef, WorkspaceRef
from pyromind_runtime.contracts.tools import ToolSpec
from pyromind_runtime.product.commands import (
    CancelCommand,
    PermissionResponseCommand,
    ProductCommand,
    UserMessageCommand,
)
from pyromind_runtime.product.event_store import (
    ConversationNotFoundError,
    EventStoreError,
    FileConversationEventStore,
)
from pyromind_runtime.product.models import (
    CommandReceipt,
    ConversationMetadata,
    ConversationSnapshot,
)
from pyromind_runtime.product.registry import HarnessRegistry
from pyromind_runtime.projectors import (
    ConversationSnapshotProjector,
    ProductEventProjector,
    SnapshotProjectionError,
    WorkflowProductProjector,
)
from pyromind_runtime.tool_host import SessionToolContextStore, ToolRequestContext


logger = logging.getLogger(__name__)


class ProductRuntimeError(RuntimeError):
    pass


class ProductConversationNotActiveError(ProductRuntimeError):
    pass


class CapabilityNotSupportedError(ProductRuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class ProductRuntimeSettings:
    storage_root: Path
    default_harness_id: str
    default_workspace_root: str
    default_model_profile_id: str = "default"


@dataclass(slots=True)
class _ActiveSession:
    adapter: HarnessProtocol
    handle: SessionHandle
    event_task: asyncio.Task[None]


class _ProductEventBus:
    def __init__(self) -> None:
        self._subscribers: dict[str, set[asyncio.Queue[ProductEvent]]] = {}

    def add(self, conversation_id: str) -> asyncio.Queue[ProductEvent]:
        queue: asyncio.Queue[ProductEvent] = asyncio.Queue()
        self._subscribers.setdefault(conversation_id, set()).add(queue)
        return queue

    def remove(
        self,
        conversation_id: str,
        queue: asyncio.Queue[ProductEvent],
    ) -> None:
        subscribers = self._subscribers.get(conversation_id)
        if subscribers is None:
            return
        subscribers.discard(queue)
        if not subscribers:
            self._subscribers.pop(conversation_id)

    def publish(self, event: ProductEvent) -> None:
        for queue in tuple(self._subscribers.get(event.conversation_id, ())):
            queue.put_nowait(event)


class ProductRuntimeService:
    def __init__(
        self,
        settings: ProductRuntimeSettings,
        registry: HarnessRegistry | None = None,
        *,
        default_tools_by_harness: Mapping[str, tuple[ToolSpec, ...]] | None = None,
        tool_context_store: SessionToolContextStore | None = None,
    ) -> None:
        self.settings = settings
        self.registry = registry or HarnessRegistry()
        self.product_projector = ProductEventProjector()
        self.snapshot_projector = ConversationSnapshotProjector()
        self.workflow_projector = WorkflowProductProjector()
        self._default_tools_by_harness = dict(default_tools_by_harness or {})
        self._tool_context_store = tool_context_store
        self._active_sessions: dict[str, _ActiveSession] = {}
        self._activation_locks: dict[str, asyncio.Lock] = {}
        self._event_bus = _ProductEventBus()

    async def create_conversation(
        self,
        *,
        user_id: str,
        workspace: WorkspaceRef | None = None,
        sandbox: SandboxRef | None = None,
        model_profile: ModelProfile | None = None,
        tools: tuple[ToolSpec, ...] | None = None,
        required_capabilities: frozenset[CapabilityName] = frozenset(),
        conversation_id: str | None = None,
        tool_context: ToolRequestContext | None = None,
    ) -> ConversationSnapshot:
        product_session_id = conversation_id or uuid4().hex
        adapter, _ = await self.registry.resolve(self.settings.default_harness_id)
        spec = SessionSpec(
            product_session_id=product_session_id,
            user_id=user_id,
            workspace=workspace
            or WorkspaceRef(
                workspace_id=product_session_id,
                root=str(
                    Path(self.settings.default_workspace_root) / product_session_id
                ),
            ),
            sandbox=sandbox
            or SandboxRef(
                sandbox_id=product_session_id,
                backend="managed",
                lease_id=product_session_id,
            ),
            model_profile=model_profile
            or ModelProfile(profile_id=self.settings.default_model_profile_id),
            tools=(
                tools
                if tools is not None
                else self._default_tools_by_harness.get(
                    self.settings.default_harness_id,
                    (),
                )
            ),
            required_capabilities=required_capabilities,
        )
        if tool_context is not None and self._tool_context_store is not None:
            self._tool_context_store.bind(product_session_id, tool_context)
        try:
            handle = await adapter.create_session(spec)
        except Exception:
            if self._tool_context_store is not None:
                self._tool_context_store.remove(product_session_id)
            raise
        missing = handle.capabilities.missing(required_capabilities)
        if missing:
            await adapter.close(handle.session_id)
            if self._tool_context_store is not None:
                self._tool_context_store.remove(product_session_id)
            missing_names = ", ".join(sorted(missing))
            raise CapabilityNotSupportedError(
                f"harness does not satisfy required capabilities: {missing_names}"
            )
        metadata = ConversationMetadata(
            conversation_id=product_session_id,
            user_id=user_id,
            harness_id=handle.harness_id,
            adapter_session_ref=handle.adapter_session_ref,
            capabilities=handle.capabilities,
            workspace=spec.workspace,
            sandbox=spec.sandbox,
        )
        snapshot = ConversationSnapshot(
            conversation_id=product_session_id,
            capabilities=handle.capabilities,
        )
        store = self._store(product_session_id)
        try:
            await asyncio.to_thread(store.create, metadata, snapshot)
            creation_event = ProductEvent(
                conversation_id=product_session_id,
                type="conversation.created",
                payload={
                    "harness_id": handle.harness_id,
                    "capabilities": handle.capabilities.model_dump(mode="json"),
                },
            )
            _, snapshot = await self._append(creation_event)
            event_stream = adapter.subscribe(handle.session_id)
            event_task = asyncio.create_task(
                self._pump_events(product_session_id, event_stream),
                name=f"product-events-{product_session_id}",
            )
            self._active_sessions[product_session_id] = _ActiveSession(
                adapter=adapter,
                handle=handle,
                event_task=event_task,
            )
        except Exception:
            if self._tool_context_store is not None:
                self._tool_context_store.remove(product_session_id)
            await adapter.close(handle.session_id)
            raise
        return snapshot

    async def list_conversations(
        self,
        user_id: str,
    ) -> tuple[ConversationSnapshot, ...]:
        return await asyncio.to_thread(self._list_conversations_sync, user_id)

    async def get_snapshot(
        self,
        conversation_id: str,
        user_id: str,
    ) -> ConversationSnapshot:
        store = self._store(conversation_id)
        await asyncio.to_thread(self._authorize, store, user_id)
        return await asyncio.to_thread(
            store.load_snapshot,
            self.snapshot_projector.reduce,
        )

    async def fork_conversation(
        self,
        conversation_id: str,
        user_id: str,
        *,
        after_seq: int | None = None,
    ) -> ConversationSnapshot:
        source_store = self._store(conversation_id)
        metadata = await asyncio.to_thread(self._authorize, source_store, user_id)
        source_snapshot = await asyncio.to_thread(
            source_store.load_snapshot,
            self.snapshot_projector.reduce,
        )
        fork_seq = source_snapshot.through_seq if after_seq is None else after_seq
        if fork_seq != source_snapshot.through_seq:
            raise CapabilityNotSupportedError(
                "historical forks require workspace snapshots and are not supported "
                "in the first version"
            )
        if source_snapshot.status != "idle":
            raise ProductRuntimeError("only idle conversations can be forked")
        if not metadata.capabilities.fork:
            raise CapabilityNotSupportedError(
                f"harness {metadata.harness_id} does not support fork"
            )

        source_session = await self._ensure_active(conversation_id, source_store)
        adapter = source_session.adapter
        if not isinstance(adapter, ForkableHarnessProtocol):
            raise CapabilityNotSupportedError(
                f"harness {metadata.harness_id} does not implement fork"
            )

        fork_id = uuid4().hex
        spec = SessionSpec(
            product_session_id=fork_id,
            user_id=user_id,
            workspace=metadata.workspace,
            sandbox=metadata.sandbox,
            model_profile=ModelProfile(
                profile_id=self.settings.default_model_profile_id
            ),
            tools=self._default_tools_by_harness.get(metadata.harness_id, ()),
            required_capabilities=frozenset({"fork"}),
        )
        source_events = await asyncio.to_thread(source_store.replay, 0)
        handle = await adapter.fork_session(source_session.handle.session_id, spec)
        if handle.harness_id != metadata.harness_id:
            await adapter.close(handle.session_id)
            raise ProductRuntimeError("forked harness does not preserve harness_id")

        fork_metadata = ConversationMetadata(
            conversation_id=fork_id,
            user_id=user_id,
            harness_id=handle.harness_id,
            adapter_session_ref=handle.adapter_session_ref,
            capabilities=handle.capabilities,
            workspace=spec.workspace,
            sandbox=spec.sandbox,
        )
        fork_snapshot = ConversationSnapshot(
            conversation_id=fork_id,
            capabilities=handle.capabilities,
        )
        fork_store = self._store(fork_id)
        try:
            await asyncio.to_thread(fork_store.create, fork_metadata, fork_snapshot)
            _, fork_snapshot = await self._append(
                ProductEvent(
                    conversation_id=fork_id,
                    type="conversation.created",
                    payload={
                        "harness_id": handle.harness_id,
                        "capabilities": handle.capabilities.model_dump(mode="json"),
                        "forked_from": conversation_id,
                        "forked_through_seq": fork_seq,
                    },
                )
            )
            for source_event in source_events:
                if (
                    source_event.seq > fork_seq
                    or source_event.type == "conversation.created"
                ):
                    continue
                copied_event = source_event.model_copy(
                    update={
                        "event_id": uuid4().hex,
                        "conversation_id": fork_id,
                        "seq": 0,
                    }
                )
                _, fork_snapshot = await self._append(copied_event)

            current_source = await asyncio.to_thread(
                source_store.load_snapshot,
                self.snapshot_projector.reduce,
            )
            if current_source.through_seq != fork_seq:
                raise ProductRuntimeError(
                    "source conversation changed while the fork was being created"
                )
            event_stream = adapter.subscribe(handle.session_id)
            event_task = asyncio.create_task(
                self._pump_events(fork_id, event_stream),
                name=f"product-events-{fork_id}",
            )
            self._active_sessions[fork_id] = _ActiveSession(
                adapter=adapter,
                handle=handle,
                event_task=event_task,
            )
            return fork_snapshot
        except Exception:
            await adapter.close(handle.session_id)
            raise

    async def submit_command(
        self,
        conversation_id: str,
        command: ProductCommand,
        user_id: str,
        tool_context: ToolRequestContext | None = None,
    ) -> CommandReceipt:
        store = self._store(conversation_id)
        await asyncio.to_thread(self._authorize, store, user_id)
        if tool_context is not None and self._tool_context_store is not None:
            self._tool_context_store.bind(conversation_id, tool_context)
        session = await self._ensure_active(conversation_id, store)
        receipt, created = await asyncio.to_thread(
            store.claim_command,
            command.command_id,
        )
        if not created:
            return receipt

        try:
            if isinstance(command, UserMessageCommand):
                await session.adapter.send(session.handle.session_id, command)
            elif isinstance(command, CancelCommand):
                if not session.handle.capabilities.cancel:
                    raise CapabilityNotSupportedError("cancel is not supported")
                await session.adapter.cancel(session.handle.session_id)
            elif isinstance(command, PermissionResponseCommand):
                if not session.handle.capabilities.permission_reply:
                    raise CapabilityNotSupportedError(
                        "permission replies are not supported"
                    )
                await session.adapter.respond_permission(
                    session.handle.session_id,
                    PermissionResponse(
                        permission_id=command.permission_id,
                        decision=command.decision,
                        reason=command.reason,
                    ),
                )
        except Exception as exc:
            failed = CommandReceipt(
                command_id=command.command_id,
                status="failed",
                response={"error": str(exc)},
            )
            await asyncio.to_thread(store.complete_command, failed)
            raise

        completed = CommandReceipt(
            command_id=command.command_id,
            status="completed",
            response={"accepted": True},
        )
        return await asyncio.to_thread(store.complete_command, completed)

    async def stream_events(
        self,
        conversation_id: str,
        after_seq: int,
        user_id: str,
    ) -> AsyncGenerator[ProductEvent]:
        await asyncio.to_thread(self._authorize, self._store(conversation_id), user_id)
        queue = self._event_bus.add(conversation_id)
        cursor = after_seq
        try:
            replay = await asyncio.to_thread(
                self._store(conversation_id).replay,
                cursor,
            )
            for event in replay:
                cursor = event.seq
                yield event
            while True:
                event = await queue.get()
                if event.seq <= cursor:
                    continue
                cursor = event.seq
                yield event
        finally:
            self._event_bus.remove(conversation_id, queue)

    async def close(self) -> None:
        sessions = tuple(self._active_sessions.values())
        self._active_sessions.clear()
        self._activation_locks.clear()
        for session in sessions:
            session.event_task.cancel()
        await asyncio.gather(
            *(session.event_task for session in sessions),
            return_exceptions=True,
        )
        await asyncio.gather(
            *(session.adapter.close(session.handle.session_id) for session in sessions),
            return_exceptions=True,
        )
        if self._tool_context_store is not None:
            self._tool_context_store.clear()

    async def _pump_events(
        self,
        conversation_id: str,
        event_stream: AsyncIterator[HarnessEvent],
    ) -> None:
        try:
            async for harness_event in event_stream:
                if harness_event.type == "resource.updated":
                    metadata = await asyncio.to_thread(
                        self._store(conversation_id).load_metadata
                    )
                    workflow_event = await asyncio.to_thread(
                        self.workflow_projector.project,
                        conversation_id,
                        metadata.workspace,
                        harness_event,
                    )
                    if workflow_event is not None:
                        await self._append(workflow_event)
                    continue
                for product_event in self.product_projector.project(
                    conversation_id,
                    harness_event,
                ):
                    await self._append(product_event)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception(
                "Harness event pump failed for conversation %s",
                conversation_id,
            )
            failure = ProductEvent(
                conversation_id=conversation_id,
                type="run.failed",
                payload={"error_code": "event_pump_failed", "message": str(exc)},
            )
            try:
                await self._append(failure)
            except Exception:
                logger.exception(
                    "Failed to persist event pump failure for conversation %s",
                    conversation_id,
                )

    async def _append(
        self,
        event: ProductEvent,
    ) -> tuple[ProductEvent, ConversationSnapshot]:
        persisted, snapshot = await asyncio.to_thread(
            self._store(event.conversation_id).append,
            event,
            self.snapshot_projector.reduce,
        )
        self._event_bus.publish(persisted)
        return persisted, snapshot

    def _store(self, conversation_id: str) -> FileConversationEventStore:
        return FileConversationEventStore(self.settings.storage_root, conversation_id)

    async def _ensure_active(
        self,
        conversation_id: str,
        store: FileConversationEventStore,
    ) -> _ActiveSession:
        active = self._active_sessions.get(conversation_id)
        if active is not None:
            return active

        lock = self._activation_locks.setdefault(conversation_id, asyncio.Lock())
        async with lock:
            active = self._active_sessions.get(conversation_id)
            if active is not None:
                return active

            metadata = await asyncio.to_thread(store.load_metadata)
            if not metadata.capabilities.resume:
                raise ProductConversationNotActiveError(
                    f"conversation {conversation_id} is inactive and harness "
                    f"{metadata.harness_id} does not support resume"
                )

            adapter, _ = await self.registry.resolve(metadata.harness_id)
            spec = SessionSpec(
                product_session_id=metadata.conversation_id,
                user_id=metadata.user_id,
                workspace=metadata.workspace,
                sandbox=metadata.sandbox,
                model_profile=ModelProfile(
                    profile_id=self.settings.default_model_profile_id
                ),
                tools=self._default_tools_by_harness.get(metadata.harness_id, ()),
                required_capabilities=frozenset({"resume"}),
            )
            handle = await adapter.create_session(spec)
            try:
                if handle.harness_id != metadata.harness_id:
                    raise ProductRuntimeError(
                        "resumed harness does not match persisted harness_id"
                    )
                if handle.session_id != metadata.conversation_id:
                    raise ProductRuntimeError(
                        "resumed harness does not preserve the product session id"
                    )
                if handle.adapter_session_ref != metadata.adapter_session_ref:
                    raise ProductRuntimeError(
                        "resumed harness does not match the persisted adapter session"
                    )
                event_stream = adapter.subscribe(handle.session_id)
                event_task = asyncio.create_task(
                    self._pump_events(conversation_id, event_stream),
                    name=f"product-events-{conversation_id}",
                )
                active = _ActiveSession(
                    adapter=adapter,
                    handle=handle,
                    event_task=event_task,
                )
                self._active_sessions[conversation_id] = active
                return active
            except Exception:
                await adapter.close(handle.session_id)
                raise

    def _list_conversations_sync(
        self,
        user_id: str,
    ) -> tuple[ConversationSnapshot, ...]:
        if not self.settings.storage_root.exists():
            return ()
        snapshots: list[ConversationSnapshot] = []
        for directory in sorted(self.settings.storage_root.iterdir()):
            if not directory.is_dir():
                continue
            try:
                store = self._store(directory.name)
                metadata = store.load_metadata()
                if metadata.user_id != user_id:
                    continue
                snapshots.append(store.load_snapshot(self.snapshot_projector.reduce))
            except (EventStoreError, SnapshotProjectionError, ValueError):
                logger.warning(
                    "Skipping invalid product conversation directory %s",
                    directory,
                )
        return tuple(snapshots)

    @staticmethod
    def _authorize(
        store: FileConversationEventStore,
        user_id: str,
    ) -> ConversationMetadata:
        metadata = store.load_metadata()
        if metadata.user_id != user_id:
            raise ConversationNotFoundError(store.conversation_id)
        return metadata
