from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest
from pyromind_runtime.contracts import (
    HarnessCapabilities,
    HarnessCommand,
    HarnessDescriptor,
    HarnessEvent,
    PermissionResponse,
    SessionHandle,
    SessionSpec,
)
from pyromind_runtime.product import (
    HarnessRegistry,
    ProductRuntimeService,
    ProductRuntimeSettings,
)


class FakeHarness:
    def __init__(self) -> None:
        self.capabilities = HarnessCapabilities(
            cancel=True,
            partial_message=True,
            custom_tools=True,
            native_workspace_tools=frozenset({"read", "write", "edit", "bash"}),
        )
        self.queues: dict[str, asyncio.Queue[HarnessEvent | None]] = {}
        self.sent: list[tuple[str, HarnessCommand]] = []
        self.cancelled: list[str] = []
        self.closed: list[str] = []
        self.created_specs: list[SessionSpec] = []
        self.forked: list[tuple[str, str]] = []

    async def describe(self) -> HarnessDescriptor:
        return HarnessDescriptor(
            harness_id="fake",
            display_name="Fake Harness",
            capabilities=self.capabilities,
        )

    async def create_session(self, spec: SessionSpec) -> SessionHandle:
        self.created_specs.append(spec)
        self.queues[spec.product_session_id] = asyncio.Queue()
        return SessionHandle(
            session_id=spec.product_session_id,
            harness_id="fake",
            adapter_session_ref=f"fake-{spec.product_session_id}",
            capabilities=self.capabilities,
        )

    async def send(self, session_id: str, command: HarnessCommand) -> None:
        self.sent.append((session_id, command))

    async def cancel(self, session_id: str) -> None:
        self.cancelled.append(session_id)

    async def respond_permission(
        self,
        session_id: str,
        response: PermissionResponse,
    ) -> None:
        raise AssertionError(
            f"unexpected permission response for {session_id}: {response}"
        )

    def subscribe(self, session_id: str) -> AsyncIterator[HarnessEvent]:
        queue = self.queues[session_id]

        async def stream() -> AsyncIterator[HarnessEvent]:
            while True:
                event = await queue.get()
                if event is None:
                    return
                yield event

        return stream()

    async def close(self, session_id: str) -> None:
        self.closed.append(session_id)
        queue = self.queues.get(session_id)
        if queue is not None:
            queue.put_nowait(None)

    async def fork_session(
        self,
        source_session_id: str,
        spec: SessionSpec,
    ) -> SessionHandle:
        self.forked.append((source_session_id, spec.product_session_id))
        return await self.create_session(spec)

    def emit(self, session_id: str, event: HarnessEvent) -> None:
        self.queues[session_id].put_nowait(event)


@pytest.fixture
def fake_harness() -> FakeHarness:
    return FakeHarness()


@pytest.fixture
def product_runtime(tmp_path, fake_harness: FakeHarness) -> ProductRuntimeService:
    registry = HarnessRegistry()
    registry.register("fake", fake_harness)
    return ProductRuntimeService(
        ProductRuntimeSettings(
            storage_root=tmp_path / "product-conversations",
            default_harness_id="fake",
            default_workspace_root=str(tmp_path / "workspace"),
        ),
        registry,
    )
