import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from openhands.agent_server.pyromind_constants import (
    PYROMIND_APP_TAG_KEY,
    PYROMIND_APP_TAG_VALUE,
)
from openhands.agent_server.workflow_task_poller import WorkflowTaskPoller
from openhands.sdk.conversation.state import ActiveLongTask
from openhands.tools.embodied_data.platform_submit import RunEmbodiedSandboxTool


class _State(SimpleNamespace):
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None


class _SecretRegistry:
    def get_secret_value(self, name: str) -> str:
        return "session-token"


class _EventService:
    def __init__(self, status: str = "Pending") -> None:
        self.stored = SimpleNamespace(
            id=uuid4(),
            tags={PYROMIND_APP_TAG_KEY: PYROMIND_APP_TAG_VALUE},
        )
        self.state = _State(
            active_long_tasks=[
                ActiveLongTask(
                    task_id="8176",
                    kind="embodied_data_cleaning",
                    status=status,
                )
            ],
            agent=SimpleNamespace(
                tools=[
                    SimpleNamespace(
                        name=RunEmbodiedSandboxTool.name,
                        params={
                            "env": "pre",
                            "cluster": "test-cluster",
                            "headers": {"request-app": "openhands"},
                        },
                    )
                ]
            ),
            secret_registry=_SecretRegistry(),
        )
        self.updated_statuses: list[tuple[str, str]] = []

    def get_conversation(self):
        return SimpleNamespace(_state=self.state)

    async def update_active_long_task_status(
        self,
        task_id: str,
        status: str,
    ) -> None:
        self.updated_statuses.append((task_id, status))


class _ConversationService:
    def __init__(self, event_service: _EventService) -> None:
        self.event_service = event_service

    def _loaded_event_services_snapshot(self):
        return [self.event_service]


@pytest.mark.asyncio
async def test_poller_delivers_terminal_status_through_shared_callback() -> None:
    event_service = _EventService()
    service = _ConversationService(event_service)
    get_task = MagicMock(return_value=SimpleNamespace(status="Succeeded"))
    client_factory = MagicMock(
        return_value=SimpleNamespace(studio=SimpleNamespace(get_task=get_task))
    )
    status_callback = AsyncMock(return_value=SimpleNamespace(outcome="delivered_async"))
    poller = WorkflowTaskPoller(
        service,
        interval_seconds=1,
        client_factory=client_factory,
        status_callback=status_callback,
    )

    await poller.run_once()

    client_factory.assert_called_once_with(
        env="pre",
        cluster="test-cluster",
        auth_token="session-token",
        headers={"request-app": "openhands"},
        timeout=30,
    )
    get_task.assert_called_once_with("8176")
    status_callback.assert_awaited_once_with(
        task_id="8176",
        status="Succeeded",
        conversation_id=str(event_service.stored.id),
        auto_run=True,
        conversation_service=service,
    )
    assert event_service.updated_statuses == []


@pytest.mark.asyncio
async def test_poller_updates_non_terminal_status_without_waking_agent() -> None:
    event_service = _EventService()
    service = _ConversationService(event_service)
    client = SimpleNamespace(
        studio=SimpleNamespace(
            get_task=MagicMock(return_value=SimpleNamespace(status="Running"))
        )
    )
    status_callback = AsyncMock()
    poller = WorkflowTaskPoller(
        service,
        interval_seconds=1,
        client_factory=MagicMock(return_value=client),
        status_callback=status_callback,
    )

    await poller.run_once()

    assert event_service.updated_statuses == [("8176", "Running")]
    status_callback.assert_not_awaited()


@pytest.mark.asyncio
async def test_poller_ignores_stopped_tasks() -> None:
    event_service = _EventService(status="Stopped")
    client_factory = MagicMock()
    poller = WorkflowTaskPoller(
        _ConversationService(event_service),
        interval_seconds=1,
        client_factory=client_factory,
        status_callback=AsyncMock(),
    )

    await poller.run_once()

    client_factory.assert_not_called()


@pytest.mark.asyncio
async def test_poller_background_task_stops_cleanly() -> None:
    service = SimpleNamespace(_loaded_event_services_snapshot=lambda: [])
    poller = WorkflowTaskPoller(service, interval_seconds=60)

    poller.start()
    await asyncio.sleep(0)

    assert poller._loop_task is not None
    assert not poller._loop_task.done()

    await poller.stop()

    assert poller._loop_task is None
