"""Tests for bash_router.py endpoints."""

import asyncio
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from openhands.agent_server.api import create_app
from openhands.agent_server.bash_service import BashEventService
from openhands.agent_server.config import Config
from openhands.agent_server.dependencies import (
    get_bash_event_service,
    get_conversation_service,
)
from openhands.agent_server.models import BashCommand, BashEventPage, BashOutput


@pytest.fixture
def test_bash_service():
    """Create a BashEventService instance for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        yield BashEventService(
            bash_events_dir=temp_path / "bash_events",
        )


@pytest.fixture
def client():
    """Create a test client for the FastAPI app without authentication."""
    config = Config(
        session_api_keys=[],
        enable_pyromind_jwt_auth=False,
    )
    return TestClient(create_app(config))


def test_clear_all_bash_events_empty_storage():
    """Test clearing bash events when storage is empty."""
    mock_service = MagicMock(spec=BashEventService)
    mock_service.clear_all_events = AsyncMock(return_value=0)

    config = Config(
        session_api_keys=[],
        enable_pyromind_jwt_auth=False,
    )
    app = create_app(config)
    app.dependency_overrides[get_bash_event_service] = lambda: mock_service

    client = TestClient(app)
    response = client.delete("/api/bash/bash_events")

    assert response.status_code == 200
    assert response.json() == {"cleared_count": 0}
    mock_service.clear_all_events.assert_called_once()


def test_clear_all_bash_events_with_data():
    """Test clearing bash events when storage contains data."""
    mock_service = MagicMock(spec=BashEventService)
    mock_service.clear_all_events = AsyncMock(return_value=5)

    config = Config(
        session_api_keys=[],
        enable_pyromind_jwt_auth=False,
    )
    app = create_app(config)
    app.dependency_overrides[get_bash_event_service] = lambda: mock_service

    client = TestClient(app)
    response = client.delete("/api/bash/bash_events")

    assert response.status_code == 200
    assert response.json() == {"cleared_count": 5}
    mock_service.clear_all_events.assert_called_once()


def test_start_bash_command_disabled_in_multi_tenant():
    mock_service = MagicMock(spec=BashEventService)
    mock_service.start_bash_command = AsyncMock()

    config = Config(
        session_api_keys=[],
        enable_pyromind_jwt_auth=False,
        command_policy_mode="multi_tenant_strict",
    )
    app = create_app(config)
    app.dependency_overrides[get_bash_event_service] = lambda: mock_service

    client = TestClient(app)
    response = client.post(
        "/api/bash/start_bash_command",
        json={"command": "echo blocked"},
    )

    assert response.status_code == 403
    assert response.json()["detail"]["code"] == "direct_bash_api_disabled"
    mock_service.start_bash_command.assert_not_called()


def test_execute_bash_command_disabled_in_multi_tenant():
    mock_service = MagicMock(spec=BashEventService)
    mock_service.start_bash_command = AsyncMock()

    config = Config(
        session_api_keys=[],
        enable_pyromind_jwt_auth=False,
        command_policy_mode="multi_tenant_strict",
    )
    app = create_app(config)
    app.dependency_overrides[get_bash_event_service] = lambda: mock_service

    client = TestClient(app)
    response = client.post(
        "/api/bash/execute_bash_command",
        json={"command": "echo blocked"},
    )

    assert response.status_code == 403
    assert response.json()["detail"]["code"] == "direct_bash_api_disabled"
    mock_service.start_bash_command.assert_not_called()


def test_conversation_scoped_bash_executes_in_conversation_workspace(tmp_path: Path):
    conversation_id = uuid4()
    command = BashCommand(command="echo scoped", cwd=str(tmp_path))
    output = BashOutput(command_id=command.id, order=0, exit_code=0, stdout="scoped\n")
    event_service = MagicMock()
    event_service.get_conversation.return_value = SimpleNamespace(
        workspace=SimpleNamespace(working_dir=str(tmp_path))
    )
    conversation_service = MagicMock()
    conversation_service.get_event_service = AsyncMock(return_value=event_service)
    bash_service = MagicMock(spec=BashEventService)

    async def start_command(request):
        assert request.cwd == str(tmp_path)
        task = asyncio.create_task(asyncio.sleep(0))
        return command, task

    bash_service.start_bash_command = AsyncMock(side_effect=start_command)
    bash_service.search_bash_events = AsyncMock(
        return_value=BashEventPage(items=[command, output])
    )

    config = Config(session_api_keys=[], command_policy_mode="multi_tenant_strict")
    app = create_app(config)
    app.dependency_overrides[get_conversation_service] = lambda: conversation_service
    app.dependency_overrides[get_bash_event_service] = lambda: bash_service

    client = TestClient(app)
    response = client.post(
        f"/api/conversations/{conversation_id}/bash/execute",
        json={"command": "echo scoped"},
        headers={"x-pyromind-debug-user-id": "123"},
    )

    assert response.status_code == 200
    assert response.json()["stdout"] == "scoped\n"
    conversation_service.get_event_service.assert_awaited_once_with(
        conversation_id,
        user_id="123",
    )
    bash_service.start_bash_command.assert_awaited_once()


def test_conversation_scoped_bash_denies_policy_blocked_command(tmp_path: Path):
    conversation_id = uuid4()
    event_service = MagicMock()
    event_service.get_conversation.return_value = SimpleNamespace(
        workspace=SimpleNamespace(working_dir=str(tmp_path))
    )
    conversation_service = MagicMock()
    conversation_service.get_event_service = AsyncMock(return_value=event_service)
    bash_service = MagicMock(spec=BashEventService)
    bash_service.start_bash_command = AsyncMock()

    config = Config(session_api_keys=[], command_policy_mode="multi_tenant_strict")
    app = create_app(config)
    app.dependency_overrides[get_conversation_service] = lambda: conversation_service
    app.dependency_overrides[get_bash_event_service] = lambda: bash_service

    client = TestClient(app)
    response = client.post(
        f"/api/conversations/{conversation_id}/bash/execute",
        json={"command": "cat .env"},
        headers={"x-pyromind-debug-user-id": "123"},
    )

    assert response.status_code == 403
    assert response.json()["detail"]["rule_id"] == "secret-read"
    bash_service.start_bash_command.assert_not_called()


@pytest.mark.asyncio
async def test_clear_all_bash_events_integration(test_bash_service):
    """Integration test for clearing bash events."""
    # Execute some commands to create events
    commands = [
        BashCommand(command='echo "first"', cwd="/tmp"),
        BashCommand(command='echo "second"', cwd="/tmp"),
    ]

    for cmd in commands:
        await test_bash_service.start_bash_command(cmd)

    # Wait for commands to complete
    import asyncio

    await asyncio.sleep(2)

    # Verify events exist before clearing
    page = await test_bash_service.search_bash_events()
    initial_count = len(page.items)
    assert initial_count > 0

    # Clear all events
    cleared_count = await test_bash_service.clear_all_events()
    assert cleared_count == initial_count

    # Verify events are gone
    page_after = await test_bash_service.search_bash_events()
    assert len(page_after.items) == 0
