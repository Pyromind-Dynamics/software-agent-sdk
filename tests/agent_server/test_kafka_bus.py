"""Unit tests for openhands.agent_server.kafka_bus."""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiokafka import AIOKafkaConsumer
from fastapi import FastAPI

from openhands.agent_server.api import api_lifespan
from openhands.agent_server.config import Config
from openhands.agent_server.kafka_bus.handlers.studio_workflow_notify_handler import (
    StudioWorkflowNotifyHandler,
    TrainingTaskEventType,
    _parse_conversation_id_from_out_id,
)
from openhands.agent_server.kafka_bus.kafka_bus import KafkaMessageBus
from openhands.agent_server.kafka_bus.kafka_topic import KafkaTopic
from openhands.agent_server.kafka_bus.message_event import MessageEvent
from openhands.agent_server.run_workflow_callback import RunWorkflowCallbackResult


@pytest.fixture
def non_dev_bus(monkeypatch: pytest.MonkeyPatch) -> KafkaMessageBus:
    """KafkaMessageBus with ``_is_dev`` forced off (production-like path)."""
    bus = KafkaMessageBus()
    monkeypatch.setattr(bus, "_is_dev", lambda: False)
    return bus


def test_message_event_round_trip_drops_none_fields():
    event = MessageEvent(
        data={"task_id": "t1"},
        event_type="training_task_status_changed",
        message_id="m1",
        attempt=None,
    )

    payload = event.to_json()
    assert "attempt" not in payload

    restored = MessageEvent.from_dict(
        {
            "data": {"task_id": "t1"},
            "event_type": "training_task_status_changed",
            "message_id": "m1",
            "unknown_field": "ignored",
        }
    )
    assert restored.data == {"task_id": "t1"}
    assert restored.message_id == "m1"
    assert restored.event_type == "training_task_status_changed"


def test_kafka_topic_resolve_prod_and_pre(monkeypatch):
    monkeypatch.setattr(
        "openhands.agent_server.kafka_bus.kafka_topic.env_util.get_env_value",
        lambda: "prod",
    )
    monkeypatch.setattr(
        "openhands.agent_server.kafka_bus.kafka_topic.env_util.is_pre",
        lambda: False,
    )
    monkeypatch.setattr(
        "openhands.agent_server.kafka_bus.kafka_topic.env_util.get_cluster_region",
        lambda: "normal",
    )
    assert KafkaTopic.WORKFLOW_MONITOR.resolve() == "workflow_monitor_topic"
    assert KafkaTopic.DLQ.resolve() == "dead_letter_queue_topic"

    monkeypatch.setattr(
        "openhands.agent_server.kafka_bus.kafka_topic.env_util.get_env_value",
        lambda: "pre",
    )
    monkeypatch.setattr(
        "openhands.agent_server.kafka_bus.kafka_topic.env_util.is_pre",
        lambda: True,
    )
    assert KafkaTopic.WORKFLOW_MONITOR.resolve() == "workflow_monitor_topic_pre"
    assert KafkaTopic.DLQ.resolve() == "dead_letter_queue_topic_pre"


def test_parse_conversation_id_from_out_id():
    assert _parse_conversation_id_from_out_id(None) is None
    assert _parse_conversation_id_from_out_id("") is None
    assert _parse_conversation_id_from_out_id("None") is None
    assert _parse_conversation_id_from_out_id("agent1#abc") == "abc"
    assert _parse_conversation_id_from_out_id("agent1#debug#abc") == "abc"
    # Bare / foreign out_ids are not from this agent.
    assert _parse_conversation_id_from_out_id("abc") is None
    assert _parse_conversation_id_from_out_id("studio#abc") is None


def test_parse_out_id_detects_workflow_debug():
    from openhands.agent_server.kafka_bus.handlers.studio_workflow_notify_handler import (  # noqa: E501
        _parse_out_id,
    )

    assert _parse_out_id("agent1#c1") == ("c1", False)
    assert _parse_out_id("agent1#debug#c1") == ("c1", True)
    assert _parse_out_id(None) == (None, False)
    assert _parse_out_id("") == (None, False)
    assert _parse_out_id("None") == (None, False)
    # Marker without conversation id → no routeable id.
    assert _parse_out_id("agent1#debug#") == (None, True)
    # "debug" alone (no trailing #) is conversation_id="debug" under agent1#.
    assert _parse_out_id("agent1#debug") == ("debug", False)
    assert _parse_out_id("  agent1#debug#  abc  ") == ("abc", True)
    # Reject bare UUID / other systems (must start with agent1#).
    assert _parse_out_id("c1") == (None, False)
    assert _parse_out_id("other#c1") == (None, False)
    assert _parse_out_id("agent2#c1") == (None, False)


def test_is_dev_delegates_to_env_util(monkeypatch):
    bus = KafkaMessageBus()
    monkeypatch.setattr(
        "openhands.agent_server.kafka_bus.kafka_bus.env_util.is_kafka_dev",
        lambda: True,
    )
    assert bus._is_dev() is True
    monkeypatch.setattr(
        "openhands.agent_server.kafka_bus.kafka_bus.env_util.is_kafka_dev",
        lambda: False,
    )
    assert bus._is_dev() is False


@pytest.mark.asyncio
async def test_start_consumer_skips_in_dev(monkeypatch):
    bus = KafkaMessageBus()
    monkeypatch.setattr(bus, "_is_dev", lambda: True)
    register = MagicMock()
    monkeypatch.setattr(bus, "register_handler", register)

    await bus.start_consumer()

    assert bus._startup_task is None
    assert bus._handlers == {}
    register.assert_not_called()


@pytest.mark.asyncio
async def test_start_consumer_skips_when_brokers_missing(non_dev_bus, monkeypatch):
    bus = non_dev_bus
    monkeypatch.setattr(
        "openhands.agent_server.kafka_bus.kafka_bus.env_util.get_kafka_brokers",
        lambda: None,
    )

    await bus.start_consumer()

    assert bus._startup_task is None
    assert bus._handlers  # handlers registered before broker check


@pytest.mark.asyncio
async def test_start_consumer_creates_background_startup_task(non_dev_bus, monkeypatch):
    bus = non_dev_bus
    monkeypatch.setattr(
        "openhands.agent_server.kafka_bus.kafka_bus.env_util.get_kafka_brokers",
        lambda: "localhost:9092",
    )

    created: list[object] = []

    def fake_create_task(coro):
        created.append(coro)
        coro.close()
        task = MagicMock()
        task.done.return_value = False
        task.cancel = MagicMock()
        return task

    monkeypatch.setattr(
        "openhands.agent_server.kafka_bus.kafka_bus.asyncio.create_task",
        fake_create_task,
    )

    await bus.start_consumer()

    assert bus._startup_task is not None
    assert len(created) == 1
    assert any(
        isinstance(h, StudioWorkflowNotifyHandler) for h in bus._handlers.values()
    )


@pytest.mark.asyncio
async def test_create_consumer_success_starts_consume_loop(monkeypatch):
    bus = KafkaMessageBus()
    consumer = AsyncMock()
    consume_loop = AsyncMock()

    monkeypatch.setattr(
        "openhands.agent_server.kafka_bus.kafka_bus.env_util.get_kafka_brokers",
        lambda: "localhost:9092",
    )
    monkeypatch.setattr(
        "openhands.agent_server.kafka_bus.kafka_bus.AIOKafkaConsumer",
        lambda *args, **kwargs: consumer,
    )
    monkeypatch.setattr(bus, "_consume_loop", consume_loop)

    created_tasks: list[object] = []

    def fake_create_task(coro):
        created_tasks.append(coro)
        coro.close()
        return MagicMock()

    monkeypatch.setattr(
        "openhands.agent_server.kafka_bus.kafka_bus.asyncio.create_task",
        fake_create_task,
    )

    ok = await bus._create_consumer("workflow_monitor_topic", "group-1", concurrency=2)

    assert ok is True
    consumer.start.assert_awaited_once()
    assert consumer in bus._consumers
    assert len(bus._consumer_tasks) == 1
    assert len(created_tasks) == 1


@pytest.mark.asyncio
async def test_create_consumer_failure_stops_consumer(monkeypatch):
    bus = KafkaMessageBus()
    consumer = AsyncMock()
    consumer.start.side_effect = RuntimeError("boom")

    monkeypatch.setattr(
        "openhands.agent_server.kafka_bus.kafka_bus.env_util.get_kafka_brokers",
        lambda: "localhost:9092",
    )
    monkeypatch.setattr(
        "openhands.agent_server.kafka_bus.kafka_bus.AIOKafkaConsumer",
        lambda *args, **kwargs: consumer,
    )

    ok = await bus._create_consumer("workflow_monitor_topic", "group-1")

    assert ok is False
    consumer.stop.assert_awaited_once()
    assert bus._consumers == []


@pytest.mark.asyncio
async def test_restart_consumer_skips_in_dev(monkeypatch):
    bus = KafkaMessageBus()
    monkeypatch.setattr(bus, "_is_dev", lambda: True)
    old_consumer = AsyncMock()
    bus._consumers.append(old_consumer)
    create = AsyncMock(return_value=True)
    monkeypatch.setattr(bus, "_create_consumer", create)

    await bus._restart_consumer(
        old_consumer, "workflow_monitor_topic", "group-1", concurrency=2
    )

    create.assert_not_awaited()
    old_consumer.stop.assert_not_awaited()
    assert old_consumer in bus._consumers


@pytest.mark.asyncio
async def test_restart_consumer_stops_old_instance_before_recreate(
    non_dev_bus, monkeypatch
):
    bus = non_dev_bus
    old_consumer = AsyncMock()
    bus._consumers.append(old_consumer)

    create = AsyncMock(return_value=True)
    monkeypatch.setattr(bus, "_create_consumer", create)

    await bus._restart_consumer(
        old_consumer, "workflow_monitor_topic", "group-1", concurrency=3
    )

    old_consumer.stop.assert_awaited_once()
    assert old_consumer not in bus._consumers
    create.assert_awaited_once_with("workflow_monitor_topic", "group-1", 3)


@pytest.mark.asyncio
async def test_restart_consumer_continues_when_old_stop_fails(non_dev_bus, monkeypatch):
    bus = non_dev_bus
    old_consumer = AsyncMock()
    old_consumer.stop.side_effect = RuntimeError("already closed")
    bus._consumers.append(old_consumer)

    create = AsyncMock(return_value=True)
    monkeypatch.setattr(bus, "_create_consumer", create)

    await bus._restart_consumer(
        old_consumer, "workflow_monitor_topic", "group-1", concurrency=2
    )

    old_consumer.stop.assert_awaited_once()
    create.assert_awaited_once()


@pytest.mark.asyncio
async def test_consume_loop_skips_bad_messages_without_restart(monkeypatch):
    """Poison pills (bad JSON / missing data) must not rebuild the consumer."""
    bus = KafkaMessageBus()
    restart = AsyncMock()
    handle = AsyncMock()
    monkeypatch.setattr(bus, "_restart_consumer", restart)
    monkeypatch.setattr(bus, "_handle_one", handle)

    bad_json = SimpleNamespace(topic="t", partition=0, offset=1, value=b"not-json")
    missing_data = SimpleNamespace(
        topic="t", partition=0, offset=2, value=b'{"event_type":"x"}'
    )
    good = SimpleNamespace(
        topic="t",
        partition=0,
        offset=3,
        value=b'{"data":{"ok":true},"event_type":"x"}',
    )

    class _FakeConsumer:
        def __aiter__(self):
            async def _gen():
                yield bad_json
                yield missing_data
                yield good

            return _gen()

    await bus._consume_loop(
        cast(AIOKafkaConsumer, _FakeConsumer()),
        "t",
        "group-1",
        concurrency=1,
    )

    handle.assert_awaited_once()
    assert handle.await_args is not None
    assert handle.await_args.args[0].data == {"ok": True}
    restart.assert_not_awaited()


@pytest.mark.asyncio
async def test_consume_loop_handler_error_does_not_skip_as_deserialize(monkeypatch):
    """Handler failures must propagate to _handle_one retry path, not be swallowed."""
    bus = KafkaMessageBus()
    restart = AsyncMock()
    monkeypatch.setattr(bus, "_restart_consumer", restart)

    handler = AsyncMock()
    handler.handle.side_effect = RuntimeError("biz failed")
    handler.group_id.return_value = "g1"
    bus._handlers["t"] = handler
    send_to_dlq = AsyncMock(return_value=True)
    monkeypatch.setattr(bus, "_send_to_dlq", send_to_dlq)

    good = SimpleNamespace(
        topic="t",
        partition=0,
        offset=1,
        value=b'{"data":{"ok":true},"message_id":"m1","attempt":2}',
    )

    class _FakeConsumer:
        def __aiter__(self):
            async def _gen():
                yield good

            return _gen()

    await bus._consume_loop(
        cast(AIOKafkaConsumer, _FakeConsumer()),
        "t",
        "group-1",
        concurrency=1,
    )

    handler.handle.assert_awaited_once()
    send_to_dlq.assert_awaited_once()
    restart.assert_not_awaited()


@pytest.mark.asyncio
async def test_handle_one_retries_then_sends_dlq(monkeypatch):
    bus = KafkaMessageBus()
    handler = AsyncMock()
    handler.handle.side_effect = RuntimeError("handler failed")
    handler.group_id.return_value = "group-1"
    bus._handlers["topic-a"] = handler

    event = MessageEvent(data={"x": 1}, topic="topic-a", message_id="m1", attempt=2)
    send_to_dlq = AsyncMock(return_value=True)
    monkeypatch.setattr(bus, "_send_to_dlq", send_to_dlq)

    await bus._handle_one(event)

    send_to_dlq.assert_awaited_once()
    assert event.error_message is not None


@pytest.mark.asyncio
async def test_studio_workflow_handler_ignores_non_terminal_and_wrong_type():
    handler = StudioWorkflowNotifyHandler()
    deliver = AsyncMock()

    with patch(
        "openhands.agent_server.kafka_bus.handlers."
        "studio_workflow_notify_handler.deliver_run_workflow_status",
        deliver,
    ):
        await handler.handle(
            MessageEvent(
                data={"status": "Succeeded", "task_id": "t1", "out_id": "c1"},
                event_type=TrainingTaskEventType.TRAINING_NODE_STATUS_CHANGED.value,
            )
        )
        await handler.handle(
            MessageEvent(
                data={"status": "Running", "task_id": "t1", "out_id": "c1"},
                event_type=TrainingTaskEventType.TRAINING_TASK_STATUS_CHANGED.value,
            )
        )

    deliver.assert_not_awaited()


@pytest.mark.asyncio
async def test_studio_workflow_handler_delivers_terminal_status():
    handler = StudioWorkflowNotifyHandler()
    deliver = AsyncMock(
        return_value=RunWorkflowCallbackResult(
            outcome="delivered_async",
            task_id="t1",
            normalized_status="Succeeded",
            conversation_id="c1",
        )
    )

    with patch(
        "openhands.agent_server.kafka_bus.handlers."
        "studio_workflow_notify_handler.deliver_run_workflow_status",
        deliver,
    ):
        await handler.handle(
            MessageEvent(
                data={
                    "status": "Succeeded",
                    "task_id": "t1",
                    "out_id": "agent1#c1",
                    "error_msg": None,
                },
                event_type=TrainingTaskEventType.TRAINING_TASK_STATUS_CHANGED.value,
            )
        )

    deliver.assert_awaited_once_with(
        task_id="t1",
        status="Succeeded",
        error_log=None,
        conversation_id="c1",
        auto_run=True,
        from_workflow_debug=False,
    )


@pytest.mark.asyncio
async def test_studio_workflow_handler_delivers_debug_terminal_status():
    handler = StudioWorkflowNotifyHandler()
    deliver = AsyncMock(
        return_value=RunWorkflowCallbackResult(
            outcome="delivered_async",
            task_id="t1",
            normalized_status="Failed",
            conversation_id="c1",
        )
    )

    with patch(
        "openhands.agent_server.kafka_bus.handlers."
        "studio_workflow_notify_handler.deliver_run_workflow_status",
        deliver,
    ):
        await handler.handle(
            MessageEvent(
                data={
                    "status": "Failed",
                    "task_id": "t1",
                    "out_id": "agent1#debug#c1",
                    "error_msg": "boom",
                },
                event_type=TrainingTaskEventType.TRAINING_TASK_STATUS_CHANGED.value,
            )
        )

    deliver.assert_awaited_once_with(
        task_id="t1",
        status="Failed",
        error_log="boom",
        conversation_id="c1",
        auto_run=True,
        from_workflow_debug=True,
    )


@pytest.mark.asyncio
async def test_studio_workflow_handler_skips_when_conversation_not_on_pod():
    """Broadcast: pods without the conversation must exit without raising."""
    handler = StudioWorkflowNotifyHandler()
    deliver = AsyncMock(
        return_value=RunWorkflowCallbackResult(
            outcome="unknown_conversation",
            task_id="t1",
            normalized_status="Succeeded",
            conversation_id="c1",
        )
    )

    with patch(
        "openhands.agent_server.kafka_bus.handlers."
        "studio_workflow_notify_handler.deliver_run_workflow_status",
        deliver,
    ):
        await handler.handle(
            MessageEvent(
                data={
                    "status": "Succeeded",
                    "task_id": "t1",
                    "out_id": "agent1#c1",
                },
                event_type=TrainingTaskEventType.TRAINING_TASK_STATUS_CHANGED.value,
            )
        )

    deliver.assert_awaited_once()


@pytest.mark.asyncio
async def test_studio_workflow_handler_discards_missing_out_id():
    """Empty out_id → discard; do not deliver or broker-fallback."""
    handler = StudioWorkflowNotifyHandler()
    deliver = AsyncMock()

    with patch(
        "openhands.agent_server.kafka_bus.handlers."
        "studio_workflow_notify_handler.deliver_run_workflow_status",
        deliver,
    ):
        await handler.handle(
            MessageEvent(
                data={"status": "Succeeded", "task_id": "t1"},
                event_type=TrainingTaskEventType.TRAINING_TASK_STATUS_CHANGED.value,
            )
        )
        await handler.handle(
            MessageEvent(
                data={"status": "Succeeded", "task_id": "t1", "out_id": ""},
                event_type=TrainingTaskEventType.TRAINING_TASK_STATUS_CHANGED.value,
            )
        )
        await handler.handle(
            MessageEvent(
                data={"status": "Succeeded", "task_id": "t1", "out_id": None},
                event_type=TrainingTaskEventType.TRAINING_TASK_STATUS_CHANGED.value,
            )
        )

    deliver.assert_not_awaited()


@pytest.mark.asyncio
async def test_studio_workflow_handler_discards_foreign_out_id():
    """out_id not prefixed with agent1# → other system; discard."""
    handler = StudioWorkflowNotifyHandler()
    deliver = AsyncMock()

    with patch(
        "openhands.agent_server.kafka_bus.handlers."
        "studio_workflow_notify_handler.deliver_run_workflow_status",
        deliver,
    ):
        await handler.handle(
            MessageEvent(
                data={
                    "status": "Succeeded",
                    "task_id": "t1",
                    "out_id": "studio#550e8400-e29b-41d4-a716-446655440000",
                },
                event_type=TrainingTaskEventType.TRAINING_TASK_STATUS_CHANGED.value,
            )
        )
        await handler.handle(
            MessageEvent(
                data={
                    "status": "Succeeded",
                    "task_id": "t1",
                    "out_id": "550e8400-e29b-41d4-a716-446655440000",
                },
                event_type=TrainingTaskEventType.TRAINING_TASK_STATUS_CHANGED.value,
            )
        )

    deliver.assert_not_awaited()


@pytest.mark.asyncio
async def test_api_lifespan_starts_and_stops_kafka_consumer():
    mock_conversation_service = AsyncMock()
    mock_kafka = AsyncMock()

    with (
        patch(
            "openhands.agent_server.api.get_default_conversation_service",
            return_value=mock_conversation_service,
        ),
        patch("openhands.agent_server.api.get_vscode_service", return_value=None),
        patch("openhands.agent_server.api.get_desktop_service", return_value=None),
        patch(
            "openhands.agent_server.api.get_tool_preload_service",
            return_value=None,
        ),
        patch("openhands.agent_server.api.kafka_message_bus", mock_kafka),
    ):
        app = FastAPI()
        app.state.config = Config()

        async with api_lifespan(app):
            mock_kafka.start_consumer.assert_awaited_once()

        mock_kafka.stop.assert_awaited_once()
