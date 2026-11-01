"""Faults around durable Kafka acknowledgement use the existing bus."""

import asyncio
from types import SimpleNamespace

import pytest
from aiokafka.structs import TopicPartition
from openhands.agent_server.kafka_bus.kafka_bus import KafkaMessageBus
from openhands.agent_server.kafka_bus.message_event import MessageEvent
from pyromind_agent_server.execution_observations import ExecutionObservationHandler

from .test_standalone_experiments import experiment  # noqa: F401


@pytest.mark.parametrize("fault", [None, "persist", "commit"])
async def test_durable_handler_precedes_partition_ack_and_failure_replays(experiment, monkeypatch, fault):
    service, _, _ = experiment
    bus = KafkaMessageBus(acknowledge_after_handle=True)
    handler = ExecutionObservationHandler(service.store)
    bus.register_handler(handler)
    calls = []
    original = service.store.ingest

    def ingest(*args):
        calls.append("persist")
        if fault == "persist":
            raise OSError("database unavailable")
        original(*args)

    monkeypatch.setattr(service.store, "ingest", ingest)
    class Consumer:
        def __aiter__(self):
            return self.messages()

        async def messages(self):
            event = MessageEvent(event_type="training_task_status_changed", data={"task_id": "early", "status": "Succeeded", "out_id": "agent1#not-loaded"})
            yield SimpleNamespace(topic=handler.topic(), partition=2, offset=7, value=event.to_json().encode())

        async def commit(self, offsets):
            assert offsets == {TopicPartition(handler.topic(), 2): 8}
            calls.append("commit")
            if fault == "commit":
                raise OSError("commit unavailable")

    async def restart(*_):
        calls.append("restart")
    monkeypatch.setattr(bus, "_restart_consumer", restart)
    try:
        await bus._consume_loop(Consumer(), handler.topic(), handler.group_id())
        await asyncio.sleep(0)
        assert calls == (["persist", "restart"] if fault == "persist" else ["persist", "commit", "restart"] if fault == "commit" else ["persist", "commit"])
        if fault != "persist":
            # A commit failure replays the same offset; SQL keeps one observation.
            original(f"{handler.topic()}:2:7", "early", "kafka", {"status": "succeeded"})
            from pyromind_runtime.infrastructure.sql_experiment_execution import inbox
            import sqlalchemy as sa
            with service.store.engine.connect() as conn:
                assert conn.execute(sa.select(sa.func.count()).select_from(inbox)).scalar_one() == 1
            assert service.store.apply_observations() == 0  # retained until binding
    finally:
        await bus.stop()


def test_existing_bus_keeps_legacy_acknowledgement():
    assert KafkaMessageBus()._acknowledge_after_handle is False
