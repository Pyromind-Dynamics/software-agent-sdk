"""Persist platform observations without requiring an activated conversation."""

import asyncio
import os

from openhands.agent_server.kafka_bus.kafka_handler import MessageHandler
from openhands.agent_server.kafka_bus.kafka_topic import KafkaTopic
from openhands.agent_server.kafka_bus.message_event import MessageEvent
from openhands.sdk.utils import env_util
from pyromind_runtime.ports.experiment_execution import ExperimentExecutionStore


class ExecutionObservationHandler(MessageHandler):
    def __init__(self, store: ExperimentExecutionStore):
        self.store = store

    def topic(self) -> str:
        return KafkaTopic.WORKFLOW_MONITOR.resolve()

    def group_id(self) -> str:
        return os.getenv("EXPERIMENT_KAFKA_GROUP", "runtime-inbox-" + env_util.get_env_value())

    async def handle(self, event: MessageEvent):
        if event.event_type != "training_task_status_changed" or not isinstance(event.data, dict):
            return
        data = event.data
        if not str(data.get("out_id", "")).startswith("agent1#"):
            return
        status = str(data.get("status", "")).lower()
        status = {"error": "failed", "success": "succeeded", "completed": "succeeded"}.get(status, status)
        if not data.get("task_id") or status not in {"pending", "running", "succeeded", "failed", "stopped", "terminated"}:
            return
        await asyncio.to_thread(
            self.store.ingest, event.message_id, str(data["task_id"]), "kafka", {"status": status}
        )
