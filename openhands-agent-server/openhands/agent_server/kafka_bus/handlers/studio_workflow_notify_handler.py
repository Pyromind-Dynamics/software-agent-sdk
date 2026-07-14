from enum import Enum

from pyromind_sdk.client.models import StudioTaskStatus

from openhands.agent_server.kafka_bus.kafka_handler import MessageHandler
from openhands.agent_server.kafka_bus.kafka_topic import KafkaTopic
from openhands.agent_server.kafka_bus.message_event import MessageEvent
from openhands.agent_server.run_workflow_callback import (
    deliver_run_workflow_status,
)
from openhands.sdk.logger import get_logger
from openhands.sdk.utils.env_util import get_pod_name


logger = get_logger(__name__)

OUT_ID_PREFIX = "agent1#"

# Outcomes that mean "this pod should ignore the message" under Kafka broadcast
# (unique group_id per pod). Do not raise — Kafka must not retry/DLQ these.
_BROADCAST_SKIP_OUTCOMES = frozenset(
    {
        "unknown_conversation",  # conversation not loaded on this pod
        "unknown_task",  # no conversation_id to route with
        "duplicate_terminal",
        "ignored_non_terminal",
        "resolved_blocked",
    }
)
_TERMINAL_STUDIO_STATUSES = frozenset(
    {
        StudioTaskStatus.FAILED,
        StudioTaskStatus.SUCCEEDED,
        StudioTaskStatus.ERROR,
        StudioTaskStatus.TERMINATED,
    }
)


class TrainingTaskEventType(str, Enum):
    """Training task event type enumeration"""

    TRAINING_TASK_STATUS_CHANGED = "training_task_status_changed"
    TRAINING_NODE_STATUS_CHANGED = "training_node_status_changed"


class StudioWorkflowNotifyHandler(MessageHandler):
    def topic(self) -> str:
        return KafkaTopic.WORKFLOW_MONITOR.resolve()

    def group_id(self) -> str:
        # Per-pod group → every pod receives every message (broadcast).
        # Only the pod that holds the conversation will deliver; others skip.
        return get_pod_name()

    async def handle(self, event: MessageEvent):
        """Handle studio workflow notify message.

        Multi-pod broadcast: each pod sees the same Kafka message. If this pod
        does not have the conversation, ``deliver_run_workflow_status`` returns
        ``unknown_conversation`` and we exit without error (no retry/DLQ).

        Expected ``event.data``::

            {
                "task_id": ...,
                "status": ...,
                "out_id": "agent1#<conversation_uuid>",
                "error_msg": ...,
                ...
            }
        """
        if event.event_type != TrainingTaskEventType.TRAINING_TASK_STATUS_CHANGED:
            return

        message_data = event.data
        status = _get_task_status(message_data.get("status"))
        if status is None or status not in _TERMINAL_STUDIO_STATUSES:
            return

        raw_task_id = message_data.get("task_id")
        if raw_task_id is None or str(raw_task_id).strip() == "":
            logger.warning(
                "Skip workflow notify: missing task_id message_id=%s",
                event.message_id,
            )
            return
        task_id = str(raw_task_id)

        conversation_id = _parse_conversation_id_from_out_id(message_data.get("out_id"))
        if not conversation_id:
            logger.info(
                "Workflow notify has no out_id; trying generic task correlation "
                "task_id=%s message_id=%s",
                task_id,
                event.message_id,
            )

        error_log = (
            str(message_data["error_msg"])
            if message_data.get("error_msg") is not None
            else None
        )

        result = await deliver_run_workflow_status(
            task_id=task_id,
            status=status.value,
            error_log=error_log,
            conversation_id=conversation_id,
            auto_run=True,
        )

        # Broadcast: other pods normally hit unknown_conversation — that is OK.
        if result.outcome in _BROADCAST_SKIP_OUTCOMES:
            logger.info(
                "Skip workflow notify on this pod: outcome=%s task_id=%s "
                "conversation_id=%s",
                result.outcome,
                task_id,
                conversation_id,
            )
            return

        if result.outcome != "delivered_async":
            logger.warning(
                "Unexpected workflow notify outcome=%s task_id=%s conversation_id=%s",
                result.outcome,
                task_id,
                conversation_id,
            )


def _parse_conversation_id_from_out_id(out_id: object | None) -> str | None:
    """Return bare conversation id from task out_id, or None if missing."""
    if out_id is None:
        return None
    cleaned = str(out_id).strip()
    if not cleaned or cleaned == "None":
        return None
    if cleaned.startswith(OUT_ID_PREFIX):
        cleaned = cleaned.removeprefix(OUT_ID_PREFIX).strip()
    return cleaned or None


def _get_task_status(task_status_value):
    if task_status_value is None:
        return None
    for status in StudioTaskStatus:
        if (
            status.value == str(task_status_value).strip()
            or status.name == str(task_status_value).strip()
        ):
            return status
    return None
