"""Adapt dataset-cleaning tasks to the generic workflow callback pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from openhands.agent_server.conversation_service import (
    ConversationService,
    get_default_conversation_service,
)
from openhands.agent_server.event_service import EventService
from openhands.agent_server.run_workflow_callback import (
    RunWorkflowCallbackResult,
    RunWorkflowStatus,
    deliver_run_workflow_status,
    normalize_platform_status,
)
from openhands.sdk.llm import Message, TextContent
from openhands.sdk.logger import get_logger
from openhands.tools.pyromind_cleaning.task_store import (
    DatasetCleaningTaskAssociation,
    task_store_for_conversations_dir,
)


logger = get_logger(__name__)


def build_dataset_cleaning_terminal_reminder(
    *,
    association: DatasetCleaningTaskAssociation,
    status: RunWorkflowStatus,
    error_log: str | None = None,
) -> str:
    """Build the follow-up context for a completed dataset cleaning task."""
    lines = [
        "<system_reminder>",
        (
            f"Pyromind dataset cleaning task {association.task_id} reached "
            f"terminal status {status}."
        ),
        f"Cleaning run_id: {association.run_id}",
        f"Cleaning output directory: {association.output_dir}",
    ]
    if status == "Succeeded":
        lines.extend(
            [
                (
                    "Preview the output directory's stats.json before reporting "
                    "the result to the user."
                ),
                (
                    "Treat the run as incomplete if stats.json is missing even "
                    "when the workflow status is Succeeded."
                ),
            ]
        )
    else:
        lines.append(
            "Inspect checkpoint.json and errors.jsonl, then decide whether to resume."
        )
    if error_log:
        lines.extend(["Runtime error log:", error_log])
    lines.append("</system_reminder>")
    return "\n".join(lines)


@dataclass(frozen=True)
class DatasetCleaningTerminalDelivery:
    """Deliver cleaning-specific terminal context to the owning conversation."""

    association: DatasetCleaningTaskAssociation

    async def __call__(
        self,
        *,
        event_service: EventService,
        task_id: str,
        status: RunWorkflowStatus,
        error_log: str | None = None,
        auto_run: bool = True,
    ) -> None:
        if task_id != self.association.task_id:
            raise ValueError(
                "Dataset cleaning task association does not match task_id."
            )
        reminder = build_dataset_cleaning_terminal_reminder(
            association=self.association,
            status=status,
            error_log=error_log,
        )
        message = Message(
            role="user",
            content=[
                TextContent(
                    text=(
                        "Pyromind dataset cleaning finished "
                        f"(task_id={task_id}, status={status})."
                    )
                )
            ],
        )
        await event_service.send_message(
            message,
            run=auto_run,
            extended_content=[TextContent(text=reminder)],
        )


async def dispatch_run_workflow_status(
    *,
    task_id: str,
    status: str,
    error_log: str | None = None,
    conversation_id: str | None = None,
    updated_at: datetime | None = None,
    auto_run: bool = True,
    conversation_service: ConversationService | None = None,
) -> RunWorkflowCallbackResult:
    """Route known cleaning tasks through their adapter, else use generic delivery."""
    service = conversation_service or get_default_conversation_service()
    cleaning_store = task_store_for_conversations_dir(service.conversations_dir)
    association = cleaning_store.get(task_id)
    if association is None:
        return await deliver_run_workflow_status(
            task_id=task_id,
            status=status,
            error_log=error_log,
            conversation_id=conversation_id,
            updated_at=updated_at,
            auto_run=auto_run,
            conversation_service=service,
        )

    normalized_status = normalize_platform_status(status)
    try:
        updated_association = cleaning_store.update_status(task_id, normalized_status)
        if updated_association is not None:
            association = updated_association
    except OSError:
        logger.exception(
            "Failed to persist dataset cleaning status for task_id=%s",
            task_id,
        )

    if conversation_id and conversation_id != association.conversation_id:
        logger.warning(
            "Ignoring mismatched callback conversation_id=%s for dataset cleaning "
            "task_id=%s; persisted conversation_id=%s",
            conversation_id,
            task_id,
            association.conversation_id,
        )

    return await deliver_run_workflow_status(
        task_id=task_id,
        status=status,
        error_log=error_log,
        conversation_id=association.conversation_id,
        updated_at=updated_at,
        auto_run=auto_run,
        conversation_service=service,
        terminal_delivery=DatasetCleaningTerminalDelivery(association),
    )
