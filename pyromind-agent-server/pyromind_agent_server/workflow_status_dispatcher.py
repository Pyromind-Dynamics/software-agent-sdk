from __future__ import annotations

from uuid import UUID

from pyromind_runtime.application.conversation_runtime import ConversationRuntime
from pyromind_runtime.infrastructure.file_product_store import FileProductStore

from openhands.agent_server.run_workflow_callback import (
    RunWorkflowCallbackResult,
    deliver_run_workflow_status,
    normalize_platform_status,
)


class WorkflowStatusDispatcher:
    """Route Product conversations through one harness-neutral callback path."""

    def __init__(self, runtime: ConversationRuntime) -> None:
        self._runtime = runtime

    async def dispatch(
        self,
        *,
        task_id: str,
        status: str,
        error_log: str | None = None,
        conversation_id: str | None = None,
        auto_run: bool = True,
        from_workflow_debug: bool = False,
    ) -> RunWorkflowCallbackResult:
        if from_workflow_debug:
            return await deliver_run_workflow_status(
                task_id=task_id,
                status=status,
                error_log=error_log,
                conversation_id=conversation_id,
                auto_run=auto_run,
                from_workflow_debug=from_workflow_debug,
            )
        normalized = normalize_platform_status(status)
        owner = self._runtime.resolve_external_task_owner(task_id)
        supplied_conversation_id = (
            _canonical_conversation_id(conversation_id)
            if conversation_id is not None
            else None
        )
        if (
            supplied_conversation_id is not None
            and owner is not None
            and supplied_conversation_id != owner
        ):
            return RunWorkflowCallbackResult(
                outcome="unknown_task",
                task_id=task_id,
                normalized_status=normalized,
                conversation_id=supplied_conversation_id,
            )
        resolved_conversation_id = supplied_conversation_id or owner
        if resolved_conversation_id is None:
            return await deliver_run_workflow_status(
                task_id=task_id,
                status=status,
                error_log=error_log,
                conversation_id=None,
                auto_run=auto_run,
                from_workflow_debug=from_workflow_debug,
            )
        store = FileProductStore(
            self._runtime.conversation_root / resolved_conversation_id
        )
        if not store.metadata_path.is_file():
            return await deliver_run_workflow_status(
                task_id=task_id,
                status=status,
                error_log=error_log,
                conversation_id=resolved_conversation_id,
                auto_run=auto_run,
                from_workflow_debug=from_workflow_debug,
            )
        await self._runtime.deliver_external_task_status(
            resolved_conversation_id,
            task_id=task_id,
            status=status,
            error_summary=error_log,
            auto_run=auto_run,
            from_workflow_debug=from_workflow_debug,
        )
        return RunWorkflowCallbackResult(
            outcome="delivered_async",
            task_id=task_id,
            normalized_status=normalized,
            conversation_id=resolved_conversation_id,
        )


def _canonical_conversation_id(value: str) -> str:
    try:
        return UUID(value).hex
    except ValueError:
        return value
