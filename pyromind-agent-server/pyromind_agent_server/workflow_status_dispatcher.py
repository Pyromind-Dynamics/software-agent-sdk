from __future__ import annotations

from datetime import UTC, datetime

from pyromind_runtime.application.conversation_runtime import ConversationRuntime
from pyromind_runtime.infrastructure.file_product_store import FileProductStore

from openhands.agent_server.run_workflow_callback import (
    RunWorkflowCallbackResult,
    deliver_run_workflow_status,
    normalize_platform_status,
)
from openhands.tools.data_preparation.platform_submit import (
    TASK_ASSOCIATION_DIRNAME as PREPARATION_TASK_DIR,
    DataPreparationTaskStore,
)
from openhands.tools.pyromind_cleaning.task_store import (
    TASK_ASSOCIATION_DIRNAME as CLEANING_TASK_DIR,
    DatasetCleaningTaskStore,
)


class WorkflowStatusDispatcher:
    """Route Studio callbacks by the conversation's persisted harness owner."""

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
        if not conversation_id or from_workflow_debug:
            return await deliver_run_workflow_status(
                task_id=task_id,
                status=status,
                error_log=error_log,
                conversation_id=conversation_id,
                auto_run=auto_run,
                from_workflow_debug=from_workflow_debug,
            )
        store = FileProductStore(self._runtime.conversation_root / conversation_id)
        if not store.metadata_path.is_file() or store.harness_id() != "pi":
            return await deliver_run_workflow_status(
                task_id=task_id,
                status=status,
                error_log=error_log,
                conversation_id=conversation_id,
                auto_run=auto_run,
                from_workflow_debug=from_workflow_debug,
            )
        normalized = normalize_platform_status(status)
        self._ensure_product_task(store, task_id)
        await self._runtime.deliver_external_task_status(
            conversation_id,
            task_id=task_id,
            status=normalized,
            error_summary=error_log,
        )
        self._update_association(task_id, normalized)
        return RunWorkflowCallbackResult(
            outcome="delivered_async",
            task_id=task_id,
            normalized_status=normalized,
            conversation_id=conversation_id,
        )

    def _update_association(self, task_id: str, status: str) -> None:
        root = self._runtime.conversation_root
        cleaning = DatasetCleaningTaskStore(root / CLEANING_TASK_DIR)
        if cleaning.update_status(task_id, status) is not None:
            return
        preparation = DataPreparationTaskStore(root / PREPARATION_TASK_DIR)
        association = preparation.get(task_id)
        if association is None:
            return
        association.status = status
        association.updated_at = datetime.now(UTC).isoformat()
        preparation.save(association)

    def _ensure_product_task(self, store: FileProductStore, task_id: str) -> None:
        if any(
            task.task_id == task_id for task in store.load_snapshot().external_tasks
        ):
            return
        root = self._runtime.conversation_root
        cleaning = DatasetCleaningTaskStore(root / CLEANING_TASK_DIR).get(task_id)
        if cleaning is not None:
            submitted_at = cleaning.submitted_at.isoformat()
            payload = {
                "task_id": task_id,
                "kind": "data_cleaning",
                "run_id": cleaning.run_id,
                "status": _task_status(cleaning.status),
                "output_dir": cleaning.output_dir,
                "submitted_at": submitted_at,
                "updated_at": submitted_at,
                "resume_pending": False,
            }
        else:
            preparation = DataPreparationTaskStore(root / PREPARATION_TASK_DIR).get(
                task_id
            )
            if preparation is None:
                raise ValueError(f"unknown Pi external task: {task_id}")
            payload = {
                "task_id": task_id,
                "kind": "data_preparation",
                "run_id": preparation.run_id,
                "status": _task_status(preparation.status),
                "output_dir": preparation.output_dir,
                "submitted_at": preparation.submitted_at,
                "updated_at": preparation.updated_at,
                "resume_pending": False,
            }
        self._runtime.register_external_task(
            store.load_snapshot().conversation_id,
            payload,
        )


def _task_status(value: str) -> str:
    normalized = value.strip().lower()
    return {
        "success": "succeeded",
        "succeeded": "succeeded",
        "error": "failed",
        "failed": "failed",
        "terminated": "terminated",
        "stopped": "stopped",
        "running": "running",
        "pending": "pending",
    }.get(normalized, "pending")
