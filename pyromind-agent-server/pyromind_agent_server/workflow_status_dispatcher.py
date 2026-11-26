from __future__ import annotations

from typing import Protocol
from uuid import UUID

from pyromind_agent_server.workflow_status import (
    WorkflowCallbackResult,
    normalize_platform_status,
)
from pyromind_runtime.application.conversation_runtime import ConversationRuntime
from pyromind_runtime.domain.task_correlation import TaskCorrelation
from pyromind_runtime.infrastructure.file_product_store import FileProductStore


class GraphTaskHandler(Protocol):
    async def __call__(
        self,
        *,
        task_id: str,
        status: str,
        conversation_id: str,
        task_kind: str | None,
    ) -> str | None: ...


class WorkflowStatusDispatcher:
    """Route Product conversations through one harness-neutral callback path."""

    def __init__(
        self,
        runtime: ConversationRuntime,
        *,
        graph_task_handler: GraphTaskHandler | None = None,
    ) -> None:
        self._runtime = runtime
        self._graph_task_handler = graph_task_handler

    async def dispatch(
        self,
        *,
        task_id: str,
        status: str,
        error_log: str | None = None,
        conversation_id: str | None = None,
        auto_run: bool = True,
        from_workflow_debug: bool = False,  # noqa: ARG002
        correlation: TaskCorrelation | None = None,
        trusted_platform: bool = False,
    ) -> WorkflowCallbackResult:
        if correlation is not None:
            if (
                conversation_id is not None
                and _canonical_conversation_id(conversation_id)
                != correlation.conversation_id
            ):
                raise PermissionError("callback conversation does not match out_id")
            conversation_id = correlation.conversation_id
        normalized = normalize_platform_status(status)
        graph_correlation = correlation is not None and (
            correlation.type == "main" and correlation.task_kind.startswith("echomind_")
        )
        if (
            self._graph_task_handler is not None
            and conversation_id is not None
            and (correlation is None or graph_correlation)
        ):
            graph_conversation_id = await self._graph_task_handler(
                task_id=task_id,
                status=status,
                conversation_id=conversation_id,
                task_kind=correlation.task_kind if correlation is not None else None,
            )
            if graph_conversation_id is not None:
                if (
                    correlation is not None
                    and _canonical_conversation_id(graph_conversation_id)
                    != correlation.conversation_id
                ):
                    return WorkflowCallbackResult(
                        outcome="unknown_conversation",
                        task_id=task_id,
                        normalized_status=normalized,
                        conversation_id=graph_conversation_id,
                    )
                return WorkflowCallbackResult(
                    outcome="delivered_async",
                    task_id=task_id,
                    normalized_status=normalized,
                    conversation_id=graph_conversation_id,
                )
        if graph_correlation:
            return WorkflowCallbackResult(
                outcome="unknown_task",
                task_id=task_id,
                normalized_status=normalized,
                conversation_id=conversation_id,
            )
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
            return WorkflowCallbackResult(
                outcome="unknown_task",
                task_id=task_id,
                normalized_status=normalized,
                conversation_id=supplied_conversation_id,
            )
        resolved_conversation_id = supplied_conversation_id or owner
        if resolved_conversation_id is None:
            return WorkflowCallbackResult(
                outcome="unknown_task",
                task_id=task_id,
                normalized_status=normalized,
                conversation_id=None,
            )
        store = FileProductStore(
            self._runtime.conversation_root / resolved_conversation_id
        )
        if not store.metadata_path.is_file():
            return WorkflowCallbackResult(
                outcome="unknown_conversation",
                task_id=task_id,
                normalized_status=normalized,
                conversation_id=resolved_conversation_id,
            )
        await self._runtime.deliver_external_task_status(
            resolved_conversation_id,
            task_id=task_id,
            status=status,
            error_summary=error_log,
            auto_run=auto_run,
            correlation=correlation,
            trusted_platform=trusted_platform,
        )
        return WorkflowCallbackResult(
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
