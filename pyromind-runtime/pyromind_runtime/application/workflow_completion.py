from __future__ import annotations

import logging

from pyromind_runtime.domain.content import JsonObject
from pyromind_runtime.domain.events import (
    HarnessEvent,
    ProductEvent,
    ProductEventType,
    WorkflowRunState,
)
from pyromind_runtime.ports.harness import HarnessAdapter, SessionHandle
from pyromind_runtime.ports.product_store import ProductStore


logger = logging.getLogger(__name__)


class WorkflowCompletionHook:
    def __init__(
        self, store: ProductStore, adapter: HarnessAdapter, handle: SessionHandle
    ) -> None:
        self.store = store
        self.adapter = adapter
        self.handle = handle

    async def recover(self) -> tuple[ProductEvent, ...]:
        output: list[ProductEvent] = []
        for state in self.store.load_workflow_runs().values():
            if state.completion is not None and not state.completed:
                output.extend(await self.accept(state.completion))
        return tuple(output)

    async def accept(self, event: HarnessEvent) -> tuple[ProductEvent, ...]:
        if not event.run_id:
            raise ValueError("workflow lifecycle events require a run_id")
        completion_id = event.payload.get("completion_id")
        key = completion_id if isinstance(completion_id, str) else event.run_id
        state = self.store.load_workflow_runs().get(key, WorkflowRunState(run_id=key))
        if state.completed:
            return ()
        if event.type == "workflow.modified":
            self.store.save_workflow_run(state.model_copy(update={"modified": True}))
            return ()
        # Capture the immutable completion before external snapshot I/O so a
        # restart can retry the same file contents and native checkpoint.
        state = state.model_copy(update={"completion": event})
        self.store.save_workflow_run(state)
        output: list[ProductEvent] = []
        workflow_id = f"{event.event_id}:workflow"

        def append(kind: ProductEventType, payload: JsonObject, event_id: str) -> None:
            persisted, _ = self.store.append(
                ProductEvent(
                    event_id=event_id,
                    conversation_id=self.handle.session_id,
                    run_id=event.run_id,
                    occurred_at=event.occurred_at,
                    type=kind,
                    payload=payload,
                    source_event_id=event.source_event_id,
                )
            )
            output.append(persisted)

        succeeded = False
        try:
            workflow = await self.adapter.finalize_run(
                self.handle, event, workflow_id if state.modified else None
            )
            if workflow is not None:
                append(
                    "workflow.updated", workflow.model_dump(mode="json"), workflow_id
                )
            succeeded = True
        except Exception:
            logger.exception(
                "Workflow completion failed for %s", self.handle.session_id
            )
            append(
                "notice.raised",
                {
                    "severity": "warning",
                    "code": "workflow_snapshot_failed",
                    "message": "Could not save the final workflow snapshot.",
                },
                f"{event.event_id}:workflow-error",
            )
        error = event.payload.get("error")
        if isinstance(error, dict):
            append("notice.raised", error, f"{event.event_id}:error")
        status = event.payload.get("status")
        if isinstance(status, str):
            append("status.changed", {"status": status}, f"{event.event_id}:status")
        if succeeded:
            self.store.save_workflow_run(
                state.model_copy(update={"completed": True, "completion": None})
            )
        return tuple(output)
