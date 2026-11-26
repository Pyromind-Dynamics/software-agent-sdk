from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

from pyromind_runtime.domain.snapshot import ExternalTaskState
from pyromind_runtime.infrastructure.file_product_store import (
    FileProductStore,
    ProductStoreError,
)
from pyromind_runtime.ports.task_monitor import (
    TaskMonitorHost,
    TaskMonitorStore,
    TaskStatusQuery,
)


logger = logging.getLogger(__name__)
_PLATFORM_KINDS = frozenset(
    {
        "workflow_run",
        "workflow_debug",
        "data_cleaning",
        "data_preparation",
        "environment_processing",
        "environment_processing_render",
        "environment_processing_aggregate",
        "graph_await",
    }
)
_TERMINAL = frozenset({"succeeded", "failed", "terminated", "stopped"})


class TaskMonitor:
    """Run bounded status checks without involving a harness or model."""

    def __init__(
        self,
        store: TaskMonitorStore,
        query: TaskStatusQuery,
        host: TaskMonitorHost,
        *,
        first_delay: float = 60,
        interval: float = 180,
        query_timeout: float = 30,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if min(first_delay, interval, query_timeout) <= 0:
            raise ValueError("monitor delays and timeout must be positive")
        self.store = store
        self.query = query
        self.host = host
        self.first_delay = first_delay
        self.interval = interval
        self.query_timeout = query_timeout
        self.clock = clock

    @staticmethod
    def tracks(task: ExternalTaskState) -> bool:
        return task.kind in _PLATFORM_KINDS and task.status not in _TERMINAL

    def register(self, conversation_id: str, task: ExternalTaskState) -> None:
        if self.tracks(task):
            self.store.register(
                conversation_id, task.task_id, self.clock() + self.first_delay
            )

    def recover(self, root: Path) -> None:
        for path in root.glob("*/product/snapshot.json"):
            try:
                snapshot = FileProductStore(path.parent.parent).load_snapshot()
                for task in snapshot.external_tasks:
                    self.register(snapshot.conversation_id, task)
            except (OSError, ProductStoreError):
                logger.warning("Could not recover task monitoring for %s", path.parent)

    async def check_due(self, *, limit: int = 10) -> int:
        checked = 0
        for _ in range(limit):
            claimed = self.store.claim(self.clock(), self.query_timeout + 60)
            if claimed is None:
                break
            try:
                task = self.host.task_monitor_state(
                    claimed.conversation_id, claimed.task_id
                )
                if task is None:
                    self.store.remove(claimed)
                    continue
                if task.status in _TERMINAL:
                    await self.host.deliver_external_task_status(
                        claimed.conversation_id,
                        task_id=task.task_id,
                        status=task.status,
                        error_summary=task.error_summary,
                        auto_run=task.resume_auto_run,
                        trusted_platform=True,
                    )
                    if task.resume_auto_run:
                        await self.host.resume_monitored_task(claimed.conversation_id)
                    self.store.remove(claimed)
                    continue
                context = self.host.task_monitor_context(claimed.conversation_id)
                if context is None:
                    self.store.postpone(claimed, self.clock() + self.interval, 0)
                    continue
                if task.x_cluster is not None:
                    context = replace(context, x_cluster=task.x_cluster)
                result = await asyncio.wait_for(
                    self.query.query(task.task_id, context), self.query_timeout
                )
                checked += 1
                if result.status not in _TERMINAL | {"pending", "running"}:
                    raise ValueError("platform returned an unknown task status")
                if result.status != task.status or result.status in _TERMINAL:
                    await self.host.deliver_external_task_status(
                        claimed.conversation_id,
                        task_id=task.task_id,
                        status=result.status,
                        error_summary=result.error_summary,
                        auto_run=task.resume_auto_run,
                        trusted_platform=True,
                    )
                if result.status in _TERMINAL:
                    if task.resume_auto_run:
                        await self.host.resume_monitored_task(claimed.conversation_id)
                    self.store.remove(claimed)
                else:
                    self.store.postpone(claimed, self.clock() + self.interval, 0)
            except asyncio.CancelledError:
                self.store.postpone(claimed, self.clock(), claimed.failures)
                raise
            except Exception as exc:
                failures = claimed.failures + 1
                delay = min(300, self.interval * 2 ** min(failures - 1, 4))
                self.store.postpone(claimed, self.clock() + delay, failures)
                logger.warning(
                    "Task check deferred task_id=%s error_type=%s",
                    claimed.task_id,
                    type(exc).__name__,
                )
        return checked
