"""Archive platform-accepted SDK graphs without resubmitting an external task."""

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import replace

from pyromind_runtime.application.sdk_execution_metadata import (
    bind_sdk_execution_metadata,
    submitted_sdk_dataset_recipe,
)
from pyromind_runtime.application.versions import Versions
from pyromind_runtime.domain.content import JsonObject
from pyromind_runtime.domain.context import RequestContext
from pyromind_runtime.domain.snapshot import ExternalTaskState
from pyromind_runtime.domain.workflow_execution import WorkflowExecutionRequest
from pyromind_runtime.infrastructure.file_product_store import FileProductStore


type AcceptedWorkflowReader = Callable[[str, RequestContext], Awaitable[JsonObject]]
type TaskKey = tuple[str, str]

logger = logging.getLogger(__name__)


class SdkWorkflowArchive:
    def __init__(
        self,
        reader: AcceptedWorkflowReader,
        versions: Versions,
        register_task: Callable[[str, JsonObject], object],
    ):
        self.register_task = register_task
        self.reader = reader
        self.versions = versions
        self._workers: dict[TaskKey, asyncio.Task[None]] = {}
        self._complete: set[TaskKey] = set()
        self._retry: dict[TaskKey, tuple[float, int]] = {}
        self._capacity = asyncio.Semaphore(4)

    def schedule(
        self,
        store: FileProductStore,
        context: RequestContext,
        tasks: Iterable[ExternalTaskState],
    ) -> None:
        if not (context.cookie or context.authorization):
            return
        credentials = hash((context.cookie, context.authorization))
        for task in tasks:
            correlation = task.correlation
            if (
                correlation is None
                or correlation.type != "main"
                or not correlation.signature
                or (
                    not correlation.execution
                    and correlation.task_kind != "data_preparation"
                )
                or not correlation.cluster
                or correlation.task_kind
                not in {"workflow_run", "evaluation", "data_preparation"}
                or task.graph_run_id is not None
            ):
                continue
            key = correlation.conversation_id, task.task_id
            if key in self._complete or key in self._workers:
                continue
            next_try, previous_credentials = self._retry.get(key, (0.0, 0))
            if credentials == previous_credentials and time.monotonic() < next_try:
                continue
            self._workers[key] = asyncio.create_task(
                self._recover(
                    key,
                    store,
                    replace(context, x_cluster=correlation.cluster),
                    task,
                    credentials,
                )
            )

    async def _recover(
        self,
        key: TaskKey,
        store: FileProductStore,
        context: RequestContext,
        task: ExternalTaskState,
        credentials: int,
    ) -> None:
        try:
            async with self._capacity:
                receipt = await asyncio.to_thread(self._existing, store, task, context)
                if receipt is None:
                    workflow = await self.reader(task.task_id, context)
                    if task.correlation and not task.correlation.execution:
                        recipe = submitted_sdk_dataset_recipe(workflow)
                        if recipe is None or not recipe.replay_of_task_id:
                            self._complete.add(key)
                            self._retry.pop(key, None)
                            return
                    receipt, _ = await asyncio.to_thread(
                        bind_sdk_execution_metadata,
                        store,
                        self.versions,
                        {"task_id": task.task_id, "workflow": workflow},
                        context,
                        platform_archive=True,
                    )
                if receipt.dataset_recipe is not None and task.output_dir is None:
                    self.register_task(
                        key[0],
                        {
                            "task_id": task.task_id,
                            "output_dir": receipt.dataset_recipe.output_directory,
                        },
                    )
                self._complete.add(key)
                self._retry.pop(key, None)
        except Exception as exc:
            self._retry[key] = time.monotonic() + 30, credentials
            logger.warning(
                "SDK workflow archive pending: conversation=%s task=%s error=%s",
                *key,
                type(exc).__name__,
            )
        finally:
            self._workers.pop(key, None)

    @staticmethod
    def _existing(
        store: FileProductStore, task: ExternalTaskState, context: RequestContext
    ) -> WorkflowExecutionRequest | None:
        store.authorize(context.user_id, context.profile_id)
        correlation = task.correlation
        if not correlation or not store.verify_task_correlation(correlation):
            raise PermissionError("Unverified SDK task")
        reference = correlation.execution
        snapshot = store.load_snapshot()
        if reference is not None and not any(
            run.role_id == "main"
            and run.project_id == reference.project_id
            and run.task_id == reference.task_id
            and run.plan_sequence == reference.plan_sequence
            and run.plan_attempt == reference.attempt
            and task.task_id in (run.platform_task_id, *run.platform_task_ids)
            for run in snapshot.experiment_executions or ()
        ):
            raise ValueError("SDK task has no matching approved execution")
        return next(
            (
                receipt
                for receipt in store.load_workflow_execution_requests().values()
                if receipt.task_id == task.task_id
                and receipt.correlation == correlation
                and receipt.status == "submitted"
            ),
            None,
        )

    async def close(self) -> None:
        tasks = tuple(self._workers.values())
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
