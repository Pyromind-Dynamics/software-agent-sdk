"""Fallback polling for Pyromind workflow tasks awaiting terminal callbacks."""

from __future__ import annotations

import asyncio
import os
from collections.abc import Awaitable, Callable
from contextlib import suppress
from dataclasses import dataclass
from typing import Any

from pyromind_sdk import PyroMindAPIClient

from openhands.agent_server.event_service import EventService
from openhands.agent_server.pyromind_constants import (
    PYROMIND_APP_TAG_KEY,
    PYROMIND_APP_TAG_VALUE,
)
from openhands.agent_server.run_workflow_callback import (
    TERMINAL_STATUSES,
    RunWorkflowCallbackResult,
    deliver_run_workflow_status,
    normalize_platform_status,
)
from openhands.sdk import get_logger
from openhands.sdk.conversation.state import ActiveLongTask
from openhands.tools.embodied_data.platform_submit import RunEmbodiedSandboxTool
from openhands.tools.workflow.task_submission import (
    PYROMIND_WORKFLOW_AUTH_TOKEN_SECRET,
    create_workflow_api_client,
)


logger = get_logger(__name__)

DEFAULT_POLL_INTERVAL_SECONDS = 10.0
POLL_INTERVAL_ENV = "PYROMIND_WORKFLOW_TASK_POLL_INTERVAL_SECONDS"
POLLABLE_TASK_KINDS = frozenset({"embodied_data_cleaning"})

StatusCallback = Callable[..., Awaitable[RunWorkflowCallbackResult]]
ClientFactory = Callable[..., PyroMindAPIClient]


@dataclass(frozen=True)
class _PollContext:
    event_service: EventService
    conversation_id: str
    tasks: tuple[ActiveLongTask, ...]
    env: str
    cluster: str
    headers: dict[str, str]
    timeout: int
    auth_token: str


@dataclass(repr=False)
class _CachedClient:
    env: str
    cluster: str
    headers: dict[str, str]
    timeout: int
    auth_token: str
    client: PyroMindAPIClient

    def matches(self, context: _PollContext) -> bool:
        return (
            self.env == context.env
            and self.cluster == context.cluster
            and self.headers == context.headers
            and self.timeout == context.timeout
            and self.auth_token == context.auth_token
        )


class WorkflowTaskPoller:
    def __init__(
        self,
        conversation_service: Any,
        *,
        interval_seconds: float | None = None,
        client_factory: ClientFactory = create_workflow_api_client,
        status_callback: StatusCallback = deliver_run_workflow_status,
    ) -> None:
        self._conversation_service = conversation_service
        self._interval_seconds = (
            _poll_interval_from_env() if interval_seconds is None else interval_seconds
        )
        if self._interval_seconds <= 0:
            raise ValueError("workflow task poll interval must be greater than 0")
        self._client_factory = client_factory
        self._status_callback = status_callback
        self._clients: dict[str, _CachedClient] = {}
        self._loop_task: asyncio.Task[None] | None = None

    def start(self) -> None:
        if self._loop_task is None or self._loop_task.done():
            self._loop_task = asyncio.create_task(
                self._run_loop(),
                name="pyromind-workflow-task-poller",
            )

    async def stop(self) -> None:
        if self._loop_task is None:
            return
        self._loop_task.cancel()
        with suppress(asyncio.CancelledError):
            await self._loop_task
        self._loop_task = None
        self._clients.clear()

    async def run_once(self) -> None:
        event_services = self._conversation_service._loaded_event_services_snapshot()
        snapshots = await asyncio.gather(
            *(
                asyncio.to_thread(_snapshot_poll_context, item)
                for item in event_services
            ),
            return_exceptions=True,
        )
        contexts: list[_PollContext] = []
        for snapshot in snapshots:
            if isinstance(snapshot, BaseException):
                logger.warning(
                    "Failed to snapshot workflow polling context: %s", snapshot
                )
            elif snapshot is not None:
                contexts.append(snapshot)

        active_conversations = {context.conversation_id for context in contexts}
        self._clients = {
            conversation_id: entry
            for conversation_id, entry in self._clients.items()
            if conversation_id in active_conversations
        }
        for context in contexts:
            await self._poll_context(context)

    async def _run_loop(self) -> None:
        while True:
            try:
                await self.run_once()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Workflow task polling iteration failed")
            await asyncio.sleep(self._interval_seconds)

    async def _poll_context(self, context: _PollContext) -> None:
        try:
            client = await self._client_for(context)
        except Exception as exc:
            logger.warning(
                "Failed to create workflow polling client for conversation %s: %s",
                context.conversation_id,
                exc,
            )
            return

        for task in context.tasks:
            try:
                response = await asyncio.to_thread(client.studio.get_task, task.task_id)
            except Exception as exc:
                self._clients.pop(context.conversation_id, None)
                logger.warning(
                    "Failed to poll workflow task %s: %s",
                    task.task_id,
                    exc,
                )
                continue

            try:
                status = normalize_platform_status(response.status)
            except (AttributeError, ValueError) as exc:
                logger.warning(
                    "Workflow task %s returned an unsupported status: %s",
                    task.task_id,
                    exc,
                )
                continue

            if status not in TERMINAL_STATUSES:
                if task.status != status:
                    await context.event_service.update_active_long_task_status(
                        task.task_id,
                        status,
                    )
                continue

            await self._status_callback(
                task_id=task.task_id,
                status=status,
                conversation_id=context.conversation_id,
                auto_run=True,
                conversation_service=self._conversation_service,
            )

    async def _client_for(self, context: _PollContext) -> PyroMindAPIClient:
        cached = self._clients.get(context.conversation_id)
        if cached is not None and cached.matches(context):
            return cached.client
        client = await asyncio.to_thread(
            self._client_factory,
            env=context.env,
            cluster=context.cluster,
            auth_token=context.auth_token,
            headers=context.headers,
            timeout=context.timeout,
        )
        self._clients[context.conversation_id] = _CachedClient(
            env=context.env,
            cluster=context.cluster,
            headers=context.headers,
            timeout=context.timeout,
            auth_token=context.auth_token,
            client=client,
        )
        return client


def _snapshot_poll_context(event_service: EventService) -> _PollContext | None:
    if event_service.stored.tags.get(PYROMIND_APP_TAG_KEY) != PYROMIND_APP_TAG_VALUE:
        return None
    conversation = event_service.get_conversation()
    with conversation._state as state:
        tasks = tuple(
            task
            for task in state.active_long_tasks
            if task.kind in POLLABLE_TASK_KINDS and task.status != "Stopped"
        )
        if not tasks:
            return None
        tool = next(
            (
                item
                for item in state.agent.tools
                if item.name == RunEmbodiedSandboxTool.name
            ),
            None,
        )
        if tool is None:
            return None
        auth_token = state.secret_registry.get_secret_value(
            PYROMIND_WORKFLOW_AUTH_TOKEN_SECRET
        )
        params = tool.params

    env = params.get("env")
    cluster = params.get("cluster")
    if (
        not isinstance(env, str)
        or not env
        or not isinstance(cluster, str)
        or not cluster
    ):
        return None
    if not auth_token:
        return None
    raw_headers = params.get("headers", {})
    headers = (
        {str(name): str(value) for name, value in raw_headers.items()}
        if isinstance(raw_headers, dict)
        else {}
    )
    raw_timeout = params.get("timeout", 30)
    timeout = raw_timeout if isinstance(raw_timeout, int) and raw_timeout > 0 else 30
    return _PollContext(
        event_service=event_service,
        conversation_id=str(event_service.stored.id),
        tasks=tasks,
        env=env,
        cluster=cluster,
        headers=headers,
        timeout=timeout,
        auth_token=auth_token,
    )


def _poll_interval_from_env() -> float:
    raw = os.environ.get(POLL_INTERVAL_ENV)
    if raw is None:
        return DEFAULT_POLL_INTERVAL_SECONDS
    try:
        value = float(raw)
    except ValueError:
        logger.warning(
            "Ignoring invalid %s=%r; using %.1fs",
            POLL_INTERVAL_ENV,
            raw,
            DEFAULT_POLL_INTERVAL_SECONDS,
        )
        return DEFAULT_POLL_INTERVAL_SECONDS
    if value <= 0:
        logger.warning(
            "Ignoring non-positive %s=%r; using %.1fs",
            POLL_INTERVAL_ENV,
            raw,
            DEFAULT_POLL_INTERVAL_SECONDS,
        )
        return DEFAULT_POLL_INTERVAL_SECONDS
    return value
