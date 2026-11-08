"""Shared Pyromind Studio workflow submission helpers."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field

import httpx
from pyromind_runtime.domain.task_correlation import TaskCorrelation, TaskOrigin
from pyromind_sdk import PyroMindAPIClient
from pyromind_sdk.client.models import (
    TrainingTaskCreateRequest,
    TrainingTaskCreateResponse,
)

from openhands.tools.utils.platform_env import (
    platform_client_route,
    platform_cluster_target,
    platform_headers,
)
from openhands.tools.utils.pyromind_api_client import (
    get_api_key,
    get_pyromind_api_client,
)


PYROMIND_WORKFLOW_AUTH_TOKEN_SECRET = "auth_token"

# get_api_key raises httpx request errors while pyromind_sdk raises requests
# errors (OSError subclasses); both cover SSL EOF / connection resets, which
# a fresh attempt usually survives.
_TRANSIENT_NETWORK_ERRORS = (
    httpx.RequestError,
    httpx.RemoteProtocolError,
    OSError,
)


@dataclass
class WorkflowSubmissionScope:
    origin: TaskOrigin
    submissions: dict[str, TaskCorrelation] = field(default_factory=dict)
    direct_tasks: set[str] = field(default_factory=set)


_current_correlation: ContextVar[TaskCorrelation | None] = ContextVar(
    "current_task_correlation", default=None
)


def current_task_correlation() -> TaskCorrelation | None:
    return _current_correlation.get()


_submission_scope: ContextVar[WorkflowSubmissionScope | None] = ContextVar(
    "workflow_submission_scope", default=None
)


@contextmanager
def workflow_submission_scope(origin: TaskOrigin) -> Iterator[WorkflowSubmissionScope]:
    scope = WorkflowSubmissionScope(origin)
    token = _submission_scope.set(scope)
    try:
        yield scope
    finally:
        _submission_scope.reset(token)


class WorkflowTaskSubmissionError(RuntimeError):
    """Raised when Pyromind does not create a workflow task."""


def _retry_transient[T](
    operation: Callable[[], T], *, attempts: int = 3, delay: float = 1.0
) -> T:
    for attempt in range(attempts):
        try:
            return operation()
        except _TRANSIENT_NETWORK_ERRORS:
            if attempt == attempts - 1:
                raise
            time.sleep(delay * 2**attempt)
    raise AssertionError("unreachable")


def create_workflow_api_client(
    *,
    env: str | None,
    cluster: str | None,
    auth_token: str | None,
    headers: dict[str, str],
    timeout: int = 30,
) -> PyroMindAPIClient:
    """Create the authenticated client used by every workflow-producing tool."""
    if not auth_token:
        raise ValueError("API key is required.")
    route = platform_client_route(headers, cluster=cluster, env=env)
    env, _ = platform_cluster_target(route)

    access_key = _retry_transient(
        lambda: get_api_key(
            env=env,
            auth_token=auth_token,
            origin_headers=platform_headers(headers, route=route),
            timeout=timeout,
        )
    )
    return get_pyromind_api_client(
        env=env,
        cluster=route,
        api_key=access_key,
        timeout=timeout,
    )


def submit_workflow_task(
    *,
    client: PyroMindAPIClient,
    workflow: dict,
    name: str,
    conversation_id: str,
    test_mode: bool = False,
    task_kind: str = "workflow_run",
    origin: TaskOrigin | None = None,
    retry_create: bool = True,
    submission_handler: Callable[[dict, str, str | None], TrainingTaskCreateResponse]
    | None = None,
    source_code: str | None = None,
) -> TrainingTaskCreateResponse:
    """Submit a Studio task carrying the caller's return address as JSON text."""
    scope = _submission_scope.get()
    recipient = origin or (scope.origin if scope else TaskOrigin())
    correlation = TaskCorrelation(
        conversation_id=conversation_id,
        task_kind="workflow_debug" if test_mode else task_kind,
        **recipient.model_dump(),
    )
    if submission_handler is not None and not test_mode:
        token = _current_correlation.set(correlation)
        try:
            response = submission_handler(workflow, name, source_code)
        finally:
            _current_correlation.reset(token)
    else:
        request = TrainingTaskCreateRequest(
            name=name, workflow=workflow, out_id=correlation.to_out_id()
        )
        response = (
            _retry_transient(lambda: client.studio.create(request))
            if retry_create
            else client.studio.create(request)
        )
    if response is None:
        raise WorkflowTaskSubmissionError("Workflow create failed, response is None")
    if not response.task_id:
        raise WorkflowTaskSubmissionError("Workflow create failed, task_id is None")
    if scope is not None:
        scope.submissions[str(response.task_id)] = correlation
        if submission_handler is None or test_mode:
            scope.direct_tasks.add(str(response.task_id))
    return response
