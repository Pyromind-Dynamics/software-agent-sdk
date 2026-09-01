"""Shared Pyromind Studio workflow submission helpers."""

from __future__ import annotations

import time
from collections.abc import Callable

import httpx
from pyromind_sdk import PyroMindAPIClient
from pyromind_sdk.client.models import (
    TrainingTaskCreateRequest,
    TrainingTaskCreateResponse,
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
    if not env:
        raise ValueError("env is required.")
    if not cluster:
        raise ValueError("cluster is required.")

    access_key = _retry_transient(
        lambda: get_api_key(
            env=env,
            auth_token=auth_token,
            origin_headers=headers,
            timeout=timeout,
        )
    )
    return get_pyromind_api_client(
        env=env,
        cluster=cluster,
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
) -> TrainingTaskCreateResponse:
    """Create one Studio task with the standard conversation correlation ID.

    When ``test_mode`` is True (``workflow_debug`` path), ``out_id`` is
    ``agent1#debug#<conversation_id>`` so Kafka callbacks can apply
    debug-only agent guidance. Production runs keep ``agent1#<conversation_id>``.
    """
    if test_mode:
        out_id = f"agent1#debug#{conversation_id}"
    else:
        out_id = f"agent1#{conversation_id}"
    request = TrainingTaskCreateRequest(
        name=name,
        workflow=workflow,
        out_id=out_id,
    )
    response = _retry_transient(lambda: client.studio.create(request))
    if response is None:
        raise WorkflowTaskSubmissionError("Workflow create failed, response is None")
    if not response.task_id:
        raise WorkflowTaskSubmissionError("Workflow create failed, task_id is None")
    return response
