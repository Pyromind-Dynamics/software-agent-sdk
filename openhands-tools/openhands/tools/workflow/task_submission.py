"""Shared Pyromind Studio workflow submission helpers."""

from __future__ import annotations

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


class WorkflowTaskSubmissionError(RuntimeError):
    """Raised when Pyromind does not create a workflow task."""


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

    access_key = get_api_key(
        env=env,
        auth_token=auth_token,
        origin_headers=headers,
        timeout=timeout,
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
) -> TrainingTaskCreateResponse:
    """Create one Studio task with the standard conversation correlation ID."""
    request = TrainingTaskCreateRequest(
        name=name,
        workflow=workflow,
        out_id=f"agent1#{conversation_id}",
    )
    response = client.studio.create(request)
    if response is None:
        raise WorkflowTaskSubmissionError("Workflow create failed, response is None")
    if not response.task_id:
        raise WorkflowTaskSubmissionError("Workflow create failed, task_id is None")
    return response
