from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta

import httpx
from pyromind_runtime.domain.context import RequestContext
from pyromind_sdk import PyroMindAPIClient
from pyromind_sdk.client.base import PyroMindAPIError
from pyromind_sdk.client.models import TrainingTaskNodeInfo

from openhands.agent_server.echomind_projects import ProjectInfo
from openhands.tools.utils.platform_env import (
    platform_cluster_route,
    platform_cluster_target,
)
from openhands.tools.utils.pyromind_api_client import get_pyromind_api_client
from openhands.tools.workflow.task_submission import create_workflow_api_client
from pyromind_agent_server.echomind_runtime_access import platform_request_token
from pyromind_agent_server.echomind_runtime_credentials import RuntimeCredentialStore
from pyromind_agent_server.echomind_train_backend import _without_sensitive_sdk_logs


_MAX_NODES = 4
_MAX_LOG_LINES = 200
_MAX_EXCERPT = 4096
_TIMEOUT_SECONDS = 4
_ACCESS_ERRORS = (PyroMindAPIError, httpx.HTTPError, OSError, RuntimeError, ValueError)
_DIAGNOSTIC_LINE = re.compile(
    r"\b(?:[a-z_]\w*)?(?:error|exception):|"
    r"\b(?:traceback|fatal|oomkilled|"
    r"out of memory|permission denied|no such file|not a directory|"
    r"exit (?:status|code)|unschedulable|failed to)\b",
    re.IGNORECASE,
)
_URL_QUERY = re.compile(r"(https?://[^\s\"'<>?]+)\?[^\s\"'<>]+", re.IGNORECASE)
_URL_CREDENTIAL = re.compile(r"([a-z][\w+.-]*://)[^\s/'\"@]+@", re.IGNORECASE)
_AUTH_HEADER = re.compile(r"(\b(?:authorization|cookie)[\"']?\s*[:=]\s*).*", re.I)
_CREDENTIAL = re.compile(
    r"(\b[\w.-]*(?:api[_-]?key|access[_-]?key|secret(?:[_-]?key)?|password|"
    r"token)[\"']?\s*[:=]\s*)"
    r"(\"[^\"]*\"|'[^']*'|[^\s,;]+)",
    re.IGNORECASE,
)
_KNOWN_TOKEN = re.compile(
    r"\b(?:Bearer\s+[\w.+/=-]+|wandb_[\w-]{20,}|"
    r"sk-[\w-]{16,}|eyJ[\w-]+\.[\w-]+\.[\w-]+|[0-9a-fA-F]{40})\b",
    re.IGNORECASE,
)


def read_graph_failure_diagnostics(
    project: ProjectInfo,
    task_id: str,
    context: RequestContext,
    *,
    credential_store: RuntimeCredentialStore | None = None,
) -> str | None:
    owner = project.project.owner_user_id
    route = platform_cluster_route(project.project.cluster)
    if (
        owner is None
        or owner != context.user_id
        or (context.x_cluster and platform_cluster_route(context.x_cluster) != route)
        or re.fullmatch(r"[0-9]{1,20}", task_id) is None
    ):
        return None

    auth_token = platform_request_token(context)
    access_key: str | None = None
    env, cluster = platform_cluster_target(route)
    try:
        with _without_sensitive_sdk_logs(True):
            if auth_token:
                client = create_workflow_api_client(
                    env=env,
                    cluster=cluster,
                    auth_token=auth_token,
                    headers={"x-cluster": route},
                    timeout=_TIMEOUT_SECONDS,
                )
            else:
                access_key = (
                    credential_store.load(owner, route) if credential_store else None
                )
                if not access_key:
                    return None
                client = get_pyromind_api_client(
                    env=env,
                    cluster=route,
                    api_key=access_key,
                    timeout=_TIMEOUT_SECONDS,
                    max_retries=0,
                )
            try:
                excerpt = _studio_excerpt(client, task_id)
            finally:
                client.close()
    except _ACCESS_ERRORS:
        return None
    if not excerpt:
        return None
    excerpt = redact_graph_failure_text(excerpt, context, access_key=access_key)
    return excerpt[:_MAX_EXCERPT] or None


def _studio_excerpt(client: PyroMindAPIClient, task_id: str) -> str:
    task = client.studio.get_task(task_id)
    if task.status.casefold() not in {"failed", "error"}:
        return ""
    excerpts: list[str] = []
    if task.error_message:
        excerpts.append(task.error_message)
    for node in (task.nodes or [])[:_MAX_NODES]:
        logs = _read_node_logs(client, task_id, node)
        lines = [
            line
            for line in logs
            if isinstance(line, str) and _DIAGNOSTIC_LINE.search(line)
        ]
        if lines:
            excerpts.append(f"Studio node {node.node_id}:\n" + "\n".join(lines[-8:]))
    return "\n".join(excerpts)


def _read_node_logs(
    client: PyroMindAPIClient, task_id: str, node: TrainingTaskNodeInfo
) -> list[object]:
    params: dict[str, str | int] = {"expect_count": _MAX_LOG_LINES}
    end_at = node.end_at
    if isinstance(end_at, str):
        try:
            end_at = datetime.fromisoformat(end_at)
        except ValueError:
            end_at = None
    if end_at is not None:
        if end_at.tzinfo is None:
            end_at = end_at.replace(tzinfo=UTC)
        params["timestamp"] = (end_at - timedelta(minutes=1)).isoformat()
    path = f"/studio/tasks/{task_id}/nodes/{node.node_id}/logs/scroll"
    try:
        response = client.studio.get(path, params=params)
    except _ACCESS_ERRORS:
        return []
    data = response.get("data")
    logs = data.get("logs") if isinstance(data, dict) else data
    return logs if isinstance(logs, list) else []


def redact_graph_failure_text(
    text: str, context: RequestContext, *, access_key: str | None = None
) -> str:
    cookie_values = [
        value.strip()
        for part in (context.cookie or "").split(";")
        if (value := part.partition("=")[2]).strip()
    ]
    for value in (
        *cookie_values,
        context.authorization,
        platform_request_token(context),
        access_key,
    ):
        if value:
            text = text.replace(value, "[redacted]")
    text = _URL_QUERY.sub(r"\1?[redacted]", text)
    text = _URL_CREDENTIAL.sub(r"\1[redacted]@", text)
    text = _AUTH_HEADER.sub(r"\1[redacted]", text)
    text = _CREDENTIAL.sub(r"\1[redacted]", text)
    return _KNOWN_TOKEN.sub("[redacted]", text)
