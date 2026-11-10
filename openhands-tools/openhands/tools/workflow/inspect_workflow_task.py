"""Inspect the status and logs of any Pyromind workflow task node."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal, Self, cast

import httpx
from pydantic import BaseModel, Field, model_validator
from rich.text import Text

from openhands.sdk.tool.registry import register_tool
from openhands.sdk.tool.tool import (
    Action,
    Observation,
    ToolAnnotations,
    ToolDefinition,
    ToolExecutor,
)
from openhands.tools.workflow.validate_workflow_dsl import (
    PYROMIND_VALIDATE_AUTH_COOKIE_SECRET,
    PYROMIND_VALIDATE_HEADERS_STATE_KEY,
)


if TYPE_CHECKING:
    from openhands.sdk.conversation.base import BaseConversation
    from openhands.sdk.conversation.state import ConversationState


PRE_API_BASE = "https://pre-api-portal.pyromind.ai/std2/studio_api/"
PROD_API_BASE = "https://api-portal.pyromind.ai/std2/studio_api/"
_PROD_APP_ENVS = {"prod", "production", "online"}
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)
_EMPTY_NODE_LOG = "<empty node log>"
_NODE_ID_KEYS = ("node_code", "node_id", "id", "nodeId", "code")
_NODE_TYPE_KEYS = ("nodeType", "node_type", "type", "node_class")
_NODE_NAME_KEYS = ("name", "label", "display_name", "title")
_NODE_STATUS_KEYS = (
    "dystatus",
    "status",
    "state",
    "runStatus",
    "run_status",
    "nodeStatus",
    "node_status",
    "executionStatus",
    "execution_status",
    "taskStatus",
    "task_status",
)


def _default_api_base() -> str:
    app_env = os.getenv("APP_ENV", "dev").strip().lower()
    return PROD_API_BASE if app_env in _PROD_APP_ENVS else PRE_API_BASE


def _resolve_api_base() -> str:
    return os.getenv("PYROMIND_API_BASE", "").strip() or _default_api_base()


def _find_nodes(payload: dict[str, Any]) -> list[dict[str, Any]]:
    candidates: list[Any] = []
    for key in ("nodes", "node_list", "node_infos"):
        value = payload.get(key)
        if isinstance(value, list):
            candidates.append(value)
    workflow = payload.get("workflow")
    if isinstance(workflow, dict):
        for key in ("nodes", "node_list"):
            value = workflow.get(key)
            if isinstance(value, list):
                candidates.append(value)
    data = payload.get("data")
    if isinstance(data, dict):
        for key in ("nodes", "node_list"):
            value = data.get(key)
            if isinstance(value, list):
                candidates.append(value)
    if isinstance(data, list):
        candidates.append(data)
    for candidate in candidates:
        if candidate and all(isinstance(item, dict) for item in candidate):
            return candidate
    return []


def _scalar_text(value: Any) -> str | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (str, int)):
        text = str(value).strip()
        return text or None
    return None


def _first_text(node: dict[str, Any], keys: Sequence[str]) -> str | None:
    for key in keys:
        text = _scalar_text(node.get(key))
        if text:
            return text
    data = node.get("data")
    if isinstance(data, dict):
        for key in keys:
            text = _scalar_text(data.get(key))
            if text:
                return text
    return None


def _node_identifier(node: dict[str, Any]) -> str | None:
    return _first_text(node, _NODE_ID_KEYS)


def _status_text(value: Any) -> str | None:
    if isinstance(value, (bool, dict, list)) or value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"none", "null", "undefined"}:
        return None
    return text


def _node_type(node: dict[str, Any]) -> str | None:
    data = node.get("data")
    if isinstance(data, dict):
        for key in _NODE_TYPE_KEYS:
            text = _status_text(data.get(key))
            if text and text.lower() != "default":
                return text
    for key in ("nodeType", "node_type", "node_class"):
        text = _status_text(node.get(key))
        if text and text.lower() != "default":
            return text
    return None


def _node_status(node: dict[str, Any]) -> str | None:
    properties = node.get("properties")
    if isinstance(properties, dict):
        for key in _NODE_STATUS_KEYS:
            text = _status_text(properties.get(key))
            if text is not None:
                return text
    for container in (node, node.get("data")):
        if not isinstance(container, dict):
            continue
        for key in _NODE_STATUS_KEYS:
            text = _status_text(container.get(key))
            if text is not None:
                return text
    return None


def _extract_log_text(data: Any) -> str:
    if isinstance(data, str):
        return data
    if isinstance(data, dict):
        entries = data.get("entries")
        if isinstance(entries, list):
            parts = [
                entry.get("m", "")
                for entry in entries
                if isinstance(entry, dict) and isinstance(entry.get("m"), str)
            ]
            if parts:
                return "".join(
                    part if part.endswith("\n") else f"{part}\n" for part in parts
                )
        for key in ("log", "logs", "content", "data", "stdout", "text", "raw"):
            if isinstance(data.get(key), str):
                return data[key]
    return ""


def _tail_log(text: str, tail_lines: int, max_chars: int) -> str:
    lines = text.splitlines()
    joined = "\n".join(lines[-tail_lines:])
    return joined if len(joined) <= max_chars else joined[-max_chars:]


class _WorkflowTaskApiClient:
    def __init__(
        self,
        *,
        api_base: str | None = None,
        headers: Mapping[str, str] | None = None,
        secret_headers: Mapping[str, str] | None = None,
        timeout: float = 30.0,
    ) -> None:
        self._api_base = (api_base or _resolve_api_base()).rstrip("/")
        self._headers = dict(headers or {})
        self._secret_headers = dict(secret_headers or {})
        self._timeout = timeout

    def _resolved_headers(
        self, conversation: BaseConversation | None
    ) -> dict[str, str] | None:
        headers = {
            "accept": "application/json",
            "user-agent": USER_AGENT,
            **self._headers,
        }
        if conversation is not None:
            state = cast("ConversationState", conversation.state)
            conversation_headers = state.agent_state.get(
                PYROMIND_VALIDATE_HEADERS_STATE_KEY
            )
            if isinstance(conversation_headers, dict):
                headers.update(
                    {
                        str(name): str(value)
                        for name, value in conversation_headers.items()
                        if value is not None
                    }
                )
            secret_headers = dict(self._secret_headers)
            registry = state.secret_registry
            if registry.get_secret_value(PYROMIND_VALIDATE_AUTH_COOKIE_SECRET):
                secret_headers.setdefault(
                    "cookie", PYROMIND_VALIDATE_AUTH_COOKIE_SECRET
                )
            for header_name, secret_name in secret_headers.items():
                value = registry.get_secret_value(secret_name)
                if not value:
                    return None
                headers[header_name] = value
        return headers

    def _get_json(
        self, path: str, params: dict[str, str], headers: dict[str, str]
    ) -> dict[str, Any] | str:
        try:
            response = httpx.get(
                f"{self._api_base}/{path}",
                params=params,
                headers=headers,
                timeout=self._timeout,
            )
        except httpx.RequestError as exc:
            return f"Platform API {path} unreachable: {type(exc).__name__}: {exc}"
        if response.status_code >= 400:
            guidance = (
                "\nDo not retry: the request is unauthenticated or forbidden "
                "(stop probing credentials)."
                if response.status_code in {401, 403}
                else ""
            )
            return (
                f"Platform API {path} returned HTTP {response.status_code}: "
                f"{response.text[:500]}{guidance}"
            )
        try:
            payload = response.json()
        except (json.JSONDecodeError, ValueError):
            return f"Platform API {path} returned invalid JSON: {response.text[:500]}"
        if not isinstance(payload, dict):
            return f"Platform API {path} returned a non-object JSON payload."
        return payload


WorkflowExecutionState = Literal[
    "succeeded",
    "failed",
    "running",
    "pending",
    "cancelled",
    "skipped",
    "unknown",
]

_FAILED_MARKERS = ("fail", "error", "exception", "timeout")
_CANCELLED_MARKERS = ("cancel", "abort", "terminated", "stopped")
_SKIPPED_MARKERS = ("skip", "omit", "bypass")
_SUCCEEDED_MARKERS = (
    "success",
    "succeed",
    "completed",
    "complete",
    "finished",
    "done",
)
_PENDING_MARKERS = (
    "pending",
    "queued",
    "waiting",
    "created",
    "ready",
    "submitted",
    "not started",
    "not_started",
)
_RUNNING_MARKERS = ("running", "processing", "in progress", "started", "executing")


def workflow_execution_state(status: str | None) -> WorkflowExecutionState:
    """Normalize a platform free-text status without guessing unknown values."""
    if not status:
        return "unknown"
    lowered = status.lower()
    state_markers: tuple[tuple[WorkflowExecutionState, tuple[str, ...]], ...] = (
        ("failed", _FAILED_MARKERS),
        ("cancelled", _CANCELLED_MARKERS),
        ("skipped", _SKIPPED_MARKERS),
        ("succeeded", _SUCCEEDED_MARKERS),
        ("pending", _PENDING_MARKERS),
        ("running", _RUNNING_MARKERS),
    )
    for state, markers in state_markers:
        if any(marker in lowered for marker in markers):
            return state
    return "unknown"


class InspectWorkflowTaskAction(Action):
    """Inspect all node states or one node and its log."""

    task_id: str = Field(description="Platform workflow task id to inspect.")
    node_id: str | None = Field(
        default=None,
        description=(
            "Optional node_code or node id. When omitted, the tool returns every "
            "node and does not fetch logs."
        ),
    )
    include_log: bool = Field(
        default=False,
        description=(
            "Fetch the selected node's stdout log. Requires node_id and works for "
            "normal, running, and failed nodes."
        ),
    )
    tail_lines: int = Field(
        default=100,
        ge=1,
        le=1000,
        description="Number of trailing log lines to return (max 1000).",
    )
    max_log_chars: int = Field(
        default=20000,
        ge=1,
        le=100000,
        description="Hard cap on returned log characters.",
    )

    @model_validator(mode="after")
    def _validate_log_selection(self) -> Self:
        if self.include_log and not self.node_id:
            raise ValueError("node_id is required when include_log=true")
        return self

    @property
    def visualize(self) -> Text:
        content = Text()
        content.append("Inspect workflow task: ", style="bold blue")
        content.append(self.task_id)
        if self.node_id:
            content.append(f" node={self.node_id}")
        return content


class WorkflowNodeStatus(BaseModel):
    """Normalized status for one workflow node."""

    node_id: str = Field(description="Node identifier (node_code / node id).")
    node_type: str | None = Field(default=None, description="Business node type.")
    node_name: str | None = Field(default=None, description="Node display name.")
    raw_status: str | None = Field(
        default=None, description="Unmodified platform node status."
    )
    state: WorkflowExecutionState = Field(description="Normalized node state.")


class InspectWorkflowTaskObservation(Observation):
    """Task status, all node states, and an optional selected-node log."""

    task_status: str | None = Field(
        default=None, description="Unmodified platform task status."
    )
    task_state: WorkflowExecutionState = Field(
        default="unknown", description="Normalized task state."
    )
    nodes: list[WorkflowNodeStatus] = Field(
        default_factory=list, description="Every workflow node and its state."
    )
    failed_nodes: list[WorkflowNodeStatus] = Field(
        default_factory=list, description="Nodes whose normalized state is failed."
    )
    selected_node: WorkflowNodeStatus | None = Field(
        default=None, description="Node selected by node_id, when provided."
    )
    log: str | None = Field(
        default=None, description="Selected node's trailing log when requested."
    )
    node_log_available: bool = Field(
        default=False, description="Whether the requested node returned a usable log."
    )
    error_message: str | None = Field(
        default=None, description="API, lookup, or parse error detail."
    )

    @property
    def visualize(self) -> Text:
        content = Text()
        if self.is_error:
            content.append("Workflow task inspection failed", style="bold red")
        else:
            content.append("Workflow task inspection", style="bold blue")
            content.append(f" ({len(self.nodes)} node(s))")
        if self.task_status:
            content.append(f" task_status={self.task_status}")
        return content


class InspectWorkflowTaskExecutor(
    _WorkflowTaskApiClient,
    ToolExecutor[InspectWorkflowTaskAction, InspectWorkflowTaskObservation],
):
    """Fetch task and optional node-log data from the Pyromind studio API."""

    def __call__(
        self,
        action: InspectWorkflowTaskAction,
        conversation: BaseConversation | None = None,
    ) -> InspectWorkflowTaskObservation:
        headers = self._resolved_headers(conversation)
        if headers is None:
            return self._error(
                "Cannot resolve platform API auth headers: a required header "
                "secret is missing."
            )

        payload = self._get_json(
            "api/task_workflow_result", {"task_id": action.task_id}, headers
        )
        if isinstance(payload, str):
            return self._error(payload)

        task_status = _status_text(payload.get("task_status"))
        task_state = workflow_execution_state(task_status)
        nodes = [
            WorkflowNodeStatus(
                node_id=node_id,
                node_type=_node_type(node),
                node_name=_first_text(node, _NODE_NAME_KEYS),
                raw_status=(raw_status := _node_status(node)),
                state=workflow_execution_state(raw_status),
            )
            for node in _find_nodes(payload)
            if (node_id := _node_identifier(node))
        ]
        if not nodes:
            return self._error(
                "task_workflow_result returned no workflow nodes; the payload "
                "shape may have changed.",
                task_status=task_status,
                task_state=task_state,
            )

        failed_nodes = [node for node in nodes if node.state == "failed"]
        selected_node = None
        if action.node_id:
            selected_node = next(
                (node for node in nodes if node.node_id == action.node_id), None
            )
            if selected_node is None:
                available = ", ".join(node.node_id for node in nodes)
                return self._error(
                    f"Node {action.node_id!r} was not found. Available nodes: "
                    f"{available}",
                    task_status=task_status,
                    task_state=task_state,
                    nodes=nodes,
                    failed_nodes=failed_nodes,
                )

        log = None
        log_available = False
        if action.include_log and selected_node is not None:
            raw = self._get_json(
                "internal/logs/node/raw",
                {"nodeId": selected_node.node_id, "taskId": action.task_id},
                headers,
            )
            if isinstance(raw, str):
                return self._error(
                    raw,
                    task_status=task_status,
                    task_state=task_state,
                    nodes=nodes,
                    failed_nodes=failed_nodes,
                    selected_node=selected_node,
                )
            log = _tail_log(
                _extract_log_text(raw), action.tail_lines, action.max_log_chars
            )
            log_available = bool(log)
            if not log:
                log = _EMPTY_NODE_LOG

        text = _format_inspection_text(
            task_status, task_state, nodes, selected_node, log
        )
        return InspectWorkflowTaskObservation.from_text(
            text=text,
            task_status=task_status,
            task_state=task_state,
            nodes=nodes,
            failed_nodes=failed_nodes,
            selected_node=selected_node,
            log=log,
            node_log_available=log_available,
        )

    def _error(
        self,
        message: str,
        *,
        task_status: str | None = None,
        task_state: WorkflowExecutionState = "unknown",
        nodes: list[WorkflowNodeStatus] | None = None,
        failed_nodes: list[WorkflowNodeStatus] | None = None,
        selected_node: WorkflowNodeStatus | None = None,
    ) -> InspectWorkflowTaskObservation:
        return InspectWorkflowTaskObservation.from_text(
            text=message,
            is_error=True,
            task_status=task_status,
            task_state=task_state,
            nodes=nodes or [],
            failed_nodes=failed_nodes or [],
            selected_node=selected_node,
            error_message=message,
        )


def _format_inspection_text(
    task_status: str | None,
    task_state: WorkflowExecutionState,
    nodes: list[WorkflowNodeStatus],
    selected_node: WorkflowNodeStatus | None,
    log: str | None,
) -> str:
    parts = [f"task_status: {task_status or 'unknown'} ({task_state})", "nodes:"]
    for node in nodes:
        name = node.node_name or node.node_type or "?"
        raw = node.raw_status or "unknown"
        parts.append(f"  - {node.node_id}: {name} [{node.state}; raw={raw}]")
    if selected_node is not None:
        parts.append(f"selected_node: {selected_node.node_id} [{selected_node.state}]")
    if log is not None and selected_node is not None:
        parts.append(f"--- node {selected_node.node_id} log tail ---\n{log}")
    return "\n".join(parts)


TOOL_DESCRIPTION = """Inspect a Pyromind workflow task and any node state.

Use this general read-only tool for running, successful, failed, cancelled,
skipped, or pending tasks. With only task_id, it returns the task state and all
workflow nodes with both normalized and raw statuses, without making one log
request per node. To inspect one node, pass node_id. Set include_log=true only
when its log is needed; logs are available for normal and failed nodes alike.

Do not infer failure from node selection: failed_nodes contains only nodes whose
platform status is actually classified as failed.
"""


class InspectWorkflowTaskTool(
    ToolDefinition[InspectWorkflowTaskAction, InspectWorkflowTaskObservation]
):
    """Tool for inspecting any Pyromind workflow task node."""

    @classmethod
    def create(
        cls,
        conv_state: ConversationState | None = None,  # noqa: ARG003
        **params: Any,
    ) -> Sequence[Self]:
        api_base = params.pop("api_base", None)
        headers = params.pop("headers", None)
        secret_headers = params.pop("secret_headers", None)
        timeout = float(params.pop("timeout", 30.0))
        if params:
            names = ", ".join(sorted(params))
            raise ValueError(f"InspectWorkflowTaskTool got unknown params: {names}")
        if headers is not None and not isinstance(headers, Mapping):
            raise ValueError("headers must be a mapping when provided")
        if secret_headers is not None and not isinstance(secret_headers, Mapping):
            raise ValueError("secret_headers must be a mapping when provided")
        if timeout <= 0:
            raise ValueError("timeout must be greater than 0")

        normalized_headers = (
            {str(key): str(value) for key, value in headers.items()}
            if headers
            else None
        )
        normalized_secret_headers = (
            {str(key): str(value) for key, value in secret_headers.items()}
            if secret_headers
            else None
        )
        return [
            cls(
                description=TOOL_DESCRIPTION,
                action_type=InspectWorkflowTaskAction,
                observation_type=InspectWorkflowTaskObservation,
                executor=InspectWorkflowTaskExecutor(
                    api_base=str(api_base) if api_base is not None else None,
                    headers=normalized_headers,
                    secret_headers=normalized_secret_headers,
                    timeout=timeout,
                ),
                annotations=ToolAnnotations(
                    title="inspect_workflow_task",
                    readOnlyHint=True,
                    destructiveHint=False,
                    idempotentHint=True,
                    openWorldHint=True,
                ),
            )
        ]


register_tool(InspectWorkflowTaskTool.name, InspectWorkflowTaskTool)
