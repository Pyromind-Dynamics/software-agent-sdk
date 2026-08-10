"""Analyze a failed Pyromind workflow task via platform APIs.

Fetches the task workflow result, locates failed nodes (by status field or an
explicit node_id), and returns the tail of each target node's stdout log so the
agent can diagnose the failure. The platform API contract is calibrated in
``.agents/skills/training-analysis/references/platform-data-contract.md``.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal, Self, cast

import httpx
from pydantic import BaseModel, Field
from rich.text import Text

from openhands.sdk.tool.registry import register_tool
from openhands.sdk.tool.tool import (
    Action,
    Observation,
    ToolAnnotations,
    ToolDefinition,
    ToolExecutor,
)
from openhands.tools.utils.pyromind_api_client import _build_access_key_request_headers
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

# Cloudflare rejects non-browser User-Agents with HTTP 403 browser_signature_banned.
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)

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
_FAILURE_MARKERS = ("fail", "error", "exception")

# Function-signature API secret name, same as get_node_function_signature.
PYROMIND_AUTH_TOKEN_SECRET = "auth_token"


def _default_api_base() -> str:
    app_env = os.getenv("APP_ENV", "dev").strip().lower()
    if app_env in _PROD_APP_ENVS:
        return PROD_API_BASE
    return PRE_API_BASE


def _resolve_api_base() -> str:
    return os.getenv("PYROMIND_API_BASE", "").strip() or _default_api_base()


# ---------------------------------------------------------------------------
# Action / Observation
# ---------------------------------------------------------------------------


class AnalyzeTaskFailureAction(Action):
    """Inspect a platform workflow task and diagnose failed node logs."""

    task_id: str = Field(description="Platform workflow task id to analyze.")
    node_id: str | None = Field(
        default=None,
        description=(
            "Target a specific node directly (node_code or node id from the "
            "workflow). When omitted, the tool auto-detects failed nodes from "
            "the task workflow result."
        ),
    )
    tail_lines: int = Field(
        default=100,
        ge=1,
        le=1000,
        description="Number of trailing log lines to return per node (max 1000).",
    )
    max_log_chars: int = Field(
        default=20000,
        ge=1,
        le=100000,
        description=(
            "Hard cap on returned log characters per node; the tail is trimmed "
            "to this many characters."
        ),
    )
    include_source: bool = Field(
        default=True,
        description=(
            "Whether to also fetch each analyzed node's operator source code "
            "from the function-signature API, alongside the log tail."
        ),
    )
    source_lines: int = Field(
        default=200,
        ge=1,
        le=300,
        description="Maximum source code lines returned per node.",
    )

    @property
    def visualize(self) -> Text:
        content = Text()
        content.append("Analyze task failure: ", style="bold red")
        content.append(self.task_id)
        if self.node_id:
            content.append(f" node={self.node_id}")
        return content


class TaskNodeInfo(BaseModel):
    """Minimal node summary parsed from the task workflow result."""

    node_id: str = Field(description="Node identifier (node_code / node id).")
    node_type: str | None = Field(
        default=None, description="Node type, e.g. ModelTrainSFTNode."
    )
    node_name: str | None = Field(default=None, description="Node display name.")
    status: str | None = Field(
        default=None, description="Raw node status text, when present."
    )


class AnalyzeTaskFailureObservation(Observation):
    """Result of a task failure analysis."""

    task_status: str | None = Field(
        default=None, description="task_status from the task workflow result."
    )
    nodes: list[TaskNodeInfo] = Field(
        default_factory=list, description="All workflow nodes found in the task result."
    )
    failed_nodes: list[TaskNodeInfo] = Field(
        default_factory=list,
        description=(
            "Nodes whose status looks failed (status contains fail/error/exception). "
            "Empty when the payload carries no node status or node_id was explicit."
        ),
    )
    logs: dict[str, str] = Field(
        default_factory=dict,
        description="node_id -> trailing log text fetched for each analyzed node.",
    )
    node_sources: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "node_id -> operator source code fetched from the function-signature "
            "API when include_source=true and an auth token is available."
        ),
    )
    source: Literal["explicit", "status", "all"] = Field(
        description=(
            "How the analyzed nodes were chosen: 'explicit' from node_id, "
            "'status' from failed-node detection, or 'all' meaning no logs were "
            "fetched because no failed node could be identified."
        ),
    )
    error_message: str | None = Field(
        default=None, description="API or parse error detail, when the call failed."
    )

    @property
    def visualize(self) -> Text:
        content = Text()
        if self.is_error:
            content.append("Task failure analysis failed", style="bold red")
        elif self.logs:
            content.append("Task failure analysis", style="bold yellow")
            content.append(f" ({len(self.logs)} node log(s))")
        else:
            content.append("Task failure analysis", style="bold blue")
        if self.task_status:
            content.append(f" task_status={self.task_status}")
        if self.failed_nodes:
            content.append(f" failed={len(self.failed_nodes)}")
        if self.node_sources:
            content.append(f" sources={len(self.node_sources)}")
        return content


# ---------------------------------------------------------------------------
# Payload parsing helpers
# ---------------------------------------------------------------------------


def _find_nodes(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Locate the node list in a task_workflow_result payload.

    The calibrated shape nests nodes under ``workflow.nodes``; tolerate top-level
    or ``data``-wrapped arrays and alternate key names.
    """
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


def _first_string(node: dict[str, Any], keys: Sequence[str]) -> str | None:
    for key in keys:
        value = node.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    data = node.get("data")
    if isinstance(data, dict):
        for key in keys:
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def _node_identifier(node: dict[str, Any]) -> str | None:
    return _first_string(node, _NODE_ID_KEYS)


def _node_type(node: dict[str, Any]) -> str | None:
    """Extract the business node type from a task workflow result node.

    The calibrated payload stores the real type in ``data.nodeType``; the
    top-level ``type`` is only a React Flow render hint (usually "default")
    and must be ignored.
    """
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


def _status_text(value: Any) -> str | None:
    if isinstance(value, (bool, dict, list)) or value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"none", "null", "undefined"}:
        return None
    return text


def _looks_failed(status: str | None) -> bool:
    if not status:
        return False
    lowered = status.lower()
    return any(marker in lowered for marker in _FAILURE_MARKERS)


def _extract_log_text(data: Any) -> str:
    """Join node log text from the raw API response (entries[].m array)."""
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
    tail = lines[-tail_lines:] if tail_lines > 0 else lines
    joined = "\n".join(tail)
    if len(joined) <= max_chars:
        return joined
    return joined[-max_chars:]


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------


class AnalyzeTaskFailureExecutor(
    ToolExecutor[AnalyzeTaskFailureAction, AnalyzeTaskFailureObservation]
):
    """Fetch task workflow result and node logs from the Pyromind studio API."""

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

    def __call__(
        self,
        action: AnalyzeTaskFailureAction,
        conversation: BaseConversation | None = None,
    ) -> AnalyzeTaskFailureObservation:
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
        nodes = [
            TaskNodeInfo(
                node_id=node_id,
                node_type=_node_type(node),
                node_name=_first_string(node, _NODE_NAME_KEYS),
                status=_node_status(node),
            )
            for node in _find_nodes(payload)
            if (node_id := _node_identifier(node))
        ]
        if not nodes:
            return self._error(
                "task_workflow_result returned no workflow nodes; the payload "
                "shape may have changed.",
                task_status=task_status,
            )

        if action.node_id:
            target_ids = [action.node_id]
            source: Literal["explicit", "status", "all"] = "explicit"
        else:
            failed = [node for node in nodes if _looks_failed(node.status)]
            if failed:
                target_ids = [node.node_id for node in failed]
                source = "status"
            else:
                target_ids = []
                source = "all"

        auth_token = self._auth_token(conversation)
        node_by_id = {node.node_id: node for node in nodes}
        logs: dict[str, str] = {}
        node_sources: dict[str, str] = {}
        for target in target_ids:
            raw = self._get_json(
                "internal/logs/node/raw",
                {"nodeId": target, "taskId": action.task_id},
                headers,
            )
            if isinstance(raw, str):
                logs[target] = f"<log fetch failed: {raw}>"
                continue
            log_text = _tail_log(
                _extract_log_text(raw), action.tail_lines, action.max_log_chars
            )
            if not log_text:
                logs[target] = "<empty node log>"
            else:
                logs[target] = log_text
            if not action.include_source or auth_token is None:
                continue
            node = node_by_id.get(target)
            node_type = node.node_type if node else None
            if not node_type:
                continue
            source_code = self._fetch_node_source(
                node_type, auth_token, action.source_lines
            )
            if source_code:
                node_sources[target] = source_code

        failed_nodes = (
            [node for node in nodes if node.node_id in target_ids]
            if source == "explicit"
            else [node for node in nodes if _looks_failed(node.status)]
        )
        text = _format_observation_text(
            task_status, nodes, failed_nodes, logs, node_sources, source
        )
        return AnalyzeTaskFailureObservation.from_text(
            text=text,
            task_status=task_status,
            nodes=nodes,
            failed_nodes=failed_nodes,
            logs=logs,
            node_sources=node_sources,
            source=source,
        )

    # -- HTTP ---------------------------------------------------------------

    def _resolved_headers(
        self, conversation: BaseConversation | None
    ) -> dict[str, str] | None:
        headers = {
            "accept": "application/json",
            "user-agent": USER_AGENT,
            **self._headers,
        }
        if conversation is not None:
            headers.update(self._resolve_conversation_headers(conversation))
            resolved_secrets = self._resolve_secret_headers(conversation)
            if resolved_secrets is None:
                return None
            headers.update(resolved_secrets)
        return headers

    def _resolve_conversation_headers(
        self, conversation: BaseConversation
    ) -> dict[str, str]:
        state = cast("ConversationState", conversation.state)
        headers = state.agent_state.get(PYROMIND_VALIDATE_HEADERS_STATE_KEY)
        if not isinstance(headers, dict):
            return {}
        return {
            str(name): str(value)
            for name, value in headers.items()
            if value is not None
        }

    def _resolve_secret_headers(
        self, conversation: BaseConversation
    ) -> dict[str, str] | None:
        secret_headers = dict(self._secret_headers)
        state = cast("ConversationState", conversation.state)
        secret_registry = state.secret_registry
        if secret_registry.get_secret_value(PYROMIND_VALIDATE_AUTH_COOKIE_SECRET):
            secret_headers.setdefault("cookie", PYROMIND_VALIDATE_AUTH_COOKIE_SECRET)
        if not secret_headers:
            return {}
        resolved: dict[str, str] = {}
        for header_name, secret_name in secret_headers.items():
            value = secret_registry.get_secret_value(secret_name)
            if not value:
                return None
            resolved[header_name] = value
        return resolved

    def _auth_token(self, conversation: BaseConversation | None) -> str | None:
        """Read the function-signature auth token from the conversation registry."""
        if conversation is None:
            return None
        state = cast("ConversationState", conversation.state)
        value = state.secret_registry.get_secret_value(PYROMIND_AUTH_TOKEN_SECRET)
        return value if value else None

    def _fetch_node_source(
        self,
        node_type: str,
        auth_token: str,
        source_lines: int,
    ) -> str | None:
        """Fetch a node operator's source code; best-effort, never raises."""
        url = f"{self._api_base}/api/agent/nodes/function_signature/batch"
        headers = _build_access_key_request_headers(
            self._headers, auth_token=auth_token
        )
        try:
            response = httpx.post(
                url,
                json={
                    "node_names": [node_type],
                    "node_type": None,
                    "include_source": True,
                    "max_source_lines": source_lines,
                },
                headers=headers,
                timeout=self._timeout,
            )
        except httpx.RequestError:
            return None
        if response.status_code >= 400:
            return None
        try:
            payload = response.json()
        except (json.JSONDecodeError, ValueError):
            return None
        if not isinstance(payload, dict) or payload.get("success") is not True:
            return None
        data = payload.get("data")
        if not isinstance(data, dict):
            return None
        entry = data.get(node_type)
        if not isinstance(entry, dict) or entry.get("success") is not True:
            return None
        entry_data = entry.get("data")
        if not isinstance(entry_data, dict):
            return None
        source = entry_data.get("source_code")
        return str(source) if isinstance(source, str) and source else None

    def _get_json(
        self, path: str, params: dict[str, str], headers: dict[str, str]
    ) -> dict[str, Any] | str:
        url = f"{self._api_base}/{path}"
        try:
            response = httpx.get(
                url, params=params, headers=headers, timeout=self._timeout
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

    # -- Errors -------------------------------------------------------------

    def _error(
        self,
        message: str,
        *,
        task_status: str | None = None,
    ) -> AnalyzeTaskFailureObservation:
        return AnalyzeTaskFailureObservation.from_text(
            text=message,
            is_error=True,
            task_status=task_status,
            error_message=message,
            source="all",
        )


def _format_observation_text(
    task_status: str | None,
    nodes: list[TaskNodeInfo],
    failed_nodes: list[TaskNodeInfo],
    logs: dict[str, str],
    node_sources: dict[str, str],
    source: Literal["explicit", "status", "all"],
) -> str:
    parts: list[str] = []
    if task_status:
        parts.append(f"task_status: {task_status}")
    parts.append(f"workflow nodes: {len(nodes)}")
    if failed_nodes:
        ids = ", ".join(node.node_id for node in failed_nodes)
        parts.append(f"failed nodes (status-based): {ids}")
    if logs:
        for node_id, log_text in logs.items():
            parts.append(f"--- node {node_id} (last log tail) ---\n{log_text}")
    if node_sources:
        for node_id, source_code in node_sources.items():
            parts.append(f"--- node {node_id} operator source ---\n{source_code}")
    elif source == "all":
        parts.append(
            "No failed node could be identified from the task result (the payload "
            "has no readable node status). Pass `node_id` to fetch a specific "
            "node's log. Available nodes:"
        )
        for node in nodes:
            name = node.node_name or node.node_type or "?"
            parts.append(f"  - {node.node_id}: {name}")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Tool Definition
# ---------------------------------------------------------------------------


TOOL_DESCRIPTION = """Analyze a failed Pyromind workflow task on the platform.

Use this tool when a workflow run failed or errored (task_id from
workflow_debug / run_workflow / stop_task / the user) and you need to see which
node failed and why. The tool:

1. Calls `api/task_workflow_result?task_id=...` to list the workflow nodes and
   task_status. Nodes carry id, nodeType (e.g. ModelTrainSFTNode), and config.
2. Auto-detects failed nodes by their status field (status text containing
   fail/error/exception). If the payload has no readable node status, pass
   `node_id` explicitly to analyze one node.
3. Fetches each target node's stdout via
   `internal/logs/node/raw?nodeId=...&taskId=...` and returns the last
   `tail_lines` lines (default 100, max 1000).

The observation returns task_status, all nodes, failed_nodes, logs
(node_id -> trailing log text), and node_sources (node_id -> operator source
code, fetched from the function-signature API when include_source=true and an
auth token is available). Diagnose the failure by comparing the failing log
lines against the operator source (parameter mismatch, operator-internal
exceptions, etc.); fix the workflow DSL if the error is deterministic, then
re-run via workflow_debug. Source fetching is best-effort: a failed source
fetch never fails the whole analysis.

Auth mirrors validate_workflow_dsl: cookie / x-cluster / authorization headers
are forwarded, and the request carries a browser-style User-Agent (the platform
rejects other UAs with HTTP 403). HTTP 401/403 responses mean the credentials
are stale — stop and do not probe credentials with terminal commands.
"""  # noqa: E501


class AnalyzeTaskFailureTool(
    ToolDefinition[AnalyzeTaskFailureAction, AnalyzeTaskFailureObservation]
):
    """Tool for diagnosing failed Pyromind workflow tasks."""

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
            raise ValueError(f"AnalyzeTaskFailureTool got unknown params: {names}")
        if headers is not None and not isinstance(headers, dict):
            raise ValueError("headers must be a dictionary when provided")
        if secret_headers is not None and not isinstance(secret_headers, dict):
            raise ValueError("secret_headers must be a dictionary when provided")
        if timeout <= 0:
            raise ValueError("timeout must be greater than 0")

        normalized_headers = (
            {str(k): str(v) for k, v in headers.items()} if headers else None
        )
        normalized_secret_headers = (
            {str(k): str(v) for k, v in secret_headers.items()}
            if secret_headers
            else None
        )

        return [
            cls(
                description=TOOL_DESCRIPTION,
                action_type=AnalyzeTaskFailureAction,
                observation_type=AnalyzeTaskFailureObservation,
                executor=AnalyzeTaskFailureExecutor(
                    api_base=str(api_base) if api_base is not None else None,
                    headers=normalized_headers,
                    secret_headers=normalized_secret_headers,
                    timeout=timeout,
                ),
                annotations=ToolAnnotations(
                    title="analyze_task_failure",
                    readOnlyHint=True,
                    destructiveHint=False,
                    idempotentHint=True,
                    openWorldHint=True,
                ),
            )
        ]


register_tool(AnalyzeTaskFailureTool.name, AnalyzeTaskFailureTool)
