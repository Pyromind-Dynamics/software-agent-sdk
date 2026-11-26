"""Read Studio task facts and node logs without registering an Agent Tool."""

from __future__ import annotations

from typing import Literal

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    ValidationError,
)
from pyromind_sdk import PyroMindAPIClient
from pyromind_sdk.client.base import PyroMindAPIError
from requests import RequestException

from pyromind_common.platform.workflows import sdk_request_logging


WorkflowExecutionState = Literal[
    "succeeded", "failed", "running", "pending", "cancelled", "skipped", "unknown"
]
_IDENTIFIER_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}$"
_STATUS_GROUPS: dict[WorkflowExecutionState, tuple[str, ...]] = {
    "succeeded": ("success", "succeeded", "completed", "complete", "finished", "done"),
    "failed": ("failed", "failure", "error", "exception", "timeout", "timed out"),
    "running": ("running", "processing", "in progress", "started", "executing"),
    "pending": (
        "pending",
        "queued",
        "waiting",
        "created",
        "ready",
        "submitted",
        "not started",
    ),
    "cancelled": ("cancelled", "canceled", "aborted", "terminated", "stopped"),
    "skipped": ("skipped", "omitted", "bypassed"),
}
_STATUS_KEYS = (
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


def workflow_execution_state(status: str | None) -> WorkflowExecutionState:
    value = status.strip().lower().replace("_", " ") if status else ""
    for state, values in _STATUS_GROUPS.items():
        if value in values:
            return state
    return "unknown"


class WorkflowNodeStatus(BaseModel):
    node_id: str
    node_code: str | None = None
    node_name: str | None = None
    node_type: str | None = None
    raw_status: str | None = None
    state: WorkflowExecutionState = "unknown"


class WorkflowTaskInspection(BaseModel):
    task_id: str
    task_status: str | None = None
    task_state: WorkflowExecutionState = "unknown"
    nodes: list[WorkflowNodeStatus] = Field(default_factory=list)
    node_status_available: bool = False
    failed_nodes: list[WorkflowNodeStatus] = Field(default_factory=list)
    selected_node: WorkflowNodeStatus | None = None
    log: str | None = None
    node_log_available: bool = False
    log_error: str | None = None


class WorkflowInspectionError(RuntimeError):
    def __init__(self, status_code: int | None = None) -> None:
        self.status_code = status_code
        super().__init__(
            "Studio task inspection failed; check task access and API availability"
            + (f" (HTTP {status_code})" if status_code is not None else "")
        )


class _ResponseModel(BaseModel):
    model_config = ConfigDict(coerce_numbers_to_str=True, hide_input_in_errors=True)


class _NodeProperties(_ResponseModel):
    dystatus: str | None = Field(
        default=None, validation_alias=AliasChoices(*_STATUS_KEYS)
    )


class _NodeData(_ResponseModel):
    node_type: str | None = Field(
        default=None, validation_alias=AliasChoices("nodeType", "node_type")
    )
    display_name: str | None = None
    status: str | None = Field(
        default=None, validation_alias=AliasChoices(*_STATUS_KEYS)
    )


class _Node(_ResponseModel):
    node_id: str = Field(
        validation_alias=AliasChoices("node_id", "node_code", "id"),
        pattern=_IDENTIFIER_PATTERN,
    )
    node_code: str | None = None
    node_name: str | None = Field(
        default=None, validation_alias=AliasChoices("node_name", "name")
    )
    node_type: str | None = Field(
        default=None, validation_alias=AliasChoices("node_type", "nodeType")
    )
    status: str | None = Field(
        default=None, validation_alias=AliasChoices(*_STATUS_KEYS)
    )
    properties: _NodeProperties = Field(default_factory=_NodeProperties)
    data: _NodeData = Field(default_factory=_NodeData)

    def inspect(self, definition: _Node | None = None) -> WorkflowNodeStatus:
        raw = self.properties.dystatus or self.status or self.data.status
        if raw is None and definition is not None:
            raw = (
                definition.properties.dystatus
                or definition.status
                or definition.data.status
            )
        return WorkflowNodeStatus(
            node_id=self.node_id,
            node_code=self.node_code,
            node_name=self.node_name
            or self.data.display_name
            or (
                definition.node_name or definition.data.display_name
                if definition is not None
                else None
            ),
            node_type=self.node_type
            or self.data.node_type
            or (
                definition.node_type or definition.data.node_type
                if definition is not None
                else None
            ),
            raw_status=raw,
            state=workflow_execution_state(raw),
        )


class _Workflow(_ResponseModel):
    nodes: list[_Node] = Field(default_factory=list)


class _Task(_ResponseModel):
    task_id: str
    status: str | None = None
    nodes: list[_Node] | None = None
    workflow: _Workflow | None = None

    def inspect(self) -> WorkflowTaskInspection:
        definitions = self.workflow.nodes if self.workflow is not None else []
        by_code: dict[str, list[_Node]] = {}
        for definition in definitions:
            by_code.setdefault(definition.node_code or definition.node_id, []).append(
                definition
            )
        nodes = []
        for node in self.nodes or definitions:
            matches = by_code.get(node.node_code, []) if node.node_code else []
            nodes.append(node.inspect(matches[0] if len(matches) == 1 else None))
        return WorkflowTaskInspection(
            task_id=self.task_id,
            task_status=self.status,
            task_state=workflow_execution_state(self.status),
            nodes=nodes,
            node_status_available=bool(nodes)
            and all(node.raw_status is not None for node in nodes),
            failed_nodes=[node for node in nodes if node.state == "failed"],
        )


class _InspectionOptions(BaseModel):
    model_config = ConfigDict(hide_input_in_errors=True)

    task_id: str = Field(pattern=_IDENTIFIER_PATTERN)
    node_id: str | None = Field(default=None, pattern=_IDENTIFIER_PATTERN)
    tail_lines: int = Field(default=100, ge=1, le=1000)
    max_log_chars: int = Field(default=20000, ge=1, le=100000)


def inspect_workflow_task(
    client: PyroMindAPIClient,
    task_id: str,
    *,
    node_id: str | None = None,
    include_log: bool = False,
    tail_lines: int = 100,
    max_log_chars: int = 20000,
) -> WorkflowTaskInspection:
    """Read raw SDK task data so unmodelled execution status is retained.

    Missing node status remains unknown. Logs are fetched only for an explicitly
    selected node; output is bounded after the SDK's full-log request.
    """
    options = _InspectionOptions(
        task_id=task_id,
        node_id=node_id,
        tail_lines=tail_lines,
        max_log_chars=max_log_chars,
    )
    if include_log and options.node_id is None:
        raise ValueError("node_id is required when include_log=True")
    try:
        with sdk_request_logging():
            payload = client.studio.get(f"/studio/tasks/{options.task_id}")
        if payload.get("success") is False:
            raise WorkflowInspectionError()
        task = _Task.model_validate(payload.get("data", payload))
    except PyroMindAPIError as exc:
        raise WorkflowInspectionError(exc.status_code) from None
    except (RequestException, ValidationError, ValueError, OSError):
        raise WorkflowInspectionError() from None
    if task.task_id != options.task_id:
        raise WorkflowInspectionError()
    result = task.inspect()
    if options.node_id is None:
        return result
    selected = [
        node
        for node in result.nodes
        if options.node_id in {node.node_id, node.node_code}
    ]
    if len(selected) != 1:
        raise ValueError("Requested node must identify exactly one node in the task")
    result.selected_node = selected[0]
    if include_log:
        try:
            with sdk_request_logging():
                lines = client.studio.get_node_logs(
                    options.task_id, selected[0].node_id
                )
            lines = TypeAdapter(list[str]).validate_python(lines)
            text = "".join(
                line if line.endswith("\n") else line + "\n" for line in lines
            )
            result.log = "\n".join(text.splitlines()[-options.tail_lines :])[
                -options.max_log_chars :
            ]
            result.node_log_available = bool(result.log)
        except (PyroMindAPIError, RequestException, ValueError, OSError):
            result.log_error = "Node log unavailable; task and node facts remain valid"
    return result
