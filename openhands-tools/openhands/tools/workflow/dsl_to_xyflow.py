from __future__ import annotations

import ast
import json
from collections.abc import Callable, Sequence
from copy import deepcopy
from importlib import import_module
from typing import TYPE_CHECKING, Any, Self

from pydantic import Field
from rich.text import Text

from openhands.sdk.tool.registry import register_tool
from openhands.sdk.tool.tool import (
    Action,
    Observation,
    ToolAnnotations,
    ToolDefinition,
    ToolExecutor,
)


if TYPE_CHECKING:
    from openhands.sdk.conversation.base import BaseConversation
    from openhands.sdk.conversation.state import ConversationState


DslConverterFactory = Callable[[], Any]


def create_dsl_converter() -> Any:
    try:
        workflow_module = import_module("pyromind_sdk.client.workflow")
        converter_type = getattr(workflow_module, "DslConverter")
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(
            "pyromind-sdk with pyromind_sdk.client.workflow.DslConverter is "
            "required for workflow DSL/xyflow conversion."
        ) from exc
    return converter_type()


def convert_dsl_to_xyflow(
    dsl: str,
    *,
    name: str = "workflow",
    converter_factory: DslConverterFactory | None = None,
) -> dict[str, Any]:
    converter = (converter_factory or create_dsl_converter)()
    xyflow = converter.from_python(dsl, name=name)
    if not isinstance(xyflow, dict):
        raise TypeError(
            f"Workflow DSL converter returned {type(xyflow).__name__}; expected dict."
        )
    return _normalize_xyflow(xyflow)


def convert_xyflow_to_dsl(
    xyflow: dict[str, Any],
    *,
    converter_factory: DslConverterFactory | None = None,
) -> str:
    converter = (converter_factory or create_dsl_converter)()
    dsl = converter.to_python(_normalize_xyflow(xyflow))
    if not isinstance(dsl, str):
        raise TypeError(
            f"Workflow xyflow converter returned {type(dsl).__name__}; expected str."
        )
    ast.parse(dsl)
    return dsl


def _normalize_xyflow(xyflow: dict[str, Any]) -> dict[str, Any]:
    normalized = deepcopy(xyflow)
    nodes = normalized.get("nodes")
    if not isinstance(nodes, list):
        return normalized

    edges = normalized.get("edges")
    if not isinstance(edges, list):
        edges = []

    incoming_handles = _edge_handles_by_node(edges, "target", "targetHandle")
    outgoing_handles = _edge_handles_by_node(edges, "source", "sourceHandle")
    for node in nodes:
        if not isinstance(node, dict):
            continue
        data = node.get("data")
        if not isinstance(data, dict):
            continue
        if data.get("nodeType") and not node.get("type"):
            node["type"] = "default"
        node_id = str(node.get("id"))
        _ensure_minimal_node_definition(
            data,
            incoming_handles.get(node_id, []),
            outgoing_handles.get(node_id, []),
        )
    return normalized


def _edge_handles_by_node(
    edges: list[Any],
    node_key: str,
    handle_key: str,
) -> dict[str, list[str]]:
    handles_by_node: dict[str, list[str]] = {}
    for edge in edges:
        if not isinstance(edge, dict):
            continue
        node_id = edge.get(node_key)
        handle = edge.get(handle_key)
        if node_id is None or not isinstance(handle, str) or not handle:
            continue
        handles_by_node.setdefault(str(node_id), []).append(handle)
    return handles_by_node


def _ensure_minimal_node_definition(
    data: dict[str, Any],
    incoming_handles: list[str],
    outgoing_handles: list[str],
) -> None:
    input_names = _ordered_unique(
        [
            *incoming_handles,
            *_node_config_keys(data.get("config")),
        ]
    )
    output_names = _ordered_unique(outgoing_handles)
    if not input_names and not output_names:
        return

    node_definition = data.get("nodeDefinition")
    if not isinstance(node_definition, dict):
        node_definition = {}
        data["nodeDefinition"] = node_definition

    input_definition = node_definition.get("input")
    if not isinstance(input_definition, dict):
        input_definition = {}
        node_definition["input"] = input_definition

    required = input_definition.get("required")
    if not isinstance(required, dict):
        required = {}
        input_definition["required"] = required
    optional = input_definition.get("optional")
    if not isinstance(optional, dict):
        optional = {}
        input_definition["optional"] = optional

    known_inputs = set(required) | set(optional)
    for name in input_names:
        if name not in known_inputs:
            optional[name] = {}
            known_inputs.add(name)

    existing_outputs = node_definition.get("output_name")
    if not isinstance(existing_outputs, list):
        existing_outputs = []
        node_definition["output_name"] = existing_outputs

    known_outputs = {name for name in existing_outputs if isinstance(name, str)}
    for name in output_names:
        if name not in known_outputs:
            existing_outputs.append(name)
            known_outputs.add(name)


def _node_config_keys(config: Any) -> list[str]:
    if not isinstance(config, dict):
        return []
    return [
        key
        for key, value in config.items()
        if isinstance(key, str) and key != "controlMode" and value not in ("", None)
    ]


def _ordered_unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


class DslToXyflowAction(Action):
    dsl: str = Field(description="Pyromind workflow Python DSL source to convert.")
    name: str = Field(default="workflow", description="Workflow name.")

    @property
    def visualize(self) -> Text:
        content = Text()
        content.append("Convert workflow DSL to xyflow: ", style="bold blue")
        content.append(self.name)
        return content


class DslToXyflowObservation(Observation):
    xyflow: dict[str, Any] | None = Field(
        default=None,
        description="Converted xyflow JSON workflow.",
    )
    workflow_name: str | None = Field(default=None, description="Workflow name.")
    node_count: int | None = Field(default=None, description="Number of nodes.")
    edge_count: int | None = Field(default=None, description="Number of edges.")

    @property
    def visualize(self) -> Text:
        content = Text()
        if self.is_error:
            content.append("Workflow DSL conversion failed", style="bold red")
        else:
            content.append("Workflow DSL converted to xyflow", style="bold green")
        if self.workflow_name:
            content.append(f" ({self.workflow_name})")
        if self.node_count is not None:
            content.append(f"\nNodes: {self.node_count}")
        if self.edge_count is not None:
            content.append(f"\nEdges: {self.edge_count}")
        return content


TOOL_DESCRIPTION = """Convert Pyromind workflow Python DSL to xyflow JSON.

Use this tool after creating or editing a Pyromind workflow DSL script when you
need the SDK-standard xyflow JSON representation. It calls
`pyromind_sdk.client.workflow.DslConverter().from_python(code, name)` and
returns the converted workflow JSON containing fields such as `name`, `nodes`,
and `edges`.
"""


class DslToXyflowExecutor(ToolExecutor):
    def __init__(
        self,
        converter_factory: DslConverterFactory | None = None,
    ) -> None:
        self._converter_factory = converter_factory or create_dsl_converter

    def __call__(
        self,
        action: DslToXyflowAction,
        conversation: BaseConversation | None = None,  # noqa: ARG002
    ) -> DslToXyflowObservation:
        if not action.dsl.strip():
            return DslToXyflowObservation.from_text(
                text="Workflow DSL source must be a non-empty string.",
                is_error=True,
            )
        if not action.name.strip():
            return DslToXyflowObservation.from_text(
                text="Workflow name must be a non-empty string.",
                is_error=True,
            )

        try:
            xyflow = convert_dsl_to_xyflow(
                action.dsl,
                name=action.name,
                converter_factory=self._converter_factory,
            )
        except Exception as exc:
            return DslToXyflowObservation.from_text(
                text=(
                    "Failed to convert workflow DSL to xyflow: "
                    f"{type(exc).__name__}: {exc}"
                ),
                is_error=True,
            )

        try:
            xyflow_json = json.dumps(xyflow, ensure_ascii=False, indent=2)
        except TypeError as exc:
            return DslToXyflowObservation.from_text(
                text=f"Workflow DSL converter returned non-JSON data: {exc}",
                is_error=True,
            )

        workflow_name = _optional_str(xyflow.get("name")) or action.name
        node_count = _count_list(xyflow.get("nodes"))
        edge_count = _count_list(xyflow.get("edges"))
        summary = _format_summary(workflow_name, node_count, edge_count)

        return DslToXyflowObservation.from_text(
            text=f"{summary}\n{xyflow_json}",
            xyflow=xyflow,
            workflow_name=workflow_name,
            node_count=node_count,
            edge_count=edge_count,
        )


class DslToXyflowTool(ToolDefinition[DslToXyflowAction, DslToXyflowObservation]):
    """Tool for converting Pyromind workflow DSL to xyflow JSON."""

    @classmethod
    def create(
        cls,
        conv_state: ConversationState | None = None,  # noqa: ARG003
        **params: Any,
    ) -> Sequence[Self]:
        if params:
            names = ", ".join(sorted(params))
            raise ValueError(f"DslToXyflowTool got unknown params: {names}")

        return [
            cls(
                description=TOOL_DESCRIPTION,
                action_type=DslToXyflowAction,
                observation_type=DslToXyflowObservation,
                executor=DslToXyflowExecutor(),
                annotations=ToolAnnotations(
                    title="dsl_to_xyflow",
                    readOnlyHint=True,
                    destructiveHint=False,
                    idempotentHint=True,
                    openWorldHint=False,
                ),
            )
        ]


def _count_list(value: Any) -> int | None:
    if isinstance(value, list):
        return len(value)
    return None


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _format_summary(
    workflow_name: str,
    node_count: int | None,
    edge_count: int | None,
) -> str:
    node_text = str(node_count) if node_count is not None else "unknown"
    edge_text = str(edge_count) if edge_count is not None else "unknown"
    return (
        "Workflow DSL converted to xyflow JSON. "
        f"name={workflow_name}. nodes={node_text}. edges={edge_text}."
    )


register_tool(DslToXyflowTool.name, DslToXyflowTool)
