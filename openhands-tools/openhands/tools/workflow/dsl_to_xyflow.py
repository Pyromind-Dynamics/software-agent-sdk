from __future__ import annotations

import json
from collections.abc import Callable, Sequence
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


class DslToXyflowAction(Action):
    dsl: str = Field(
        description=(
            "Pyromind workflow Python DSL source code to convert (not a file path)."
        ),
    )
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
        self._converter_factory = converter_factory or _create_dsl_converter

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
            converter = self._converter_factory()
            xyflow = converter.from_python(action.dsl, name=action.name)
        except Exception as exc:
            return DslToXyflowObservation.from_text(
                text=(
                    "Failed to convert workflow DSL to xyflow: "
                    f"{type(exc).__name__}: {exc}"
                ),
                is_error=True,
            )

        if not isinstance(xyflow, dict):
            return DslToXyflowObservation.from_text(
                text=(
                    "Workflow DSL converter returned "
                    f"{type(xyflow).__name__}; expected dict."
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


def _create_dsl_converter() -> Any:
    try:
        workflow_module = import_module("pyromind_sdk.client.workflow")
        converter_type = getattr(workflow_module, "DslConverter")
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(
            "pyromind-sdk with pyromind_sdk.client.workflow.DslConverter is "
            "required for DSL to xyflow conversion."
        ) from exc
    return converter_type()


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
