"""Definition of the get_node_function_signature tool.

This tool retrieves the entry function signature and docstring of a workflow node
from the Pyromind middleware API. It helps the Agent understand what parameters
a node expects without needing to read the full source code.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

from pydantic import Field

from openhands.sdk.llm.message import ImageContent, TextContent
from openhands.sdk.tool import (
    Action,
    Observation,
    ToolAnnotations,
    ToolDefinition,
    register_tool,
)


if TYPE_CHECKING:
    from openhands.sdk.conversation.state import ConversationState


class NodeSignatureAction(Action):
    """Request the function signature of one or more workflow nodes."""

    node_name: str | None = Field(
        default=None,
        description=(
            "Legacy single-node alias; merged into the batch query when "
            "node_names is empty. Prefer node_names."
        ),
    )
    node_names: list[str] = Field(
        default_factory=list,
        description=(
            "Batch mode: list of node type names to query in a single call. "
            "Preferred over node_name when multiple nodes are needed."
        ),
    )
    node_type: str | None = Field(
        default=None,
        description=(
            "Optional node type filter: 'system', 'share', or 'user'. "
            "Used to disambiguate same-name nodes. If omitted, same-name "
            "candidates are tried by priority system > share > user."
        ),
    )
    include_source: bool = Field(
        default=True,
        description="Whether to include full source code (default: True)",
    )


class NodeSignatureObservation(Observation):
    """Result of fetching node function signatures."""

    status: Literal["success", "error"] = Field(
        description="'success' if signature was retrieved, 'error' otherwise."
    )
    node_name: str | None = Field(
        default=None, description="The node type name that was queried."
    )
    results: list[dict] | None = Field(
        default=None,
        description="Batch results, one entry per queried node.",
    )
    function_signature: str | None = Field(
        default=None,
        description="The entry function signature (e.g., 'def train(model_path, ...)')",
    )
    docstring: str | None = Field(
        default=None,
        description="The function's docstring explaining parameters.",
    )
    parameters: list[dict] | None = Field(
        default=None,
        description="Parsed parameter list with name, type, default for each param.",
    )
    source_code: str | None = Field(
        default=None,
        description="Full source code if include_source=True was requested.",
    )
    error_message: str | None = Field(
        default=None,
        description="Error message if status='error'.",
    )

    @staticmethod
    def _format_entry(entry: dict) -> str:
        node_name = entry.get("node_name", "?")
        if not entry.get("success"):
            return f"Node: {node_name}\nError: {entry.get('error_message') or 'Unknown error'}"

        parts: list[str] = [f"Node: {node_name}"]

        if entry.get("function_signature"):
            parts.append(f"Signature: {entry['function_signature']}")

        if entry.get("docstring"):
            parts.append(f"Docstring: {entry['docstring']}")

        if entry.get("parameters"):
            parts.append("Parameters:")
            for p in entry["parameters"]:
                req = "required" if p.get("required") else "optional"
                default = (
                    f", default={p['default']}" if p.get("default") is not None else ""
                )
                parts.append(
                    f"  - {p['name']} ({p.get('type', 'STRING')}, {req}{default})"
                )

        if entry.get("source_code"):
            parts.append(f"\nSource Code:\n```python\n{entry['source_code']}\n```")

        return "\n".join(parts)

    @property
    def to_llm_content(self) -> Sequence[TextContent | ImageContent]:
        """Format observation data into text for the LLM."""
        if self.results:
            return [
                TextContent(
                    text="\n\n".join(self._format_entry(e) for e in self.results)
                )
            ]

        if self.status == "error":
            return [TextContent(text=f"Error: {self.error_message or 'Unknown error'}")]

        return [
            TextContent(
                text=self._format_entry(
                    {
                        "node_name": self.node_name,
                        "success": True,
                        "function_signature": self.function_signature,
                        "docstring": self.docstring,
                        "parameters": self.parameters,
                        "source_code": self.source_code,
                    }
                )
            )
        ]


_NODE_SIGNATURE_DESCRIPTION = """Retrieve the entry function signature, docstring, and source code of one or more workflow nodes.

Use this when you need to understand what parameters a node expects.
Returns the function signature, parameter types/defaults, docstring, and full
source code for each requested node.

The goal is to help you understand what parameters the node expects and how it
processes them, so you can fix workflow.py accordingly.

Args:
    node_names: List of node type names to query in one call
        (e.g., ["LoadDataset", "ModelTrainGRPONode"]). Use this when you need
        signatures for one or more nodes at once.
    node_name: Legacy single-node alias, converted into a one-element batch
        query. Only used when node_names is empty.
    node_type: Optional filter - "system", "share", or "user". If omitted,
        same-name candidates are tried by priority system > share > user.
    include_source: Set to False if you only need the signature without source code
"""


class GetNodeFunctionSignatureTool(
    ToolDefinition[NodeSignatureAction, NodeSignatureObservation]
):
    """Tool that fetches a node's function signature from the middleware API."""

    @classmethod
    def create(
        cls,
        conv_state: ConversationState | None = None,
        **params,
    ) -> Sequence[ToolDefinition]:
        del conv_state
        from openhands.tools.node_signature.impl import NodeSignatureExecutor

        env = str(params.pop("env", "")) or None
        headers = params.pop("headers", None)
        if isinstance(headers, dict):
            headers = {str(k): str(v) for k, v in headers.items()}
        else:
            headers = None

        executor = NodeSignatureExecutor(env=env, headers=headers)
        return [
            cls(
                description=_NODE_SIGNATURE_DESCRIPTION,
                action_type=NodeSignatureAction,
                observation_type=NodeSignatureObservation,
                annotations=ToolAnnotations(
                    title="get_node_function_signature",
                    readOnlyHint=True,
                    destructiveHint=False,
                    idempotentHint=True,
                    openWorldHint=True,
                ),
                executor=executor,
            )
        ]


register_tool(GetNodeFunctionSignatureTool.name, GetNodeFunctionSignatureTool)
