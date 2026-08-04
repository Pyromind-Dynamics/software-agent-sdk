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
    """Request the function signature of a workflow node."""

    node_name: str = Field(
        description="The node type name (e.g., 'SFTTrain', 'DataProcess')"
    )
    node_type: str | None = Field(
        default=None,
        description=(
            "Optional node type filter: 'system', 'share', or 'user'. "
            "Used to disambiguate same-name nodes. If omitted, priority is "
            "system > share > user."
        ),
    )
    include_source: bool = Field(
        default=True,
        description="Whether to include full source code (default: True)",
    )


class NodeSignatureObservation(Observation):
    """Result of fetching a node's function signature."""

    status: Literal["success", "error"] = Field(
        description="'success' if signature was retrieved, 'error' otherwise."
    )
    node_name: str = Field(description="The node type name that was queried.")
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

    @property
    def to_llm_content(self) -> Sequence[TextContent | ImageContent]:
        """Format observation data into text for the LLM."""
        if self.status == "error":
            return [TextContent(text=f"Error: {self.error_message or 'Unknown error'}")]

        parts: list[str] = [f"Node: {self.node_name}"]

        if self.function_signature:
            parts.append(f"Signature: {self.function_signature}")

        if self.docstring:
            parts.append(f"Docstring: {self.docstring}")

        if self.parameters:
            parts.append("Parameters:")
            for p in self.parameters:
                req = "required" if p.get("required") else "optional"
                default = (
                    f", default={p['default']}" if p.get("default") is not None else ""
                )
                parts.append(
                    f"  - {p['name']} ({p.get('type', 'STRING')}, {req}{default})"
                )

        if self.source_code:
            parts.append(f"\nSource Code:\n```python\n{self.source_code}\n```")

        return [TextContent(text="\n".join(parts))]


_NODE_SIGNATURE_DESCRIPTION = """Retrieve the entry function signature, docstring, and source code of a workflow node.

Use this when you need to understand what parameters a node expects.
Returns the function signature, parameter types/defaults, docstring, and full
source code.

The goal is to help you understand what parameters the node expects and how it
processes them, so you can fix workflow.py accordingly.

Args:
    node_name: The node type name (e.g., "SFTTrain", "DataProcess")
    node_type: Optional filter - "system", "share", or "user". If omitted,
        priority is system > share > user.
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
