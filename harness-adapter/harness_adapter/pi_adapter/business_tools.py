from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

from pyromind_runtime.domain.context import RequestContext

from openhands.sdk.conversation.secret_registry import SecretRegistry
from openhands.sdk.workspace.local import LocalWorkspace
from openhands.tools.workflow.validate_workflow_dsl import (
    ValidateWorkflowDslAction,
    ValidateWorkflowDslExecutor,
    ValidateWorkflowDslTool,
)


def validation_tool_spec() -> dict[str, Any]:
    tool = ValidateWorkflowDslTool.create()[0]
    definition = tool.to_mcp_tool()
    schema = definition["inputSchema"]
    if not isinstance(schema, dict):
        raise TypeError("validate_workflow_dsl schema must be an object")
    return {
        "name": tool.name,
        "description": tool.description,
        "input_schema": schema,
    }


async def execute_validation_tool(
    workspace_root: Path,
    arguments: dict[str, Any],
    context: RequestContext,
) -> dict[str, Any]:
    action = ValidateWorkflowDslAction.model_validate(arguments)
    headers = {
        name: value
        for name, value in (
            ("cookie", context.cookie),
            ("authorization", context.authorization),
            ("x-cluster", context.x_cluster),
            ("accept-language", context.accept_language),
        )
        if value
    }
    executor = ValidateWorkflowDslExecutor(headers=headers)
    conversation = SimpleNamespace(
        workspace=LocalWorkspace(working_dir=workspace_root),
        state=SimpleNamespace(agent_state={}, secret_registry=SecretRegistry()),
    )
    observation = await asyncio.to_thread(executor, action, cast(Any, conversation))
    content = [block.model_dump(mode="json") for block in observation.to_llm_content]
    details = observation.model_dump(
        mode="json",
        exclude={"content", "is_error"},
    )
    return {
        "is_error": observation.is_error,
        "content": content,
        "details": details,
    }
