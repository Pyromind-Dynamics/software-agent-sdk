from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from pyromind_runtime.domain.context import RequestContext

from openhands.tools.workflow.validate_workflow_dsl import (
    ValidateWorkflowDslAction,
    ValidateWorkflowDslExecutor,
    ValidateWorkflowDslTool,
)


_MAX_DSL_BYTES = 2 * 1024 * 1024


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
    if action.dsl_path is None:
        raise ValueError("dsl_path is required")
    dsl = _read_workspace_file(workspace_root, action.dsl_path)
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
    legacy_action = ValidateWorkflowDslAction(dsl=dsl, name=action.name)
    observation = await asyncio.to_thread(executor, legacy_action, None)
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


def _read_workspace_file(workspace_root: Path, relative: str) -> str:
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(
            f"Workflow DSL path must stay inside the workspace: {relative!r}"
        )
    current = workspace_root.resolve()
    for part in path.parts:
        current /= part
        if current.is_symlink():
            raise ValueError(
                f"Workflow DSL path must not contain symlinks: {relative!r}"
            )
    resolved = (workspace_root / path).resolve()
    if not resolved.is_relative_to(workspace_root.resolve()):
        raise ValueError(
            f"Workflow DSL path must stay inside the workspace: {relative!r}"
        )
    if not resolved.is_file():
        raise ValueError(f"Cannot read workflow DSL file: {relative!r} does not exist.")
    if resolved.stat().st_size > _MAX_DSL_BYTES:
        raise ValueError(
            f"Workflow DSL file exceeds {_MAX_DSL_BYTES} bytes: {relative!r}"
        )
    try:
        return resolved.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"Workflow DSL file must be UTF-8: {relative!r}") from exc
