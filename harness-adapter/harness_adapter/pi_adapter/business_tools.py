from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from types import SimpleNamespace
from typing import Any, cast

from harness_adapter.workflow_validation import (
    RESOURCE_TOOL_DESCRIPTION,
    ResourceValidatedWorkflowDslExecutor,
)
from pyromind_common.sdk.conversation.secret_registry import SecretRegistry
from pyromind_common.sdk.workspace.base import BaseWorkspace
from pyromind_common.tools.utils.platform_env import platform_cluster_route
from pyromind_common.tools.workflow.dsl_to_xyflow import convert_dsl_to_xyflow
from pyromind_common.tools.workflow.validate_workflow_dsl import (
    ValidateWorkflowDslAction,
    ValidateWorkflowDslObservation,
    ValidateWorkflowDslTool,
)
from pyromind_runtime.application.reward_workflows import (
    training_workflow,
    validate_reward_workflow,
)
from pyromind_runtime.domain.context import RequestContext
from pyromind_runtime.domain.reward_bundle import RewardBundle


def validation_tool_spec() -> dict[str, Any]:
    tool = ValidateWorkflowDslTool.create()[0]
    definition = tool.to_mcp_tool()
    schema = definition["inputSchema"]
    if not isinstance(schema, dict):
        raise TypeError("validate_workflow_dsl schema must be an object")
    return {
        "name": tool.name,
        "description": RESOURCE_TOOL_DESCRIPTION,
        "input_schema": schema,
    }


type ProjectRewardReader = Callable[
    [str, RequestContext], Awaitable[RewardBundle | None]
]


async def execute_validation_tool(
    workspace: BaseWorkspace,
    arguments: dict[str, Any],
    context: RequestContext,
    reward: RewardBundle | None = None,
) -> dict[str, Any]:
    action = ValidateWorkflowDslAction.model_validate(arguments)
    headers = {
        name: value
        for name, value in (
            ("cookie", context.cookie),
            ("authorization", context.authorization),
            ("x-cluster", platform_cluster_route(context.x_cluster)),
            ("accept-language", context.accept_language),
        )
        if value
    }
    executor = ResourceValidatedWorkflowDslExecutor(headers=headers)
    conversation = SimpleNamespace(
        workspace=workspace,
        state=SimpleNamespace(agent_state={}, secret_registry=SecretRegistry()),
    )

    def validate():
        if reward is not None:
            try:
                dsl = executor._resolve_dsl(action, cast(Any, conversation))
                workflow = convert_dsl_to_xyflow(dsl, name=action.name)
                if training_workflow(workflow):
                    validate_reward_workflow(workflow, reward)
                action_to_validate = ValidateWorkflowDslAction(
                    dsl=dsl, name=action.name
                )
            except ValueError as exc:
                return ValidateWorkflowDslObservation.from_text(
                    text=str(exc), is_error=True, valid=False
                )
        else:
            action_to_validate = action
        return executor(action_to_validate, cast(Any, conversation))

    observation = await asyncio.to_thread(validate)
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
