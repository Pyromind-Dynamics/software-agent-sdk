"""Native tool wrappers for the shared Product Runtime version API."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Coroutine
from typing import Any, Literal, Self

from pydantic import Field
from pyromind_runtime.domain.context import RequestContext
from pyromind_runtime.domain.versions import VersionToolRequest

from openhands.sdk.tool import (
    Action,
    Observation,
    ToolAnnotations,
    ToolDefinition,
    ToolExecutor,
    register_tool,
)


type VersionCall = Callable[
    [str, VersionToolRequest, str, RequestContext], Coroutine[Any, Any, dict[str, Any]]
]

VERSION_DESCRIPTION = """Manage this conversation's artifact versions.
Actions: list (no arguments); save (kind, logical_key, source with storage_ref OR
text and filename, optional parents and summary); compare (left_version_id,
right_version_id); select (slot, version_id, expected_revision, optional reason);
new_iteration (goal, expected_revision); recover (retry unfinished snapshots).
Use the revision returned by list for selections and new iterations. Reuse the
same command_id when retrying the same operation. A reference_only record is not
a restorable snapshot. Selection affects future executions, not running tasks.
Use version://<slot> in workflow path fields to consume a selected version.
"""


class ProductVersionsAction(Action):
    action: Literal["save", "list", "compare", "select", "new_iteration", "recover"]
    command_id: str = Field(
        min_length=1, description="Stable unique operation ID; reuse on retries."
    )
    arguments: dict[str, Any] = Field(default_factory=dict)


class ProductVersionsObservation(Observation):
    """Result of a product version operation."""


class ProductVersionsExecutor(
    ToolExecutor[ProductVersionsAction, ProductVersionsObservation]
):
    def __init__(
        self,
        callback: VersionCall,
        scope: str,
        context: RequestContext,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        self.callback, self.scope, self.context, self.loop = (
            callback,
            scope,
            context,
            loop,
        )

    def __call__(
        self,
        action: ProductVersionsAction,
        conversation: Any = None,  # noqa: ARG002
    ) -> ProductVersionsObservation:
        request = VersionToolRequest(action=action.action, arguments=action.arguments)
        try:
            result = asyncio.run_coroutine_threadsafe(
                self.callback(self.scope, request, action.command_id, self.context),
                self.loop,
            ).result()
            return ProductVersionsObservation.from_text(
                json.dumps(result, ensure_ascii=False)
            )
        except Exception as exc:
            return ProductVersionsObservation.from_text(
                str(exc) if isinstance(exc, ValueError) else type(exc).__name__,
                is_error=True,
            )


class ProductVersionsTool(
    ToolDefinition[ProductVersionsAction, ProductVersionsObservation]
):
    @classmethod
    def create(cls, conv_state: Any = None, **params: Any) -> list[Self]:  # noqa: ARG003
        executor = (
            ProductVersionsExecutor(
                params["callback"], params["scope"], params["context"], params["loop"]
            )
            if params
            else None
        )
        return [
            cls(
                description=VERSION_DESCRIPTION,
                action_type=ProductVersionsAction,
                observation_type=ProductVersionsObservation,
                executor=executor,
                annotations=ToolAnnotations(
                    readOnlyHint=False,
                    destructiveHint=False,
                    idempotentHint=True,
                    openWorldHint=True,
                ),
            )
        ]


register_tool(ProductVersionsTool.name, ProductVersionsTool)


type ExecutionCall = Callable[
    [str, str, dict[str, Any], str | None, RequestContext],
    Coroutine[Any, Any, dict[str, Any]],
]


def submission_bridge(
    callback: ExecutionCall,
    scope: str,
    request_id: Callable[[], str],
    context: RequestContext,
    loop: asyncio.AbstractEventLoop,
):
    from pyromind_sdk.client.models import TrainingTaskCreateResponse

    def submit(workflow: dict, name: str, code: str | None):
        effective = {**workflow, "name": name}
        result = asyncio.run_coroutine_threadsafe(
            callback(scope, request_id(), effective, code, context), loop
        ).result()
        return TrainingTaskCreateResponse.model_validate({"name": name, **result})

    return submit
