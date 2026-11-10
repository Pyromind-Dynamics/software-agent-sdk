"""Platform operations used by the harness-neutral product workflow submission port."""

from __future__ import annotations

import asyncio
import hashlib
from pathlib import PurePosixPath

from pydantic import JsonValue
from pyromind_runtime.domain.content import JsonObject
from pyromind_runtime.domain.context import RequestContext
from pyromind_runtime.domain.task_correlation import TaskCorrelation

from openhands.agent_server.pyromind_auth import parse_auth_token_from_cookie_header
from openhands.tools.pyromind_dataset.definition import (
    _default_storage_base_url,
    download_file_from_pyromind,
)
from openhands.tools.utils.platform_env import (
    platform_cluster_route,
    platform_cluster_target,
)
from openhands.tools.utils.storage_paths import (
    StoragePathError,
    to_tool_storage_path,
)
from openhands.tools.workflow.dsl_to_xyflow import convert_dsl_to_xyflow
from openhands.tools.workflow.task_submission import (
    create_workflow_api_client,
    submit_workflow_task,
)
from openhands.tools.workflow.validate_workflow_dsl import (
    ValidateWorkflowDslAction,
    ValidateWorkflowDslExecutor,
)


def _storage_path(workflow_path: str) -> str:
    """Restate the spelling the model used as the Storage path the platform reads."""
    path = PurePosixPath(workflow_path)
    if path.suffix != ".py" or ".." in path.parts:
        raise ValueError("workflow_path must be a workflow .py file inside Storage")
    try:
        storage_path = to_tool_storage_path(workflow_path, field="workflow_path")
    except StoragePathError as exc:
        raise ValueError(str(exc)) from exc
    parts = PurePosixPath(storage_path).parts[1:]
    if len(parts) < 2:
        raise ValueError(
            "workflow_path must be a workflow .py file inside Storage, "
            "such as /workspace/<dir>/workflow.py"
        )
    return storage_path


class PlatformWorkflowSubmission:
    async def query(self, task_id: str, context: RequestContext) -> JsonObject:
        def fetch():
            if not context.cookie and not context.authorization:
                raise PermissionError("platform credentials need renewal")
            route = platform_cluster_route(context.x_cluster)
            env, cluster = platform_cluster_target(route)
            client = create_workflow_api_client(
                env=env,
                cluster=cluster,
                auth_token=parse_auth_token_from_cookie_header(context.cookie),
                headers={
                    key: value
                    for key, value in {
                        "authorization": context.authorization,
                        "x-cluster": route,
                    }.items()
                    if value
                },
            )
            try:
                task = client.studio.get_task(task_id)
            except Exception as exc:
                response = getattr(exc, "response", None)
                if getattr(
                    exc, "status_code", getattr(response, "status_code", None)
                ) in {401, 403}:
                    raise PermissionError("platform credentials need renewal") from exc
                raise
            status = getattr(task, "status", "")
            result: JsonObject = {
                "task_id": task_id,
                "status": str(getattr(status, "value", status)),
            }
            return result

        return await asyncio.to_thread(fetch)

    async def prepare(self, workflow_path: str, context: RequestContext) -> JsonObject:
        return await asyncio.to_thread(self._prepare, workflow_path, context)

    @staticmethod
    def _prepare(workflow_path: str, context: RequestContext) -> JsonObject:
        path = PurePosixPath(workflow_path)
        storage_path = _storage_path(workflow_path)
        headers = {
            name: value
            for name, value in (
                ("cookie", context.cookie),
                ("authorization", context.authorization),
                ("x-cluster", platform_cluster_route(context.x_cluster)),
            )
            if value
        }
        source = download_file_from_pyromind(
            storage_path=storage_path,
            storage_base_url=_default_storage_base_url(),
            headers=headers,
            timeout=30,
            max_bytes=2 * 1024 * 1024,
        )
        dsl = source.decode("utf-8")
        name = path.stem
        validation = ValidateWorkflowDslExecutor(headers=headers)(
            ValidateWorkflowDslAction(dsl=dsl, name=name)
        )
        if validation.is_error or validation.valid is not True:
            raise ValueError(validation.text or "workflow DSL validation failed")
        workflow = convert_dsl_to_xyflow(dsl, name=name)
        nodes = workflow.get("nodes", [])
        node_types: list[JsonValue] = [
            str(node.get("data", {}).get("nodeType", ""))
            for node in nodes
            if isinstance(node, dict) and isinstance(node.get("data"), dict)
        ]
        summary: JsonObject = {
            "name": workflow.get("name") or name,
            "workflow_path": workflow_path,
            "node_count": len(node_types),
            "node_types": node_types,
            "has_training_node": any(
                isinstance(value, str) and "Train" in value for value in node_types
            ),
        }
        return {
            "workflow": workflow,
            "fingerprint": hashlib.sha256(source).hexdigest(),
            "summary": summary,
            "source": dsl,
        }

    async def submit(
        self,
        workflow: JsonObject,
        conversation_id: str,
        context: RequestContext,
        *,
        correlation: TaskCorrelation | None = None,
    ) -> str:
        return await asyncio.to_thread(
            self._submit, workflow, conversation_id, context, correlation=correlation
        )

    @staticmethod
    def _submit(
        workflow: JsonObject,
        conversation_id: str,
        context: RequestContext,
        *,
        correlation: TaskCorrelation | None = None,
    ) -> str:
        route = platform_cluster_route(context.x_cluster)
        env, cluster = platform_cluster_target(route)
        client = create_workflow_api_client(
            env=env,
            cluster=cluster,
            auth_token=parse_auth_token_from_cookie_header(context.cookie),
            headers={"x-cluster": route},
        )
        response = submit_workflow_task(
            client=client,
            workflow=dict(workflow),
            name=str(workflow.get("name") or "workflow"),
            conversation_id=conversation_id,
            retry_create=False,
            origin=correlation.origin if correlation else context.task_origin,
            task_kind=correlation.task_kind if correlation else "workflow_run",
        )
        return str(response.task_id)
