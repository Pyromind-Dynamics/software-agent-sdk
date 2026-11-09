"""Platform operations used by the harness-neutral product workflow submission port."""

from __future__ import annotations

import asyncio
import hashlib
import os
from pathlib import PurePosixPath

from pydantic import JsonValue
from pyromind_runtime.domain.content import JsonObject
from pyromind_runtime.domain.context import RequestContext

from openhands.agent_server.pyromind_auth import parse_auth_token_from_cookie_header
from openhands.tools.pyromind_dataset.definition import (
    _default_storage_base_url,
    download_file_from_pyromind,
)
from openhands.tools.utils.storage_paths import StoragePathError, to_tool_storage_path
from openhands.tools.workflow.dsl_to_xyflow import convert_dsl_to_xyflow
from openhands.tools.workflow.task_submission import (
    create_workflow_api_client,
    submit_workflow_task,
)
from openhands.tools.workflow.validate_workflow_dsl import (
    ValidateWorkflowDslAction,
    ValidateWorkflowDslExecutor,
)


_PERSISTENT_OUTPUT_PATH_FIELDS = {
    "CloneAndCacheDataset": ("target_path",),
    "CloneAndCacheModel": ("target_path",),
    "DataPreprocessLabelNode": ("output_path",),
    "DataPreprocessMergeNode": ("output_path",),
    "DataPreprocessPromptExpandNode": ("output_path",),
    "DataPreprocessReasoningNode": ("output_path",),
    "DataPreprocessSplitNode": ("output_path",),
    "DownloadAndCacheDataset": ("cache_dir",),
    "DownloadAndCacheModel": ("cache_dir",),
    "DownloadURLNode": ("output_path",),
    "ModelEvalApiNode": ("output_path",),
    "ModelMergeLoraNode": ("output_path",),
    "ModelTrainDPONode": ("output_path",),
    "ModelTrainGRPONode": ("output_path",),
    "ModelTrainOPDNode": ("output_path",),
    "ModelTrainSFTNode": ("output_path",),
    "PythonFileProcessorNode": ("output_file_path",),
}


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


def _validate_persistent_output_paths(nodes: object) -> None:
    if not isinstance(nodes, list):
        return
    for node in nodes:
        if not isinstance(node, dict):
            continue
        data = node.get("data")
        if not isinstance(data, dict):
            continue
        node_type = data.get("nodeType")
        if not isinstance(node_type, str):
            continue
        fields = _PERSISTENT_OUTPUT_PATH_FIELDS.get(node_type)
        if fields is None:
            continue
        config = data.get("config")
        if not isinstance(config, dict):
            continue
        for field in fields:
            output_path = config.get(field)
            if not isinstance(output_path, str) or not output_path.strip():
                continue
            normalized = PurePosixPath(output_path.strip())
            if (
                not normalized.is_absolute()
                or normalized.parts[:2] != ("/", "workspace")
                or len(normalized.parts) < 3
                or ".." in normalized.parts
            ):
                node_id = node.get("id", "unknown")
                raise ValueError(
                    f"Node {node_id} {field} must be under /workspace/ so its "
                    "output persists in user Storage."
                )


class PlatformWorkflowSubmission:
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
                ("x-cluster", context.x_cluster),
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
        _validate_persistent_output_paths(nodes)
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
        self, workflow: JsonObject, conversation_id: str, context: RequestContext
    ) -> str:
        return await asyncio.to_thread(self._submit, workflow, conversation_id, context)

    @staticmethod
    def _submit(
        workflow: JsonObject, conversation_id: str, context: RequestContext
    ) -> str:
        route = (context.x_cluster or "").strip()
        cluster, separator, routed_env = route.partition("#")
        env = (
            routed_env.strip().lower()
            if separator and routed_env.strip()
            else "prod"
            if route
            else os.getenv("PYROMIND_ENV") or os.getenv("APP_ENV")
        )
        client = create_workflow_api_client(
            env=env,
            cluster=cluster.strip() or os.getenv("PYROMIND_CLUSTER") or "default",
            auth_token=parse_auth_token_from_cookie_header(context.cookie),
            headers={"x-cluster": route} if route else {},
        )
        response = submit_workflow_task(
            client=client,
            workflow=dict(workflow),
            name=str(workflow.get("name") or "workflow"),
            conversation_id=conversation_id,
            retry_create=False,
        )
        return str(response.task_id)
