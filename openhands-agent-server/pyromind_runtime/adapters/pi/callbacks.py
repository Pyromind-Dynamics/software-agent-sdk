from __future__ import annotations

import asyncio
import base64
import os
import posixpath
import tempfile
from pathlib import Path

from pydantic import JsonValue

from pyromind_runtime.adapters.pi.sandbox import BoundedSandboxGateway
from pyromind_runtime.contracts.content import JsonObject
from pyromind_runtime.contracts.harness import SessionSpec
from pyromind_runtime.tool_host import PythonToolHost, ToolExecutionContext


_WORKFLOW_RELATIVE_PATH = "public_data/workflow_canvas/workflow.py"
_MAX_WORKFLOW_BYTES = 2 * 1024 * 1024


class PiWorkflowMirror:
    """Mirror only the canonical workflow artifact from the remote workspace."""

    def __init__(
        self, *, remote_workspace_root: str, local_workspace_root: str
    ) -> None:
        self._remote_workspace_root = posixpath.normpath(remote_workspace_root)
        self._remote_path = posixpath.join(
            self._remote_workspace_root,
            _WORKFLOW_RELATIVE_PATH,
        )
        self._local_workspace_root = Path(local_workspace_root)

    def targets_workflow(self, path: str) -> bool:
        candidate = (
            path
            if path.startswith("/")
            else posixpath.join(self._remote_workspace_root, path)
        )
        return posixpath.normpath(candidate) == self._remote_path

    async def write(self, content: bytes) -> None:
        if len(content) > _MAX_WORKFLOW_BYTES:
            raise ValueError("Workflow resource exceeds the product projection limit")
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("Workflow resource must be UTF-8") from exc
        await asyncio.to_thread(self._write_sync, text)

    async def remove(self) -> None:
        await asyncio.to_thread(self._remove_sync)

    def _write_sync(self, content: str) -> None:
        root = self._local_workspace_root.absolute()
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
        resolved_root = root.resolve()
        if resolved_root != root:
            raise ValueError("Local workflow mirror root must not contain symlinks")
        target = resolved_root / _WORKFLOW_RELATIVE_PATH
        target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        if target.parent.resolve() != target.parent:
            raise ValueError("Local workflow mirror parent escapes workspace")
        if target.is_symlink():
            raise ValueError("Local workflow mirror target must not be a symlink")
        state_path = target.parent / "state.json"
        if state_path.is_symlink():
            raise ValueError("Local workflow state target must not be a symlink")
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".workflow-",
            suffix=".tmp",
            dir=target.parent,
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            os.chmod(temporary_path, 0o600)
            os.replace(temporary_path, target)
            state_path.unlink(missing_ok=True)
        finally:
            temporary_path.unlink(missing_ok=True)

    def _remove_sync(self) -> None:
        configured_root = self._local_workspace_root.absolute()
        root = configured_root.resolve()
        if root != configured_root:
            raise ValueError("Local workflow mirror root must not contain symlinks")
        target = root / _WORKFLOW_RELATIVE_PATH
        if target.is_symlink():
            raise ValueError("Local workflow mirror target must not be a symlink")
        target.unlink(missing_ok=True)


class PiRunnerCallbackRouter:
    def __init__(
        self,
        spec: SessionSpec,
        sandbox: BoundedSandboxGateway | None,
        tool_host: PythonToolHost,
        workflow_mirror: PiWorkflowMirror | None = None,
    ) -> None:
        self._spec = spec
        self._sandbox = sandbox
        self._tool_host = tool_host
        self._workflow_mirror = workflow_mirror
        self._allowed_tools = frozenset(tool.name for tool in spec.tools)

    async def handle(self, method: str, params: JsonObject) -> JsonValue:
        if method.startswith("env."):
            if self._sandbox is None:
                raise ValueError(
                    "Environment callbacks are unavailable for a local Pi session"
                )
            result = await self._sandbox.handle(method, params)
            await self._capture_workflow(method, params, result)
            return result
        if method != "tool.execute":
            raise ValueError(f"Unsupported Pi callback method: {method}")
        tool_name = params.get("tool_name")
        arguments = params.get("arguments")
        if not isinstance(tool_name, str) or tool_name not in self._allowed_tools:
            raise ValueError("Tool is not enabled for this Pi session")
        if not isinstance(arguments, dict):
            raise ValueError("Tool arguments must be an object")
        result = await self._tool_host.execute(
            tool_name,
            arguments,
            ToolExecutionContext(
                session_id=self._spec.product_session_id,
                user_id=self._spec.user_id,
                workspace=self._spec.workspace,
                sandbox=self._spec.sandbox,
            ),
        )
        return result.model_dump(mode="json")

    async def _capture_workflow(
        self,
        method: str,
        params: JsonObject,
        result: JsonValue,
    ) -> None:
        mirror = self._workflow_mirror
        sandbox = self._sandbox
        if mirror is None or sandbox is None or not _successful_result(result):
            return
        if method == "env.writeFile":
            path = params.get("path")
            if isinstance(path, str) and mirror.targets_workflow(path):
                await mirror.write(_content_bytes(params))
            return
        if method not in {"env.appendFile", "env.exec"}:
            return
        if method == "env.appendFile":
            path = params.get("path")
            relevant = isinstance(path, str) and mirror.targets_workflow(path)
        else:
            command = params.get("command")
            relevant = isinstance(command, str) and _WORKFLOW_RELATIVE_PATH in command
        if not relevant:
            return
        read_result = await sandbox.handle(
            "env.readTextFile",
            {"path": _WORKFLOW_RELATIVE_PATH},
        )
        content = _successful_value(read_result)
        if isinstance(content, str):
            await mirror.write(content.encode())
        elif _error_code(read_result) == "not_found":
            await mirror.remove()

    async def close(self) -> None:
        if self._sandbox is not None:
            await self._sandbox.close()


def _content_bytes(params: JsonObject) -> bytes:
    content = params.get("content")
    encoding = params.get("encoding")
    if not isinstance(content, str):
        raise ValueError("Workflow mirror content is invalid")
    if encoding == "base64":
        return base64.b64decode(content, validate=True)
    if encoding in {None, "utf8"}:
        return content.encode()
    raise ValueError("Workflow mirror encoding is invalid")


def _successful_result(value: JsonValue) -> bool:
    return isinstance(value, dict) and value.get("ok") is True


def _successful_value(value: JsonValue) -> JsonValue:
    if not _successful_result(value) or not isinstance(value, dict):
        return None
    return value.get("value")


def _error_code(value: JsonValue) -> str | None:
    if not isinstance(value, dict):
        return None
    error = value.get("error")
    if not isinstance(error, dict):
        return None
    code = error.get("code")
    return code if isinstance(code, str) else None
