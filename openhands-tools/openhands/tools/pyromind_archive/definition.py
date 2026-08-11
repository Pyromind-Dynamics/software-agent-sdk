"""Submit archive extraction tasks to Pyromind Studio."""

from __future__ import annotations

import json
import shlex
import uuid
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal, Self, cast

import httpx
from pydantic import Field
from rich.text import Text

from openhands.sdk.tool import (
    Action,
    Observation,
    ToolAnnotations,
    ToolDefinition,
    ToolExecutor,
    register_tool,
)
from openhands.tools.pyromind_dataset.definition import (
    PYROMIND_AGENT_STORAGE_ROOT,
    _default_storage_base_url,
    _resolve_conversation_headers,
    _resolve_secret_headers,
)
from openhands.tools.workflow.task_submission import (
    PYROMIND_WORKFLOW_AUTH_TOKEN_SECRET,
    create_workflow_api_client,
    submit_workflow_task,
)


if TYPE_CHECKING:
    from openhands.sdk.conversation.base import BaseConversation
    from openhands.sdk.conversation.state import ConversationState


class ExtractArchiveAction(Action):
    """Submit an archive extraction task as a Studio workflow."""

    archive_path: str = Field(
        description="Storage path of the archive file to extract.",
    )
    format: Literal["auto", "zip", "tar", "tar.gz", "tgz"] = Field(
        default="auto",
        description=(
            "Archive format. 'auto' infers from the filename extension. "
            "Supported: zip, tar, tar.gz, tgz."
        ),
    )
    output_dir: str | None = Field(
        default=None,
        description=(
            "Optional target storage directory for extracted files. "
            "Defaults to a unique run directory under "
            "`/.pyromind-agent/<conversation_id>/extracted/<run_id>/`."
        ),
    )
    cpu: int = Field(default=1, ge=1, le=64)
    memory: int = Field(default=2, ge=1, le=256)

    @property
    def visualize(self) -> Text:
        content = Text()
        content.append("Extract archive: ", style="bold blue")
        content.append(self.archive_path)
        return content


class ExtractArchiveObservation(Observation):
    """Initial result of a submitted archive extraction task."""

    status: str = Field(description="Initial Studio task status.")
    task_id: str | None = Field(default=None)
    run_id: str | None = Field(default=None)
    output_dir: str | None = Field(default=None)

    @property
    def visualize(self) -> Text:
        content = Text()
        if self.is_error:
            content.append("Archive extraction submission failed", style="bold red")
            return content
        content.append("Archive extraction submitted", style="bold green")
        if self.task_id:
            content.append(f"\ntask_id={self.task_id}")
        if self.output_dir:
            content.append(f"\noutput_dir={self.output_dir}")
        return content


TOOL_DESCRIPTION = """Extract a compressed archive stored in Pyromind storage.

The tool submits a one-node CustomCommandNode workflow; do not construct or run
shell commands yourself. Supported formats: zip, tar, tar.gz, tgz. The format
is auto-detected from the filename extension unless specified explicitly.

The submission is asynchronous. A new run gets a unique result directory under
`/.pyromind-agent/<conversation_id>/extracted/<run_id>`. When the terminal
workflow callback resumes the conversation, use `preview_dataset` with the
`output_dir` returned by this tool to inspect the extracted files.

v1 extracts archives in a single pass. Nested archives inside the extracted
output require a separate call.
"""


def _detect_format(archive_path: str, format: str) -> str:  # noqa: A002
    if format != "auto":
        return format
    lower = archive_path.lower()
    if lower.endswith(".zip"):
        return "zip"
    if lower.endswith(".tar.gz") or lower.endswith(".tgz"):
        return "tar.gz"
    if lower.endswith(".tar"):
        return "tar"
    raise ValueError(
        f"Cannot detect archive format from path: {archive_path}. "
        "Specify the format parameter explicitly."
    )


def _build_extract_command(
    *,
    archive_path: str,
    output_dir: str,
    format: str,
) -> str:
    pod_archive = _pod_path(archive_path)
    pod_output_dir = _pod_path(output_dir)

    mkdir = f"mkdir -p {shlex.quote(pod_output_dir)}"

    if format == "zip":
        extract = (
            f"python3 -m zipfile -e {shlex.quote(pod_archive)} "
            f"{shlex.quote(pod_output_dir)}"
        )
    elif format in ("tar", "tar.gz", "tgz"):
        cmd = (
            "import tarfile,sys;tarfile.open(sys.argv[1],'r:*').extractall(sys.argv[2])"
        )
        extract = (
            f"python3 -c {shlex.quote(cmd)} "
            f"{shlex.quote(pod_archive)} {shlex.quote(pod_output_dir)}"
        )
    else:
        raise ValueError(f"Unsupported archive format: {format}")

    return f"{mkdir} && {extract}"


def _build_archive_workflow(
    action: ExtractArchiveAction,
    run_id: uuid.UUID,
    command: str,
) -> dict[str, Any]:
    return {
        "id": str(run_id),
        "name": f"agent-extract-{str(run_id)[:8]}",
        "nodes": [
            {
                "id": "1",
                "type": "default",
                "position": {"x": 0, "y": 0},
                "data": {
                    "display_name": "Custom Command",
                    "nodeType": "CustomCommandCPUNode",
                    "config": {
                        "command": command,
                        "cpu": action.cpu,
                        "memory": action.memory,
                    },
                },
            }
        ],
        "edges": [],
        "viewport": {"x": 0, "y": 0, "zoom": 1},
        "timestamp": datetime.now(UTC).isoformat(),
    }


def _normalize_headers(value: Any) -> dict[str, str] | None:
    if not value:
        return None
    if not isinstance(value, dict):
        raise ValueError("headers must be a dictionary when provided")
    return {str(name): str(header_value) for name, header_value in value.items()}


def _resolve_execution_target(
    env: str | None,
    cluster: str | None,
    headers: dict[str, str] | None,
) -> tuple[str | None, str | None]:
    routed_cluster = next(
        (
            value
            for name, value in (headers or {}).items()
            if name.lower() == "x-cluster"
        ),
        None,
    )
    if not routed_cluster:
        return env, cluster

    cluster_part, separator, env_part = routed_cluster.partition("#")
    resolved_cluster = cluster or cluster_part.strip() or None
    resolved_env = env or (env_part.strip().lower() if separator else "prod")
    return resolved_env, resolved_cluster


def _normalize_storage_path(value: str, field_name: str) -> str:
    raw = value.strip()
    if not raw:
        raise ValueError(f"{field_name} must be a non-empty storage path.")
    if any(ord(character) < 32 for character in raw):
        raise ValueError(f"{field_name} contains control characters.")
    parts = [part for part in raw.split("/") if part not in {"", "."}]
    if not parts or ".." in parts:
        raise ValueError(f"{field_name} must not be the root or contain '..'.")
    return "/" + "/".join(parts)


def _pod_path(storage_path: str) -> str:
    return f"/target-workspace{storage_path}"


def _strip_workspace_prefix(path: str) -> str:
    """Strip /workspace prefix if present (agent workspace path -> storage path)."""
    if path.startswith("/workspace/"):
        return path[len("/workspace") :]
    return path


class ExtractArchiveExecutor(
    ToolExecutor[ExtractArchiveAction, ExtractArchiveObservation]
):
    """Build and submit a CustomCommandNode archive extraction workflow."""

    def __init__(
        self,
        *,
        env: str | None = None,
        cluster: str | None = None,
        output_root: str | None = None,
        headers: dict[str, str] | None = None,
        storage_base_url: str | None = None,
        storage_headers: dict[str, str] | None = None,
        storage_secret_headers: dict[str, str] | None = None,
        timeout: int = 30,
    ) -> None:
        self._env = env
        self._cluster = cluster
        self._output_root = (
            _normalize_storage_path(output_root, "output_root")
            if output_root is not None
            else None
        )
        self._headers = dict(headers or {})
        self._storage_base_url = (
            storage_base_url or _default_storage_base_url()
        ).rstrip("/")
        self._storage_headers = dict(storage_headers or {})
        self._storage_secret_headers = dict(storage_secret_headers or {})
        self._timeout = timeout

    def __call__(
        self,
        action: ExtractArchiveAction,
        conversation: BaseConversation | None = None,
    ) -> ExtractArchiveObservation:
        try:
            if conversation is None:
                raise ValueError("extract_archive requires an active conversation.")

            archive_path = _normalize_storage_path(action.archive_path, "archive_path")
            archive_path = _strip_workspace_prefix(archive_path)
            format = _detect_format(archive_path, action.format)  # noqa: A001

            # Verify the archive file exists in storage before submitting
            error = self._check_storage_file_exists(archive_path, conversation)
            if error is not None:
                raise ValueError(error)

            output_root = (
                self._output_root
                or f"{PYROMIND_AGENT_STORAGE_ROOT}/{conversation.id}/extracted"
            )
            run_id = uuid.uuid4()
            if action.output_dir is not None:
                output_dir = _normalize_storage_path(action.output_dir, "output_dir")
            else:
                output_dir = f"{output_root}/{run_id}"

            command = _build_extract_command(
                archive_path=archive_path,
                output_dir=output_dir,
                format=format,
            )
            workflow = _build_archive_workflow(action, run_id, command)
        except ValueError as exc:
            return ExtractArchiveObservation.from_text(
                text=str(exc),
                status="Failed",
                is_error=True,
            )

        try:
            state = cast("ConversationState", conversation.state)
            auth_token = state.secret_registry.get_secret_value(
                PYROMIND_WORKFLOW_AUTH_TOKEN_SECRET
            )
            client = create_workflow_api_client(
                env=self._env,
                cluster=self._cluster,
                auth_token=auth_token,
                headers=self._headers,
                timeout=self._timeout,
            )
            response = submit_workflow_task(
                client=client,
                workflow=workflow,
                name=str(workflow["name"]),
                conversation_id=str(conversation.id),
            )
            task_id = response.task_id
        except Exception as exc:
            return ExtractArchiveObservation.from_text(
                text=f"Failed to submit extraction workflow: {exc}",
                status="Failed",
                run_id=str(run_id),
                output_dir=output_dir,
                is_error=True,
            )

        return ExtractArchiveObservation.from_text(
            text=(
                "Archive extraction workflow submitted. "
                f"task_id={task_id}, run_id={run_id}, output_dir={output_dir}. "
                "After the terminal callback, preview "
                f"{output_dir}/ to inspect the extracted files."
            ),
            status=response.status,
            task_id=task_id,
            run_id=str(run_id),
            output_dir=output_dir,
        )

    def _resolved_storage_headers(
        self,
        conversation: BaseConversation,
    ) -> dict[str, str]:
        headers = {"accept": "*/*", **self._storage_headers}
        headers.update(_resolve_conversation_headers(conversation))
        headers.update(
            _resolve_secret_headers(conversation, self._storage_secret_headers)
        )
        return headers

    def _check_storage_file_exists(
        self,
        path: str,
        conversation: BaseConversation,
    ) -> str | None:
        """Check if a storage path exists. Returns error message or None."""
        headers = self._resolved_storage_headers(conversation)
        try:
            response = httpx.post(
                f"{self._storage_base_url}/get_url",
                headers=headers,
                json={"path": path},
                timeout=self._timeout,
            )
            if response.status_code >= 400:
                return (
                    f"Archive path not found in storage: {path}. "
                    "Check that the path is correct and the file exists."
                )
            payload = response.json()
            if not isinstance(payload, dict) or payload.get("success") is not True:
                return (
                    f"Archive path not found in storage: {path}. "
                    "Check that the path is correct and the file exists."
                )
            return None
        except (httpx.RequestError, json.JSONDecodeError) as exc:
            return f"Failed to verify archive path in storage: {exc}"


class ExtractArchiveTool(
    ToolDefinition[ExtractArchiveAction, ExtractArchiveObservation]
):
    """Tool definition for asynchronous archive extraction submissions."""

    @classmethod
    def create(
        cls,
        conv_state: ConversationState | None = None,  # noqa: ARG003
        **params: Any,
    ) -> Sequence[Self]:
        env_value = params.pop("env", None)
        env = str(env_value) if env_value is not None else None
        cluster_value = params.pop("cluster", None)
        cluster = str(cluster_value) if cluster_value is not None else None
        params.pop("current_user", None)
        params.pop("runtime_dir", None)
        output_root_value = params.pop("output_root", None)
        output_root = str(output_root_value) if output_root_value is not None else None
        headers = _normalize_headers(params.pop("headers", None))
        storage_base_url_value = params.pop("storage_base_url", None)
        storage_base_url = (
            str(storage_base_url_value) if storage_base_url_value is not None else None
        )
        storage_headers = _normalize_headers(params.pop("storage_headers", None))
        legacy_secret_headers = params.pop("secret_headers", None)
        storage_secret_headers = _normalize_headers(
            params.pop("storage_secret_headers", legacy_secret_headers)
        )
        env, cluster = _resolve_execution_target(env, cluster, headers)
        params.pop("endpoint_url", None)
        timeout = int(params.pop("timeout", 30))
        if params:
            names = ", ".join(sorted(params))
            raise ValueError(f"ExtractArchiveTool got unknown params: {names}")
        if timeout <= 0:
            raise ValueError("timeout must be greater than 0")
        if output_root is not None:
            _normalize_storage_path(output_root, "output_root")
        return [
            cls(
                description=TOOL_DESCRIPTION,
                action_type=ExtractArchiveAction,
                observation_type=ExtractArchiveObservation,
                executor=ExtractArchiveExecutor(
                    env=env,
                    cluster=cluster,
                    output_root=output_root,
                    headers=headers,
                    storage_base_url=storage_base_url,
                    storage_headers=storage_headers,
                    storage_secret_headers=storage_secret_headers,
                    timeout=timeout,
                ),
                annotations=ToolAnnotations(
                    title="extract_archive",
                    readOnlyHint=False,
                    destructiveHint=False,
                    idempotentHint=False,
                    openWorldHint=True,
                ),
            )
        ]


register_tool(ExtractArchiveTool.name, ExtractArchiveTool)
