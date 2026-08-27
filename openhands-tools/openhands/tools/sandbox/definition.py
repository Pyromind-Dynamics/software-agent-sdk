"""Definitions of the Pyromind Sandbox service tools.

The Pyromind Sandbox service provisions disposable headless containers
(CUSTOM) for experiments, debugging and reproducible demos. These tools let an
agent create and delete sandboxes and read files (``SandboxClient.read_file``)
via pyromind-sdk >= 0.1.9. Commands inside a sandbox are executed through the
agent session's terminal (``pyromind terminal`` TTY bridge), never through the
exec API.

Auth/env wiring mirrors ``workflow_debug``: every executor builds an
authenticated ``PyroMindAPIClient`` from conversation secrets plus the
env/cluster/header context injected by the agent-server Pyromind router.
"""

from __future__ import annotations

import base64
import json
import re
import ssl
import time
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Self, cast

import certifi
from pydantic import BaseModel, Field
from pyromind_sdk import PyroMindAPIClient
from pyromind_sdk.client.base import resolve_base_url_from_cluster
from pyromind_sdk.client.models import (
    PortMapping,
    ResourceConfig,
    SandboxRequest,
    SandboxResponse,
    SandboxType,
    VolumeMount,
)
from websockets.sync.client import connect

from openhands.sdk.tool import (
    Action,
    Observation,
    ToolAnnotations,
    ToolDefinition,
    ToolExecutor,
    register_tool,
)
from openhands.tools.workflow.task_submission import (
    PYROMIND_WORKFLOW_AUTH_TOKEN_SECRET,
    create_workflow_api_client,
)


if TYPE_CHECKING:
    from pyromind_sdk.client.sandbox import SandboxClient

    from openhands.sdk.conversation.base import BaseConversation
    from openhands.sdk.conversation.state import ConversationState


DEFAULT_CPU = 4
DEFAULT_WAIT_TIMEOUT = 600
MAX_FILE_TEXT_CHARS = 20000


def _default_image(cluster: str | None) -> str:
    """Cluster default Jupyter-lab image when no image is pinned."""
    if cluster and "us-west-2" in cluster:
        return "pyrominddynamics/jupyter-lab-with-ssh:v0.9-aws"
    return "pyrominddynamics/jupyter-lab-with-ssh:v0.9"


def create_sandbox_api_client(
    *,
    env: str | None,
    cluster: str | None,
    auth_token: str | None,
    headers: dict[str, str],
) -> PyroMindAPIClient:
    """Create an authenticated client for the Pyromind Sandbox API.

    Reuses the workflow submission auth flow (auth token -> access key) so a
    sandbox tool only needs the same ``auth_token`` conversation secret.
    """
    return create_workflow_api_client(
        env=env,
        cluster=cluster,
        auth_token=auth_token,
        headers=headers,
    )


def _sandbox_type_value(sandbox_type: SandboxType | str) -> str:
    return sandbox_type.value if isinstance(sandbox_type, SandboxType) else sandbox_type


def _sandbox_lines(sandbox: SandboxResponse) -> list[str]:
    lines = [
        f"sandbox_id: {sandbox.id}",
        f"name: {sandbox.name}",
        f"type: {_sandbox_type_value(sandbox.type)}",
        f"status: {sandbox.status}",
    ]
    if sandbox.endpoint_url:
        lines.append(f"endpoint_url: {sandbox.endpoint_url}")
    if sandbox.endpoint:
        lines.append(f"endpoint: {sandbox.endpoint}")
    if sandbox.web_vnc_url:
        lines.append(f"web_vnc_url: {sandbox.web_vnc_url}")
    return lines


class SandboxObservation(Observation):
    """Snapshot of one sandbox in a lifecycle observation."""

    sandbox_id: str | None = Field(default=None, description="Sandbox identifier.")
    status: str | None = Field(
        default=None,
        description=(
            "Sandbox lifecycle status: 'creating', 'pending', 'starting', "
            "'running', 'stopped', or 'error'."
        ),
    )
    name: str | None = Field(default=None, description="Sandbox display name.")
    type: str | None = Field(default=None, description="Sandbox type, e.g. 'osworld'.")
    endpoint_url: str | None = Field(
        default=None, description="Optional sandbox endpoint URL."
    )
    web_vnc_url: str | None = Field(
        default=None, description="Web VNC viewer URL for the sandbox desktop."
    )
    count: int | None = Field(
        default=None, description="Number of sandboxes in a list result."
    )
    cpu_percent: float | None = Field(
        default=None, description="Current CPU usage in percent (when reported)."
    )
    memory_used: str | None = Field(
        default=None, description="Current memory usage (when reported)."
    )


def _observation_from_sandbox[O: SandboxObservation](
    observation_type: type[O],
    sandbox: SandboxResponse,
    *,
    text: str,
    is_error: bool = False,
) -> O:
    return observation_type.from_text(
        text=text,
        is_error=is_error,
        sandbox_id=sandbox.id,
        status=sandbox.status,
        name=sandbox.name,
        type=_sandbox_type_value(sandbox.type),
        endpoint_url=sandbox.endpoint_url or sandbox.endpoint,
        web_vnc_url=sandbox.web_vnc_url,
        cpu_percent=sandbox.usage.cpu_percent if sandbox.usage else None,
        memory_used=sandbox.usage.memory_used if sandbox.usage else None,
    )


class _SandboxExecutorMixin:
    """Shared plumbing: build the authenticated sandbox client per tool call."""

    cluster: str | None
    env: str | None
    headers: dict[str, str]

    def __init__(
        self,
        *,
        cluster: str | None = None,
        env: str | None = None,
        current_user: object | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.cluster = cluster
        self.env = env
        self.current_user = current_user
        self.headers = headers or {}

    def _sandbox_client(self, conversation: BaseConversation | None) -> SandboxClient:
        if conversation is None:
            raise ValueError("Sandbox tools require an active conversation.")
        state = cast("ConversationState", conversation.state)
        auth_token = state.secret_registry.get_secret_value(
            PYROMIND_WORKFLOW_AUTH_TOKEN_SECRET
        )
        client = create_sandbox_api_client(
            env=self.env,
            cluster=self.cluster,
            auth_token=auth_token,
            headers=self.headers,
        )
        return client.sandboxes


# ---------------------------------------------------------------------------
# sandbox_create / sandbox_delete / sandbox_exec / sandbox_read_file
# ---------------------------------------------------------------------------


class SandboxMountInput(BaseModel):
    """Host-to-container volume mount for custom sandboxes."""

    host_path: str = Field(description="Path on the node, e.g. '/workspace'.")
    mount_path: str = Field(
        description="Mount point inside the container, e.g. '/data'."
    )
    read_only: bool = Field(
        default=False, description="Mount as read-only inside the container."
    )


class SandboxPortInput(BaseModel):
    """Container port exposed via a node port (custom sandboxes)."""

    container_port: int = Field(
        description="Port the container listens on (1-65535).",
        ge=1,
        le=65535,
    )
    host_port: int | None = Field(
        default=None,
        description="Optional port exposed on the node (defaults to container_port).",
        ge=1,
        le=65535,
    )
    protocol: str = Field(default="TCP", description="Port protocol: 'TCP' or 'UDP'.")


class SandboxCreateAction(Action):
    """Create a disposable Pyromind custom (headless) sandbox."""

    name: str | None = Field(
        default=None, description="Optional display name for the sandbox."
    )
    image: str | None = Field(
        default=None,
        description=(
            "Container image to run. Defaults to the cluster Jupyter-lab "
            "image (us-west-2: pyrominddynamics/jupyter-lab-with-ssh:v0.9-aws; "
            "other clusters: pyrominddynamics/jupyter-lab-with-ssh:v0.9)."
        ),
    )
    volume_mounts: list[SandboxMountInput] | None = Field(
        default=None,
        description=(
            "Host path mounts for custom sandboxes; use the mount_path as "
            "the target in sandbox_read_file calls."
        ),
    )
    port_mappings: list[SandboxPortInput] | None = Field(
        default=None,
        description="Ports to expose for custom sandboxes (docker -p style).",
    )
    cpu: int = Field(
        default=DEFAULT_CPU,
        description="vCPU count for the sandbox (platform minimum is 4).",
        ge=1,
    )
    memory: str | None = Field(
        default=None,
        description=(
            "Memory requested for the sandbox, e.g. '16Gi'. Defaults to "
            "cpu x 2 Gi (the platform 1:2 ratio) when omitted."
        ),
    )
    wait_timeout: int = Field(
        default=DEFAULT_WAIT_TIMEOUT,
        description=(
            "Seconds to wait for the sandbox to become 'running' after "
            "submission (0 returns right after creation)."
        ),
        ge=0,
    )


class SandboxCreateObservation(SandboxObservation):
    """Result of creating a sandbox."""


class SandboxCreateExecutor(
    _SandboxExecutorMixin,
    ToolExecutor[SandboxCreateAction, SandboxCreateObservation],
):
    """Create a custom sandbox and optionally wait until it is running."""

    def _build_request(self, action: SandboxCreateAction) -> SandboxRequest:
        resources = ResourceConfig(
            cpu=str(action.cpu),
            memory=action.memory or f"{action.cpu * 2}Gi",
        )
        return SandboxRequest(
            name=action.name,
            sandbox_type=SandboxType.CUSTOM,
            resources=resources,
            image=action.image or _default_image(self.cluster),
            volume_mounts=[
                VolumeMount(
                    host_path=mount.host_path,
                    mount_path=mount.mount_path,
                    read_only=mount.read_only,
                )
                for mount in action.volume_mounts or []
            ]
            or None,
            port_mappings=[
                PortMapping(
                    container_port=port.container_port,
                    host_port=port.host_port,
                    protocol=port.protocol,
                )
                for port in action.port_mappings or []
            ]
            or None,
        )

    def __call__(
        self,
        action: SandboxCreateAction,
        conversation: BaseConversation | None = None,
    ) -> SandboxCreateObservation:
        try:
            sandboxes = self._sandbox_client(conversation)
            request = self._build_request(action)
            created = sandboxes.create(request)
            if action.wait_timeout <= 0:
                return _observation_from_sandbox(
                    SandboxCreateObservation,
                    created,
                    text="Created sandbox: " + "\n".join(_sandbox_lines(created)),
                )
            reached = sandboxes.wait_for_sandbox_status(
                created.id,
                target_status="running",
                timeout=action.wait_timeout,
            )
            latest = sandboxes.get_sandbox(created.id)
            if not reached:
                return _observation_from_sandbox(
                    SandboxCreateObservation,
                    latest,
                    text=(
                        "Sandbox was created but did not reach 'running' "
                        f"within {action.wait_timeout}s (status: "
                        f"{latest.status}). Poll the sandbox status to "
                        "continue."
                    ),
                )
            return _observation_from_sandbox(
                SandboxCreateObservation,
                latest,
                text="Created sandbox: " + "\n".join(_sandbox_lines(latest)),
            )
        except Exception as exc:  # noqa: BLE001
            return SandboxCreateObservation.from_text(
                text=f"Failed to create sandbox: {exc}",
                is_error=True,
            )


class SandboxDeleteAction(Action):
    """Permanently destroy a sandbox and release its resources."""

    sandbox_id: str = Field(description="The sandbox id to delete.")


class SandboxDeleteObservation(SandboxObservation):
    """Acknowledgment that the sandbox was deleted."""


class SandboxDeleteExecutor(
    _SandboxExecutorMixin,
    ToolExecutor[SandboxDeleteAction, SandboxDeleteObservation],
):
    """Delete a sandbox."""

    def __call__(
        self,
        action: SandboxDeleteAction,
        conversation: BaseConversation | None = None,
    ) -> SandboxDeleteObservation:
        client = self._sandbox_client(conversation)
        try:
            try:
                client.delete(action.sandbox_id)
            except Exception as exc:  # noqa: BLE001
                if "can not delete" not in str(exc).lower():
                    raise
                # Platform rejects DELETE while running: pause first, then retry.
                client.pause(action.sandbox_id)
                client.delete(action.sandbox_id)
            return SandboxDeleteObservation.from_text(
                text=f"Sandbox {action.sandbox_id} deleted.",
                sandbox_id=action.sandbox_id,
            )
        except Exception as exc:  # noqa: BLE001
            return SandboxDeleteObservation.from_text(
                text=f"Failed to delete sandbox {action.sandbox_id}: {exc}",
                is_error=True,
            )


# ---------------------------------------------------------------------------
# sandbox_read_file
# ---------------------------------------------------------------------------


def _decode_file_text(raw: bytes) -> tuple[str, str]:
    try:
        return raw.decode("utf-8"), "utf-8"
    except UnicodeDecodeError:
        return base64.b64encode(raw).decode("ascii"), "base64"


class SandboxReadFileAction(Action):
    """Read a file (custom sandbox) as text, or base64 for binary files."""

    sandbox_id: str = Field(description="The id of the custom sandbox.")
    path: str = Field(
        description="Absolute path of the file inside the container (use the "
        "mount_path of a volume_mount for mounted directories)."
    )


class SandboxReadFileObservation(Observation):
    """Content of a file read from a custom sandbox."""

    sandbox_id: str | None = Field(
        default=None, description="The sandbox the file was read from."
    )
    path: str | None = Field(
        default=None, description="Path of the file inside the container."
    )
    file_content: str | None = Field(
        default=None, description="File content (utf-8 text, or base64 for binary)."
    )
    encoding: str | None = Field(
        default=None, description="How content is encoded: 'utf-8' or 'base64'."
    )
    truncated: bool | None = Field(
        default=None,
        description="Whether the content was truncated because it is very large.",
    )


class SandboxReadFileExecutor(
    _SandboxExecutorMixin,
    ToolExecutor[SandboxReadFileAction, SandboxReadFileObservation],
):
    """Read one file from a custom sandbox."""

    def __call__(
        self,
        action: SandboxReadFileAction,
        conversation: BaseConversation | None = None,
    ) -> SandboxReadFileObservation:
        try:
            raw = self._sandbox_client(conversation).read_file(
                action.sandbox_id, action.path
            )
            content, encoding = _decode_file_text(raw)
            truncated = len(content) > MAX_FILE_TEXT_CHARS
            if truncated:
                content = (
                    content[:MAX_FILE_TEXT_CHARS]
                    + f"\n...[truncated, full size {len(raw)} bytes]"
                )
            return SandboxReadFileObservation.from_text(
                text=content,
                sandbox_id=action.sandbox_id,
                path=action.path,
                file_content=content,
                encoding=encoding,
                truncated=truncated,
            )
        except Exception as exc:  # noqa: BLE001
            return SandboxReadFileObservation.from_text(
                text=(
                    f"Failed to read file {action.path} from sandbox "
                    f"{action.sandbox_id}: {exc}"
                ),
                is_error=True,
            )


# ---------------------------------------------------------------------------
# sandbox_write_file
# ---------------------------------------------------------------------------


class SandboxWriteFileAction(Action):
    """Write a file into a custom sandbox."""

    sandbox_id: str = Field(description="The id of the custom sandbox.")
    path: str = Field(
        description="Absolute path of the file inside the container (parent "
        "directories are created automatically)."
    )
    content: str = Field(description="File content to write (UTF-8 text).")


class SandboxWriteFileObservation(Observation):
    """Confirmation that a file was written."""

    sandbox_id: str | None = Field(
        default=None, description="The sandbox the file was written to."
    )
    path: str | None = Field(
        default=None, description="Path of the file inside the container."
    )
    size: int | None = Field(default=None, description="Number of bytes written.")


class SandboxWriteFileExecutor(
    _SandboxExecutorMixin,
    ToolExecutor[SandboxWriteFileAction, SandboxWriteFileObservation],
):
    """Write one file into a custom sandbox."""

    def __call__(
        self,
        action: SandboxWriteFileAction,
        conversation: BaseConversation | None = None,
    ) -> SandboxWriteFileObservation:
        try:
            raw = action.content.encode("utf-8")
            self._sandbox_client(conversation).write_file(
                action.sandbox_id, action.path, raw
            )
            return SandboxWriteFileObservation.from_text(
                text=f"Wrote {len(raw)} bytes to {action.path}.",
                sandbox_id=action.sandbox_id,
                path=action.path,
                size=len(raw),
            )
        except Exception as exc:  # noqa: BLE001
            return SandboxWriteFileObservation.from_text(
                text=(
                    f"Failed to write file {action.path} to sandbox "
                    f"{action.sandbox_id}: {exc}"
                ),
                is_error=True,
            )


# ---------------------------------------------------------------------------
# sandbox_delete_file
# ---------------------------------------------------------------------------


class SandboxDeleteFileAction(Action):
    """Delete a file or directory from a custom sandbox."""

    sandbox_id: str = Field(description="The id of the custom sandbox.")
    path: str = Field(
        description="Absolute path of the file or directory inside the container."
    )
    recursive: bool = Field(
        default=False,
        description="Recursively delete a directory; required when path is a "
        "non-empty directory.",
    )


class SandboxDeleteFileObservation(Observation):
    """Confirmation that a file was deleted."""

    sandbox_id: str | None = Field(
        default=None, description="The sandbox the file was deleted from."
    )
    path: str | None = Field(
        default=None, description="Path of the deleted file or directory."
    )


class SandboxDeleteFileExecutor(
    _SandboxExecutorMixin,
    ToolExecutor[SandboxDeleteFileAction, SandboxDeleteFileObservation],
):
    """Delete one file or directory from a custom sandbox."""

    def __call__(
        self,
        action: SandboxDeleteFileAction,
        conversation: BaseConversation | None = None,
    ) -> SandboxDeleteFileObservation:
        try:
            self._sandbox_client(conversation).delete_file(
                action.sandbox_id, action.path, recursive=action.recursive
            )
            return SandboxDeleteFileObservation.from_text(
                text=f"Deleted {action.path}.",
                sandbox_id=action.sandbox_id,
                path=action.path,
            )
        except Exception as exc:  # noqa: BLE001
            return SandboxDeleteFileObservation.from_text(
                text=(
                    f"Failed to delete {action.path} from sandbox "
                    f"{action.sandbox_id}: {exc}"
                ),
                is_error=True,
            )


# ---------------------------------------------------------------------------


_EXIT_MARKER = "__PYROMIND_TERMINAL_EXIT__"
_TERMINAL_COLS = 160
_TERMINAL_ROWS = 48
_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]|\x1b\][^\x07]*(?:\x07|\x1b\\)")
_PROD_ENVS = frozenset({"prod", "production", "online"})


def _terminal_websocket_url(
    *,
    base_url: str,
    sandbox_id: str,
    api_key: str,
) -> str:
    base = base_url.rstrip("/")
    for http_scheme, ws_scheme in (("https://", "wss://"), ("http://", "ws://")):
        if base.startswith(http_scheme):
            base = ws_scheme + base[len(http_scheme) :]
            break
    query = f"cols={_TERMINAL_COLS}&rows={_TERMINAL_ROWS}"
    if api_key:
        query = f"token={api_key}&" + query
    return f"{base}/sandboxes/{sandbox_id}/terminal?{query}"


def _strip_ansi(text: str) -> str:
    return _ANSI_ESCAPE_RE.sub("", text)


def _terminal_exit_code(text: str) -> int | None:
    matches = re.findall(rf"{re.escape(_EXIT_MARKER)}:\s*(\d+)", text)
    if not matches:
        return None
    return int(matches[-1])


def _run_terminal_command(
    *,
    base_url: str,
    api_key: str,
    sandbox_id: str,
    command: str,
    timeout_seconds: int,
) -> tuple[str, int | None, bool]:
    """Run one command through the sandbox terminal WebSocket (TTY bridge).

    The command runs in a fresh interactive shell, so cwd is folded into the
    command line. The exit code is captured with an echo marker because the
    terminal protocol is a raw byte stream.
    """
    url = _terminal_websocket_url(
        base_url=base_url,
        sandbox_id=sandbox_id,
        api_key=api_key,
    )
    # The REST client (requests) verifies TLS with the certifi bundle; the
    # websockets library defaults to the system CA store, which is empty on
    # some hosts (e.g. Homebrew OpenSSL without a populated CA file) and
    # fails with CERTIFICATE_VERIFY_FAILED. Align with the REST path.
    ssl_context = ssl.create_default_context(cafile=certifi.where())
    buffer = bytearray()
    timed_out = False
    with connect(url, ssl=ssl_context, open_timeout=30, close_timeout=5) as terminal:
        terminal.send((command + "\r\n").encode("utf-8", "replace"))
        terminal.send(f"echo {_EXIT_MARKER}:$?\r\n".encode("ascii"))
        deadline = time.monotonic() + timeout_seconds
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or _EXIT_MARKER.encode("ascii") in buffer:
                break
            try:
                message = terminal.recv(timeout=remaining)
            except TimeoutError:
                timed_out = True
                break
            if isinstance(message, str):
                try:
                    control = json.loads(message)
                    if isinstance(control, dict) and control.get("type") == "pong":
                        continue
                except (json.JSONDecodeError, TypeError):
                    pass
                buffer.extend(message.encode("utf-8", "replace"))
            else:
                buffer.extend(message)
        try:
            terminal.send(b"exit\r\n")
        except Exception:  # noqa: BLE001
            pass
    text = _strip_ansi(buffer.decode("utf-8", "replace"))
    return text, _terminal_exit_code(text), timed_out


class SandboxTerminalAction(Action):
    """Run one command inside a sandbox through its terminal TTY channel."""

    sandbox_id: str = Field(description="The sandbox id to run the command in.")
    command: str = Field(
        description="Shell command to run inside the sandbox (via the sandbox "
        "terminal WebSocket; runs in a fresh shell)."
    )
    cwd: str | None = Field(
        default=None,
        description="Optional working directory; prefixed as 'cd <cwd> && ...'.",
    )
    timeout_seconds: int | None = Field(
        default=None,
        description="Command timeout in seconds (1-600). Defaults to 60.",
        ge=1,
        le=600,
    )


class SandboxTerminalObservation(Observation):
    """Result of a sandbox terminal command."""

    sandbox_id: str | None = Field(
        default=None, description="The sandbox the command ran in."
    )
    returncode: int | None = Field(
        default=None, description="Command exit code (0 = success)."
    )
    output: str | None = Field(
        default=None, description="Combined terminal output of the command."
    )
    timed_out: bool | None = Field(
        default=None, description="Whether the command hit the timeout."
    )


class SandboxTerminalExecutor(
    _SandboxExecutorMixin,
    ToolExecutor[SandboxTerminalAction, SandboxTerminalObservation],
):
    """Execute a command in a sandbox over the terminal TTY WebSocket."""

    def _terminal_base_url(self, client: SandboxClient) -> str:
        """Resolve the per-cluster base URL used by the terminal WebSocket.

        The portal domain (client.base_url) does not proxy WebSocket, so the
        terminal endpoint must be reached through the per-cluster direct
        domain, matching the ``pyromind terminal`` CLI (cluster[#env]).
        """
        cluster = (self.cluster or client.cluster or "").strip()
        env = (self.env or "").strip()
        if cluster and "#" not in cluster and env and env not in _PROD_ENVS:
            cluster = f"{cluster}#{env}"
        if cluster:
            try:
                return resolve_base_url_from_cluster(cluster)
            except ValueError:
                pass
        return client.base_url

    def __call__(
        self,
        action: SandboxTerminalAction,
        conversation: BaseConversation | None = None,
    ) -> SandboxTerminalObservation:
        try:
            client = self._sandbox_client(conversation)
            command = (
                f"cd {action.cwd} && {action.command}" if action.cwd else action.command
            )
            output, returncode, timed_out = _run_terminal_command(
                base_url=self._terminal_base_url(client),
                api_key=client.api_key,
                sandbox_id=action.sandbox_id,
                command=command,
                timeout_seconds=action.timeout_seconds or 60,
            )
            truncated = len(output) > MAX_FILE_TEXT_CHARS
            if truncated:
                output = output[:MAX_FILE_TEXT_CHARS] + "\n...[truncated]"
            lines = [
                f"sandbox_id: {action.sandbox_id}",
                f"returncode: {returncode}",
            ]
            if timed_out:
                lines.append("timed_out: True")
            lines.append(f"output:\n{output}")
            return SandboxTerminalObservation.from_text(
                text="\n".join(lines),
                sandbox_id=action.sandbox_id,
                returncode=returncode,
                output=output,
                timed_out=timed_out,
            )
        except Exception as exc:  # noqa: BLE001
            return SandboxTerminalObservation.from_text(
                text=(
                    f"Failed to run terminal command in sandbox "
                    f"{action.sandbox_id}: {exc}"
                ),
                is_error=True,
            )


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------


# Tool definitions
# ---------------------------------------------------------------------------


class SandboxCreateTool(ToolDefinition[SandboxCreateAction, SandboxCreateObservation]):
    """Create a disposable Pyromind custom (headless) sandbox."""

    @classmethod
    def create(
        cls,
        conv_state: Any = None,  # noqa: ARG003
        **params: Any,
    ) -> Sequence[Self]:
        return [
            cls(
                description=(
                    "Create a disposable Pyromind custom (headless) sandbox "
                    "and wait until it is running. If image is omitted the "
                    "cluster default Jupyter-lab image is used. The sandbox "
                    "can run commands (sandbox_exec / agent terminal) and "
                    "read files (sandbox_read_file)."
                ),
                action_type=SandboxCreateAction,
                observation_type=SandboxCreateObservation,
                executor=SandboxCreateExecutor(
                    cluster=params.get("cluster"),
                    env=params.get("env"),
                    current_user=params.get("current_user"),
                    headers=params.get("headers", {}),
                ),
                annotations=ToolAnnotations(
                    title="Sandbox Create",
                    readOnlyHint=False,
                    destructiveHint=False,
                    idempotentHint=False,
                    openWorldHint=True,
                ),
            )
        ]


class SandboxDeleteTool(ToolDefinition[SandboxDeleteAction, SandboxDeleteObservation]):
    """Delete a sandbox."""

    @classmethod
    def create(
        cls,
        conv_state: Any = None,  # noqa: ARG003
        **params: Any,
    ) -> Sequence[Self]:
        return [
            cls(
                description=(
                    "Permanently delete a Pyromind sandbox and release its resources."
                ),
                action_type=SandboxDeleteAction,
                observation_type=SandboxDeleteObservation,
                executor=SandboxDeleteExecutor(
                    cluster=params.get("cluster"),
                    env=params.get("env"),
                    current_user=params.get("current_user"),
                    headers=params.get("headers", {}),
                ),
                annotations=ToolAnnotations(
                    title="Sandbox Delete",
                    readOnlyHint=False,
                    destructiveHint=True,
                    idempotentHint=False,
                    openWorldHint=True,
                ),
            )
        ]


class SandboxReadFileTool(
    ToolDefinition[SandboxReadFileAction, SandboxReadFileObservation]
):
    """Read a file (custom sandbox) as text or base64."""

    @classmethod
    def create(
        cls,
        conv_state: Any = None,  # noqa: ARG003
        **params: Any,
    ) -> Sequence[Self]:
        return [
            cls(
                description=(
                    "Read a file from a custom (headless) Pyromind sandbox. "
                    "Text files are returned as utf-8, binary files as "
                    "base64; very large files are truncated."
                ),
                action_type=SandboxReadFileAction,
                observation_type=SandboxReadFileObservation,
                executor=SandboxReadFileExecutor(
                    cluster=params.get("cluster"),
                    env=params.get("env"),
                    current_user=params.get("current_user"),
                    headers=params.get("headers", {}),
                ),
                annotations=ToolAnnotations(
                    title="Sandbox read file",
                    readOnlyHint=True,
                    destructiveHint=False,
                    idempotentHint=True,
                    openWorldHint=True,
                ),
            )
        ]


class SandboxWriteFileTool(
    ToolDefinition[SandboxWriteFileAction, SandboxWriteFileObservation]
):
    """Write a file into a custom sandbox."""

    @classmethod
    def create(
        cls,
        conv_state: Any = None,  # noqa: ARG003
        **params: Any,
    ) -> Sequence[Self]:
        return [
            cls(
                description=(
                    "Write a file into a custom (headless) Pyromind sandbox. "
                    "The content is written as UTF-8 text; parent directories "
                    "are created automatically."
                ),
                action_type=SandboxWriteFileAction,
                observation_type=SandboxWriteFileObservation,
                executor=SandboxWriteFileExecutor(
                    cluster=params.get("cluster"),
                    env=params.get("env"),
                    current_user=params.get("current_user"),
                    headers=params.get("headers", {}),
                ),
                annotations=ToolAnnotations(
                    title="Sandbox write file",
                    readOnlyHint=False,
                    destructiveHint=False,
                    idempotentHint=False,
                    openWorldHint=True,
                ),
            )
        ]


class SandboxDeleteFileTool(
    ToolDefinition[SandboxDeleteFileAction, SandboxDeleteFileObservation]
):
    """Delete a file or directory from a custom sandbox."""

    @classmethod
    def create(
        cls,
        conv_state: Any = None,  # noqa: ARG003
        **params: Any,
    ) -> Sequence[Self]:
        return [
            cls(
                description=(
                    "Delete a file or directory from a custom (headless) "
                    "Pyromind sandbox. Use recursive=True for non-empty "
                    "directories."
                ),
                action_type=SandboxDeleteFileAction,
                observation_type=SandboxDeleteFileObservation,
                executor=SandboxDeleteFileExecutor(
                    cluster=params.get("cluster"),
                    env=params.get("env"),
                    current_user=params.get("current_user"),
                    headers=params.get("headers", {}),
                ),
                annotations=ToolAnnotations(
                    title="Sandbox delete file",
                    readOnlyHint=False,
                    destructiveHint=True,
                    idempotentHint=False,
                    openWorldHint=True,
                ),
            )
        ]


class SandboxTerminalTool(
    ToolDefinition[SandboxTerminalAction, SandboxTerminalObservation]
):
    """Run one command in a sandbox over its terminal TTY WebSocket."""

    @classmethod
    def create(
        cls,
        conv_state: Any = None,  # noqa: ARG003
        **params: Any,
    ) -> Sequence[Self]:
        return [
            cls(
                description=(
                    "Run one shell command inside a running Pyromind sandbox "
                    "through its terminal WebSocket (the same TTY bridge the "
                    "'pyromind terminal' CLI uses). Runs in a fresh shell, "
                    "returns the combined output and exit code."
                ),
                action_type=SandboxTerminalAction,
                observation_type=SandboxTerminalObservation,
                executor=SandboxTerminalExecutor(
                    cluster=params.get("cluster"),
                    env=params.get("env"),
                    current_user=params.get("current_user"),
                    headers=params.get("headers", {}),
                ),
                annotations=ToolAnnotations(
                    title="Sandbox terminal command",
                    readOnlyHint=False,
                    destructiveHint=False,
                    idempotentHint=False,
                    openWorldHint=True,
                ),
            )
        ]


register_tool(SandboxCreateTool.name, SandboxCreateTool)
register_tool(SandboxDeleteTool.name, SandboxDeleteTool)
register_tool(SandboxReadFileTool.name, SandboxReadFileTool)
register_tool(SandboxTerminalTool.name, SandboxTerminalTool)
register_tool(SandboxWriteFileTool.name, SandboxWriteFileTool)
register_tool(SandboxDeleteFileTool.name, SandboxDeleteFileTool)
