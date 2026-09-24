"""Session-scoped sandbox lifecycle for the Pi execution plane.

The control plane (agent-server) owns credentials and container lifecycle; the
Pi runner only receives an endpoint when it is about to touch the execution
workspace. The access key therefore never reaches the runner's start frame, the
persisted ``sandbox.json``, or the logs.
"""

from __future__ import annotations

import asyncio
import logging
import os
import secrets
import shlex
import tarfile
import tempfile
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, Protocol

from pyromind_sdk.client.models import (
    ResourceConfig,
    SandboxRequest,
    SandboxResponse,
    SandboxType,
    VolumeMount,
)

from harness_adapter.pi_adapter.business_tool_host import (
    ToolExecutionContext,
    _auth_token,
    _execution_target,
    _forward_headers,
)
from harness_adapter.pi_adapter.persistence import PiSessionFiles
from harness_adapter.pi_adapter.sandbox_workspace import SandboxWorkspace
from openhands.tools.sandbox import (
    create_sandbox_api_client,
    default_sandbox_image,
    run_terminal_command,
    terminal_base_url,
    write_sandbox_file,
)


logger = logging.getLogger(__name__)

DEFAULT_MOUNT_PATH = "/target-workspace"
# Node-side path the platform exposes the user Storage on. The container only
# sees it when the create request asks for the mount.
STORAGE_HOST_PATH = "/workspace"
PYROMIND_AGENT_STORAGE_ROOT = "/.pyromind-agent"
# Workspace-relative alias for the mounted Storage root. The model reads data
# through normal file/terminal tools instead of the Storage HTTP API.
STORAGE_ALIAS = "storage"
_WORKSPACE_LAYOUT_VERSION = 2
# The platform rejects a create request without memory; mirror the sandbox
# tool's own fallback (platform minimum 4 vCPU, 1:2 vCPU:memory ratio).
DEFAULT_CPU = "4"
DEFAULT_MEMORY = "8Gi"
_READY_STATUSES = frozenset({"running"})
_RESUMABLE_STATUSES = frozenset({"paused", "stopped"})
# Exit codes reported by the workspace scripts so a failure can be attributed
# without grepping terminal output (the TTY bridge echoes the command line).
_MOUNT_MISSING_EXIT_CODE = 3
_FORK_MISSING_EXIT_CODE = 42
_CREATE_TIMEOUT_SECONDS = 600
_PREPARE_TIMEOUT_SECONDS = 120
# A fork copies an existing conversation inside the mount, so it scales with
# the amount of data the source conversation wrote to Storage.
_FORK_TIMEOUT_SECONDS = 600
_FAILURE_OUTPUT_CHARS = 500
# The skills tree is hundreds of files, so it travels as one archive instead of
# one HTTP round trip per file.
_RESOURCE_ARCHIVE_NAME = "pi-session-resources.tgz"


@dataclass(frozen=True, slots=True)
class SandboxSettings:
    """Session-scoped sandbox configuration from ``SessionSpec.extra["sandbox"]``.

    Every field falls back to the ``PYROMIND_SANDBOX_*`` environment variables and
    finally to the platform defaults, so a session only has to state what it
    wants to override.
    """

    image: str | None = None
    cpu: str | None = None
    memory: str | None = None
    gpu: str | None = None
    gpu_card: str | None = None
    mount_path: str = DEFAULT_MOUNT_PATH
    host_path: str = STORAGE_HOST_PATH

    @classmethod
    def from_extra(cls, extra: dict[str, Any]) -> SandboxSettings:
        raw = extra.get("sandbox")
        configured = raw if isinstance(raw, dict) else {}
        cpu = (
            _configured_value(configured, "cpu", "PYROMIND_SANDBOX_CPU") or DEFAULT_CPU
        )
        memory = _configured_value(
            configured, "memory", "PYROMIND_SANDBOX_MEMORY"
        ) or _memory_for_cpu(cpu)
        return cls(
            image=_configured_value(configured, "image", "PYROMIND_SANDBOX_IMAGE"),
            cpu=cpu,
            memory=memory,
            gpu=_configured_value(configured, "gpu", "PYROMIND_SANDBOX_GPU"),
            gpu_card=_configured_value(
                configured, "gpu_card", "PYROMIND_SANDBOX_GPU_CARD"
            ),
            mount_path=(
                _configured_value(
                    configured, "mount_path", "PYROMIND_SANDBOX_MOUNT_PATH"
                )
                or DEFAULT_MOUNT_PATH
            ),
            host_path=(
                _configured_value(configured, "host_path", "PYROMIND_SANDBOX_HOST_PATH")
                or STORAGE_HOST_PATH
            ),
        )


class SandboxClientLike(Protocol):
    """The slice of ``SandboxClient`` this manager drives.

    Naming the seam keeps the lifecycle testable without a live platform.
    """

    base_url: str
    api_key: str
    cluster: str

    def list(self) -> list[SandboxResponse]: ...

    def get_sandbox(self, sandbox_id: str) -> SandboxResponse: ...

    def create_and_wait(
        self, request: SandboxRequest, *, target_status: str, timeout: int
    ) -> SandboxResponse: ...

    def resume(self, sandbox_id: str) -> SandboxResponse: ...

    def delete(self, sandbox_id: str) -> None: ...

    def pause(self, sandbox_id: str) -> SandboxResponse: ...

    def read_file(self, sandbox_id: str, path: str) -> bytes: ...

    def write_file(
        self, sandbox_id: str, path: str, source: str | os.PathLike[str] | bytes
    ) -> dict[str, Any]: ...


@dataclass(slots=True)
class _SandboxState:
    client: SandboxClientLike
    endpoint: dict[str, Any]


def _configured_value(values: dict[str, Any], key: str, env: str) -> str | None:
    value = values.get(key)
    if isinstance(value, (str, int)) and not isinstance(value, bool):
        text = str(value).strip()
        if text:
            return text
    fallback = os.getenv(env, "").strip()
    return fallback or None


def _memory_for_cpu(cpu: str) -> str:
    """Mirror the sandbox tool's 1:2 vCPU:memory platform ratio."""
    return f"{int(cpu) * 2}Gi" if cpu.isdigit() and int(cpu) > 0 else DEFAULT_MEMORY


def _failure_detail(output: str, exit_code: int | None, timed_out: bool) -> str:
    """Attach the terminal outcome to a script failure message."""
    detail = [f"exit_code={exit_code}", f"timed_out={timed_out}"]
    text = output.strip()
    if text:
        detail.append(f"output={text[-_FAILURE_OUTPUT_CHARS:]}")
    return " (" + ", ".join(detail) + ")"


@contextmanager
def _resource_archive(
    files: Sequence[tuple[str, str]],
    workspace_path: str,
) -> Iterator[Path]:
    """Pack resource files into one archive mirroring their sandbox layout."""
    root = PurePosixPath(workspace_path)
    with tempfile.TemporaryDirectory(prefix="pi-session-resources-") as directory:
        archive = Path(directory) / _RESOURCE_ARCHIVE_NAME
        with tarfile.open(archive, "w:gz") as tar:
            for destination, local_path in files:
                arcname = PurePosixPath(destination).relative_to(root).as_posix()
                tar.add(local_path, arcname=arcname)
        yield archive


class SandboxExecutionManager:
    """Create, resume, and describe the sandbox backing one conversation."""

    def __init__(
        self,
        *,
        client_factory: Callable[[ToolExecutionContext], SandboxClientLike]
        | None = None,
        resource_roots: Sequence[tuple[str, os.PathLike[str]]] | None = None,
    ) -> None:
        self._client_factory = client_factory or _sandbox_client
        self._resource_roots = list(resource_roots or [])
        self._states: dict[str, _SandboxState] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    async def ensure(
        self,
        context: ToolExecutionContext,
        files: PiSessionFiles,
        *,
        refresh: bool = False,
    ) -> dict[str, Any]:
        state = await self._ensure_state(context, files, refresh=refresh)
        return dict(state.endpoint)

    async def workspace(
        self, context: ToolExecutionContext, files: PiSessionFiles
    ) -> SandboxWorkspace:
        state = await self._ensure_state(context, files)
        endpoint = state.endpoint
        return SandboxWorkspace(
            working_dir=endpoint["workspace_path"],
            storage_path=endpoint["storage_path"],
            sandbox_id=endpoint["sandbox_id"],
            ws_base_url=endpoint["ws_base_url"],
            client=state.client,
        )

    async def delete(
        self, context: ToolExecutionContext, files: PiSessionFiles
    ) -> None:
        state = self._states.pop(context.conversation_id, None)
        record = files.load_sandbox() or {}
        sandbox_id = state.endpoint["sandbox_id"] if state is not None else None
        if sandbox_id is None:
            recorded_id = record.get("sandbox_id")
            sandbox_id = recorded_id if isinstance(recorded_id, str) else None
        if sandbox_id is None:
            files.clear_sandbox()
            self._locks.pop(context.conversation_id, None)
            return
        client = state.client if state is not None else self._client_factory(context)
        try:
            await asyncio.to_thread(client.delete, sandbox_id)
        except Exception as delete_error:  # noqa: BLE001
            logger.warning(
                "Pi sandbox delete failed; pausing before retry "
                "conversation_id=%s sandbox_id=%s error=%s",
                context.conversation_id,
                sandbox_id,
                type(delete_error).__name__,
            )
            try:
                await asyncio.to_thread(client.pause, sandbox_id)
                await asyncio.to_thread(client.delete, sandbox_id)
            except Exception as retry_error:  # noqa: BLE001
                logger.warning(
                    "Pi sandbox cleanup failed conversation_id=%s sandbox_id=%s "
                    "error=%s",
                    context.conversation_id,
                    sandbox_id,
                    type(retry_error).__name__,
                )
        files.clear_sandbox()
        self._locks.pop(context.conversation_id, None)

    async def pause(self, context: ToolExecutionContext, files: PiSessionFiles) -> None:
        """Stop the container while keeping the execution workspace resumable."""
        state = self._states.pop(context.conversation_id, None)
        record = files.load_sandbox() or {}
        sandbox_id = state.endpoint["sandbox_id"] if state is not None else None
        if sandbox_id is None:
            recorded_id = record.get("sandbox_id")
            sandbox_id = recorded_id if isinstance(recorded_id, str) else None
        if sandbox_id is None:
            return
        client = state.client if state is not None else self._client_factory(context)
        await asyncio.to_thread(client.pause, sandbox_id)

    def execution_workspace_path(self, conversation_id: str) -> str | None:
        state = self._states.get(conversation_id)
        if state is None:
            return None
        value = state.endpoint.get("workspace_path")
        return value if isinstance(value, str) else None

    async def _ensure_state(
        self,
        context: ToolExecutionContext,
        files: PiSessionFiles,
        *,
        refresh: bool = False,
    ) -> _SandboxState:
        lock = self._locks.setdefault(context.conversation_id, asyncio.Lock())
        async with lock:
            if refresh:
                # A failed call means the container may have been rebuilt or
                # re-keyed behind this process, so re-read the platform state and
                # mint a fresh access key instead of returning the cached bundle.
                self._states.pop(context.conversation_id, None)
            pending_fork = files.load_pending_sandbox_fork()
            existing = self._states.get(context.conversation_id)
            if existing is not None and pending_fork is None:
                return existing
            state = await self._ensure_state_locked(context, files, pending_fork)
            self._states[context.conversation_id] = state
            return state

    async def _ensure_state_locked(
        self,
        context: ToolExecutionContext,
        files: PiSessionFiles,
        pending_fork: dict[str, Any] | None,
    ) -> _SandboxState:
        settings = SandboxSettings.from_extra(context.extra)
        record = files.load_sandbox() or {}
        client = await asyncio.to_thread(self._client_factory, context)
        mount_path = settings.mount_path
        workspace_path = (
            f"{mount_path}{PYROMIND_AGENT_STORAGE_ROOT}/{context.conversation_id}"
        )
        created_at = record.get("created_at") or datetime.now(UTC).isoformat()
        recorded_id = record.get("sandbox_id")
        record_needs_prepare = (
            record.get("workspace_version") != _WORKSPACE_LAYOUT_VERSION
        )
        # The Storage mount is fixed when the container is created, so a
        # container recorded under a different mount layout cannot be repaired
        # in place.
        record_layout_matches = (
            record.get("mount_path") == mount_path
            and record.get("storage_host_path") == settings.host_path
        )
        sandbox = await self._lookup(
            client, recorded_id if isinstance(recorded_id, str) else None
        )
        prepare = False
        if sandbox is None:
            sandbox = await self._create(client, settings, context.conversation_id)
            prepare = True
        elif not record_layout_matches:
            logger.warning(
                "Recreating Pi sandbox whose Storage mount no longer matches "
                "conversation_id=%s sandbox_id=%s",
                context.conversation_id,
                sandbox.id,
            )
            await asyncio.to_thread(client.delete, sandbox.id)
            sandbox = await self._create(client, settings, context.conversation_id)
            prepare = True
        elif sandbox.status.strip().lower() in _RESUMABLE_STATUSES:
            sandbox = await asyncio.to_thread(client.resume, sandbox.id)
            prepare = True
        elif sandbox.status.strip().lower() not in _READY_STATUSES:
            logger.warning(
                "Recreating Pi sandbox in unexpected state conversation_id=%s "
                "sandbox_id=%s status=%s",
                context.conversation_id,
                sandbox.id,
                sandbox.status,
            )
            await asyncio.to_thread(client.delete, sandbox.id)
            sandbox = await self._create(client, settings, context.conversation_id)
            prepare = True

        if prepare:
            # The sandbox name is derived from the conversation, so record the
            # container before its workspace is ready: a retry has to repair
            # this sandbox instead of creating a second one with the same name.
            files.save_sandbox(
                {
                    "sandbox_id": sandbox.id,
                    "workspace_path": workspace_path,
                    "mount_path": mount_path,
                    "storage_host_path": settings.host_path,
                    "created_at": created_at,
                }
            )

        ws_base_url = terminal_base_url(
            cluster=client.cluster,
            env=_execution_target(context)[0],
            fallback=client.base_url,
        )
        if pending_fork is not None:
            await self._apply_pending_fork(
                client,
                sandbox.id,
                ws_base_url,
                mount_path,
                workspace_path,
                pending_fork,
            )
            files.clear_pending_sandbox_fork()
        if prepare or record_needs_prepare:
            logger.info(
                "Preparing Pi sandbox workspace conversation_id=%s sandbox_id=%s "
                "workspace=%s",
                context.conversation_id,
                sandbox.id,
                workspace_path,
            )
            await self._prepare_workspace(
                client, sandbox.id, ws_base_url, mount_path, workspace_path
            )
            await self._upload_resources(
                client, sandbox.id, ws_base_url, workspace_path
            )
            logger.info(
                "Pi sandbox workspace ready conversation_id=%s sandbox_id=%s",
                context.conversation_id,
                sandbox.id,
            )
        files.save_sandbox(
            {
                "sandbox_id": sandbox.id,
                "workspace_path": workspace_path,
                "mount_path": mount_path,
                "storage_host_path": settings.host_path,
                "created_at": created_at,
                "workspace_version": _WORKSPACE_LAYOUT_VERSION,
            }
        )
        endpoint = {
            "base_url": client.base_url,
            "ws_base_url": ws_base_url,
            "sandbox_id": sandbox.id,
            "api_key": client.api_key,
            "workspace_path": workspace_path,
            "storage_path": mount_path,
            "cluster": client.cluster,
        }
        return _SandboxState(client=client, endpoint=endpoint)

    async def _lookup(
        self, client: SandboxClientLike, sandbox_id: str | None
    ) -> SandboxResponse | None:
        if not sandbox_id:
            return None
        try:
            return await asyncio.to_thread(client.get_sandbox, sandbox_id)
        except Exception as exc:  # noqa: BLE001
            logger.info(
                "Pi sandbox lookup failed; recreating sandbox_id=%s error=%s",
                sandbox_id,
                type(exc).__name__,
            )
            return None

    async def _create(
        self, client: SandboxClientLike, settings: SandboxSettings, conversation_id: str
    ) -> SandboxResponse:
        request = SandboxRequest(
            sandbox_type=SandboxType.CUSTOM,
            image=settings.image or default_sandbox_image(client.cluster),
            resources=ResourceConfig(
                cpu=settings.cpu,
                memory=settings.memory,
                gpu=settings.gpu,
                gpu_card=settings.gpu_card,
            ),
            name=f"pi-{conversation_id}",
            volume_mounts=[
                VolumeMount(
                    host_path=settings.host_path,
                    mount_path=settings.mount_path,
                    read_only=False,
                )
            ],
        )
        try:
            sandbox = await asyncio.to_thread(
                client.create_and_wait,
                request,
                target_status="running",
                timeout=_CREATE_TIMEOUT_SECONDS,
            )
        except Exception as exc:  # noqa: BLE001
            sandbox = await self._adopt_named(client, request.name, conversation_id)
            if sandbox is None:
                raise
            logger.warning(
                "Pi sandbox create failed; adopting the existing container "
                "conversation_id=%s sandbox_id=%s error=%s",
                conversation_id,
                sandbox.id,
                type(exc).__name__,
            )
        return await self._running(client, sandbox)

    async def _adopt_named(
        self,
        client: SandboxClientLike,
        name: str | None,
        conversation_id: str,
    ) -> SandboxResponse | None:
        """Find the container a previous attempt left behind under ``name``."""
        if not name:
            return None
        try:
            sandboxes = await asyncio.to_thread(client.list)
        except Exception as exc:  # noqa: BLE001
            logger.info(
                "Pi sandbox list failed conversation_id=%s error=%s",
                conversation_id,
                type(exc).__name__,
            )
            return None
        return next((sandbox for sandbox in sandboxes if sandbox.name == name), None)

    async def _running(
        self, client: SandboxClientLike, sandbox: SandboxResponse
    ) -> SandboxResponse:
        if sandbox.status.strip().lower() in _RESUMABLE_STATUSES:
            sandbox = await asyncio.to_thread(client.resume, sandbox.id)
        if sandbox.status.strip().lower() != "running":
            raise RuntimeError(
                f"PI_SANDBOX_NOT_RUNNING: sandbox {sandbox.id} is "
                f"{sandbox.status or 'unknown'}"
            )
        return sandbox

    async def _prepare_workspace(
        self,
        client: SandboxClientLike,
        sandbox_id: str,
        ws_base_url: str,
        mount_path: str,
        workspace_path: str,
    ) -> None:
        storage_link = f"{workspace_path}/{STORAGE_ALIAS}"
        script = (
            f"if [ ! -d {shlex.quote(mount_path)} ]; then "
            f"exit {_MOUNT_MISSING_EXIT_CODE}; fi\n"
            f"mkdir -p {shlex.quote(workspace_path)} && "
            f"rm -rf {shlex.quote(storage_link)} && "
            f"ln -s {shlex.quote(mount_path)} {shlex.quote(storage_link)}\n"
        )
        output, exit_code, timed_out = await self._run_script(
            client, sandbox_id, ws_base_url, script
        )
        if exit_code == 0:
            return
        detail = _failure_detail(output, exit_code, timed_out)
        if exit_code == _MOUNT_MISSING_EXIT_CODE:
            raise RuntimeError(
                f"PI_SANDBOX_MOUNT_MISSING: user Storage is not mounted at {mount_path}"
                f"{detail}"
            )
        raise RuntimeError(
            f"PI_SANDBOX_PREPARE_FAILED: cannot prepare {workspace_path}{detail}"
        )

    async def _run_script(
        self,
        client: SandboxClientLike,
        sandbox_id: str,
        ws_base_url: str,
        script: str,
        timeout_seconds: int = _PREPARE_TIMEOUT_SECONDS,
    ) -> tuple[str, int | None, bool]:
        """Run one shell script inside the sandbox and return its exit status.

        The terminal bridge echoes the command line back, so a marker embedded
        in the command text is indistinguishable from the command's own output.
        Shipping the script as a file keeps the echoed line marker-free.
        """
        remote_path = f"/tmp/pi-sandbox-{secrets.token_hex(8)}.sh"
        await asyncio.to_thread(
            client.write_file, sandbox_id, remote_path, script.encode("utf-8")
        )
        quoted = shlex.quote(remote_path)
        # The subshell propagates the script's status without ending the shell
        # that the terminal bridge reuses for the exit-code echo.
        command = f"sh {quoted}; rc=$?; rm -f {quoted}; (exit $rc)"
        return await asyncio.to_thread(
            run_terminal_command,
            base_url=ws_base_url,
            api_key=client.api_key,
            sandbox_id=sandbox_id,
            command=command,
            timeout_seconds=timeout_seconds,
        )

    async def _apply_pending_fork(
        self,
        client: SandboxClientLike,
        sandbox_id: str,
        ws_base_url: str,
        mount_path: str,
        workspace_path: str,
        pending: dict[str, Any],
    ) -> None:
        source_id = pending.get("source_conversation_id")
        if source_id is not None and (
            not isinstance(source_id, str)
            or not source_id
            or "/" in source_id
            or "\\" in source_id
            or source_id == workspace_path.rsplit("/", 1)[-1]
        ):
            raise RuntimeError(
                "PI_SANDBOX_FORK_INVALID: invalid source conversation id"
            )
        if source_id is not None:
            source_path = f"{mount_path}{PYROMIND_AGENT_STORAGE_ROOT}/{source_id}"
            temporary = f"{workspace_path}.fork"
            script = (
                f"if [ ! -d {shlex.quote(source_path)} ]; then "
                f"exit {_FORK_MISSING_EXIT_CODE}; fi\n"
                f"rm -rf {shlex.quote(temporary)} && "
                f"mkdir -p {shlex.quote(temporary)} && "
                f"cp -a {shlex.quote(source_path)}/. {shlex.quote(temporary)}/ && "
                f"rm -rf {shlex.quote(workspace_path)} && "
                f"mv {shlex.quote(temporary)} {shlex.quote(workspace_path)}\n"
            )
            output, exit_code, timed_out = await self._run_script(
                client,
                sandbox_id,
                ws_base_url,
                script,
                timeout_seconds=_FORK_TIMEOUT_SECONDS,
            )
            if exit_code != 0:
                raise RuntimeError(
                    "PI_SANDBOX_FORK_FAILED: cannot copy "
                    f"{source_path} to {workspace_path}"
                    f"{_failure_detail(output, exit_code, timed_out)}"
                )
        workflow_dsl = pending.get("workflow_dsl")
        if not isinstance(workflow_dsl, str):
            return
        workflow_path = f"{workspace_path}/public_data/workflow_canvas/workflow.py"
        if workflow_dsl.strip():
            await asyncio.to_thread(
                client.write_file,
                sandbox_id,
                workflow_path,
                workflow_dsl.encode("utf-8"),
            )
            return
        await asyncio.to_thread(
            run_terminal_command,
            base_url=ws_base_url,
            api_key=client.api_key,
            sandbox_id=sandbox_id,
            command=f"rm -f {shlex.quote(workflow_path)}",
            timeout_seconds=_PREPARE_TIMEOUT_SECONDS,
        )

    async def _upload_resources(
        self,
        client: SandboxClientLike,
        sandbox_id: str,
        ws_base_url: str,
        workspace_path: str,
    ) -> None:
        files = self._resource_files(workspace_path)
        total = len(files)
        if not total:
            return
        logger.info(
            "Uploading Pi session resources sandbox_id=%s files=%d",
            sandbox_id,
            total,
        )
        archive_path = f"/tmp/{_RESOURCE_ARCHIVE_NAME}"
        with _resource_archive(files, workspace_path) as archive:
            await asyncio.to_thread(
                write_sandbox_file,
                client,
                ws_base_url=ws_base_url,
                sandbox_id=sandbox_id,
                path=archive_path,
                source=archive,
            )
            script = (
                "set -e\n"
                f"mkdir -p {shlex.quote(workspace_path)}\n"
                f"tar -xzf {shlex.quote(archive_path)} "
                f"-C {shlex.quote(workspace_path)}\n"
                f"rm -f {shlex.quote(archive_path)}\n"
            )
            output, exit_code, timed_out = await self._run_script(
                client,
                sandbox_id,
                ws_base_url,
                script,
            )
        if exit_code != 0:
            raise RuntimeError(
                "PI_SANDBOX_RESOURCE_UPLOAD_FAILED: cannot unpack session "
                f"resources into {workspace_path}"
                f"{_failure_detail(output, exit_code, timed_out)}"
            )
        logger.info(
            "Pi session resource upload complete sandbox_id=%s uploaded=%d",
            sandbox_id,
            total,
        )

    def _resource_files(self, workspace_path: str) -> list[tuple[str, str]]:
        """Local resource files paired with the sandbox path each one targets."""
        files: list[tuple[str, str]] = []
        for destination_root, source_root in self._resource_roots:
            source = os.fspath(source_root)
            if not os.path.isdir(source):
                continue
            for directory, _, names in os.walk(source):
                for name in names:
                    local_path = os.path.join(directory, name)
                    if os.path.islink(local_path) or not os.path.isfile(local_path):
                        continue
                    relative = os.path.relpath(local_path, source)
                    files.append(
                        (
                            f"{workspace_path}/{destination_root}/"
                            f"{relative.replace(os.sep, '/')}",
                            local_path,
                        )
                    )
        return files


def _sandbox_client(context: ToolExecutionContext) -> SandboxClientLike:
    env, cluster = _execution_target(context)
    api = create_sandbox_api_client(
        env=env,
        cluster=cluster,
        auth_token=_auth_token(context.request_context),
        headers=_forward_headers(context.request_context, include_cookie=False),
    )
    return api.sandboxes
