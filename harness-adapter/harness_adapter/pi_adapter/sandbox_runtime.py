"""Session-scoped sandbox lifecycle for the Pi execution plane.

The control plane (agent-server) owns credentials and container lifecycle; the
Pi runner only receives an endpoint when it is about to touch the execution
workspace. The access key therefore never reaches the runner's start frame, the
persisted ``sandbox.json``, or the logs.
"""

from __future__ import annotations

import asyncio
import hashlib
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
# Resource archives are cached inside the user's own Storage mount, next to the
# conversation workspaces. The skills tree is identical for every conversation
# of one deployment, so a conversation which finds its digest already there
# unpacks it from the mount instead of uploading it again.
PYROMIND_AGENT_RESOURCES_ROOT = "/.pyromind-agent-resources"
# The workspace symlink that used to expose the Storage mount under a short
# relative name. Storage is addressed by its mount path now, so the link is only
# remembered here to clear it out of workspaces laid out by an older runtime.
LEGACY_STORAGE_ALIAS = "storage"
# Bumped when the workspace layout changes, so an existing sandbox re-runs the
# prepare script. 4: the Storage symlink is no longer created, and workspaces
# laid out by version 3 still carry it until they are prepared again.
_WORKSPACE_LAYOUT_VERSION = 4
# The platform rejects a create request without memory; mirror the sandbox
# tool's own fallback (platform minimum 4 vCPU, 1:2 vCPU:memory ratio).
DEFAULT_CPU = "4"
DEFAULT_MEMORY = "8Gi"
_READY_STATUSES = frozenset({"running"})
_RESUMABLE_STATUSES = frozenset({"paused", "stopped"})
# A resumed container reports these statuses while its pod is still coming up;
# resume() only accepts the request, so `running` is the first usable status.
_STARTING_STATUSES = frozenset({"creating", "pending", "starting"})
_START_TIMEOUT_SECONDS = 180
_START_POLL_SECONDS = 3
# Exit codes reported by the workspace scripts so a failure can be attributed
# without grepping terminal output (the TTY bridge echoes the command line).
_MOUNT_MISSING_EXIT_CODE = 3
_FORK_MISSING_EXIT_CODE = 42
_CREATE_TIMEOUT_SECONDS = 600
_PREPARE_TIMEOUT_SECONDS = 120
# A recorded sandbox can outlive its data-plane endpoint: the control plane still
# reports it running while connections to the terminal port never complete. One
# short command tells the two apart before the session depends on it.
_PROBE_TIMEOUT_SECONDS = 15
# A fork copies an existing conversation inside the mount, so it scales with
# the amount of data the source conversation wrote to Storage.
_FORK_TIMEOUT_SECONDS = 600
_FAILURE_OUTPUT_CHARS = 500
# The skills tree is hundreds of files, so it travels as one archive instead of
# one HTTP round trip per file.
_RESOURCE_ARCHIVE_NAME = "pi-session-resources.tgz"
# Interpreter caches the agent host may have produced next to skill scripts;
# they are stale bytecode for the sandbox and only bloat the upload.
_SKIPPED_DIRECTORIES = frozenset({"__pycache__"})
# Skills are staged under this workspace-relative alias.
_SKILLS_ALIAS = ".agents/skills"
# The DataFlow runtime is pre-installed as soon as the sandbox is prepared, so
# the first data-processing run does not pay for it. The warm-up and the run
# path take the venv's own install lock, so a run that arrives while the
# warm-up is still installing waits for it instead of installing a second copy.
_DATAFLOW_SKILL = "data-processing"
_DATAFLOW_PREPARE_SCRIPT = "scripts/preparation/sandbox_sample.py"
_DATAFLOW_WARMUP_HEARTBEAT_SECONDS = 20
# Mirrors the data-processing tool's own bootstrap budget.
_DATAFLOW_WARMUP_TIMEOUT_SECONDS = 1800
_DATAFLOW_WARMUP_SWITCH = "PYROMIND_SANDBOX_DATAFLOW_WARMUP"
# Creating the container, laying out the workspace, and staging the resources
# takes tens of seconds, so a session which is about to run work starts that in
# the background instead of making its first tool call pay for it. Set to 0 to
# create the sandbox on first touch only.
_PREWARM_SWITCH = "PYROMIND_SANDBOX_PREWARM"
_DATAFLOW_WARMUP_ENV = (
    "PYROMIND_SANDBOX_DATAFLOW_PYTHON",
    "PYROMIND_SANDBOX_DATAFLOW_VENV",
    "PYROMIND_SANDBOX_SYSTEM_PYTHON",
)
_DEFAULT_SANDBOX_SYSTEM_PYTHON = "python3"


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

    def wait_for_sandbox_status(
        self,
        sandbox_id: str,
        target_status: str,
        timeout: int = 300,
        check_interval: int = 3,
        intermediate_statuses: list[str] | None = None,
    ) -> bool: ...

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


def _dataflow_warmup_env() -> dict[str, str]:
    """Operator overrides the DataFlow runner would read, when they are set.

    The runner resolves the same values from the run spec, so forwarding them
    keeps the warm-up on the interpreter and venv path a run would pick.
    """
    values: dict[str, str] = {}
    for name in _DATAFLOW_WARMUP_ENV:
        value = os.environ.get(name, "").strip()
        if value:
            values[name] = value
    return values


def _resource_arcname(destination: str, workspace_path: str) -> str:
    """Sandbox path of one resource file, relative to the workspace root."""
    return PurePosixPath(destination).relative_to(workspace_path).as_posix()


@contextmanager
def _resource_archive(
    files: Sequence[tuple[str, str]],
    workspace_path: str,
) -> Iterator[Path]:
    """Pack resource files into one archive mirroring their sandbox layout."""
    with tempfile.TemporaryDirectory(prefix="pi-session-resources-") as directory:
        archive = Path(directory) / _RESOURCE_ARCHIVE_NAME
        with tarfile.open(archive, "w:gz") as tar:
            for destination, local_path in files:
                tar.add(
                    local_path,
                    arcname=_resource_arcname(destination, workspace_path),
                )
        yield archive


def _resource_digest(files: Sequence[tuple[str, str]], workspace_path: str) -> str:
    """Content hash naming one version of the resource tree.

    The archive bytes cannot name it: gzip stamps the packing time, so the same
    tree packs differently on every build. Hashing the workspace-relative layout
    with the file bytes gives a name a later conversation can match before it
    uploads anything, and one the host can compare without unpacking.
    """
    digest = hashlib.sha256()
    for destination, local_path in files:
        digest.update(_resource_arcname(destination, workspace_path).encode("utf-8"))
        digest.update(b"\0")
        with open(local_path, "rb") as source:
            for chunk in iter(lambda: source.read(1 << 20), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()[:32]


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
        self._warmups: dict[str, asyncio.Task[None]] = {}
        self._prewarms: dict[str, asyncio.Task[None]] = {}

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

    def prewarm(self, context: ToolExecutionContext, files: PiSessionFiles) -> None:
        """Start the execution plane while the model reads the first message.

        The container, the workspace layout, and the resource staging all happen
        before the first tool call needs them, so that call finds a ready
        endpoint instead of paying for the whole preparation. Failures only log:
        the first touch takes the same path and reports its own error.
        """
        if os.getenv(_PREWARM_SWITCH, "").strip() == "0":
            return
        conversation_id = context.conversation_id
        running = self._prewarms.get(conversation_id)
        if running is not None and not running.done():
            return
        task = asyncio.create_task(self._prewarm(context, files))
        self._prewarms[conversation_id] = task
        task.add_done_callback(
            lambda finished: self._forget_prewarm(conversation_id, finished)
        )

    async def delete(
        self, context: ToolExecutionContext, files: PiSessionFiles
    ) -> None:
        self._cancel_warmup(context.conversation_id)
        self._cancel_prewarm(context.conversation_id)
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
        self._cancel_warmup(context.conversation_id)
        self._cancel_prewarm(context.conversation_id)
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
                self._cancel_warmup(context.conversation_id)
                self._cancel_prewarm(context.conversation_id)
                self._states.pop(context.conversation_id, None)
            pending_staging = files.load_pending_sandbox_staging()
            existing = self._states.get(context.conversation_id)
            if existing is not None and pending_staging is None:
                return existing
            state = await self._ensure_state_locked(context, files, pending_staging)
            self._states[context.conversation_id] = state
            return state

    async def _ensure_state_locked(
        self,
        context: ToolExecutionContext,
        files: PiSessionFiles,
        pending_staging: dict[str, Any] | None,
    ) -> _SandboxState:
        settings = SandboxSettings.from_extra(context.extra)
        record = files.load_sandbox() or {}
        client = await asyncio.to_thread(self._client_factory, context)
        ws_base_url = terminal_base_url(
            cluster=client.cluster,
            env=_execution_target(context)[0],
            fallback=client.base_url,
        )
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
            sandbox = await self._start(client, sandbox)
            prepare = True
        elif sandbox.status.strip().lower() in _STARTING_STATUSES:
            # A sandbox whose pod is still starting (for example a resume that
            # is already in flight) must be waited for, not deleted.
            sandbox = await self._start(client, sandbox)
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

        if (
            not prepare
            and not record_needs_prepare
            and not await self._reachable(client, sandbox.id, ws_base_url)
        ):
            # The control plane reports the container running even after its
            # terminal port stops answering, so a reuse that is never checked
            # turns every later call into a wait for a connection that will not
            # open. Rebuild it here instead.
            logger.warning(
                "Recreating Pi sandbox whose terminal stopped answering "
                "conversation_id=%s sandbox_id=%s",
                context.conversation_id,
                sandbox.id,
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

        if pending_staging is not None:
            await self._apply_pending_staging(
                client,
                sandbox.id,
                ws_base_url,
                mount_path,
                workspace_path,
                pending_staging,
            )
            files.clear_pending_sandbox_staging()
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
                client, sandbox.id, ws_base_url, mount_path, workspace_path
            )
            self._start_dataflow_warmup(
                context.conversation_id,
                client,
                sandbox.id,
                ws_base_url,
                workspace_path,
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
                "Pi sandbox lookup failed; recreating sandbox_id=%s error=%s: %s",
                sandbox_id,
                type(exc).__name__,
                exc,
            )
            return None

    async def _reachable(
        self, client: SandboxClientLike, sandbox_id: str, ws_base_url: str
    ) -> bool:
        """Whether the sandbox's terminal answers right now.

        The control plane keeps reporting a container as running after its data
        plane stops accepting connections, so a recorded sandbox has to be
        asked, not assumed. Anything but a clean exit means "not usable" and the
        caller rebuilds it.
        """
        try:
            _, exit_code, timed_out = await asyncio.wait_for(
                asyncio.to_thread(
                    run_terminal_command,
                    base_url=ws_base_url,
                    api_key=client.api_key,
                    sandbox_id=sandbox_id,
                    command="true",
                    timeout_seconds=_PROBE_TIMEOUT_SECONDS,
                ),
                _PROBE_TIMEOUT_SECONDS + 5,
            )
        except Exception as error:  # noqa: BLE001 - any failure means "rebuild it"
            logger.info(
                "Pi sandbox probe failed sandbox_id=%s error=%s: %s",
                sandbox_id,
                type(error).__name__,
                error,
            )
            return False
        if timed_out or exit_code != 0:
            logger.info(
                "Pi sandbox probe answered badly sandbox_id=%s exit_code=%s "
                "timed_out=%s",
                sandbox_id,
                exit_code,
                timed_out,
            )
            return False
        return True

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
        return await self._start(client, sandbox)

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

    async def _start(
        self, client: SandboxClientLike, sandbox: SandboxResponse
    ) -> SandboxResponse:
        """Return the sandbox once its pod is actually serving.

        ``SandboxClient.resume`` and the platform's status both lag the pod:
        resume accepts the request before the container exists, so writing the
        workspace right after it fails with ``No Running pod found``.
        """
        if sandbox.status.strip().lower() in _READY_STATUSES:
            return sandbox
        if sandbox.status.strip().lower() in _RESUMABLE_STATUSES:
            await asyncio.to_thread(client.resume, sandbox.id)
        await asyncio.to_thread(
            client.wait_for_sandbox_status,
            sandbox.id,
            target_status="running",
            timeout=_START_TIMEOUT_SECONDS,
            check_interval=_START_POLL_SECONDS,
            intermediate_statuses=sorted(_RESUMABLE_STATUSES | _STARTING_STATUSES),
        )
        latest = await asyncio.to_thread(client.get_sandbox, sandbox.id)
        if latest.status.strip().lower() not in _READY_STATUSES:
            raise RuntimeError(
                f"PI_SANDBOX_NOT_RUNNING: sandbox {sandbox.id} is "
                f"{latest.status or 'unknown'}"
            )
        return latest

    async def _prepare_workspace(
        self,
        client: SandboxClientLike,
        sandbox_id: str,
        ws_base_url: str,
        mount_path: str,
        workspace_path: str,
    ) -> None:
        legacy_link = f"{workspace_path}/{LEGACY_STORAGE_ALIAS}"
        public_data = f"{workspace_path}/public_data"
        script = (
            f"if [ ! -d {shlex.quote(mount_path)} ]; then "
            f"exit {_MOUNT_MISSING_EXIT_CODE}; fi\n"
            f"mkdir -p {shlex.quote(workspace_path)} && "
            f"mkdir -p {shlex.quote(public_data)} && "
            # Drop the alias an older runtime left behind. Only a symlink is
            # removed, so a real directory of that name is never touched.
            f"if [ -L {shlex.quote(legacy_link)} ]; then "
            f"rm -f {shlex.quote(legacy_link)}; fi\n"
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
            write_sandbox_file,
            client,
            ws_base_url=ws_base_url,
            sandbox_id=sandbox_id,
            path=remote_path,
            source=script.encode("utf-8"),
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

    async def _apply_pending_staging(
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
                write_sandbox_file,
                client,
                ws_base_url=ws_base_url,
                sandbox_id=sandbox_id,
                path=workflow_path,
                source=workflow_dsl.encode("utf-8"),
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
        mount_path: str,
        workspace_path: str,
    ) -> None:
        files = self._resource_files(workspace_path)
        total = len(files)
        if not total:
            return
        digest = _resource_digest(files, workspace_path)
        cached = (
            f"{mount_path}{PYROMIND_AGENT_RESOURCES_ROOT}/{digest}/"
            f"{_RESOURCE_ARCHIVE_NAME}"
        )
        reused = await self._resource_archive_cached(
            client, sandbox_id, ws_base_url, cached
        )
        staged: str | None = None
        if not reused:
            logger.info(
                "Uploading Pi session resources sandbox_id=%s files=%d digest=%s",
                sandbox_id,
                total,
                digest,
            )
            # The archive lands under a name of its own and is renamed into
            # place, so a conversation which reads the cache while another one
            # uploads it never opens a half-written archive.
            staged = f"{cached}.partial-{secrets.token_hex(4)}"
            with _resource_archive(files, workspace_path) as archive:
                try:
                    await asyncio.to_thread(
                        write_sandbox_file,
                        client,
                        ws_base_url=ws_base_url,
                        sandbox_id=sandbox_id,
                        path=staged,
                        source=archive,
                    )
                except Exception:
                    await self._remove_sandbox_file(
                        client, sandbox_id, ws_base_url, staged
                    )
                    raise
        script = (
            "set -e\n"
            f"mkdir -p {shlex.quote(workspace_path)}\n"
            + (
                f"mv {shlex.quote(staged)} {shlex.quote(cached)}\n"
                if staged is not None
                else ""
            )
            + f"tar -xzf {shlex.quote(cached)} -C {shlex.quote(workspace_path)}\n"
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
            "Pi session resource upload complete sandbox_id=%s uploaded=%d reused=%s",
            sandbox_id,
            total,
            reused,
        )

    async def _resource_archive_cached(
        self,
        client: SandboxClientLike,
        sandbox_id: str,
        ws_base_url: str,
        cached: str,
    ) -> bool:
        """Whether the Storage mount already holds this resource archive."""
        try:
            _, exit_code, _ = await self._run_script(
                client,
                sandbox_id,
                ws_base_url,
                f"test -f {shlex.quote(cached)}\n",
            )
        except Exception as error:  # noqa: BLE001 - a failed probe re-uploads
            logger.info(
                "Pi session resource cache probe failed; staging again error=%s",
                type(error).__name__,
            )
            return False
        return exit_code == 0

    async def _remove_sandbox_file(
        self,
        client: SandboxClientLike,
        sandbox_id: str,
        ws_base_url: str,
        path: str,
    ) -> None:
        """Best-effort removal; cleanup never replaces the original failure."""
        try:
            await self._run_script(
                client,
                sandbox_id,
                ws_base_url,
                f"rm -f {shlex.quote(path)}\n",
            )
        except Exception as error:  # noqa: BLE001 - see the docstring
            logger.warning(
                "Pi sandbox file cleanup failed path=%s error=%s",
                path,
                type(error).__name__,
            )

    def _start_dataflow_warmup(
        self,
        conversation_id: str,
        client: SandboxClientLike,
        sandbox_id: str,
        ws_base_url: str,
        workspace_path: str,
    ) -> None:
        """Pre-install the DataFlow runtime without holding up the session.

        The venv lives in the sandbox, so it survives for as long as the
        container does and every later run reuses it. A warm-up which fails only
        logs: the run path still installs on demand.
        """
        if os.getenv(_DATAFLOW_WARMUP_SWITCH, "").strip() == "0":
            return
        running = self._warmups.get(conversation_id)
        if running is not None and not running.done():
            return
        command = self._dataflow_warmup_command(workspace_path)
        if command is None:
            return
        task = asyncio.create_task(
            self._warm_dataflow(
                conversation_id, client, sandbox_id, ws_base_url, command
            )
        )
        self._warmups[conversation_id] = task
        task.add_done_callback(
            lambda finished: self._forget_warmup(conversation_id, finished)
        )

    async def _warm_dataflow(
        self,
        conversation_id: str,
        client: SandboxClientLike,
        sandbox_id: str,
        ws_base_url: str,
        command: str,
    ) -> None:
        try:
            output, exit_code, timed_out = await asyncio.to_thread(
                run_terminal_command,
                base_url=ws_base_url,
                api_key=client.api_key,
                sandbox_id=sandbox_id,
                command=command,
                timeout_seconds=_DATAFLOW_WARMUP_TIMEOUT_SECONDS,
            )
        except asyncio.CancelledError:
            raise
        except Exception as error:  # noqa: BLE001 - the run path still installs
            logger.warning(
                "Pi DataFlow warm-up failed conversation_id=%s error=%s",
                conversation_id,
                type(error).__name__,
            )
            return
        if exit_code != 0:
            logger.warning(
                "Pi DataFlow warm-up did not finish conversation_id=%s%s",
                conversation_id,
                _failure_detail(output, exit_code, timed_out),
            )
            return
        logger.info(
            "Pi DataFlow environment ready conversation_id=%s report=%s",
            conversation_id,
            output.strip()[-_FAILURE_OUTPUT_CHARS:],
        )

    def _dataflow_warmup_command(self, workspace_path: str) -> str | None:
        """One-line command preparing the DataFlow venv, when the skill is staged."""
        for destination, source in self._resource_roots:
            if destination != _SKILLS_ALIAS:
                continue
            script = Path(source) / _DATAFLOW_SKILL / _DATAFLOW_PREPARE_SCRIPT
            if not script.is_file():
                continue
            remote = (
                f"{workspace_path}/{_SKILLS_ALIAS}/{_DATAFLOW_SKILL}/"
                f"{_DATAFLOW_PREPARE_SCRIPT}"
            )
            values = _dataflow_warmup_env()
            system_python = (
                values.get("PYROMIND_SANDBOX_SYSTEM_PYTHON")
                or _DEFAULT_SANDBOX_SYSTEM_PYTHON
            )
            # ``env`` keeps the overrides as plain assignments instead of
            # quoting each ``NAME=value`` pair into a command of its own.
            prepare = " ".join(
                [
                    "env",
                    *(f"{name}={shlex.quote(value)}" for name, value in values.items()),
                    shlex.quote(system_python),
                    shlex.quote(remote),
                    "--prepare",
                ]
            )
            # The install runs while the session is already serving commands, so
            # it yields both CPU and disk to them instead of making the first
            # interactive command wait behind a pip install. `ionice` is
            # optional because a container may refuse the idle class, so the
            # guard falls back to `nice` alone.
            throttled = "nice -n 19 sh -c " + shlex.quote(
                "if command -v ionice >/dev/null 2>&1 && "
                "ionice -c 3 -n 7 true 2>/dev/null; then "
                f"exec ionice -c 3 -n 7 {prepare}; "
                f"else exec {prepare}; fi"
            )
            # The terminal bridge drops an idle connection and a first install is
            # silent for minutes, so the install runs in the background and the
            # foreground keeps printing progress until the status file appears.
            status = f"/tmp/pi-dataflow-warmup-{secrets.token_hex(8)}.rc"
            quoted_status = shlex.quote(status)
            return (
                f"({throttled}; printf '%s' \"$?\" > {quoted_status}) & "
                f"pm_warm_rc=''; "
                f'while [ -z "$pm_warm_rc" ]; do '
                f"sleep {_DATAFLOW_WARMUP_HEARTBEAT_SECONDS}; "
                f"pm_warm_rc=$(cat {quoted_status} 2>/dev/null); "
                f"[ -n \"$pm_warm_rc\" ] || printf '%s\\n' '...'; done; "
                f'rm -f {quoted_status}; exit "$pm_warm_rc"'
            )
        return None

    def _cancel_warmup(self, conversation_id: str) -> None:
        task = self._warmups.pop(conversation_id, None)
        if task is not None:
            task.cancel()

    def _forget_warmup(self, conversation_id: str, task: asyncio.Task[None]) -> None:
        if self._warmups.get(conversation_id) is task:
            self._warmups.pop(conversation_id, None)

    async def _prewarm(
        self, context: ToolExecutionContext, files: PiSessionFiles
    ) -> None:
        try:
            await self.ensure(context, files)
        except asyncio.CancelledError:
            raise
        except Exception as error:  # noqa: BLE001 - the first touch retries
            logger.warning(
                "Pi sandbox pre-warm failed conversation_id=%s error=%s: %s",
                context.conversation_id,
                type(error).__name__,
                error,
            )

    def _cancel_prewarm(self, conversation_id: str) -> None:
        task = self._prewarms.pop(conversation_id, None)
        if task is not None:
            task.cancel()

    def _forget_prewarm(self, conversation_id: str, task: asyncio.Task[None]) -> None:
        if self._prewarms.get(conversation_id) is task:
            self._prewarms.pop(conversation_id, None)

    def _resource_files(self, workspace_path: str) -> list[tuple[str, str]]:
        """Local resource files paired with the sandbox path each one targets."""
        files: list[tuple[str, str]] = []
        for destination_root, source_root in self._resource_roots:
            source = os.fspath(source_root)
            if not os.path.isdir(source):
                continue
            for directory, directories, names in os.walk(source):
                directories[:] = [
                    name for name in directories if name not in _SKIPPED_DIRECTORIES
                ]
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
    auth_token = _auth_token(context.request_context)
    if not auth_token:
        raise ValueError(
            "Sandbox execution requires a Pyromind auth_token cookie or "
            "Bearer authorization."
        )
    api = create_sandbox_api_client(
        env=env,
        cluster=cluster,
        auth_token=auth_token,
        headers=_forward_headers(context.request_context, include_cookie=False),
    )
    return api.sandboxes
