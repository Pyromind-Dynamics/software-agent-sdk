from __future__ import annotations

import asyncio
import logging
import posixpath
import re
import shlex
import stat
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from pyromind_runtime.adapters.pi.adapter import PiRunnerLaunch, PiRunnerLauncher
from pyromind_runtime.adapters.pi.callbacks import (
    PiRunnerCallbackRouter,
    PiWorkflowMirror,
)
from pyromind_runtime.adapters.pi.sandbox import BoundedSandboxGateway
from pyromind_runtime.contracts.content import JsonObject
from pyromind_runtime.contracts.harness import SessionSpec
from pyromind_runtime.contracts.sandbox import SandboxExecutionBackend
from pyromind_runtime.tool_host import PythonToolHost


type SandboxLeaseRelease = Callable[[], Awaitable[None]]


logger = logging.getLogger(__name__)

_SKILL_NAME = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_SKILL_REMOTE_ROOT = ".pyromind/skills"
_SKILL_IGNORED_PARTS = frozenset({"__pycache__", ".DS_Store"})
_MAX_SKILL_FILE_BYTES = 4 * 1024 * 1024
_MAX_SKILLS_TOTAL_BYTES = 32 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class PiSandboxLease:
    sandbox_id: str
    workspace_root: str
    backend: SandboxExecutionBackend
    release: SandboxLeaseRelease
    environment: Mapping[str, str]


class PiSandboxLeaseProvider(Protocol):
    async def acquire(self, spec: SessionSpec) -> PiSandboxLease: ...


class ManagedSandboxExecutionBackend(SandboxExecutionBackend, Protocol):
    async def create_sandbox(self, request: JsonObject) -> str: ...

    async def wait_until_running(
        self,
        sandbox_id: str,
        *,
        timeout_seconds: float = 300.0,
    ) -> None: ...

    async def delete_sandbox(self, sandbox_id: str) -> None: ...


type ManagedSandboxBackendFactory = Callable[
    [SessionSpec], ManagedSandboxExecutionBackend
]


class PyromindSandboxLeaseProvider:
    def __init__(
        self,
        backend_factory: ManagedSandboxBackendFactory,
        *,
        create_request: JsonObject,
        workspace_root: str = "/workspace",
        environment: Mapping[str, str] | None = None,
        ready_timeout_seconds: float = 300.0,
    ) -> None:
        self._backend_factory = backend_factory
        self._create_request = dict(create_request)
        self._workspace_root = workspace_root
        self._environment = dict(environment or {})
        self._ready_timeout_seconds = ready_timeout_seconds

    async def acquire(self, spec: SessionSpec) -> PiSandboxLease:
        backend = self._backend_factory(spec)
        sandbox_id: str | None = None
        request = dict(self._create_request)
        request.setdefault("name", f"pi-{spec.product_session_id[:40]}")
        try:
            sandbox_id = await backend.create_sandbox(request)
            await backend.wait_until_running(
                sandbox_id,
                timeout_seconds=self._ready_timeout_seconds,
            )
            await backend.create_dir(
                sandbox_id,
                self._workspace_root,
                recursive=True,
            )
        except Exception:
            if sandbox_id is not None:
                try:
                    await backend.delete_sandbox(sandbox_id)
                except Exception:
                    pass
            await backend.close()
            raise

        release_lock = asyncio.Lock()
        released = False

        async def release() -> None:
            nonlocal released
            async with release_lock:
                if released:
                    return
                released = True
                await backend.delete_sandbox(sandbox_id)

        return PiSandboxLease(
            sandbox_id=sandbox_id,
            workspace_root=self._workspace_root,
            backend=backend,
            release=release,
            environment=self._environment,
        )


class PiModelConfigResolver(Protocol):
    async def resolve(self, spec: SessionSpec) -> JsonObject: ...


class StaticPiModelConfigResolver:
    def __init__(
        self,
        *,
        profile_id: str,
        provider: str,
        model_id: str,
        api_key: str,
        base_url: str | None = None,
        thinking_level: str = "off",
    ) -> None:
        self._profile_id = profile_id
        self._config: JsonObject = {
            "provider": provider,
            "id": model_id,
            "api_key": api_key,
            "thinking_level": thinking_level,
        }
        if base_url is not None:
            self._config["base_url"] = base_url

    async def resolve(self, spec: SessionSpec) -> JsonObject:
        if spec.model_profile.profile_id != self._profile_id:
            raise ValueError(
                f"Pi model profile is not configured: {spec.model_profile.profile_id}"
            )
        return dict(self._config)


class LocalPiRunnerLauncher(PiRunnerLauncher):
    """Launch Pi with its local NodeExecutionEnv instead of a remote sandbox."""

    def __init__(
        self,
        *,
        command: tuple[str, ...],
        workspace_root: str | Path,
        model_resolver: PiModelConfigResolver,
        tool_host: PythonToolHost,
        system_prompt: str,
        skill_directories: tuple[str | Path, ...] = (),
        cwd: str | None = None,
        environment: Mapping[str, str] | None = None,
    ) -> None:
        self._command = command
        self._workspace_root = Path(workspace_root)
        self._model_resolver = model_resolver
        self._tool_host = tool_host
        self._system_prompt = system_prompt
        self._skill_directories = tuple(Path(path) for path in skill_directories)
        self._cwd = cwd
        self._environment = dict(environment or {})

    async def prepare(self, spec: SessionSpec) -> PiRunnerLaunch:
        workspace_root = await asyncio.to_thread(self._prepare_workspace, spec)
        model = await self._model_resolver.resolve(spec)
        router = PiRunnerCallbackRouter(spec, None, self._tool_host)
        skill_dirs = _local_skill_directories(self._skill_directories)
        return PiRunnerLaunch(
            command=self._command,
            cwd=self._cwd,
            environment=self._environment,
            runtime_config={
                "model": model,
                "system_prompt": self._system_prompt,
                "workspace_root": workspace_root,
                "execution_env": "node",
                "native_tools": ["read", "write", "edit", "bash"],
                "skill_dirs": list(skill_dirs),
            },
            request_handler=router.handle,
            cleanup=router.close,
        )

    def _prepare_workspace(self, spec: SessionSpec) -> str:
        configured_root = self._workspace_root.resolve()
        configured_root.mkdir(mode=0o700, parents=True, exist_ok=True)
        configured_root = configured_root.resolve()

        requested = Path(spec.workspace.root)
        if not requested.is_absolute():
            requested = Path.cwd() / requested
        requested = requested.resolve(strict=False)
        if requested == configured_root or not requested.is_relative_to(
            configured_root
        ):
            raise ValueError(
                f"Pi workspace must be a session directory under {configured_root}"
            )
        requested.mkdir(mode=0o700, parents=True, exist_ok=True)
        resolved = requested.resolve()
        if resolved != requested:
            raise ValueError("Pi workspace must not be a symlink")
        return str(resolved)


class SandboxedPiRunnerLauncher(PiRunnerLauncher):
    def __init__(
        self,
        *,
        command: tuple[str, ...],
        lease_provider: PiSandboxLeaseProvider,
        model_resolver: PiModelConfigResolver,
        tool_host: PythonToolHost,
        system_prompt: str,
        skill_directories: tuple[str | Path, ...] = (),
        cwd: str | None = None,
        environment: Mapping[str, str] | None = None,
    ) -> None:
        self._command = command
        self._lease_provider = lease_provider
        self._model_resolver = model_resolver
        self._tool_host = tool_host
        self._system_prompt = system_prompt
        self._skill_directories = tuple(Path(path) for path in skill_directories)
        self._cwd = cwd
        self._environment = dict(environment or {})

    async def prepare(self, spec: SessionSpec) -> PiRunnerLaunch:
        lease = await self._lease_provider.acquire(spec)
        try:
            gateway = BoundedSandboxGateway(
                lease.backend,
                sandbox_id=lease.sandbox_id,
                workspace_root=lease.workspace_root,
                environment=lease.environment,
            )
            router = PiRunnerCallbackRouter(
                spec,
                gateway,
                self._tool_host,
                workflow_mirror=PiWorkflowMirror(
                    remote_workspace_root=lease.workspace_root,
                    local_workspace_root=spec.workspace.root,
                ),
            )
            skill_dirs = await _publish_skills(
                lease,
                self._skill_directories,
            )
            model = await self._model_resolver.resolve(spec)
        except Exception:
            try:
                await lease.release()
            finally:
                await lease.backend.close()
            raise

        async def cleanup() -> None:
            try:
                await lease.release()
            finally:
                await router.close()

        return PiRunnerLaunch(
            command=self._command,
            cwd=self._cwd,
            environment=self._environment,
            runtime_config={
                "model": model,
                "system_prompt": self._system_prompt,
                "workspace_root": lease.workspace_root,
                "execution_env": "remote",
                "native_tools": ["read", "write", "edit", "bash"],
                "skill_dirs": list(skill_dirs),
            },
            request_handler=router.handle,
            cleanup=cleanup,
        )


def _local_skill_directories(
    skill_directories: tuple[Path, ...],
) -> tuple[str, ...]:
    resolved: list[str] = []
    names: set[str] = set()
    for source in skill_directories:
        source = source.resolve()
        name = source.name
        if not _SKILL_NAME.fullmatch(name):
            raise ValueError(f"Invalid Pi skill directory name: {name}")
        if name in names:
            raise ValueError(f"Duplicate Pi skill directory: {name}")
        if not source.is_dir() or not (source / "SKILL.md").is_file():
            raise FileNotFoundError(f"Pi skill is missing SKILL.md: {source}")
        names.add(name)
        resolved.append(str(source))
    return tuple(resolved)


async def _publish_skills(
    lease: PiSandboxLease,
    skill_directories: tuple[Path, ...],
) -> tuple[str, ...]:
    """Copy trusted, configured skill resources into the leased workspace."""
    if not skill_directories:
        return ()

    remote_root = posixpath.join(lease.workspace_root, _SKILL_REMOTE_ROOT)
    await lease.backend.create_dir(
        lease.sandbox_id,
        remote_root,
        recursive=True,
    )
    total_bytes = 0
    executable_paths: list[str] = []
    published_names: set[str] = set()

    for source in skill_directories:
        source = source.resolve()
        name = source.name
        if not _SKILL_NAME.fullmatch(name):
            raise ValueError(f"Invalid Pi skill directory name: {name}")
        if name in published_names:
            raise ValueError(f"Duplicate Pi skill directory: {name}")
        if not source.is_dir() or not (source / "SKILL.md").is_file():
            raise FileNotFoundError(f"Pi skill is missing SKILL.md: {source}")
        published_names.add(name)

        for local_path in sorted(source.rglob("*")):
            relative = local_path.relative_to(source)
            if _ignore_skill_path(relative):
                continue
            if local_path.is_symlink():
                raise ValueError(
                    f"Pi skill resources must not be symlinks: {local_path}"
                )
            if local_path.is_dir():
                continue
            if not local_path.is_file():
                raise ValueError(f"Unsupported Pi skill resource: {local_path}")

            content = local_path.read_bytes()
            if len(content) > _MAX_SKILL_FILE_BYTES:
                raise ValueError(f"Pi skill resource is too large: {local_path}")
            total_bytes += len(content)
            if total_bytes > _MAX_SKILLS_TOTAL_BYTES:
                raise ValueError("Pi skill resources exceed the total size limit")

            remote_path = posixpath.join(
                remote_root,
                name,
                *relative.parts,
            )
            await lease.backend.write_file(
                lease.sandbox_id,
                remote_path,
                content,
            )
            if local_path.stat().st_mode & stat.S_IXUSR:
                executable_paths.append(remote_path)

    if executable_paths:
        command = "chmod u+x " + " ".join(map(shlex.quote, executable_paths))
        result = await lease.backend.exec(
            lease.sandbox_id,
            command,
            cwd=lease.workspace_root,
            environment=lease.environment,
            timeout_seconds=30,
        )
        if result.exit_code != 0:
            raise RuntimeError("Failed to preserve executable Pi skill resources")

    logger.info(
        "Published %d Pi skills (%d bytes) into %s",
        len(published_names),
        total_bytes,
        remote_root,
    )
    return (remote_root,)


def _ignore_skill_path(path: Path) -> bool:
    return path.suffix == ".pyc" or any(
        part in _SKILL_IGNORED_PARTS for part in path.parts
    )
