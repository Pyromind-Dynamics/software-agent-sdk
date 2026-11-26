from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, Never, Protocol
from uuid import UUID

from pyromind_sdk.client.models import (
    SandboxExecResponse,
    SandboxResponse,
    SandboxType,
)

from pyromind_agent_server.echomind_projects import ProjectStore
from pyromind_agent_server.echomind_runtime_access import (
    platform_request_token,
    refresh_runtime_access,
)
from pyromind_agent_server.echomind_runtime_credentials import RuntimeCredentialStore
from pyromind_agent_server.evaluation_storage import (
    REPORT_MAX_BYTES,
    build_evaluation_storage,
)
from pyromind_common.tools.utils.platform_env import (
    platform_cluster_route,
    platform_cluster_target,
)
from pyromind_common.tools.utils.pyromind_api_client import (
    get_api_key,
    get_pyromind_api_client,
)
from pyromind_runtime.application.project_storage import ProjectStorageBindings
from pyromind_runtime.domain.context import RequestContext
from pyromind_runtime.domain.project_storage import (
    ProjectStorageBinding,
    StorageProxyConfig,
)
from pyromind_runtime.domain.storage_paths import to_storage_api_path
from pyromind_runtime.infrastructure.file_project_storage import (
    FileProjectStorageBindingStore,
)
from pyromind_runtime.ports.project_storage import StorageBindingBlocked
from pyromind_runtime.ports.storage import (
    EvaluationStorage,
    EvaluationStorageMissingError,
    EvaluationStorageUnavailableError,
)


_PROBE_SCRIPT = """
import json, os, pathlib, sys
root, target, operation = sys.argv[1:]
try:
    mount = pathlib.Path(root)
    root = mount.resolve(strict=True)
    requested = root / pathlib.Path(target).relative_to(mount)
    target = requested.resolve(strict=False)
    if not root.is_dir() or not target.is_relative_to(root):
        result = {'error': 'outside_mount'}
    elif operation == 'write':
        if requested != target or target.is_dir():
            result = {'error': 'unsafe_write_target'}
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            result = {'ok': True}
    elif not target.exists():
        result = {'error': 'missing'}
    elif operation == 'mount':
        result = {'ok': target.is_dir()}
    elif operation == 'list':
        if not target.is_dir():
            result = {'error': 'not_directory'}
        else:
            names = []
            with os.scandir(target) as entries:
                for entry in entries:
                    names.append(entry.name)
                    if len(names) > 10000:
                        raise ValueError('listing limit')
            result = {'names': names}
    elif target.is_file():
        result = {'size': target.stat().st_size}
    else:
        result = {'error': 'not_file'}
except FileNotFoundError:
    result = {'error': 'missing'}
except Exception:
    result = {'error': 'unavailable'}
print(json.dumps(result))
"""


class SandboxStorageClient(Protocol):
    def get_sandbox(self, sandbox_id: str) -> SandboxResponse: ...

    def exec_command(
        self, sandbox_id: str, command: list[str], *, timeout: int
    ) -> SandboxExecResponse: ...

    def read_file(self, sandbox_id: str, path: str) -> bytes: ...

    def write_file(
        self, sandbox_id: str, path: str, source: bytes
    ) -> dict[str, Any]: ...


class _ScopedSandboxClient:
    def __init__(
        self, factory: Callable[[], AbstractContextManager[SandboxStorageClient]]
    ) -> None:
        self.factory = factory

    def get_sandbox(self, sandbox_id: str) -> SandboxResponse:
        with self.factory() as client:
            return client.get_sandbox(sandbox_id)

    def exec_command(
        self, sandbox_id: str, command: list[str], *, timeout: int
    ) -> SandboxExecResponse:
        with self.factory() as client:
            return client.exec_command(sandbox_id, command, timeout=timeout)

    def read_file(self, sandbox_id: str, path: str) -> bytes:
        with self.factory() as client:
            return client.read_file(sandbox_id, path)

    def write_file(self, sandbox_id: str, path: str, source: bytes) -> dict[str, Any]:
        with self.factory() as client:
            return client.write_file(sandbox_id, path, source)


def _require_sandbox(
    client: SandboxStorageClient,
    config: StorageProxyConfig,
    *,
    require_writable: bool = True,
) -> None:
    try:
        sandbox = client.get_sandbox(config.sandbox_id)
    except Exception:
        raise StorageBindingBlocked(
            "The bound Sandbox is unavailable in this Project cluster; "
            "check its existence and owner access."
        ) from None
    if sandbox.id != config.sandbox_id or sandbox.type != SandboxType.CUSTOM:
        raise StorageBindingBlocked(
            "Storage requires the explicitly bound CUSTOM Sandbox."
        )
    if sandbox.status.lower() != "running":
        raise StorageBindingBlocked(
            "The bound Storage Sandbox is not running. "
            "Restore it or explicitly bind another Sandbox."
        )
    mount = next(
        (
            item
            for item in sandbox.volume_mounts or []
            if item.mount_path.rstrip("/") == config.mount_root
        ),
        None,
    )
    if mount is None or mount.host_path.rstrip("/") != "/workspace":
        raise StorageBindingBlocked(
            "Sandbox must mount the owner's /workspace Storage."
        )
    if require_writable and mount.read_only:
        raise StorageBindingBlocked(
            "Sandbox must mount the owner's /workspace Storage read-write "
            "at the configured mount root."
        )


def _probe(
    client: SandboxStorageClient,
    config: StorageProxyConfig,
    path: str,
    operation: str,
) -> dict[str, Any]:
    try:
        response = client.exec_command(
            config.sandbox_id,
            ["python3", "-c", _PROBE_SCRIPT, config.mount_root, path, operation],
            timeout=20,
        )
        if response.returncode != 0 or len(response.output) > 2 * 1024 * 1024:
            raise ValueError("Invalid metadata response")
        data = json.loads(response.output)
        if not isinstance(data, dict):
            raise ValueError("Invalid metadata response")
    except Exception:
        raise EvaluationStorageUnavailableError(
            "Storage Sandbox metadata is unavailable; check owner access, "
            "Python and the server TLS trust configuration."
        ) from None
    if data.get("error") == "missing":
        raise EvaluationStorageMissingError("Storage object does not exist")
    if data.get("error"):
        raise EvaluationStorageUnavailableError(
            "Storage object is inaccessible or resolves outside the bound mount"
        )
    return data


class SandboxBindingVerifier:
    def __init__(
        self, cluster: str, client_factory: Callable[[], SandboxStorageClient]
    ) -> None:
        self.cluster = cluster
        self.client_factory = client_factory

    def verify(
        self, config: StorageProxyConfig, *, require_writable: bool = True
    ) -> None:
        client = self.client_factory()
        _require_sandbox(client, config, require_writable=require_writable)
        result = _probe(client, config, config.mount_root, "mount")
        if result.get("ok") is not True:
            raise StorageBindingBlocked("The configured Storage mount is unavailable")


class SandboxEvaluationStorage:
    def __init__(
        self,
        *,
        client: SandboxStorageClient,
        binding: ProjectStorageBinding,
        write_prefixes: tuple[str, ...] = (),
        max_read_bytes: int = 8 * 1024 * 1024,
        on_unavailable: Callable[[str], None] | None = None,
        require_writable_mount: bool = True,
    ) -> None:
        self.client = client
        self.binding = binding
        self.write_prefixes = tuple(to_storage_api_path(p) for p in write_prefixes)
        self.max_read_bytes = max_read_bytes
        self.on_unavailable = on_unavailable
        self.require_writable_mount = require_writable_mount

    def _path(self, path: str) -> str:
        return self.binding.config.mount_root + to_storage_api_path(path)

    def _check(self, path: str, operation: str) -> dict[str, Any]:
        try:
            _require_sandbox(
                self.client,
                self.binding.config,
                require_writable=self.require_writable_mount,
            )
            return _probe(self.client, self.binding.config, path, operation)
        except StorageBindingBlocked as exc:
            self._unavailable(str(exc))
        except EvaluationStorageUnavailableError as exc:
            self._unavailable(str(exc))

    def _unavailable(self, message: str) -> Never:
        if self.on_unavailable is not None:
            self.on_unavailable(message)
        raise EvaluationStorageUnavailableError(message)

    def read(self, path: str) -> bytes:
        mapped = self._path(path)
        metadata = self._check(mapped, "read")
        size = metadata.get("size")
        if type(size) is not int or not 0 <= size <= self.max_read_bytes:
            raise EvaluationStorageUnavailableError(
                "Storage object exceeds the read limit"
            )
        try:
            payload = self.client.read_file(self.binding.config.sandbox_id, mapped)
        except Exception:
            self._unavailable(
                "Storage Sandbox read failed; check owner access and status"
            )
        if len(payload) != size or len(payload) > self.max_read_bytes:
            raise EvaluationStorageUnavailableError(
                "Storage object changed while reading"
            )
        return payload

    def write(self, path: str, payload: bytes) -> None:
        storage_path = to_storage_api_path(path)
        if not any(
            PurePosixPath(storage_path).is_relative_to(prefix)
            and storage_path != prefix
            for prefix in self.write_prefixes
        ):
            raise EvaluationStorageUnavailableError(
                "Storage writes are restricted to this Project's batch artifacts"
            )
        if len(payload) > self.max_read_bytes:
            raise EvaluationStorageUnavailableError(
                "Storage payload exceeds the write limit"
            )
        mapped = self._path(path)
        self._check(mapped, "write")
        try:
            result = self.client.write_file(
                self.binding.config.sandbox_id, mapped, payload
            )
        except Exception:
            result = None

        # The platform can write the file before its upload response fails.
        try:
            stored = self.read(path)
        except EvaluationStorageMissingError:
            stored = None
        if stored == payload:
            return
        if result is None:
            self._unavailable("Storage Sandbox write failed; its result is unconfirmed")
        if result.get("path") != mapped or result.get("size") != len(payload):
            raise EvaluationStorageUnavailableError(
                "Storage write acknowledgment is invalid"
            )
        raise EvaluationStorageUnavailableError("Storage write verification failed")

    def list(self, prefix: str) -> set[str]:
        result = self._check(self._path(prefix), "list")
        names = result.get("names")
        if not isinstance(names, list) or any(
            not isinstance(name, str)
            or name in {"", ".", ".."}
            or "/" in name
            or "\\" in name
            for name in names
        ):
            raise EvaluationStorageUnavailableError(
                "Storage directory listing is invalid"
            )
        return set(names)


def build_project_storage_bindings(
    *,
    projects_root: Path | str,
    project_id: str,
    context: RequestContext,
    credential_store: RuntimeCredentialStore | None,
) -> ProjectStorageBindings:
    projects = ProjectStore(Path(projects_root))
    project = projects.load(project_id, owner_user_id=context.user_id)
    if context.x_cluster:
        projects.require_cluster(project, context.x_cluster)
    if not project.project.cluster:
        raise ValueError("Select the Project cluster before binding Storage")
    route = platform_cluster_route(project.project.cluster)

    @contextmanager
    def sdk_client() -> Iterator[SandboxStorageClient]:
        refresh_runtime_access(credential_store, project, context)
        key = (
            credential_store.load(context.user_id, route) if credential_store else None
        )
        env, _ = platform_cluster_target(route)
        if not key:
            token = platform_request_token(context)
            if not token:
                raise StorageBindingBlocked(
                    "Current owner platform credentials are required"
                )
            headers = {
                name: value
                for name, value in (
                    ("cookie", context.cookie),
                    ("authorization", context.authorization),
                    ("x-cluster", route),
                )
                if value
            }
            key = get_api_key(env, token, headers, timeout=5)
        with get_pyromind_api_client(
            env=env, cluster=route, api_key=key, max_retries=0
        ) as client:
            yield client.sandboxes

    def client_factory() -> SandboxStorageClient:
        return _ScopedSandboxClient(sdk_client)

    def verifier_factory(requested_project: str, owner: str) -> SandboxBindingVerifier:
        if UUID(requested_project) != UUID(project_id) or owner != context.user_id:
            raise PermissionError("Storage binding context belongs to another Project")
        return SandboxBindingVerifier(route, client_factory)

    return ProjectStorageBindings(
        FileProjectStorageBindingStore(projects_root), verifier_factory
    )


def build_bound_evaluation_storage(
    *,
    projects_root: Path | str,
    project_id: str,
    context: RequestContext,
    credential_store: RuntimeCredentialStore | None,
    max_read_bytes: int = 8 * 1024 * 1024,
    write_prefixes: tuple[str, ...] = (),
    acquisition_simulation_id: str | None = None,
    development_proxy: StorageProxyConfig | None = None,
) -> EvaluationStorage | None:
    if write_prefixes or acquisition_simulation_id is not None:
        project = ProjectStore(Path(projects_root)).load(
            project_id, owner_user_id=context.user_id
        )
        conversation = UUID(project.project.conversation_id).hex
        allowed = PurePosixPath(f"/.pyromind-agent/{conversation}/project-batches")
        if any(
            not PurePosixPath(to_storage_api_path(prefix)).is_relative_to(allowed)
            for prefix in write_prefixes
        ):
            raise ValueError(
                "Storage write prefix must belong to this Project's batches"
            )
        if acquisition_simulation_id is not None:
            simulation = UUID(acquisition_simulation_id)
            write_prefixes = (
                *write_prefixes,
                f"/.pyromind-agent/{conversation}/tracer-batches/{simulation}",
            )
    service = build_project_storage_bindings(
        projects_root=projects_root,
        project_id=project_id,
        context=context,
        credential_store=credential_store,
    )
    if service.get(project_id, context.user_id) is None:
        if development_proxy is None:
            return None
        if write_prefixes:
            raise ValueError("A development Storage proxy is read-only.")
        verifier = service.verifier_factory(project_id, context.user_id)
        assert isinstance(verifier, SandboxBindingVerifier)
        try:
            verifier.verify(development_proxy, require_writable=False)
        except StorageBindingBlocked as exc:
            raise EvaluationStorageUnavailableError(str(exc)) from exc
        return SandboxEvaluationStorage(
            client=verifier.client_factory(),
            binding=ProjectStorageBinding(
                project_id=project_id,
                owner_user_id=context.user_id,
                cluster=verifier.cluster,
                config=development_proxy,
                status="ready",
                last_checked_at=datetime.now(UTC),
            ),
            max_read_bytes=max_read_bytes,
            require_writable_mount=False,
        )
    binding = service.refresh(project_id, context.user_id)
    if binding.status != "ready":
        raise EvaluationStorageUnavailableError(
            binding.last_error or "Project Storage binding is not ready."
        )
    verifier = service.verifier_factory(project_id, context.user_id)
    assert isinstance(verifier, SandboxBindingVerifier)

    def mark_blocked(message: str) -> None:
        with service.store.lease(project_id):
            current = service.get(project_id, context.user_id)
            if current is not None and current.config == binding.config:
                service.store.save(
                    current.model_copy(
                        update={
                            "status": "blocked",
                            "last_error": message,
                            "last_checked_at": datetime.now(UTC),
                        }
                    )
                )

    return SandboxEvaluationStorage(
        client=verifier.client_factory(),
        binding=binding,
        write_prefixes=write_prefixes,
        max_read_bytes=max_read_bytes,
        on_unavailable=mark_blocked,
    )


def build_acquisition_storage(
    *,
    projects_root: Path | str,
    project_id: str,
    simulation_id: str,
    context: RequestContext,
    credential_store: RuntimeCredentialStore | None,
) -> EvaluationStorage:
    storage = build_bound_evaluation_storage(
        projects_root=projects_root,
        project_id=project_id,
        acquisition_simulation_id=simulation_id,
        context=context,
        credential_store=credential_store,
        max_read_bytes=REPORT_MAX_BYTES,
    )
    return storage if storage is not None else build_evaluation_storage(context)
