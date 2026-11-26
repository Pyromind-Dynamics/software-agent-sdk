from contextlib import AbstractContextManager
from typing import Protocol

from pyromind_runtime.domain.project_storage import (
    ProjectStorageBinding,
    StorageProxyConfig,
)


class StorageBindingBlocked(RuntimeError):
    """A credential-free reason why the explicitly bound Sandbox is unusable."""


class StorageBindingVerifier(Protocol):
    @property
    def cluster(self) -> str: ...

    def verify(self, config: StorageProxyConfig) -> None: ...


class ProjectStorageBindingStore(Protocol):
    def load(self, project_id: str) -> ProjectStorageBinding | None: ...

    def save(self, binding: ProjectStorageBinding) -> None: ...

    def lease(self, project_id: str) -> AbstractContextManager[None]: ...
