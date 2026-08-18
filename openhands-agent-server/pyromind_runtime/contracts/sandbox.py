from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, Protocol

from pydantic import Field

from pyromind_runtime.contracts.base import ContractModel
from pyromind_runtime.contracts.content import JsonObject


class WorkspaceRef(ContractModel):
    workspace_id: str = Field(min_length=1)
    root: str = Field(min_length=1)
    revision: str | None = None


class SandboxRef(ContractModel):
    sandbox_id: str = Field(min_length=1)
    backend: str = Field(min_length=1)
    lease_id: str | None = None
    metadata: JsonObject = Field(default_factory=dict)


class ModelProfile(ContractModel):
    profile_id: str = Field(min_length=1)


type SandboxFileKind = Literal["file", "directory", "symlink"]


@dataclass(frozen=True, slots=True)
class SandboxFileInfo:
    name: str
    path: str
    kind: SandboxFileKind
    size: int
    mtime_ms: float


@dataclass(frozen=True, slots=True)
class SandboxExecResult:
    stdout: str
    stderr: str
    exit_code: int


class SandboxExecutionBackend(Protocol):
    async def exists(self, sandbox_id: str, path: str) -> bool: ...

    async def canonical_path(self, sandbox_id: str, path: str) -> str: ...

    async def read_file(self, sandbox_id: str, path: str) -> bytes: ...

    async def write_file(
        self,
        sandbox_id: str,
        path: str,
        content: bytes,
    ) -> None: ...

    async def append_file(
        self,
        sandbox_id: str,
        path: str,
        content: bytes,
    ) -> None: ...

    async def rename_file(
        self,
        sandbox_id: str,
        source_path: str,
        destination_path: str,
    ) -> None: ...

    async def file_info(
        self,
        sandbox_id: str,
        path: str,
    ) -> SandboxFileInfo: ...

    async def list_dir(
        self,
        sandbox_id: str,
        path: str,
    ) -> tuple[SandboxFileInfo, ...]: ...

    async def create_dir(
        self,
        sandbox_id: str,
        path: str,
        *,
        recursive: bool,
    ) -> None: ...

    async def remove(
        self,
        sandbox_id: str,
        path: str,
        *,
        recursive: bool,
        force: bool,
    ) -> None: ...

    async def create_temp_dir(
        self,
        sandbox_id: str,
        parent: str,
        prefix: str,
    ) -> str: ...

    async def create_temp_file(
        self,
        sandbox_id: str,
        parent: str,
        prefix: str,
        suffix: str,
    ) -> str: ...

    async def exec(
        self,
        sandbox_id: str,
        command: str,
        *,
        cwd: str,
        environment: Mapping[str, str],
        timeout_seconds: float,
    ) -> SandboxExecResult: ...

    async def close(self) -> None: ...
