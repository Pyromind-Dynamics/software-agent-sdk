from __future__ import annotations

from typing import Protocol

from pyromind_runtime.domain.context import RequestContext
from pyromind_runtime.domain.versions import (
    ArtifactVersion,
    RunSource,
    RunSpec,
    SaveVersion,
    SnapshotResult,
    SnapshotSource,
    VersionProject,
    VersionRun,
)


class VersionStore(Protocol):
    """Product Store sub-port for the version ledger."""

    def recovery_scopes(self) -> list[str]: ...

    def bound_project(self, conversation_id: str) -> VersionProject | None:
        """Internal callback lookup; user-facing access must call project()."""
        ...

    def project(
        self, conversation_id: str, user_id: str, *, create: bool = False
    ) -> VersionProject | None: ...

    def list_versions(self, project_id: str) -> list[ArtifactVersion]: ...

    def get_version(self, project_id: str, version_id: str) -> ArtifactVersion: ...

    def reserve_version(
        self, project: VersionProject, key: str, request: SaveVersion
    ) -> ArtifactVersion: ...

    def complete_version(
        self, project_id: str, version_id: str, result: SnapshotResult
    ) -> ArtifactVersion: ...

    def fail_version(self, project_id: str, version_id: str, error: str) -> None: ...

    def select(
        self,
        project_id: str,
        slot: str,
        version_id: str,
        command_id: str,
        reason: str,
        revision: int,
    ) -> VersionProject: ...

    def new_iteration(
        self, project_id: str, goal: str, command_id: str, revision: int
    ) -> VersionProject: ...

    def reserve_run(
        self, project_id: str, key: str, spec: RunSpec, source: RunSource
    ) -> VersionRun: ...

    def list_runs(self, project_id: str) -> list[VersionRun]: ...

    def link_task(self, project_id: str, run_id: str, task_id: str) -> None: ...


class ArtifactSnapshotPort(Protocol):
    def save(
        self, source: SnapshotSource, target: str, context: RequestContext
    ) -> SnapshotResult: ...

    def verify(self, result: SnapshotResult, context: RequestContext) -> None: ...

    def read_text(self, result: SnapshotResult, context: RequestContext) -> str: ...
