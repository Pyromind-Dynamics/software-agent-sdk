"""Sandbox-backed execution workspace for the Pi adapter."""

from __future__ import annotations

import os
import shlex
from pathlib import Path, PurePosixPath
from typing import Any, Protocol

from pydantic import PrivateAttr

from openhands.sdk.git.models import GitChange, GitDiff
from openhands.sdk.utils.path import to_posix_path
from openhands.sdk.workspace.base import BaseWorkspace
from openhands.sdk.workspace.models import CommandResult, FileOperationResult
from openhands.tools.sandbox import run_terminal_command, write_sandbox_file


class SandboxClientLike(Protocol):
    """The slice of ``SandboxClient`` used by the Pi execution plane."""

    base_url: str
    api_key: str
    cluster: str

    def get_sandbox(self, sandbox_id: str) -> Any: ...

    def create_and_wait(
        self, request: Any, *, target_status: str, timeout: int
    ) -> Any: ...

    def resume(self, sandbox_id: str) -> Any: ...

    def delete(self, sandbox_id: str) -> None: ...

    def pause(self, sandbox_id: str) -> Any: ...

    def read_file(self, sandbox_id: str, path: str) -> bytes: ...

    def write_file(
        self, sandbox_id: str, path: str, source: str | os.PathLike[str] | bytes
    ) -> dict[str, Any]: ...


def _workspace_path(workspace: str, path: str | Path) -> str:
    value = to_posix_path(path)
    candidate = PurePosixPath(value)
    if candidate.is_absolute():
        return value
    return str(PurePosixPath(to_posix_path(workspace)) / candidate)


class SandboxWorkspace(BaseWorkspace):
    """Expose a platform Sandbox through the SDK workspace port."""

    _sandbox_id: str = PrivateAttr()
    _ws_base_url: str = PrivateAttr()
    _client: SandboxClientLike = PrivateAttr()

    def __init__(
        self,
        *,
        working_dir: str | Path,
        storage_path: str | Path,
        sandbox_id: str,
        ws_base_url: str,
        client: SandboxClientLike,
        **kwargs: Any,
    ) -> None:
        """Bind the SDK workspace port to one platform sandbox.

        ``storage_path`` is the sandbox-side mount of the user Storage that the
        conversation addresses as ``storage/``; tools that stage data into a
        local staging directory resolve the alias against it.
        """
        super().__init__(working_dir=str(working_dir), **kwargs)
        self._storage_path = str(storage_path)
        self._sandbox_id = sandbox_id
        self._ws_base_url = ws_base_url
        self._client = client

    @property
    def storage_path(self) -> str:
        """Sandbox-side Storage mount root addressed as ``storage/``."""
        return self._storage_path

    def execute_command(
        self,
        command: str,
        cwd: str | Path | None = None,
        timeout: float = 30.0,
    ) -> CommandResult:
        directory = _workspace_path(self.working_dir, cwd or self.working_dir)
        wrapped = f"cd {shlex.quote(directory)} && {command}"
        output, exit_code, timed_out = run_terminal_command(
            base_url=self._ws_base_url,
            api_key=self._client.api_key,
            sandbox_id=self._sandbox_id,
            command=wrapped,
            timeout_seconds=max(1, int(timeout)),
        )
        return CommandResult(
            command=command,
            exit_code=exit_code if exit_code is not None else -1,
            stdout=output,
            stderr="",
            timeout_occurred=timed_out,
        )

    def file_upload(
        self,
        source_path: str | Path,
        destination_path: str | Path,
    ) -> FileOperationResult:
        source = Path(source_path)
        destination = _workspace_path(self.working_dir, destination_path)
        try:
            write_sandbox_file(
                self._client,
                ws_base_url=self._ws_base_url,
                sandbox_id=self._sandbox_id,
                path=destination,
                source=source,
            )
            return FileOperationResult(
                success=True,
                source_path=str(source),
                destination_path=destination,
                file_size=source.stat().st_size,
            )
        except Exception as exc:  # noqa: BLE001
            return FileOperationResult(
                success=False,
                source_path=str(source),
                destination_path=destination,
                error=str(exc),
            )

    def file_download(
        self,
        source_path: str | Path,
        destination_path: str | Path,
    ) -> FileOperationResult:
        source = _workspace_path(self.working_dir, source_path)
        destination = Path(destination_path)
        try:
            content = self._client.read_file(self._sandbox_id, source)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(content)
            return FileOperationResult(
                success=True,
                source_path=source,
                destination_path=str(destination),
                file_size=len(content),
            )
        except Exception as exc:  # noqa: BLE001
            return FileOperationResult(
                success=False,
                source_path=source,
                destination_path=str(destination),
                error=str(exc),
            )

    def git_changes(self, path: str | Path) -> list[GitChange]:
        raise NotImplementedError(
            "SandboxWorkspace does not expose OpenHands git workspace operations"
        )

    def git_diff(self, path: str | Path) -> GitDiff:
        raise NotImplementedError(
            "SandboxWorkspace does not expose OpenHands git workspace operations"
        )

    def pause(self) -> None:
        self._client.pause(self._sandbox_id)

    def resume(self) -> None:
        self._client.resume(self._sandbox_id)
