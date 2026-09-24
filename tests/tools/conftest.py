"""Fixtures for tools that need a sandbox-shaped execution workspace."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from openhands.sdk.workspace.models import CommandResult, FileOperationResult


class HostBackedSandboxWorkspace:
    """Sandbox workspace double backed by two host directories.

    Tools reach a remote workspace only through the workspace port, so pointing
    ``working_dir`` and ``storage_path`` at temporary directories exercises the
    sandbox code paths without a platform sandbox.
    """

    is_remote = True

    def __init__(self, workspace_dir: Path, storage_dir: Path) -> None:
        self.workspace_dir = workspace_dir
        self.storage_dir = storage_dir
        self.working_dir = str(workspace_dir)
        self.storage_path = str(storage_dir)

    def execute_command(
        self,
        command: str,
        cwd: str | Path | None = None,
        timeout: float = 30.0,
    ) -> CommandResult:
        completed = subprocess.run(
            command,
            shell=True,
            cwd=str(cwd or self.working_dir),
            # macOS bsdtar stores extended attributes as AppleDouble ``._``
            # members, which a Linux sandbox never produces.
            env={**os.environ, "COPYFILE_DISABLE": "1"},
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return CommandResult(
            command=command,
            exit_code=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
            timeout_occurred=False,
        )

    def file_upload(
        self, source_path: str | Path, destination_path: str | Path
    ) -> FileOperationResult:
        return self._copy(source_path, destination_path)

    def file_download(
        self, source_path: str | Path, destination_path: str | Path
    ) -> FileOperationResult:
        return self._copy(source_path, destination_path)

    @staticmethod
    def _copy(source_path: str | Path, destination_path: str | Path):
        source = Path(source_path)
        destination = Path(destination_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        return FileOperationResult(
            success=True,
            source_path=str(source),
            destination_path=str(destination),
            file_size=source.stat().st_size,
        )


class TruncatingDownloadWorkspace:
    """Sandbox double whose file reads stop at a fixed byte budget.

    The platform streams a sandbox file through one HTTP response, which can
    return a short body for a large file. ``limit`` mirrors that cut.
    """

    is_remote = True

    def __init__(
        self,
        workspace: HostBackedSandboxWorkspace,
        limit: int,
        *,
        echo_commands: bool = False,
    ) -> None:
        self.workspace_dir = workspace.workspace_dir
        self.storage_dir = workspace.storage_dir
        self.working_dir = workspace.working_dir
        self.storage_path = workspace.storage_path
        self.limit = limit
        self.echo_commands = echo_commands
        self.downloads = 0
        self._workspace = workspace

    def execute_command(
        self,
        command: str,
        cwd: str | Path | None = None,
        timeout: float = 30.0,
    ) -> CommandResult:
        result = self._workspace.execute_command(command, cwd=cwd, timeout=timeout)
        if not self.echo_commands:
            return result
        # The sandbox terminal bridge echoes the command and appends a status
        # line, so the real output arrives wrapped in that noise.
        return CommandResult(
            command=command,
            exit_code=result.exit_code,
            stdout=(
                f"{command}\r\n{result.stdout}__OH_EXIT__test:{result.exit_code}\r\n"
            ),
            stderr=result.stderr,
            timeout_occurred=result.timeout_occurred,
        )

    def file_upload(
        self, source_path: str | Path, destination_path: str | Path
    ) -> FileOperationResult:
        return self._workspace.file_upload(source_path, destination_path)

    def file_download(
        self, source_path: str | Path, destination_path: str | Path
    ) -> FileOperationResult:
        result = self._workspace.file_download(source_path, destination_path)
        destination = Path(destination_path)
        content = destination.read_bytes()
        self.downloads += 1
        if len(content) > self.limit:
            destination.write_bytes(content[: self.limit])
        return result


@pytest.fixture
def sandbox_workspace(tmp_path: Path) -> HostBackedSandboxWorkspace:
    workspace_dir = tmp_path / "execution"
    storage_dir = tmp_path / "storage"
    workspace_dir.mkdir()
    storage_dir.mkdir()
    return HostBackedSandboxWorkspace(workspace_dir, storage_dir)


@pytest.fixture
def truncated_downloads():
    """Build a sandbox double whose downloads arrive cut off past ``limit``."""

    def build(
        workspace: HostBackedSandboxWorkspace,
        *,
        limit: int,
        echo_commands: bool = False,
    ) -> TruncatingDownloadWorkspace:
        return TruncatingDownloadWorkspace(
            workspace, limit=limit, echo_commands=echo_commands
        )

    return build
