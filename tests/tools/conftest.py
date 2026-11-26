"""Fixtures for tools that need a sandbox-shaped execution workspace."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from typing import Any

import pytest
from pyromind_sdk import PyroMindAPIClient

from pyromind_common.sdk.workspace.models import CommandResult, FileOperationResult


@dataclass
class StudioHTTPFixture:
    url: str = ""
    requests: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    headers: list[dict[str, str]] = field(default_factory=list)
    responses: deque[tuple[int, dict[str, Any]]] = field(default_factory=deque)


@pytest.fixture
def studio_http() -> Iterator[StudioHTTPFixture]:
    fixture = StudioHTTPFixture()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            size = int(self.headers["Content-Length"])
            self._respond(json.loads(self.rfile.read(size)))

        def do_GET(self) -> None:
            self._respond({})

        def _respond(self, request: dict[str, Any]) -> None:
            fixture.requests.append((self.path, request))
            fixture.headers.append(dict(self.headers.items()))
            status, document = fixture.responses.popleft()
            payload = json.dumps(document).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format: str, *args: Any) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    fixture.url = f"http://127.0.0.1:{server.server_port}"
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield fixture
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.fixture
def direct_sdk(studio_http: StudioHTTPFixture) -> Iterator[PyroMindAPIClient]:
    with PyroMindAPIClient(
        api_key="test-key",
        base_url=studio_http.url,
        cluster="us-west-1",
        max_retries=0,
    ) as client:
        client.studio.session.trust_env = False
        yield client


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


class MountPathSandboxWorkspace(HostBackedSandboxWorkspace):
    """Sandbox double whose port speaks the container's mount path.

    The container addresses mounted Storage as ``/target-workspace`` and the
    conversation workspace as a directory inside it. Neither path exists on the
    host, so a tool that writes to ``working_dir`` directly fails here exactly
    as it does against a real sandbox. Port calls map the mount onto the host
    directory that backs it.
    """

    def __init__(
        self,
        backing_root: Path,
        storage_dir: Path,
        *,
        conversation_id: str,
        mount_path: str = "/target-workspace",
    ) -> None:
        super().__init__(backing_root, storage_dir)
        self.backing_root = Path(backing_root)
        self.mount_path = mount_path
        self.storage_path = mount_path
        self.working_dir = f"{mount_path}/.pyromind-agent/{conversation_id}"

    @property
    def host_workspace_dir(self) -> Path:
        """Host directory backing :attr:`working_dir`."""
        return Path(self._to_host(self.working_dir))

    def _to_host(self, path: str | Path) -> str:
        value = str(path)
        if value == self.mount_path:
            return str(self.backing_root)
        prefix = f"{self.mount_path}/"
        if value.startswith(prefix):
            return str(self.backing_root / value[len(prefix) :])
        return value

    def execute_command(
        self,
        command: str,
        cwd: str | Path | None = None,
        timeout: float = 30.0,
    ) -> CommandResult:
        return super().execute_command(
            command.replace(self.mount_path, str(self.backing_root)),
            cwd=self._to_host(cwd or self.working_dir),
            timeout=timeout,
        )

    def file_upload(
        self, source_path: str | Path, destination_path: str | Path
    ) -> FileOperationResult:
        return super().file_upload(source_path, self._to_host(destination_path))

    def file_download(
        self, source_path: str | Path, destination_path: str | Path
    ) -> FileOperationResult:
        return super().file_download(self._to_host(source_path), destination_path)


@pytest.fixture
def sandbox_workspace(tmp_path: Path) -> HostBackedSandboxWorkspace:
    workspace_dir = tmp_path / "execution"
    storage_dir = tmp_path / "storage"
    workspace_dir.mkdir()
    storage_dir.mkdir()
    return HostBackedSandboxWorkspace(workspace_dir, storage_dir)


@pytest.fixture
def mount_path_sandbox_workspace(tmp_path: Path) -> MountPathSandboxWorkspace:
    backing_root = tmp_path / "host-storage"
    storage_dir = tmp_path / "storage"
    backing_root.mkdir()
    storage_dir.mkdir()
    return MountPathSandboxWorkspace(
        backing_root, storage_dir, conversation_id="a" * 32
    )


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
