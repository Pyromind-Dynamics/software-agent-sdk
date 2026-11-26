"""Workspace port for a pre-provisioned Substrate sandbox Actor."""

from __future__ import annotations

import base64
import math
import shlex
import time
import uuid
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import httpx
from pydantic import PrivateAttr

from harness_adapter.pi_adapter.sandbox_workspace import (
    RemoteSandboxWorkspace,
    _workspace_path,
)
from pyromind_common.sdk.workspace.models import CommandResult


@dataclass(frozen=True)
class ActorTransport:
    base_url: str
    actor_target: str

    def _client(self) -> httpx.Client:
        return httpx.Client(
            base_url=self.base_url.rstrip("/") + "/",
            headers={"ate-target-actor": self.actor_target},
            timeout=30,
        )

    def ready(self) -> None:
        with self._client() as client:
            client.get("readyz").raise_for_status()

    def read_file(
        self, path: str, *, timeout: float = 30, max_bytes: int | None = None
    ) -> bytes:
        with (
            self._client() as client,
            client.stream(
                "GET", "files", params={"path": path}, timeout=timeout
            ) as response,
        ):
            response.raise_for_status()
            if max_bytes is None:
                return response.read()
            data = bytearray()
            for chunk in response.iter_bytes(chunk_size=64 * 1024):
                if len(data) + len(chunk) > max_bytes:
                    raise ValueError("Actor file exceeds the requested size limit")
                data.extend(chunk)
            return bytes(data)

    def write_file(self, path: str, content: bytes) -> None:
        with self._client() as client:
            response = client.put("files", params={"path": path}, content=content)
            response.raise_for_status()

    def execute(self, command: str, cwd: str, timeout: float) -> CommandResult:
        seconds = max(1, min(math.ceil(timeout), 3600))
        run_id = uuid.uuid4().hex
        deadline = time.monotonic() + seconds + 10
        with self._client() as client:
            view = (
                client.post(
                    "exec",
                    json={
                        "run_id": run_id,
                        "command": ["/bin/bash", "-c", command],
                        "cwd": cwd,
                        "timeout_seconds": seconds,
                    },
                )
                .raise_for_status()
                .json()
            )
            streams = {"stdout": bytearray(), "stderr": bytearray()}
            while True:
                page_full = False
                for name, stream in streams.items():
                    if view[f"{name}_truncated"]:
                        stream.extend(b"\n[Actor output truncated]\n")
                    data = base64.b64decode(view[name], validate=True)
                    stream.extend(data)
                    page_full |= len(data) == 64 * 1024
                status = view["status"]
                if status != "running" and not page_full:
                    break
                if time.monotonic() > deadline:
                    client.post(f"exec/{run_id}/cancel").raise_for_status()
                    raise TimeoutError("Actor did not return a final command result")
                if not page_full and status == "running":
                    time.sleep(0.1)
                view = (
                    client.get(
                        f"exec/{run_id}",
                        params={
                            "stdout_offset": view["stdout_offset"],
                            "stderr_offset": view["stderr_offset"],
                        },
                    )
                    .raise_for_status()
                    .json()
                )
        return CommandResult(
            command=command,
            exit_code=view.get("exit_code", -1),
            stdout=streams["stdout"].decode(errors="replace"),
            stderr=streams["stderr"].decode(errors="replace") or view.get("error", ""),
            timeout_occurred=status == "timed_out",
        )


class ActorWorkspace(RemoteSandboxWorkspace):
    _transport: ActorTransport = PrivateAttr()

    def __init__(self, *, working_dir: str, transport: ActorTransport) -> None:
        super().__init__(working_dir=working_dir)
        self._transport = transport

    @property
    def transport(self) -> ActorTransport:
        return self._transport

    @property
    def storage_path(self) -> str:
        return "/workspace"

    def execute_command(
        self,
        command: str,
        cwd: str | Path | None = None,
        timeout: float = 30.0,
    ) -> CommandResult:
        directory = _workspace_path(self.working_dir, cwd or self.working_dir)
        return self.transport.execute(command, directory, timeout)

    def checked_command(
        self, command: str, cwd: str | Path | None = None, timeout: float = 30.0
    ) -> str:
        result = self.execute_command(command, cwd, timeout)
        if result.exit_code or result.timeout_occurred:
            raise RuntimeError(f"Actor command failed (exit code {result.exit_code})")
        return result.stdout

    def _upload_file(self, source: Path, destination: str) -> None:
        parent = PurePosixPath(destination).parent
        if not parent.is_relative_to(self.storage_path) or ".." in parent.parts:
            raise ValueError("Actor destination must be inside /workspace")
        self.checked_command(f"mkdir -p -- {shlex.quote(str(parent))}")
        if source.stat().st_size > 64 << 20:
            raise ValueError("Actor file exceeds 64 MiB")
        self.transport.write_file(destination, source.read_bytes())

    def _read_file(self, source: str) -> bytes:
        return self.transport.read_file(source)

    def pause(self) -> None:
        """A fixed Actor has an externally managed lifetime."""

    def resume(self) -> None:
        self.transport.ready()
