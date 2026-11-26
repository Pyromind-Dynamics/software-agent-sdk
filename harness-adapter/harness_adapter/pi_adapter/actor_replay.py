"""Read replay evidence from an already registered development workspace."""

from __future__ import annotations

import hashlib
import shlex
from dataclasses import dataclass
from pathlib import PurePosixPath

import httpx

from harness_adapter.pi_adapter.actor_workspace import ActorWorkspace
from pyromind_common.platform.dataset_replay import DatasetReplayPaths
from pyromind_runtime.domain.storage_paths import to_storage_api_path
from pyromind_runtime.ports.storage import (
    EvaluationStorageMissingError,
    EvaluationStorageUnavailableError,
)


_MAX_READ_BYTES = 8 * 1024 * 1024
_FILE_DIGEST = """
import hashlib, os, stat, sys
fd = os.open("/", os.O_DIRECTORY)
try:
    for part in sys.argv[1].strip("/").split("/"):
        child = os.open(part, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=fd)
        os.close(fd)
        fd = child
    attrs = os.fstat(fd)
    if not stat.S_ISREG(attrs.st_mode) or attrs.st_size > int(sys.argv[2]):
        sys.exit(2)
    with os.fdopen(fd, "rb", closefd=False) as file:
        data = file.read(int(sys.argv[2]) + 1)
    if len(data) != attrs.st_size:
        sys.exit(2)
    print(hashlib.sha256(data).hexdigest())
except FileNotFoundError:
    sys.exit(44)
finally:
    os.close(fd)
"""


@dataclass(frozen=True)
class RecordedActorReplayReader:
    workspace: ActorWorkspace

    def _mounted(self, path: str) -> PurePosixPath:
        mounted = PurePosixPath(self.workspace.storage_path + to_storage_api_path(path))
        if mounted.is_relative_to(
            PurePosixPath(self.workspace.storage_path) / ".pyromind-agent"
        ) and not mounted.is_relative_to(self.workspace.working_dir):
            raise ValueError("Replay paths must belong to the source conversation")
        return mounted

    def read(self, path: str) -> bytes:
        mounted = self._mounted(path)
        if not mounted.is_relative_to(self.workspace.working_dir):
            raise ValueError("Replay assets must belong to the source conversation")
        try:
            result = self.workspace.execute_command(
                shlex.join(
                    [
                        "python3",
                        "-I",
                        "-c",
                        _FILE_DIGEST,
                        str(mounted),
                        str(_MAX_READ_BYTES),
                    ]
                )
            )
            if result.exit_code == 44:
                raise EvaluationStorageMissingError("Replay evidence is missing")
            if result.exit_code:
                raise EvaluationStorageUnavailableError("Replay evidence is unreadable")
            content = self.workspace.transport.read_file(
                str(mounted), max_bytes=_MAX_READ_BYTES
            )
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                raise EvaluationStorageMissingError(
                    "Replay evidence is missing"
                ) from exc
            raise EvaluationStorageUnavailableError(
                "Replay workspace is unavailable"
            ) from exc
        except (httpx.RequestError, TimeoutError, ValueError) as exc:
            raise EvaluationStorageUnavailableError(
                "Replay workspace is unavailable"
            ) from exc
        if hashlib.sha256(content).hexdigest() != result.stdout.strip():
            raise EvaluationStorageUnavailableError(
                "Replay evidence changed while reading"
            )
        return content

    def verify_splits(self, paths: DatasetReplayPaths) -> tuple[int, str, str]:
        for path in (paths.source_batch, paths.approved_source, paths.fixed_validation):
            if path:
                self._mounted(path)
        if not self._mounted(paths.output_dir).is_relative_to(
            self.workspace.working_dir
        ):
            raise ValueError("Replay output must belong to the source conversation")
        try:
            result = self.workspace.execute_command(
                shlex.join(paths.verification_command(self.workspace.storage_path)),
                timeout=180,
            )
        except (httpx.HTTPError, TimeoutError) as exc:
            raise EvaluationStorageUnavailableError(
                "Replay workspace is unavailable"
            ) from exc
        if result.timeout_occurred or "[Actor output truncated]" in result.stdout:
            raise EvaluationStorageUnavailableError(
                "Workspace verification output is incomplete"
            )
        return result.exit_code, result.stdout, result.stderr
