"""Read replay evidence from an already registered development workspace."""

from __future__ import annotations

import hashlib
import json
import shlex
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any

import httpx

from harness_adapter.pi_adapter.actor_workspace import ActorWorkspace
from pyromind_common.platform.dataset_replay import DatasetReplayPaths
from pyromind_common.platform.pcb_collection import pcb_selection_command
from pyromind_runtime.domain.storage_paths import to_storage_api_path
from pyromind_runtime.ports.storage import (
    EvaluationStorageMissingError,
    EvaluationStorageUnavailableError,
)


_MAX_READ_BYTES = 8 * 1024 * 1024
_MAX_ENTRIES = 10_000
_MAX_LIST_BYTES = 60 * 1024
_INSPECT_PATH = """
import hashlib, json, os, stat, sys
action, path = sys.argv[1:3]
fd = os.open("/", os.O_DIRECTORY)
try:
    for part in path.strip("/").split("/"):
        child = os.open(part, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=fd)
        os.close(fd)
        fd = child
    attrs = os.fstat(fd)
    if action == "list":
        if not stat.S_ISDIR(attrs.st_mode):
            sys.exit(2)
        names = []
        with os.scandir(fd) as entries:
            for index, entry in enumerate(entries):
                if index >= int(sys.argv[3]):
                    sys.exit(2)
                if entry.is_file(follow_symlinks=False) or entry.is_dir(
                    follow_symlinks=False
                ):
                    names.append(entry.name)
        result = json.dumps(names, separators=(",", ":"))
        if len(result) > int(sys.argv[4]):
            sys.exit(2)
        print(result)
    else:
        limit = int(sys.argv[3])
        if not stat.S_ISREG(attrs.st_mode) or attrs.st_size > limit:
            sys.exit(2)
        digest, size = hashlib.sha256(), 0
        with os.fdopen(fd, "rb", closefd=False) as file:
            while chunk := file.read(min(1024 * 1024, limit + 1 - size)):
                size += len(chunk)
                if size > limit:
                    sys.exit(2)
                digest.update(chunk)
        if size != attrs.st_size:
            sys.exit(2)
        print(digest.hexdigest())
except FileNotFoundError:
    sys.exit(44)
finally:
    os.close(fd)
"""


@dataclass(frozen=True)
class RecordedActorReplayReader:
    workspace: ActorWorkspace
    max_read_bytes: int = _MAX_READ_BYTES

    def _mounted(self, path: str) -> PurePosixPath:
        mounted = PurePosixPath(self.workspace.storage_path + to_storage_api_path(path))
        if mounted.is_relative_to(
            PurePosixPath(self.workspace.storage_path) / ".pyromind-agent"
        ) and not mounted.is_relative_to(self.workspace.working_dir):
            raise ValueError("Replay paths must belong to the source conversation")
        return mounted

    def _artifact_path(self, path: str) -> PurePosixPath:
        mounted = self._mounted(path)
        if not mounted.is_relative_to(self.workspace.working_dir):
            raise ValueError("Replay assets must belong to the source conversation")
        return mounted

    def _inspect(self, action: str, mounted: PurePosixPath, *limits: int) -> str:
        try:
            result = self.workspace.execute_command(
                shlex.join(
                    ["python3", "-I", "-c", _INSPECT_PATH, action, str(mounted)]
                    + [str(limit) for limit in limits]
                )
            )
        except (httpx.HTTPError, TimeoutError) as exc:
            raise EvaluationStorageUnavailableError(
                "Replay workspace is unavailable"
            ) from exc
        if result.exit_code == 44:
            raise EvaluationStorageMissingError("Replay evidence is missing")
        if result.exit_code:
            raise EvaluationStorageUnavailableError("Replay evidence is unreadable")
        return result.stdout.strip()

    def list(self, prefix: str) -> set[str]:
        output = self._inspect(
            "list", self._artifact_path(prefix), _MAX_ENTRIES, _MAX_LIST_BYTES
        )
        try:
            if len(output) > _MAX_LIST_BYTES:
                raise ValueError
            names = json.loads(output)
            if (
                not isinstance(names, list)
                or len(names) > _MAX_ENTRIES
                or any(
                    not isinstance(name, str)
                    or name in {"", ".", ".."}
                    or "/" in name
                    or "\\" in name
                    for name in names
                )
            ):
                raise ValueError
        except ValueError as exc:
            raise EvaluationStorageUnavailableError(
                "Invalid Actor directory listing"
            ) from exc
        return set(names)

    def read(self, path: str) -> bytes:
        mounted = self._artifact_path(path)
        digest = self._inspect("read", mounted, self.max_read_bytes)
        try:
            content = self.workspace.transport.read_file(
                str(mounted), max_bytes=self.max_read_bytes
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
        if hashlib.sha256(content).hexdigest() != digest:
            raise EvaluationStorageUnavailableError(
                "Replay evidence changed while reading"
            )
        return content

    def select_pcb_groups(
        self, image_directory: str, validation_uri: str, offset: int
    ) -> list[dict[str, Any]]:
        raw, heldout = (
            to_storage_api_path(p) for p in (image_directory, validation_uri)
        )
        if offset < 0 or not all(p.startswith("/datasets/") for p in (raw, heldout)):
            raise ValueError(
                "PCB selection requires a datasets directory and fixed Eval file"
            )
        command = pcb_selection_command(
            self.workspace.storage_path, raw, heldout, offset
        )
        try:
            result = self.workspace.execute_command(shlex.join(command), timeout=180)
        except (httpx.HTTPError, TimeoutError) as exc:
            raise EvaluationStorageUnavailableError("PCB 选图工作区不可用") from exc
        if result.timeout_occurred:
            raise EvaluationStorageUnavailableError("PCB 图片分组或评测集校验超时")
        if result.exit_code or "[Actor output truncated]" in result.stdout:
            raise ValueError("所选范围缺少 5 组可用三图，或固定评测集图片无法校验")
        rows = json.loads(result.stdout)
        if not isinstance(rows, list) or len(rows) != 5:
            raise ValueError("PCB selection returned an incomplete page")
        return rows

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
