from __future__ import annotations

import fcntl
import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from uuid import UUID

from pyromind_runtime.domain.project_storage import ProjectStorageBinding


class FileProjectStorageBindingStore:
    def __init__(self, projects_root: Path | str) -> None:
        self.root = Path(projects_root)

    def _path(self, project_id: str) -> Path:
        return self.root / str(UUID(project_id)) / "storage_binding.json"

    def load(self, project_id: str) -> ProjectStorageBinding | None:
        try:
            binding = ProjectStorageBinding.model_validate_json(
                self._path(project_id).read_bytes()
            )
        except FileNotFoundError:
            return None
        if UUID(binding.project_id) != UUID(project_id):
            raise ValueError("Storage binding belongs to another Project")
        return binding

    def save(self, binding: ProjectStorageBinding) -> None:
        path = self._path(binding.project_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as output:
                temporary = Path(output.name)
                os.fchmod(output.fileno(), 0o600)
                output.write(binding.model_dump_json().encode())
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    @contextmanager
    def lease(self, project_id: str) -> Iterator[None]:
        path = self._path(project_id).with_suffix(".lock")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a+b") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                yield
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
