from __future__ import annotations

import fcntl
import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from uuid import UUID

from pyromind_runtime.domain.batch_schedule import BatchScheduleState


class FileBatchScheduleStore:
    def __init__(self, projects_root: Path | str) -> None:
        self.root = Path(projects_root)

    def _path(self, project_id: str) -> Path:
        return self.root / str(UUID(project_id)) / "batch_schedule.json"

    def load(self, project_id: str) -> BatchScheduleState | None:
        try:
            state = BatchScheduleState.model_validate_json(
                self._path(project_id).read_bytes()
            )
        except FileNotFoundError:
            return None
        if UUID(state.project_id) != UUID(project_id):
            raise ValueError("Batch schedule does not belong to this Project")
        return state

    def save(self, state: BatchScheduleState) -> None:
        path = self._path(state.project_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as output:
                temporary = Path(output.name)
                os.fchmod(output.fileno(), 0o600)
                output.write(state.model_dump_json().encode())
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def list_projects(self) -> list[tuple[str, str]]:
        projects: list[tuple[str, str]] = []
        for path in sorted(self.root.glob("*/batch_schedule.json")):
            state = self.load(path.parent.name)
            if state is not None and (
                state.config.enabled
                or any(item.status != "released" for item in state.receipts)
            ):
                projects.append((state.project_id, state.owner_user_id))
        return projects

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
