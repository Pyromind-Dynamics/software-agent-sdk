from __future__ import annotations

import fcntl
import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from uuid import UUID

from pydantic import TypeAdapter

from pyromind_runtime.domain.echomind_deployments import DeploymentRecord


_RECORDS = TypeAdapter(list[DeploymentRecord])


class FileDeploymentStore:
    def __init__(self, projects_root: Path | str) -> None:
        self.root = Path(projects_root)

    def _directory(self, project_id: str) -> Path:
        return self.root / str(UUID(project_id))

    @contextmanager
    def lease(self, project_id: str) -> Iterator[None]:
        directory = self._directory(project_id)
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / "runtime_slot.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError(
                    "Project resources are busy; retry shortly"
                ) from None
            try:
                yield
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def list(self, project_id: str) -> list[DeploymentRecord]:
        path = self._directory(project_id) / "deployments.json"
        if not path.exists():
            return []
        records = _RECORDS.validate_json(path.read_bytes())
        if any(UUID(record.project_id) != UUID(project_id) for record in records):
            raise ValueError("Deployment history belongs to another Project")
        return records

    def save(self, record: DeploymentRecord) -> None:
        directory = self._directory(record.project_id)
        records = self.list(record.project_id)
        records = [r for r in records if r.deployment_id != record.deployment_id]
        records.append(record)
        records.sort(key=lambda item: item.created_at)
        with tempfile.NamedTemporaryFile(dir=directory, delete=False) as temporary:
            temporary.write(_RECORDS.dump_json(records))
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary.name, directory / "deployments.json")
