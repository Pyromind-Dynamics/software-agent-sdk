"""Persistent task associations for Pyromind dataset cleaning runs."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field


TASK_ASSOCIATION_DIRNAME = ".pyromind_dataset_cleaning_tasks"


class DatasetCleaningTaskAssociation(BaseModel):
    """Durable link between a Studio task and its owning conversation."""

    schema_version: int = 1
    task_id: str
    conversation_id: str
    run_id: str
    output_dir: str
    input_path: str
    script_path: str
    limit: int | None = None
    resumed: bool = False
    status: str = "Pending"
    submitted_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class DatasetCleaningTaskStore:
    """Store task associations as atomically replaced JSON files."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def save(
        self, association: DatasetCleaningTaskAssociation
    ) -> DatasetCleaningTaskAssociation:
        self.root.mkdir(parents=True, exist_ok=True)
        updated = association.model_copy(update={"updated_at": datetime.now(UTC)})
        target = self._path(updated.task_id)
        temporary = self.root / f".{target.stem}-{uuid.uuid4().hex}.tmp"
        payload = updated.model_dump_json(indent=2) + "\n"
        try:
            with temporary.open("w", encoding="utf-8") as file_obj:
                file_obj.write(payload)
                file_obj.flush()
                os.fsync(file_obj.fileno())
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
        return updated

    def get(self, task_id: str) -> DatasetCleaningTaskAssociation | None:
        association = self._read(self._path(task_id))
        if association is None or association.task_id != task_id:
            return None
        return association

    def get_by_run_id(self, run_id: str) -> DatasetCleaningTaskAssociation | None:
        matches: list[DatasetCleaningTaskAssociation] = []
        try:
            for path in self.root.glob("*.json"):
                association = self._read(path)
                if association is not None and association.run_id == run_id:
                    matches.append(association)
        except OSError:
            return None
        return max(matches, key=lambda item: item.updated_at, default=None)

    def _read(self, path: Path) -> DatasetCleaningTaskAssociation | None:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            return DatasetCleaningTaskAssociation.model_validate(payload)
        except (FileNotFoundError, json.JSONDecodeError, OSError, ValueError):
            return None

    def update_status(
        self, task_id: str, status: str
    ) -> DatasetCleaningTaskAssociation | None:
        association = self.get(task_id)
        if association is None:
            return None
        return self.save(association.model_copy(update={"status": status}))

    def _path(self, task_id: str) -> Path:
        digest = hashlib.sha256(task_id.encode("utf-8")).hexdigest()
        return self.root / f"{digest}.json"


def task_store_for_conversations_dir(
    conversations_dir: Path,
) -> DatasetCleaningTaskStore:
    return DatasetCleaningTaskStore(conversations_dir / TASK_ASSOCIATION_DIRNAME)
