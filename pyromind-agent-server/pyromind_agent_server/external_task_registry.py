from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

from pyromind_runtime.domain.content import JsonObject

from openhands.tools.data_preparation.platform_submit import (
    TASK_ASSOCIATION_DIRNAME as PREPARATION_TASK_DIR,
    DataPreparationTaskStore,
)
from openhands.tools.pyromind_cleaning.task_store import (
    TASK_ASSOCIATION_DIRNAME as CLEANING_TASK_DIR,
    DatasetCleaningTaskStore,
)


class WorkflowExternalTaskRegistry:
    """Expose legacy task associations through one Product runtime port."""

    def __init__(self, conversation_root: Path | str) -> None:
        self._root = Path(conversation_root)

    def owner(self, task_id: str) -> str | None:
        product = self._load_product(task_id)
        if product is not None:
            return _canonical_conversation_id(str(product["conversation_id"]))
        cleaning = self._cleaning().get(task_id)
        if cleaning is not None:
            return _canonical_conversation_id(cleaning.conversation_id)
        preparation = self._preparation().get(task_id)
        if preparation is not None:
            return _canonical_conversation_id(preparation.conversation_id)
        return None

    def resolve(self, conversation_id: str, task_id: str) -> JsonObject | None:
        canonical_id = _canonical_conversation_id(conversation_id)
        product = self._load_product(task_id)
        if product is not None and (
            _canonical_conversation_id(str(product["conversation_id"])) == canonical_id
        ):
            payload = product.get("payload")
            return dict(payload) if isinstance(payload, dict) else None
        cleaning = self._cleaning().get(task_id)
        if cleaning is not None and (
            _canonical_conversation_id(cleaning.conversation_id) == canonical_id
        ):
            submitted_at = cleaning.submitted_at.isoformat()
            return {
                "task_id": task_id,
                "kind": "data_cleaning",
                "run_id": cleaning.run_id,
                "status": _task_status(cleaning.status),
                "output_dir": cleaning.output_dir,
                "submitted_at": submitted_at,
                "updated_at": cleaning.updated_at.isoformat(),
                "resume_pending": False,
            }
        preparation = self._preparation().get(task_id)
        if preparation is None or (
            _canonical_conversation_id(preparation.conversation_id) != canonical_id
        ):
            return None
        return {
            "task_id": task_id,
            "kind": "data_preparation",
            "run_id": preparation.run_id,
            "status": _task_status(preparation.status),
            "output_dir": preparation.output_dir,
            "submitted_at": preparation.submitted_at,
            "updated_at": preparation.updated_at,
            "resume_pending": False,
        }

    def register(self, conversation_id: str, payload: JsonObject) -> None:
        task_id = payload.get("task_id")
        if not isinstance(task_id, str) or not task_id:
            raise ValueError("external task_id is required")
        self._product_directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        value = {
            "conversation_id": _canonical_conversation_id(conversation_id),
            "payload": dict(payload),
        }
        path = self._product_path(task_id)
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{path.name}.", dir=self._product_directory
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(value, stream, ensure_ascii=False, separators=(",", ":"))
                stream.flush()
                os.fsync(stream.fileno())
            os.chmod(temporary, 0o600)
            os.replace(temporary, path)
        finally:
            Path(temporary).unlink(missing_ok=True)

    def update_status(
        self,
        conversation_id: str,
        task_id: str,
        status: str,
    ) -> None:
        canonical_id = _canonical_conversation_id(conversation_id)
        product = self._load_product(task_id)
        if product is not None and (
            _canonical_conversation_id(str(product["conversation_id"])) == canonical_id
        ):
            payload = product.get("payload")
            if isinstance(payload, dict):
                payload = {**payload, "status": status}
                self.register(conversation_id, payload)
        legacy_status = _legacy_status(status)
        cleaning = self._cleaning().get(task_id)
        if cleaning is not None and (
            _canonical_conversation_id(cleaning.conversation_id) == canonical_id
        ):
            self._cleaning().update_status(task_id, legacy_status)
            return
        preparation = self._preparation().get(task_id)
        if preparation is None or (
            _canonical_conversation_id(preparation.conversation_id) != canonical_id
        ):
            return
        preparation.status = legacy_status
        preparation.updated_at = datetime.now(UTC).isoformat()
        self._preparation().save(preparation)

    def _cleaning(self) -> DatasetCleaningTaskStore:
        return DatasetCleaningTaskStore(self._root / CLEANING_TASK_DIR)

    def _preparation(self) -> DataPreparationTaskStore:
        return DataPreparationTaskStore(self._root / PREPARATION_TASK_DIR)

    @property
    def _product_directory(self) -> Path:
        return self._root / ".product_external_tasks"

    def _product_path(self, task_id: str) -> Path:
        digest = hashlib.sha256(task_id.encode()).hexdigest()
        return self._product_directory / f"{digest}.json"

    def _load_product(self, task_id: str) -> dict | None:
        path = self._product_path(task_id)
        if not path.is_file():
            return None
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if not isinstance(value, dict):
            return None
        payload = value.get("payload")
        if not isinstance(payload, dict) or payload.get("task_id") != task_id:
            return None
        return value


def _task_status(value: str) -> str:
    normalized = value.strip().lower()
    return {
        "success": "succeeded",
        "succeeded": "succeeded",
        "error": "failed",
        "failed": "failed",
        "terminated": "terminated",
        "stopped": "stopped",
        "running": "running",
        "pending": "pending",
    }.get(normalized, "pending")


def _legacy_status(value: str) -> str:
    return {
        "succeeded": "Succeeded",
        "failed": "Failed",
        "terminated": "Terminated",
        "stopped": "Stopped",
        "running": "Running",
        "pending": "Pending",
    }.get(value, value)


def _canonical_conversation_id(value: str) -> str:
    try:
        return UUID(value).hex
    except ValueError:
        return value
