from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from hashlib import sha256
from pathlib import Path
from typing import Any
from uuid import UUID

from filelock import FileLock, Timeout
from pydantic import BaseModel

from pyromind_common.storage.atomic import (
    _atomic_write_json,
    _ensure_secure_directory,
)


def canonical_uuid(value: str, *, kind: str) -> str:
    try:
        return str(UUID(value))
    except ValueError as exc:
        raise ValueError(f"invalid {kind} ID: {value}") from exc


def project_directory(projects_root: Path | str, project_id: str) -> Path:
    root = Path(projects_root)
    return root / canonical_uuid(project_id, kind="project")


class ProjectLeaseTimeout(RuntimeError):
    pass


class ProjectLease:
    def __init__(
        self,
        projects_root: Path | str,
        project_id: str,
        *,
        timeout_seconds: float = 30,
    ) -> None:
        self.path = project_directory(projects_root, project_id) / (
            ".runtime_scheduler.lock"
        )
        self.timeout_seconds = timeout_seconds

    @contextmanager
    def acquire(self) -> Iterator[None]:
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        try:
            with FileLock(str(self.path), timeout=self.timeout_seconds):
                yield
        except Timeout as exc:
            raise ProjectLeaseTimeout(
                f"Could not acquire project runtime lease: {self.path.parent.name}"
            ) from exc


def canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_sha256(payload: Any) -> str:
    return f"sha256:{sha256(canonical_json_bytes(payload)).hexdigest()}"


def write_model(path: Path, model: BaseModel) -> None:
    _ensure_secure_directory(path.parent)
    _atomic_write_json(path, model.model_dump(mode="json"))


def read_model[ModelT: BaseModel](path: Path, model_type: type[ModelT]) -> ModelT:
    return model_type.model_validate_json(path.read_text(encoding="utf-8"))
