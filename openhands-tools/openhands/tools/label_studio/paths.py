"""Resolve Label Studio inputs through the shared workspace/Storage contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from openhands.tools.pyromind_dataset.definition import (
    _storage_path_for,
)
from openhands.tools.utils.storage_paths import StoragePathError, to_storage_api_path
from openhands.tools.utils.workspace_staging import (
    WorkspaceStagingError,
    resolve_workspace_path,
)


@dataclass(frozen=True)
class StorageInput:
    original: str
    resolved: str
    kind: str

    def diagnostic(self) -> str:
        return (
            f"original={self.original!r} resolved={self.resolved!r} "
            f"path_type={self.kind}"
        )


def resolve_storage_input(
    value: str, workspace: Any = None, conversation_id: str = ""
) -> StorageInput:
    """Resolve a conversation path or an absolute user Storage path."""
    raw = value.strip()
    if not raw or any(ord(character) < 32 for character in raw):
        raise ValueError(f"Invalid Storage path: original={value!r}")
    if raw == "/target-workspace" or raw.startswith("/target-workspace/"):
        raise ValueError(
            "'/target-workspace' is an execution-only path; use '/workspace/<path>'"
        )
    if workspace is not None:
        try:
            _, from_storage = resolve_workspace_path(workspace, raw)
        except WorkspaceStagingError:
            pass
        else:
            return StorageInput(
                value,
                _storage_path_for(workspace, conversation_id, raw),
                "storage_mount" if from_storage else "workspace",
            )
    try:
        resolved = to_storage_api_path(raw, field="path")
    except StoragePathError as exc:
        raise ValueError(str(exc)) from exc
    return StorageInput(
        value,
        resolved,
        "storage_object",
    )
