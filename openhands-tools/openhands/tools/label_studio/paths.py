"""Resolve Label Studio inputs through the shared workspace/Storage contract."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any

from openhands.tools.pyromind_dataset.definition import (
    _storage_path,
    _storage_path_for,
    _strip_workspace_prefix,
)
from openhands.tools.utils.storage_paths import REMOVED_ALIAS
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
    """Keep legacy relative object keys while resolving explicit workspace paths."""
    raw = value.strip()
    if not raw or any(ord(character) < 32 for character in raw):
        raise ValueError(f"Invalid Storage path: original={value!r}")
    path = PurePosixPath(raw)
    # Validate before stripping aliases or delegating to a mount resolver.
    _storage_path(path)
    if workspace is not None:
        try:
            _, from_storage = resolve_workspace_path(workspace, raw)
        except WorkspaceStagingError:
            pass
        else:
            if from_storage or path.is_absolute() or raw.startswith("public_data/"):
                return StorageInput(
                    value,
                    _storage_path_for(workspace, conversation_id, raw),
                    "storage_mount" if from_storage else "workspace",
                )
    if raw.startswith(f"{REMOVED_ALIAS}/"):
        raise ValueError(
            f"the '{REMOVED_ALIAS}/' alias was removed: original={value!r}; "
            "address Storage by its mount path or its Storage path"
        )
    return StorageInput(
        value,
        _storage_path(PurePosixPath(_strip_workspace_prefix(raw))),
        "storage_object",
    )
