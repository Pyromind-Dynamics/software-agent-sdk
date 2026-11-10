"""Tool-facing aliases for the product Storage path contract.

Models use ``/workspace/<path>``. Storage APIs and persisted tool state use
``/<path>``. A relative path is rejected because its meaning depends on the
caller's working directory.
"""

from __future__ import annotations

from pyromind_runtime.domain.storage_paths import (
    WORKSPACE_ROOT,
    StoragePathError as StoragePathError,
    from_storage_api_path,
    to_storage_api_path,
)


REMOVED_ALIAS = "storage"
DEFAULT_MOUNT_PATH = WORKSPACE_ROOT
PLATFORM_ROOT = WORKSPACE_ROOT


def to_tool_storage_path(value: str, *, field: str = "path") -> str:
    """Normalize an absolute model or internal path for a platform tool."""

    return to_storage_api_path(value, field=field)


def to_storage_path(value: str, *, field: str = "path") -> str:
    """Normalize a public or internal path to the Storage API spelling."""

    return to_storage_api_path(value, field=field)


def to_agent_storage_path(value: str, *, field: str = "path") -> str:
    """Return the single Storage spelling exposed to users and agents."""

    return from_storage_api_path(to_storage_path(value, field=field), field=field)
