"""Platform file-path spelling, independent of Agent conversations and Runtime."""

from __future__ import annotations

from pathlib import PurePosixPath


WORKSPACE_ROOT = "/workspace"
PUBLIC_WORKSPACE_DIRNAME = "public_data"


class StoragePathError(ValueError):
    """A value does not name a file below the user's Storage root."""


def to_storage_api_path(value: str, *, field: str = "path") -> str:
    """Convert a public or internal Storage path to the Storage API spelling."""

    parts = _parts(value, field=field)
    if parts[0] == "target-workspace":
        raise StoragePathError(
            f"{field}: '/target-workspace' is an execution-only path and is "
            "not accepted at this boundary."
        )
    if parts[0] == "workspace":
        parts = parts[1:]
    if not parts:
        raise StoragePathError(f"{field} must name a file below Storage.")
    return "/" + "/".join(parts)


def from_storage_api_path(value: str, *, field: str = "path") -> str:
    """Return the model-visible path for a Storage API path."""

    return WORKSPACE_ROOT + to_storage_api_path(value, field=field)


def _parts(value: str, *, field: str) -> list[str]:
    raw = value.strip()
    if not raw:
        raise StoragePathError(f"{field} must not be empty.")
    if any(ord(character) < 32 for character in raw):
        raise StoragePathError(f"{field} contains control characters.")
    if any(character in raw for character in ("\\", ":", "?", "#")):
        raise StoragePathError(f"{field} must be a filesystem path, not a URL.")
    path = PurePosixPath(raw)
    if not raw.startswith("/") and path.parts and path.parts[0] == "storage":
        raise StoragePathError(
            f"{field}: the 'storage/' alias was removed; use "
            "'/workspace/<path>' or '/<path>'."
        )
    if (
        not raw.startswith("/")
        and path.parts
        and path.parts[0] == PUBLIC_WORKSPACE_DIRNAME
    ):
        raise StoragePathError(
            f"{field}: '{PUBLIC_WORKSPACE_DIRNAME}/...' is a conversation "
            "workspace path, not a user Storage path."
        )
    if not raw.startswith("/"):
        raise StoragePathError(f"{field} must be absolute.")
    if ".." in path.parts:
        raise StoragePathError(f"{field} must not contain '..'.")
    parts = [part for part in path.parts if part not in {"", "/", "."}]
    if not parts:
        raise StoragePathError(f"{field} must not be the Storage root.")
    if parts[0] == PUBLIC_WORKSPACE_DIRNAME:
        raise StoragePathError(
            f"{field}: 'public_data/...' is a conversation workspace path, "
            "not a user Storage path."
        )
    return parts


to_tool_storage_path = to_storage_api_path
to_agent_storage_path = from_storage_api_path
