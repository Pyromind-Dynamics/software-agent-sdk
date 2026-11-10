"""The narrow boundary between model-visible and Storage API paths."""

from __future__ import annotations

from pathlib import PurePosixPath


AGENT_STORAGE_ROOT = "/.pyromind-agent"
PUBLIC_WORKSPACE_DIRNAME = "public_data"
WORKSPACE_ROOT = "/workspace"


class StoragePathError(ValueError):
    """A value does not name a file below the user's Storage root."""


def normalize_workspace_path(value: str, *, field: str = "path") -> str:
    """Return the single model-visible spelling of a Storage file path."""

    parts = _parts(value, field=field)
    if parts[0] != "workspace":
        if parts[0] == "target-workspace":
            raise StoragePathError(
                f"{field}: '/target-workspace' is not a model-visible path; "
                "use '/workspace/<path>'."
            )
        raise StoragePathError(f"{field} must start with /workspace/.")
    if len(parts) == 1:
        raise StoragePathError(f"{field} must name a file below /workspace/.")
    return f"{WORKSPACE_ROOT}/{'/'.join(parts[1:])}"


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


def resolve_conversation_storage_path(
    conversation_id: str, value: str, *, field: str = "path"
) -> str:
    """Resolve a conversation-relative path to its model-visible Storage path."""
    raw = value.strip()
    path = PurePosixPath(raw)
    if not raw or ".." in path.parts:
        raise StoragePathError(f"{field} must stay inside the conversation workspace.")
    if path.is_absolute():
        return from_storage_api_path(path.as_posix(), field=field)
    if not path.parts or path.parts[0] == "storage":
        raise StoragePathError(
            f"{field} must name a file in the conversation workspace."
        )
    conversation_id = conversation_id.strip()
    if not conversation_id or "/" in conversation_id or "\\" in conversation_id:
        raise StoragePathError("conversation id is not a valid Storage path segment.")
    return from_storage_api_path(
        str(PurePosixPath(AGENT_STORAGE_ROOT, conversation_id, *path.parts)),
        field=field,
    )


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
