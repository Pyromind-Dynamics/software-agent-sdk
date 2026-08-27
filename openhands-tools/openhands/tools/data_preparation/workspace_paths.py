from __future__ import annotations

from pathlib import Path, PurePosixPath
from typing import Any

from openhands.sdk.workspace.workspace import LocalWorkspace
from openhands.tools.utils import default_path_access_policy


def resolve_workspace_file(
    conversation: Any,
    path: str,
    *,
    allow_virtual_conversation_path: bool = False,
) -> Path:
    """Resolve an existing readable file inside the conversation workspace."""

    workspace_dir = _workspace_dir(conversation)
    candidate = _workspace_candidate(
        conversation,
        workspace_dir,
        path,
        allow_virtual_conversation_path=allow_virtual_conversation_path,
    )
    resolved = candidate.resolve()
    try:
        resolved.relative_to(workspace_dir)
    except ValueError as exc:
        raise ValueError(f"Path is outside the conversation workspace: {path}") from exc
    policy = default_path_access_policy(workspace_dir)
    if not policy.check(resolved, "read") or not resolved.is_file():
        raise ValueError(f"Missing or unreadable workspace file: {path}")
    return resolved


def workspace_relative_path(conversation: Any, path: Path) -> str:
    return path.resolve().relative_to(_workspace_dir(conversation)).as_posix()


def _workspace_dir(conversation: Any) -> Path:
    workspace = conversation.workspace
    if not isinstance(workspace, LocalWorkspace):
        raise ValueError(
            "This operation is only supported for local conversation workspaces."
        )
    return Path(workspace.working_dir).resolve()


def _workspace_candidate(
    conversation: Any,
    workspace_dir: Path,
    path: str,
    *,
    allow_virtual_conversation_path: bool,
) -> Path:
    if allow_virtual_conversation_path:
        virtual_relative = _virtual_conversation_relative_path(
            conversation, workspace_dir, path
        )
        if virtual_relative is not None:
            return workspace_dir / virtual_relative
    candidate = Path(path)
    return candidate if candidate.is_absolute() else workspace_dir / candidate


def _virtual_conversation_relative_path(
    conversation: Any,
    workspace_dir: Path,
    path: str,
) -> Path | None:
    candidate = PurePosixPath(path)
    parts = candidate.parts
    prefix = ("/", "workspace", "conversations")
    if parts[:3] != prefix:
        return None
    if len(parts) < 5:
        raise ValueError(f"Missing workspace-relative path: {path}")
    supplied_id = parts[3]
    valid_ids = {workspace_dir.name}
    conversation_id = getattr(conversation, "id", None)
    if conversation_id is not None:
        valid_ids.add(str(conversation_id))
        valid_ids.add(str(conversation_id).replace("-", ""))
    if supplied_id not in valid_ids and supplied_id.replace("-", "") not in valid_ids:
        raise ValueError(
            f"Virtual workspace path belongs to another conversation: {supplied_id}"
        )
    return Path(*parts[4:])
