"""Read workspace files through the conversation workspace port.

Local workspaces read the host filesystem directly so symlink and permission
semantics stay identical to the previous implementation. Every other workspace
implementation downloads the file through its port, which keeps tools that
resolve workspace paths (for example ``validate_workflow_dsl``) working when the
execution plane runs in a platform sandbox.
"""

from __future__ import annotations

import tempfile
from pathlib import Path, PurePosixPath
from typing import Any

from openhands.sdk.workspace.local import LocalWorkspace


class WorkspaceFileNotFoundError(ValueError):
    """Raised when a workspace-relative file does not exist."""


def read_workspace_text(
    workspace: Any,
    path: str,
    *,
    max_bytes: int | None = None,
    reject_symlinks: bool = False,
) -> str:
    """Return the UTF-8 text of a file addressed relative to ``workspace``.

    ``path`` must be workspace-relative and must not escape the workspace; the
    check is lexical because a remote workspace has no local filesystem to
    resolve against.
    """
    relative = PurePosixPath(path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Path must stay inside the workspace: {path!r}")
    if isinstance(workspace, LocalWorkspace):
        return _read_local_text(
            Path(workspace.working_dir),
            relative,
            path,
            max_bytes=max_bytes,
            reject_symlinks=reject_symlinks,
        )
    return _read_remote_text(workspace, relative, path, max_bytes=max_bytes)


def _read_local_text(
    root: Path,
    relative: PurePosixPath,
    path: str,
    *,
    max_bytes: int | None,
    reject_symlinks: bool,
) -> str:
    workspace_root = root.resolve()
    if reject_symlinks:
        current = workspace_root
        for part in relative.parts:
            current /= part
            if current.is_symlink():
                raise ValueError(f"Path must not contain symlinks: {path!r}")
    target = (workspace_root / relative).resolve()
    if not target.is_relative_to(workspace_root):
        raise ValueError(f"Path must stay inside the workspace: {path!r}")
    if not target.is_file():
        raise WorkspaceFileNotFoundError(
            f"Cannot read workspace file: {path!r} does not exist."
        )
    _check_size(target, path, max_bytes)
    return _decode_text(target.read_bytes(), path)


def _read_remote_text(
    workspace: Any,
    relative: PurePosixPath,
    path: str,
    *,
    max_bytes: int | None,
) -> str:
    with tempfile.TemporaryDirectory(prefix="workspace-read-") as directory:
        destination = Path(directory) / relative.name
        result = workspace.file_download(relative.as_posix(), destination)
        if not result.success or not destination.is_file():
            raise WorkspaceFileNotFoundError(
                f"Cannot read workspace file: {path!r} does not exist."
            )
        _check_size(destination, path, max_bytes)
        return _decode_text(destination.read_bytes(), path)


def _check_size(target: Path, path: str, max_bytes: int | None) -> None:
    if max_bytes is not None and target.stat().st_size > max_bytes:
        raise ValueError(f"Workspace file exceeds {max_bytes} bytes: {path!r}")


def _decode_text(data: bytes, path: str) -> str:
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"Workspace file must be UTF-8: {path!r}") from exc
