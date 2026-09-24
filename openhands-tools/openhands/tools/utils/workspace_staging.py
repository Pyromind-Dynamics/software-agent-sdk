"""Host-local staging for tools that spawn local processes.

DataFlow pipelines, schema validators and report generators run as host
subprocesses, so they need a host filesystem view of the files they touch. When
the conversation workspace lives in a platform sandbox, that view is built by
copying the paths the run touches into a staging directory and pushing the
results back afterwards; the sandbox stays the single authority for workspace
content.

Paths addressed as ``storage/...`` resolve against the sandbox Storage mount
instead of the workspace, which lets a run read user Storage without
materializing it through the conversation first.
"""

from __future__ import annotations

import posixpath
import shlex
import shutil
import tarfile
import tempfile
import uuid
import zlib
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import Any

from openhands.sdk.workspace.base import BaseWorkspace
from openhands.sdk.workspace.local import LocalWorkspace


STORAGE_ALIAS = "storage"
DEFAULT_TIMEOUT_SECONDS = 600.0
_ARCHIVE_NAME = "workspace.tgz"
_STAGE_PREFIX = "pyromind-stage-"
_FILE_CHANGED_MARKER = "file changed as we read it"
_ARCHIVE_ATTEMPTS = 3
# A sandbox file travels through one HTTP response, and a large archive can come
# back cut off. Splitting the member list keeps every transfer small enough to
# finish; the depth bounds how far one group is halved before it gives up.
_ARCHIVE_SPLIT_DEPTH = 4
_ARCHIVE_READ_ERRORS = (tarfile.TarError, EOFError, zlib.error, OSError)


class WorkspaceStagingError(ValueError):
    """Raised when a workspace path cannot be staged in either direction."""


class WorkspaceArchiveTruncatedError(WorkspaceStagingError):
    """Raised when a staged archive arrives cut off and cannot be split further."""


def is_remote_workspace(workspace: Any) -> bool:
    """True when ``workspace`` is served through its port rather than the host.

    A plain object with ``working_dir`` is treated as a host-backed workspace so
    lightweight local workspace doubles keep the existing filesystem behavior.
    """
    if isinstance(workspace, BaseWorkspace):
        return not isinstance(workspace, LocalWorkspace)
    return getattr(workspace, "is_remote", False) is True


def resolve_workspace_path(
    workspace: Any, path: str | Path
) -> tuple[PurePosixPath, bool]:
    """Normalize ``path`` to a workspace-relative path.

    Accepts a workspace-relative path, the ``storage/`` alias, and absolute
    paths inside either the workspace root or the Storage mount. The returned
    flag is True when the path addresses Storage rather than the workspace.
    """
    candidate = _posix(path)
    if candidate.is_absolute():
        return _split_absolute(workspace, candidate, path)
    if not candidate.parts or ".." in candidate.parts:
        raise WorkspaceStagingError(f"Path must stay inside the workspace: {path!r}")
    if candidate.parts[0] != STORAGE_ALIAS or _storage_mount(workspace) is None:
        return candidate, False
    return PurePosixPath(*candidate.parts[1:]), True


def stage_workspace_path(
    workspace: Any,
    path: str,
    *,
    destination: Path,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> Path:
    """Copy ``path`` into ``destination`` and return the staged host path.

    Workspace-relative paths keep their layout, so a pipeline that reads
    siblings through relative paths still resolves them. ``storage/...`` paths
    land under a ``storage/`` directory inside the staging root, mirroring the
    symlink the sandbox shell sees.
    """
    relative, from_storage = resolve_workspace_path(workspace, path)
    if not relative.parts:
        raise WorkspaceStagingError(
            f"Path must name a file or directory inside the workspace: {path!r}"
        )
    source_root = (
        str(workspace.storage_path) if from_storage else str(workspace.working_dir)
    )
    staged_relative = (
        PurePosixPath(STORAGE_ALIAS, *relative.parts) if from_storage else relative
    )
    target = destination / staged_relative
    if isinstance(workspace, LocalWorkspace):
        _copy_local(Path(source_root) / relative, target)
        return target
    _download_remote(
        workspace,
        posixpath.join(source_root, relative.as_posix()),
        target,
        timeout,
    )
    return target


def stage_workspace_members(
    workspace: Any,
    directory: str | Path,
    members: Sequence[str],
    *,
    destination: Path,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> None:
    """Copy selected members of one workspace directory into ``destination``.

    ``directory`` is addressed like any other workspace path, including the
    ``storage/`` alias. Only the named members travel, so a Storage tree that is
    too large to stage whole still yields the files a manifest references, and
    members keep the layout :func:`stage_workspace_path` produces.
    """
    relative, from_storage = resolve_workspace_path(workspace, directory)
    source_root = (
        str(workspace.storage_path) if from_storage else str(workspace.working_dir)
    )
    prefix = PurePosixPath(STORAGE_ALIAS) if from_storage else PurePosixPath()
    clean: list[str] = []
    for member in members:
        candidate = _posix(member)
        if candidate.is_absolute() or not candidate.parts or ".." in candidate.parts:
            raise WorkspaceStagingError(
                f"Member must stay inside {directory!r}: {member!r}"
            )
        clean.append(candidate.as_posix())
    if not clean:
        return
    target_root = destination / prefix / relative
    if isinstance(workspace, LocalWorkspace):
        for member in clean:
            _copy_local(Path(source_root) / relative / member, target_root / member)
        return
    source_dir = (
        posixpath.join(source_root, relative.as_posix())
        if relative.parts
        else source_root
    )
    _download_remote_members(
        workspace,
        source_dir,
        clean,
        destination=target_root,
        timeout=timeout,
    )


def publish_workspace_path(
    workspace: Any,
    source: Path,
    relative: str,
    *,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> None:
    """Copy staged ``source`` back to ``relative`` inside ``workspace``."""
    target = _posix(relative)
    if target.is_absolute() or not target.parts or ".." in target.parts:
        raise WorkspaceStagingError(f"Invalid workspace target: {relative!r}")
    if not source.exists():
        raise WorkspaceStagingError(f"Staged path does not exist: {source}")
    if isinstance(workspace, LocalWorkspace):
        _copy_local(source, Path(workspace.working_dir) / target)
        return
    _upload_remote(workspace, source, target.as_posix(), timeout)


@contextmanager
def staged_remote_dir() -> Iterator[Path]:
    """Yield a host staging directory that is removed on exit."""
    directory = Path(tempfile.mkdtemp(prefix=_STAGE_PREFIX)).resolve()
    try:
        yield directory
    finally:
        shutil.rmtree(directory, ignore_errors=True)


@contextmanager
def staged_remote_path(
    workspace: Any,
    path: str,
    *,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> Iterator[Path]:
    """Yield a host copy of a remote workspace path, removed on exit."""
    with staged_remote_dir() as directory:
        yield stage_workspace_path(
            workspace, path, destination=directory, timeout=timeout
        )


def _split_absolute(
    workspace: Any, candidate: PurePosixPath, path: str | Path
) -> tuple[PurePosixPath, bool]:
    workspace_root = _posix(workspace.working_dir)
    if _is_inside(workspace_root, candidate):
        return _relative_to(workspace_root, candidate), False
    mount = _storage_mount(workspace)
    if mount is not None and _is_inside(_posix(mount), candidate):
        return _relative_to(_posix(mount), candidate), True
    raise WorkspaceStagingError(f"Path is outside the execution workspace: {path!r}")


def _posix(path: str | Path) -> PurePosixPath:
    return PurePosixPath(str(path).replace("\\", "/"))


def _is_inside(root: PurePosixPath, candidate: PurePosixPath) -> bool:
    parts = candidate.parts
    return parts[: len(root.parts)] == root.parts


def _relative_to(root: PurePosixPath, candidate: PurePosixPath) -> PurePosixPath:
    return PurePosixPath(*candidate.parts[len(root.parts) :])


def _storage_mount(workspace: Any) -> str | None:
    mount = getattr(workspace, "storage_path", None)
    return mount if isinstance(mount, str) and mount.strip() else None


def _copy_local(source: Path, target: Path) -> None:
    if not source.exists():
        raise WorkspaceStagingError(f"Workspace path does not exist: {source}")
    if source.is_dir():
        shutil.copytree(source, target, dirs_exist_ok=True, symlinks=True)
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def _download_remote(
    workspace: Any,
    source: str,
    target: Path,
    timeout: float,
) -> None:
    parent, name = posixpath.split(source)
    _download_remote_members(
        workspace,
        parent,
        [name],
        destination=target.parent,
        timeout=timeout,
    )


def _download_remote_members(
    workspace: Any,
    source_dir: str,
    members: Sequence[str],
    *,
    destination: Path,
    timeout: float,
) -> None:
    _stage_member_group(
        workspace,
        source_dir,
        list(members),
        destination=destination,
        timeout=timeout,
        depth=0,
    )


def _stage_member_group(
    workspace: Any,
    source_dir: str,
    members: list[str],
    *,
    destination: Path,
    timeout: float,
    depth: int,
) -> None:
    """Stage one archive, halving the group while the transfer stays short.

    The sandbox serves a file through one HTTP response, so a large archive can
    arrive truncated. Splitting the members and retrying keeps each transfer
    small enough to complete; a single member that still arrives short is
    reported instead of staged.
    """
    if not members:
        return
    archive = _remote_archive_path()
    names = " ".join(shlex.quote(member) for member in members)
    _run_in_sandbox(
        workspace,
        f"tar -czf {shlex.quote(archive)} -C {shlex.quote(source_dir)} {names}",
        timeout=timeout,
        action=f"archive {source_dir}",
        retry_file_changed=True,
    )
    truncated: str | None = None
    try:
        with _local_archive() as local_archive:
            result = workspace.file_download(archive, local_archive)
            if not getattr(result, "success", False):
                raise WorkspaceStagingError(
                    f"Cannot stage {source_dir}: "
                    f"{getattr(result, 'error', None) or 'file download failed'}"
                )
            expected = _remote_file_size(workspace, archive, timeout)
            staged = local_archive.stat().st_size
            if expected is not None and staged != expected:
                truncated = (
                    f"{len(members)} member(s) of {source_dir} arrived truncated "
                    f"({staged} of {expected} bytes)"
                )
            else:
                try:
                    _extract_archive(local_archive, destination)
                except _ARCHIVE_READ_ERRORS as exc:
                    truncated = (
                        f"{len(members)} member(s) of {source_dir} arrived "
                        f"truncated: {exc}"
                    )
    finally:
        _remove_remote_archive(workspace, archive, timeout)
    if truncated is None:
        return
    if len(members) > 1 and depth < _ARCHIVE_SPLIT_DEPTH:
        middle = len(members) // 2
        for half in (members[:middle], members[middle:]):
            _stage_member_group(
                workspace,
                source_dir,
                half,
                destination=destination,
                timeout=timeout,
                depth=depth + 1,
            )
        return
    raise WorkspaceArchiveTruncatedError(truncated)


def _upload_remote(
    workspace: Any,
    source: Path,
    target_relative: str,
    timeout: float,
) -> None:
    archive = _remote_archive_path()
    with _local_archive() as local_archive:
        _create_archive(source, local_archive, target_relative)
        result = workspace.file_upload(local_archive, archive)
        if not getattr(result, "success", False):
            raise WorkspaceStagingError(
                f"Cannot publish workspace path {target_relative}: "
                f"{getattr(result, 'error', None) or 'file upload failed'}"
            )
    workspace_root = str(workspace.working_dir)
    try:
        _run_in_sandbox(
            workspace,
            f"mkdir -p {shlex.quote(workspace_root)} && "
            f"tar -xzf {shlex.quote(archive)} -C {shlex.quote(workspace_root)}",
            timeout=timeout,
            action=f"publish {target_relative}",
        )
    finally:
        _remove_remote_archive(workspace, archive, timeout)


def _create_archive(source: Path, archive: Path, arcname: str) -> None:
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(source, arcname=arcname)


def _extract_archive(archive: Path, directory: Path) -> None:
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar.getmembers():
            _reject_unsafe_member(member.name, archive)
        directory.mkdir(parents=True, exist_ok=True)
        tar.extractall(directory, filter="data")


def _reject_unsafe_member(name: str, archive: Path) -> None:
    parts = _posix(name).parts
    if not parts or name.startswith("/") or ".." in parts:
        raise WorkspaceStagingError(f"Refusing to extract {name!r} from {archive.name}")


def _run_in_sandbox(
    workspace: Any,
    command: str,
    *,
    timeout: float,
    action: str,
    retry_file_changed: bool = False,
) -> None:
    # JuiceFS can update a file's metadata while tar streams it, which makes GNU
    # tar report "file changed as we read it" and exit 1. The first read settles
    # the metadata, so a retry succeeds; anything else fails as before.
    attempts = _ARCHIVE_ATTEMPTS if retry_file_changed else 1
    for attempt in range(attempts):
        result = workspace.execute_command(command, timeout=timeout)
        if result.exit_code == 0:
            return
        output = f"{result.stdout or ''}{result.stderr or ''}"
        if attempt + 1 == attempts or _FILE_CHANGED_MARKER not in output:
            detail = (result.stdout or "").strip() or (result.stderr or "").strip()
            suffix = f" ({detail})" if detail else ""
            raise WorkspaceStagingError(
                f"Failed to {action} in the execution workspace: "
                f"exit code {result.exit_code}{suffix}"
            )


def _remove_remote_archive(workspace: Any, archive: str, timeout: float) -> None:
    try:
        _run_in_sandbox(
            workspace,
            f"rm -f {shlex.quote(archive)}",
            timeout=timeout,
            action=f"clean up {archive}",
        )
    except WorkspaceStagingError:
        pass


def _remote_file_size(workspace: Any, path: str, timeout: float) -> int | None:
    """Size of one sandbox file, so a short download is detectable.

    The terminal bridge echoes the command line before the output, so the size
    is the last line that is nothing but digits. The probe is best effort: when
    the workspace cannot answer, extraction still decides whether the archive
    was usable.
    """
    try:
        result = workspace.execute_command(
            f"wc -c < {shlex.quote(path)}", timeout=timeout
        )
    except Exception:  # noqa: BLE001
        return None
    if result.exit_code != 0:
        return None
    for line in reversed((result.stdout or "").splitlines()):
        candidate = line.strip()
        if candidate.isdigit():
            return int(candidate)
    return None


def _remote_archive_path() -> str:
    return posixpath.join("/tmp", f"{_STAGE_PREFIX}{uuid.uuid4().hex}.tgz")


@contextmanager
def _local_archive() -> Iterator[Path]:
    with tempfile.TemporaryDirectory(prefix=_STAGE_PREFIX) as directory:
        yield Path(directory) / _ARCHIVE_NAME
