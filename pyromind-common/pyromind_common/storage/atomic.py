"""File-based storage implementations for settings and secrets.

Following the same pattern as OpenHands app-server's FileSettingsStore
and FileSecretsStore for consistency.

File locking uses fcntl on Unix and msvcrt on Windows.
"""

from __future__ import annotations

import json
import os
import stat
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from pyromind_common.sdk.logger import get_logger


# fcntl is Unix-only; on Windows, use msvcrt for file locking
if sys.platform != "win32":
    import fcntl

    msvcrt = None
else:
    fcntl = None  # type: ignore[assignment]
    import msvcrt


logger = get_logger(__name__)

# File permission constants (owner read/write only)
_DIR_MODE = stat.S_IRWXU  # 0o700 - rwx------
_FILE_MODE = stat.S_IRUSR | stat.S_IWUSR  # 0o600 - rw-------

# Windows reserved filenames (case-insensitive)
_WINDOWS_RESERVED_NAMES = frozenset(
    {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        "COM1",
        "COM2",
        "COM3",
        "COM4",
        "COM5",
        "COM6",
        "COM7",
        "COM8",
        "COM9",
        "LPT1",
        "LPT2",
        "LPT3",
        "LPT4",
        "LPT5",
        "LPT6",
        "LPT7",
        "LPT8",
        "LPT9",
    }
)


def _validate_filename(filename: str) -> None:
    """Validate filename to prevent path traversal and injection attacks.

    Raises:
        ValueError: If filename is invalid or potentially dangerous.
    """
    # Check for empty filename (would resolve to parent directory)
    if not filename:
        raise ValueError("filename must not be empty")

    # Check for path separators
    if "/" in filename or "\\" in filename:
        raise ValueError("filename must not contain path separators")

    # Check for leading dots (hidden files, parent directory traversal)
    if filename.startswith("."):
        raise ValueError("filename must not start with '.'")

    # Check for null bytes (null byte injection)
    if "\x00" in filename:
        raise ValueError("filename must not contain null bytes")

    # Check for trailing dots/spaces (Windows path handling issues)
    if filename.endswith(".") or filename.endswith(" "):
        raise ValueError("filename must not end with '.' or space")

    # Check for Windows reserved names (split handles multi-extension files)
    # e.g., "CON.txt.json" -> "CON" not "CON.txt"
    basename = filename.split(".")[0].upper()
    if basename in _WINDOWS_RESERVED_NAMES:
        raise ValueError(f"filename '{filename}' uses a reserved name")


def _ensure_secure_directory(path: Path) -> None:
    """Ensure directory exists with secure permissions.

    Creates all parent directories with secure permissions (0o700).
    If it already exists, ensures permissions are correct.
    """
    if not path.exists():
        # Create parents with secure permissions
        current = path
        to_create: list[Path] = []
        while not current.exists():
            to_create.append(current)
            current = current.parent

        for dir_path in reversed(to_create):
            dir_path.mkdir(mode=_DIR_MODE, exist_ok=True)

    # Ensure permissions are correct even if dir already existed
    try:
        path.chmod(_DIR_MODE)
    except OSError as e:
        logger.warning(f"Failed to set permissions on {path}: {e}")


@contextmanager
def _file_lock(lock_path: Path) -> Iterator[None]:
    """Context manager for file-based locking.

    Uses Unix fcntl for exclusive locking to prevent race conditions during
    read-modify-write operations. On Windows, uses msvcrt.locking.
    """
    _ensure_secure_directory(lock_path.parent)

    # Create lock file - use O_RDWR for Windows compatibility with msvcrt
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, _FILE_MODE)
    try:
        if fcntl is not None:
            # Unix: use fcntl for file locking
            fcntl.flock(fd, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(fd, fcntl.LOCK_UN)
        elif msvcrt is not None:
            # Windows: use msvcrt for file locking
            # Lock multiple bytes for more reliable locking behavior
            os.lseek(fd, 0, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_LOCK, 100)
            try:
                yield
            finally:
                os.lseek(fd, 0, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_UNLCK, 100)
        else:
            # This should never happen on standard systems (Unix or Windows)
            # Raise an error rather than silently proceeding without locking,
            # which could cause data corruption from concurrent writes
            raise RuntimeError(
                "File locking not available on this platform. "
                "Concurrent writes may cause data corruption."
            )
    finally:
        os.close(fd)


def _atomic_write_json(path: Path, data: dict) -> None:
    """Write JSON atomically with secure permissions.

    Uses write-to-temp-then-rename pattern to prevent corruption
    if interrupted. Creates temp file with owner-only permissions from
    the start to prevent race conditions where sensitive data could
    be read before chmod.

    Note:
        The rename operation (Path.replace) is atomic on POSIX systems.
        On Windows, it may not be fully atomic in all edge cases (e.g.,
        concurrent access, network drives), but provides reasonable
        protection against corruption from interrupted writes.
    """
    import uuid

    # Use PID, time, and uuid for unique temp filename to prevent collisions
    # when multiple processes/threads write to the same file concurrently
    unique_suffix = f".tmp.{os.getpid()}.{uuid.uuid4().hex[:8]}"
    tmp_path = path.with_suffix(unique_suffix)
    # Create file with secure permissions from the start using os.open
    # O_EXCL ensures exclusive creation (fails if file exists)
    fd = os.open(tmp_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, _FILE_MODE)
    fdopen_succeeded = False
    try:
        f = os.fdopen(fd, "w", encoding="utf-8")
        fdopen_succeeded = True
        with f:
            json.dump(data, f, indent=2)
    except Exception:
        # Only close fd manually if os.fdopen() didn't take ownership
        if not fdopen_succeeded:
            try:
                os.close(fd)
            except OSError:
                pass
        # Clean up temp file on error
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise

    # Atomic rename - clean up temp file if replace() fails
    try:
        tmp_path.replace(path)  # Atomic on POSIX
    except Exception:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


# Default storage directory (relative to working directory)
