"""Per-conversation storage quotas backed by XFS project quotas.

Each conversation directory maps to its own project id so that writes made
inside the sandbox (including direct bash writes) hit the filesystem quota
hard limit instead of relying on app-level accounting. The workspace
filesystem must be mounted with ``prjquota`` and ``xfs_quota`` must exist;
otherwise calls are logged and storage stays unlimited.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import stat
import subprocess
from pathlib import Path
from uuid import UUID


logger = logging.getLogger(__name__)

OH_CONVERSATION_STORAGE_QUOTA_ENV = "OH_CONVERSATION_STORAGE_QUOTA"
_MAX_PROJECT_ID = (1 << 31) - 1
_SIZE_UNITS = {
    "": 1,
    "k": 1024,
    "m": 1024**2,
    "g": 1024**3,
    "t": 1024**4,
    "p": 1024**5,
}
_SIZE_RE = re.compile(r"^(?P<number>\d+)\s*(?P<unit>[kmgtp]?)$", re.IGNORECASE)


def parse_storage_size(text: str) -> int:
    """Parse a human readable size (``50M``, ``1G``, ``512``) into bytes."""
    match = _SIZE_RE.fullmatch(text.strip())
    if match is None:
        raise ValueError(f"invalid storage size: {text!r}")
    number = int(match.group("number"))
    unit = match.group("unit").lower()
    return number * _SIZE_UNITS[unit]


def project_id_for(conversation_id: UUID) -> int:
    """Return a stable positive XFS project id for a conversation."""
    return int.from_bytes(conversation_id.bytes, "big") % _MAX_PROJECT_ID + 1


def _mountpoint_for(path: Path) -> Path | None:
    """Return the deepest mountpoint that contains ``path``."""
    try:
        device = path.stat().st_dev
    except OSError:
        return None
    deepest: Path | None = None
    try:
        lines = Path("/proc/self/mountinfo").read_text().splitlines()
    except OSError:
        return None
    for line in lines:
        mount_point = line.split()[4]
        try:
            mount_device = os.stat(mount_point).st_dev
        except OSError:
            continue
        if mount_device != device:
            continue
        candidate = Path(mount_point)
        if deepest is None or len(candidate.parts) > len(deepest.parts):
            deepest = candidate
    return deepest


def _device_node_for(mount_point: Path) -> Path | None:
    """Return a block-device node for ``mount_point``, or ``None``.

    Containers do not expose the EBS backing device under ``/dev``, which
    ``xfs_quota`` needs to open to read/write quota state. Recreate the node
    from the ``major:minor`` pair in ``/proc/self/mountinfo`` (requires
    ``CAP_MKNOD``/``CAP_SYS_ADMIN``, which the pod already has), and fall
    back to ``None`` so callers keep passing the mount point.
    """
    major_minor: tuple[int, int] | None = None
    try:
        lines = Path("/proc/self/mountinfo").read_text().splitlines()
    except OSError:
        return None
    for line in lines:
        fields = line.split()
        if len(fields) < 7 or Path(convert_mount_point(fields[4])) != mount_point:
            continue
        parts = fields[2].split(":")
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            major_minor = (int(parts[0]), int(parts[1]))
            break
    if major_minor is None:
        return None
    device = Path("/dev") / f"openhands-quota-{major_minor[0]}-{major_minor[1]}"
    if device.exists():
        return device
    try:
        os.mknod(device, stat.S_IFBLK | 0o600, os.makedev(*major_minor))
    except OSError:
        return None
    return device


def convert_mount_point(field: str) -> str:
    """Unescape a mountinfo path field (octal `\040` etc.)."""
    return re.sub(r"\\([0-7]{3})", lambda m: chr(int(m.group(1), 8)), field)


class ConversationStorageQuota:
    """Set and remove per-conversation XFS project quota limits."""

    def __init__(self, limit_bytes: int | None, *, xfs_quota: str = "xfs_quota"):
        self._limit_bytes = limit_bytes
        self._xfs_quota = xfs_quota

    @property
    def limit_bytes(self) -> int | None:
        return self._limit_bytes

    def apply(self, directory: Path, conversation_id: UUID) -> bool:
        """Assign a project id to the conversation dir and cap its size."""
        if self._limit_bytes is None:
            return False
        mount_point = _mountpoint_for(directory)
        if mount_point is None:
            logger.warning("storage quota: cannot resolve mountpoint for %s", directory)
            return False
        project_id = project_id_for(conversation_id)
        commands = (
            f"project -s -p {directory} {project_id}",
            f"limit -p bhard={self._limit_bytes} {project_id}",
        )
        return all(self._run(mount_point, command) for command in commands)

    def remove(self, directory: Path, conversation_id: UUID) -> bool:
        """Clear the project quota when a conversation is deleted."""
        if self._limit_bytes is None:
            return False
        mount_point = _mountpoint_for(directory)
        if mount_point is None:
            return False
        project_id = project_id_for(conversation_id)
        return self._run(mount_point, f"project -c {project_id}")

    def _run(self, mount_point: Path, command: str) -> bool:
        if shutil.which(self._xfs_quota) is None:
            logger.warning(
                "storage quota requested but %s is unavailable", self._xfs_quota
            )
            return False
        target = _device_node_for(mount_point) or mount_point
        try:
            result = subprocess.run(
                [self._xfs_quota, "-x", "-c", command, str(target)],
                capture_output=True,
                text=True,
                timeout=30,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            logger.warning("storage quota command failed: %s", exc)
            return False
        if result.returncode != 0:
            logger.warning(
                "%s failed: %s", command, result.stderr.strip() or result.stdout.strip()
            )
            return False
        return True


def quota_from_env() -> ConversationStorageQuota:
    """Build a quota manager configured from the environment."""
    value = os.environ.get(OH_CONVERSATION_STORAGE_QUOTA_ENV)
    if value is None:
        return ConversationStorageQuota(None)
    return ConversationStorageQuota(parse_storage_size(value))
