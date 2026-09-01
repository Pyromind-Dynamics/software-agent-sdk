"""Per-conversation storage quotas backed by XFS project quotas.

Each conversation directory maps to its own project id so that writes made
inside the sandbox (including direct bash writes) hit the filesystem quota
hard limit instead of relying on app-level accounting. The workspace
filesystem must be mounted with ``prjquota`` and ``xfs_quota`` must exist.
Failures are recorded on :attr:`ConversationStorageQuota.last_error`; when
``OH_STORAGE_QUOTA_REQUIRED`` is set the caller fails closed, otherwise
storage stays unlimited and the failure is only logged. ``OH_STORAGE_QUOTA_DEVICE``
can point at the real block device when it is mapped into the container.
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
OH_STORAGE_QUOTA_REQUIRED_ENV = "OH_STORAGE_QUOTA_REQUIRED"
OH_STORAGE_QUOTA_DEVICE_ENV = "OH_STORAGE_QUOTA_DEVICE"
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

    Prefer the device path already exposed by the mount (e.g. a privileged
    container sees ``/dev/nvme1n1``), which ``xfs_quota`` must open to
    read/write quota state. Otherwise recreate the node from the
    ``major:minor`` pair in ``/proc/self/mountinfo`` (requires
    ``CAP_MKNOD``/``CAP_SYS_ADMIN``), and fall back to ``None`` so callers
    keep passing the mount point.
    """
    major_minor: tuple[int, int] | None = None
    source_path: Path | None = None
    try:
        lines = Path("/proc/self/mountinfo").read_text().splitlines()
    except OSError:
        return None
    for line in lines:
        fields = line.split()
        if len(fields) < 7 or Path(convert_mount_point(fields[4])) != mount_point:
            continue
        if "-" in fields:
            separator = fields.index("-")
            source = fields[separator + 2] if separator + 2 < len(fields) else ""
            if source.startswith("/"):
                source_path = Path(source)
        parts = fields[2].split(":")
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            major_minor = (int(parts[0]), int(parts[1]))
        break
    if source_path is not None:
        try:
            if stat.S_ISBLK(source_path.stat().st_mode):
                return source_path
        except OSError:
            pass
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


def _fs_type_and_options(mount_point: Path) -> tuple[str, tuple[str, ...]]:
    """Return the filesystem type and mount options for ``mount_point``."""
    try:
        lines = Path("/proc/self/mountinfo").read_text().splitlines()
    except OSError:
        return "", ()
    for line in lines:
        fields = line.split()
        if len(fields) < 6 or Path(convert_mount_point(fields[4])) != mount_point:
            continue
        if "-" not in fields:
            break
        separator = fields.index("-")
        if separator + 3 >= len(fields):
            break
        return fields[separator + 1], tuple(fields[separator + 3].split(","))
    return "", ()


def convert_mount_point(field: str) -> str:
    """Unescape a mountinfo path field (octal `\040` etc.)."""
    return re.sub(r"\\([0-7]{3})", lambda m: chr(int(m.group(1), 8)), field)


def storage_quota_required() -> bool:
    """Whether a configured storage quota must be enforced or startup fails."""
    return os.environ.get(OH_STORAGE_QUOTA_REQUIRED_ENV, "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


class ConversationStorageQuota:
    """Set and remove per-conversation XFS project quota limits."""

    def __init__(
        self,
        limit_bytes: int | None,
        *,
        xfs_quota: str = "xfs_quota",
        device: Path | None = None,
    ):
        self._limit_bytes = limit_bytes
        self._xfs_quota = xfs_quota
        self._device = device
        self._last_error: str | None = None

    @property
    def limit_bytes(self) -> int | None:
        return self._limit_bytes

    @property
    def device(self) -> Path | None:
        """Block device xfs_quota should target, if configured."""
        return self._device

    @property
    def last_error(self) -> str | None:
        """Why the most recent quota command failed, if it did."""
        return self._last_error

    def apply(self, directory: Path, conversation_id: UUID) -> bool:
        """Assign a project id to the conversation dir and cap its size."""
        if self._limit_bytes is None:
            return False
        mount_point = _mountpoint_for(directory)
        if mount_point is None:
            self._last_error = f"cannot resolve mountpoint for {directory}"
            logger.warning("storage quota: %s", self._last_error)
            return False
        project_id = project_id_for(conversation_id)
        commands = (
            f"project -s -p {directory} {project_id}",
            f"limit -p bhard={self._limit_bytes} {project_id}",
        )
        self._last_error = None
        return all(self._run(mount_point, command) for command in commands)

    def remove(self, directory: Path, conversation_id: UUID) -> bool:
        """Clear the project quota when a conversation is deleted."""
        if self._limit_bytes is None:
            return False
        mount_point = _mountpoint_for(directory)
        if mount_point is None:
            return False
        project_id = project_id_for(conversation_id)
        self._last_error = None
        return self._run(mount_point, f"project -c {project_id}")

    def _run(self, mount_point: Path, command: str) -> bool:
        if shutil.which(self._xfs_quota) is None:
            self._last_error = f"{self._xfs_quota} is unavailable"
            logger.warning("storage quota: %s", self._last_error)
            return False
        target = self._device or _device_node_for(mount_point) or mount_point
        try:
            result = subprocess.run(
                [self._xfs_quota, "-x", "-c", command, str(target)],
                capture_output=True,
                text=True,
                timeout=30,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            self._last_error = f"{command} failed: {exc}"
            logger.warning("storage quota: %s", self._last_error)
            return False
        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip()
            self._last_error = f"{command} failed: {detail}" if detail else command
            logger.warning("storage quota: %s", self._last_error)
            return False
        return True


def quota_from_env() -> ConversationStorageQuota:
    """Build a quota manager configured from the environment."""
    value = os.environ.get(OH_CONVERSATION_STORAGE_QUOTA_ENV)
    if value is None:
        return ConversationStorageQuota(None)
    device_text = os.environ.get(OH_STORAGE_QUOTA_DEVICE_ENV)
    device = Path(device_text) if device_text else None
    return ConversationStorageQuota(parse_storage_size(value), device=device)


def preflight_storage_quota(*, workspace_root: Path | None = None) -> list[str]:
    """Return deployment problems that would prevent the storage quota.

    When a quota is configured this verifies that the workspace filesystem is
    XFS with project quotas enabled (``prjquota``) and that ``xfs_quota`` can
    open the backing device. Callers decide whether the returned problems
    should fail startup or only be logged.
    """
    quota = quota_from_env()
    if quota.limit_bytes is None:
        return []
    root = workspace_root or Path(
        os.environ.get("workspace_dir")
        or os.environ.get("WORKSPACE_DIR")
        or "workspace"
    )
    mount_point = _mountpoint_for(root)
    if mount_point is None:
        return [f"cannot resolve mountpoint for {root}"]
    problems: list[str] = []
    fs_type, options = _fs_type_and_options(mount_point)
    if fs_type != "xfs":
        problems.append(f"workspace filesystem is {fs_type or 'unknown'}, XFS required")
    if "prjquota" not in options and "pquota" not in options:
        problems.append(
            f"workspace mount lacks prjquota option ({', '.join(options) or 'none'})"
        )
    if not quota._run(mount_point, "report -p"):
        problems.append(f"xfs_quota cannot open device: {quota.last_error}")
    return problems
