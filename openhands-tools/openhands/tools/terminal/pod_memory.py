"""Read-only pod memory telemetry and an admission gate for the terminal.

The container's cgroup v2 files are mounted read-only, but reads are
allowed: ``memory.current`` and ``memory.max`` expose the pod's own usage
and the kubelet-imposed limit. Gating new terminal commands on the live
headroom lets the app manage the shared pod memory pool without cgroup
write delegation.

The gate is best effort: once usage crosses
``OH_POD_MEMORY_HIGH_WATER`` (default 0.85) of ``memory.max``, new
commands are refused until memory is reclaimed. When the cgroup files are
unreadable (local dev, non-Linux hosts) the gate allows everything; the
pod-level kubelet limit remains the final barrier in every case.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from openhands.sdk.logger import get_logger


logger = get_logger(__name__)

POD_MEMORY_CGROUP_ROOT_ENV = "OH_POD_MEMORY_CGROUP_ROOT"
POD_MEMORY_HIGH_WATER_ENV = "OH_POD_MEMORY_HIGH_WATER"
DEFAULT_POD_MEMORY_HIGH_WATER = 0.85

_MEMORY_CURRENT = "memory.current"
_MEMORY_MAX = "memory.max"


@dataclass(frozen=True)
class PodMemoryStats:
    """Snapshot of the pod's current and max memory from cgroup v2."""

    current_bytes: int
    max_bytes: int

    @property
    def usage_ratio(self) -> float:
        return self.current_bytes / self.max_bytes if self.max_bytes else 0.0


def pod_memory_cgroup_root() -> str:
    """Cgroup v2 memory root, overridable via env for tests/unusual layouts."""
    return os.environ.get(POD_MEMORY_CGROUP_ROOT_ENV, "").strip() or "/sys/fs/cgroup"


def _cgroup_file_path(root: str, name: str) -> str | None:
    direct = os.path.join(root, name)
    if os.path.isfile(direct):
        return direct
    # With cgroup namespacing disabled the container sees the host tree;
    # resolve its own subtree from /proc/self/cgroup ("0::/kubepods/...").
    try:
        with open("/proc/self/cgroup", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("0::/"):
                    sub = line[3:].strip("/")
                    if sub:
                        candidate = os.path.join(root, sub, name)
                        if os.path.isfile(candidate):
                            return candidate
    except OSError:
        pass
    return None


def read_pod_memory_stats() -> PodMemoryStats | None:
    """Return the pod's live memory snapshot, or None when unavailable."""
    root = pod_memory_cgroup_root()
    current_path = _cgroup_file_path(root, _MEMORY_CURRENT)
    max_path = _cgroup_file_path(root, _MEMORY_MAX)
    if current_path is None or max_path is None:
        return None
    try:
        with open(current_path, encoding="utf-8") as fh:
            current = int(fh.read().strip())
        with open(max_path, encoding="utf-8") as fh:
            raw_max = fh.read().strip()
    except (OSError, ValueError):
        return None
    if raw_max == "max":
        return None
    try:
        return PodMemoryStats(current_bytes=current, max_bytes=int(raw_max))
    except ValueError:
        return None


def pod_memory_high_water() -> float:
    """High-water ratio for the admission gate; 0 disables the gate."""
    raw = os.environ.get(POD_MEMORY_HIGH_WATER_ENV, "").strip()
    if not raw:
        return DEFAULT_POD_MEMORY_HIGH_WATER
    try:
        value = float(raw)
    except ValueError:
        logger.warning("invalid %s=%r ignored", POD_MEMORY_HIGH_WATER_ENV, raw)
        return DEFAULT_POD_MEMORY_HIGH_WATER
    if value <= 0:
        return 0.0
    return min(value, 1.0)


def pod_memory_gate_allowed() -> tuple[bool, PodMemoryStats | None, float]:
    """Decide whether a new terminal command may run.

    Returns ``(allowed, stats, high_water)``. When telemetry is
    unavailable or the gate is disabled, ``allowed`` is always True.
    """
    stats = read_pod_memory_stats()
    if stats is None:
        return True, None, 0.0
    high_water = pod_memory_high_water()
    if high_water <= 0:
        return True, stats, high_water
    return stats.usage_ratio < high_water, stats, high_water
