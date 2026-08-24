"""Tests for the read-only pod memory gate helpers."""

from pathlib import Path

import pytest

from openhands.tools.terminal.pod_memory import (
    DEFAULT_POD_MEMORY_HIGH_WATER,
    POD_MEMORY_CGROUP_ROOT_ENV,
    POD_MEMORY_HIGH_WATER_ENV,
    pod_memory_gate_allowed,
    pod_memory_high_water,
    read_pod_memory_stats,
)


def _write_cgroup(root: Path, current: int, max_bytes: int | str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "memory.current").write_text(str(current))
    (root / "memory.max").write_text(str(max_bytes))


def test_read_stats(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv(POD_MEMORY_CGROUP_ROOT_ENV, str(tmp_path))
    _write_cgroup(tmp_path, 1024**3, 8 * 1024**3)
    stats = read_pod_memory_stats()
    assert stats is not None
    assert stats.current_bytes == 1024**3
    assert stats.max_bytes == 8 * 1024**3
    assert stats.usage_ratio == pytest.approx(0.125)


def test_stats_unavailable_without_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(POD_MEMORY_CGROUP_ROOT_ENV, str(tmp_path))
    assert read_pod_memory_stats() is None


def test_stats_unavailable_when_unlimited(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(POD_MEMORY_CGROUP_ROOT_ENV, str(tmp_path))
    _write_cgroup(tmp_path, 1024**3, "max")
    assert read_pod_memory_stats() is None


def test_high_water_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(POD_MEMORY_HIGH_WATER_ENV, raising=False)
    assert pod_memory_high_water() == DEFAULT_POD_MEMORY_HIGH_WATER


def test_high_water_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(POD_MEMORY_HIGH_WATER_ENV, "0")
    assert pod_memory_high_water() == 0.0


def test_high_water_invalid_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(POD_MEMORY_HIGH_WATER_ENV, "banana")
    assert pod_memory_high_water() == DEFAULT_POD_MEMORY_HIGH_WATER


def test_gate_allows_below_high_water(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(POD_MEMORY_CGROUP_ROOT_ENV, str(tmp_path))
    _write_cgroup(tmp_path, 4 * 1024**3, 8 * 1024**3)
    allowed, stats, high_water = pod_memory_gate_allowed()
    assert allowed is True
    assert stats is not None
    assert high_water == DEFAULT_POD_MEMORY_HIGH_WATER


def test_gate_blocks_above_high_water(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(POD_MEMORY_CGROUP_ROOT_ENV, str(tmp_path))
    _write_cgroup(tmp_path, 7 * 1024**3, 8 * 1024**3)
    allowed, stats, _ = pod_memory_gate_allowed()
    assert allowed is False
    assert stats is not None
    assert stats.usage_ratio == pytest.approx(0.875)


def test_gate_disabled_always_allows(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(POD_MEMORY_CGROUP_ROOT_ENV, str(tmp_path))
    monkeypatch.setenv(POD_MEMORY_HIGH_WATER_ENV, "0")
    _write_cgroup(tmp_path, 7 * 1024**3, 8 * 1024**3)
    allowed, stats, high_water = pod_memory_gate_allowed()
    assert allowed is True
    assert stats is not None
    assert high_water == 0.0


def test_gate_allows_when_telemetry_unavailable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(POD_MEMORY_CGROUP_ROOT_ENV, str(tmp_path / "missing"))
    allowed, stats, _ = pod_memory_gate_allowed()
    assert allowed is True
    assert stats is None
