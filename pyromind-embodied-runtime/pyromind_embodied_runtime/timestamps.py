"""Timestamp-domain helpers for self-collected robot recordings."""

from __future__ import annotations

import bisect
from typing import Any


def sensor_timestamp_ns(row: dict[str, Any]) -> int | None:
    direct = _int_or_none(row.get("stamp_ns"))
    if direct is not None:
        return direct
    seconds = _int_or_none(row.get("stamp_sec"))
    nanoseconds = _int_or_none(row.get("stamp_nsec"))
    if seconds is None or nanoseconds is None or not 0 <= nanoseconds < 10**9:
        return None
    return seconds * 10**9 + nanoseconds


def sensor_time_s(row: dict[str, Any]) -> float | None:
    timestamp_ns = sensor_timestamp_ns(row)
    return timestamp_ns / 10**9 if timestamp_ns is not None else None


def wall_time_s(row: dict[str, Any]) -> float | None:
    return _float_or_none(row.get("wall_time"))


def common_alignment_times(
    state_rows: list[dict[str, Any]],
    camera_rows: list[dict[str, Any]],
) -> tuple[list[float], list[float], str]:
    """Return state and camera times from one common clock domain."""
    state_sensor = _complete_times(state_rows, sensor_time_s, "state sensor")
    camera_sensor = _complete_times(camera_rows, sensor_time_s, "camera sensor")
    if state_sensor is not None and camera_sensor is not None:
        return state_sensor, camera_sensor, "sensor"
    if state_sensor is not None or camera_sensor is not None:
        raise ValueError(
            "state and camera streams do not share a complete sensor timestamp domain"
        )

    state_wall = _complete_times(state_rows, wall_time_s, "state wall")
    camera_wall = _complete_times(camera_rows, wall_time_s, "camera wall")
    if state_wall is None or camera_wall is None:
        raise ValueError("state and camera streams have no common timestamp domain")
    return state_wall, camera_wall, "wall"


def nearest_timestamp_indexes(
    reference_rows: list[dict[str, Any]],
    target_rows: list[dict[str, Any]],
    reference_indexes: list[int],
    *,
    max_gap_s: float,
) -> list[int]:
    """Map selected reference rows to nearest rows in a shared clock domain."""
    if max_gap_s <= 0:
        raise ValueError("max_gap_s must be positive")
    if not reference_rows or not target_rows:
        raise ValueError("reference and target timestamp rows are required")
    reference_times, target_times, _ = common_alignment_times(
        reference_rows,
        target_rows,
    )
    if any(
        current < previous
        for previous, current in zip(reference_times, reference_times[1:])
    ):
        raise ValueError("reference timestamps are decreasing")
    if any(
        current < previous for previous, current in zip(target_times, target_times[1:])
    ):
        raise ValueError("target timestamps are decreasing")

    aligned: list[int] = []
    for reference_index in reference_indexes:
        if not 0 <= reference_index < len(reference_times):
            raise ValueError(
                f"reference timestamp index is out of range: {reference_index}"
            )
        reference_time = reference_times[reference_index]
        after_index = bisect.bisect_left(target_times, reference_time)
        if after_index == 0:
            target_index = 0
        elif after_index == len(target_times):
            target_index = len(target_times) - 1
        else:
            before_index = after_index - 1
            if abs(target_times[before_index] - reference_time) <= abs(
                target_times[after_index] - reference_time
            ):
                target_index = before_index
            else:
                target_index = after_index
        gap_s = abs(target_times[target_index] - reference_time)
        if gap_s > max_gap_s:
            raise ValueError(
                f"reference timestamp row {reference_index} is {gap_s:.6f}s "
                "from the nearest target row"
            )
        aligned.append(target_index)
    return aligned


def _complete_times(
    rows: list[dict[str, Any]],
    reader,
    label: str,
) -> list[float] | None:
    values = [reader(row) for row in rows]
    available = [value is not None for value in values]
    if not any(available):
        return None
    if not all(available):
        raise ValueError(f"{label} timestamps are only partially populated")
    return [float(value) for value in values if value is not None]


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
