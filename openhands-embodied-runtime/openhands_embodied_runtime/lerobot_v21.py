"""Materialize self-collected robot episodes as LeRobotDataset v2.1."""

from __future__ import annotations

import json
import shutil
from collections.abc import Sequence
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import imageio_ffmpeg
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict, Field

from openhands_embodied_runtime.adapters import (
    S2_REFERENCE_STATE_ORDER,
    SelfCollectedAdapter,
    load_self_collected_camera_rows,
    load_self_collected_signals,
)
from openhands_embodied_runtime.models import (
    EpisodePlan,
    FeatureSpec,
    QualityStatus,
    StreamStatus,
    TimelineMapping,
)
from openhands_embodied_runtime.planning import finalize_episode_plan
from openhands_embodied_runtime.timestamps import (
    nearest_timestamp_indexes,
    sensor_time_s,
)


LEROBOT_CODEBASE_VERSION = "v2.1"
DATA_PATH_TEMPLATE = (
    "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
)
VIDEO_PATH_TEMPLATE = (
    "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
)


class LeRobotV21MaterializationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    root: str
    episode_count: int = Field(ge=1)
    frame_count: int = Field(ge=1)
    video_count: int = Field(ge=1)
    omitted_streams: list[str] = Field(default_factory=list)


class LeRobotV21ValidationReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    valid: bool
    episode_count: int = Field(ge=0)
    frame_count: int = Field(ge=0)
    video_count: int = Field(ge=0)
    structurally_valid: bool = False
    reference_profile_valid: bool = False
    errors: list[str] = Field(default_factory=list)


def materialize_lerobot_v21_episode(
    source: Path,
    plan: EpisodePlan,
    output_root: Path,
    *,
    task_text: str,
    confirm_subtasks: bool,
    confirm_derived_action: bool = False,
    robot_type: str = "s2",
) -> LeRobotV21MaterializationResult:
    """Write one accepted self-collected episode in the reference v2.1 layout."""
    prepared = finalize_episode_plan(
        source,
        plan,
        task_text=task_text,
        confirm_subtasks=confirm_subtasks,
        confirm_derived_action=confirm_derived_action,
    )
    return materialize_finalized_lerobot_v21_episode(
        source,
        prepared,
        output_root,
        robot_type=robot_type,
    )


def materialize_finalized_lerobot_v21_episode(
    source: Path,
    prepared: EpisodePlan,
    output_root: Path,
    *,
    robot_type: str = "s2",
) -> LeRobotV21MaterializationResult:
    """Write an already finalized plan without rebuilding it a second time."""
    output_root = output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise ValueError("output_root must not contain existing files")
    if output_root.exists():
        output_root.rmdir()

    if prepared.quality.status != QualityStatus.ACCEPTED:
        details = "; ".join(prepared.quality.errors + prepared.quality.warnings)
        raise ValueError(f"episode plan is not accepted: {details}")
    if not prepared.timeline_mapping:
        raise ValueError("episode plan has no retained timeline mapping")
    if prepared.source_timeline.fps is None:
        raise ValueError("source MP4 FPS is unavailable")

    adapter = SelfCollectedAdapter(source)
    episode_dir = adapter.select(prepared.episode_id)
    state_rows, _ = load_self_collected_signals(episode_dir)
    state_features = prepared.feature_schema.observation_state
    states = [
        _aligned_state(mapping, state_rows, state_features)
        for mapping in prepared.timeline_mapping
    ]
    actions = [*states[1:], states[-1].copy()]
    fps = int(round(prepared.source_timeline.fps))

    output_root.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(
        prefix=f".{output_root.name}-",
        dir=output_root.parent,
    ) as temporary_dir:
        staging_root = Path(temporary_dir)
        video_features, omitted_streams = _write_videos(
            episode_dir,
            prepared,
            staging_root,
            fps,
        )
        _write_parquet(staging_root, states, actions, fps)
        _write_metadata(
            staging_root,
            prepared,
            states,
            actions,
            fps,
            robot_type,
            video_features,
        )

        report = validate_lerobot_v21_dataset(staging_root)
        if not report.valid:
            raise RuntimeError(
                "generated LeRobot v2.1 dataset is invalid: " + "; ".join(report.errors)
            )
        staging_root.rename(output_root)
    return LeRobotV21MaterializationResult(
        root=str(output_root),
        episode_count=1,
        frame_count=len(states),
        video_count=len(video_features),
        omitted_streams=omitted_streams,
    )


def merge_lerobot_v21_datasets(
    sources: Sequence[Path],
    output_root: Path,
    *,
    task_text: str,
) -> LeRobotV21MaterializationResult:
    """Merge validated single-episode datasets using one confirmed task."""
    if not sources:
        raise ValueError("at least one input dataset is required")
    task_text = task_text.strip()
    if not task_text:
        raise ValueError("task_text must not be empty")

    output_root = output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise ValueError("output_root must not contain existing files")
    if output_root.exists():
        output_root.rmdir()

    resolved_sources = [source.resolve() for source in sources]
    if len(set(resolved_sources)) != len(resolved_sources):
        raise ValueError("input dataset paths must be unique")

    source_infos: list[dict[str, Any]] = []
    for source in resolved_sources:
        report = validate_lerobot_v21_dataset(source)
        if not report.valid:
            raise ValueError(
                f"input dataset is invalid: {source}: " + "; ".join(report.errors)
            )
        if report.episode_count != 1:
            raise ValueError(f"input dataset must contain one episode: {source}")
        source_infos.append(_read_required_info(source))
    _validate_merge_compatibility(resolved_sources, source_infos)

    base_info = source_infos[0]
    chunks_size = int(base_info["chunks_size"])
    video_keys = [
        key
        for key, value in base_info["features"].items()
        if isinstance(value, dict) and value.get("dtype") == "video"
    ]
    total_frames = 0
    episodes: list[dict[str, Any]] = []
    episode_stats: list[dict[str, Any]] = []

    output_root.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(
        prefix=f".{output_root.name}-",
        dir=output_root.parent,
    ) as temporary_dir:
        staging_root = Path(temporary_dir)
        for episode_index, source in enumerate(resolved_sources):
            source_table = pq.read_table(
                source
                / DATA_PATH_TEMPLATE.format(
                    episode_chunk=0,
                    episode_index=0,
                )
            )
            table = _reindex_episode_table(
                source_table,
                episode_index=episode_index,
                global_start=total_frames,
            )
            episode_chunk = episode_index // chunks_size
            parquet_path = staging_root / DATA_PATH_TEMPLATE.format(
                episode_chunk=episode_chunk,
                episode_index=episode_index,
            )
            parquet_path.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(table, parquet_path)

            for video_key in video_keys:
                source_video = source / VIDEO_PATH_TEMPLATE.format(
                    episode_chunk=0,
                    episode_index=0,
                    video_key=video_key,
                )
                output_video = staging_root / VIDEO_PATH_TEMPLATE.format(
                    episode_chunk=episode_chunk,
                    episode_index=episode_index,
                    video_key=video_key,
                )
                output_video.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_video, output_video)

            episodes.append(
                {
                    "episode_index": episode_index,
                    "tasks": [task_text],
                    "length": table.num_rows,
                }
            )
            episode_stats.append(
                {
                    "episode_index": episode_index,
                    "stats": _table_stats(table),
                }
            )
            total_frames += table.num_rows

        info = dict(base_info)
        info.update(
            {
                "total_episodes": len(resolved_sources),
                "total_frames": total_frames,
                "total_tasks": 1,
                "total_videos": len(video_keys) * len(resolved_sources),
                "total_chunks": (len(resolved_sources) + chunks_size - 1)
                // chunks_size,
                "splits": {"train": f"0:{len(resolved_sources)}"},
            }
        )
        meta = staging_root / "meta"
        meta.mkdir(parents=True, exist_ok=True)
        _write_json(meta / "info.json", info)
        _write_jsonl(meta / "episodes.jsonl", episodes)
        _write_jsonl(
            meta / "tasks.jsonl",
            [{"task_index": 0, "task": task_text}],
        )
        _write_jsonl(meta / "episodes_stats.jsonl", episode_stats)

        report = validate_lerobot_v21_dataset(staging_root)
        if not report.valid:
            raise RuntimeError(
                "merged LeRobot v2.1 dataset is invalid: " + "; ".join(report.errors)
            )
        staging_root.rename(output_root)

    omitted_streams = (
        [] if "observation.images.head_depth" in video_keys else ["head_depth"]
    )
    return LeRobotV21MaterializationResult(
        root=str(output_root),
        episode_count=len(resolved_sources),
        frame_count=total_frames,
        video_count=len(video_keys) * len(resolved_sources),
        omitted_streams=omitted_streams,
    )


def _read_required_info(root: Path) -> dict[str, Any]:
    errors: list[str] = []
    info = _read_json(root / "meta" / "info.json", errors)
    if errors:
        raise ValueError("; ".join(errors))
    return info


def _validate_merge_compatibility(
    sources: Sequence[Path],
    infos: Sequence[dict[str, Any]],
) -> None:
    expected = infos[0]
    fields = (
        "codebase_version",
        "robot_type",
        "fps",
        "chunks_size",
        "features",
        "s2_depth_video_colormap",
    )
    for source, info in zip(sources[1:], infos[1:], strict=True):
        mismatches = [
            field for field in fields if info.get(field) != expected.get(field)
        ]
        if mismatches:
            raise ValueError(
                f"input dataset is incompatible: {source}: " + ", ".join(mismatches)
            )


def _reindex_episode_table(
    table: pa.Table,
    *,
    episode_index: int,
    global_start: int,
) -> pa.Table:
    replacements = {
        "episode_index": pa.array(
            [episode_index] * table.num_rows,
            type=pa.int64(),
        ),
        "index": pa.array(
            range(global_start, global_start + table.num_rows),
            type=pa.int64(),
        ),
        "task_index": pa.array([0] * table.num_rows, type=pa.int64()),
    }
    arrays = [
        replacements.get(name, table[name].combine_chunks())
        for name in table.column_names
    ]
    return pa.Table.from_arrays(arrays, schema=table.schema)


def _table_stats(table: pa.Table) -> dict[str, dict[str, list[Any]]]:
    result: dict[str, dict[str, list[Any]]] = {}
    for name in table.column_names:
        dtype = (
            np.float32
            if name in {"observation.state", "action", "timestamp"}
            else np.int64
        )
        result[name] = _stats(np.asarray(table[name].to_pylist(), dtype=dtype))
    return result


def validate_lerobot_v21_dataset(root: Path) -> LeRobotV21ValidationReport:
    """Validate the required v2.1 files, row counts, indexes, and actions."""
    root = root.resolve()
    structural_errors: list[str] = []
    reference_errors: list[str] = []
    info = _read_json(root / "meta" / "info.json", structural_errors)
    episodes = _read_jsonl(root / "meta" / "episodes.jsonl", structural_errors)
    tasks = _read_jsonl(root / "meta" / "tasks.jsonl", structural_errors)
    episode_stats = _read_jsonl(
        root / "meta" / "episodes_stats.jsonl",
        structural_errors,
    )
    if info and info.get("codebase_version") != LEROBOT_CODEBASE_VERSION:
        structural_errors.append("meta/info.json codebase_version must be 'v2.1'")
    if len(tasks) != 1:
        structural_errors.append("meta/tasks.jsonl must contain exactly one task")
    if len(episode_stats) != len(episodes):
        structural_errors.append(
            "episodes_stats.jsonl count does not match episodes.jsonl"
        )
    episode_rows = _indexed_rows(
        episodes,
        "episode_index",
        "episodes.jsonl",
        structural_errors,
    )
    task_rows = _indexed_rows(
        tasks,
        "task_index",
        "tasks.jsonl",
        structural_errors,
    )
    stats_rows = _indexed_rows(
        episode_stats,
        "episode_index",
        "episodes_stats.jsonl",
        structural_errors,
    )
    if sorted(episode_rows) != list(range(len(episodes))):
        structural_errors.append("episode indexes must be zero-based and contiguous")
    if sorted(task_rows) != list(range(len(tasks))):
        structural_errors.append("task indexes must be zero-based and contiguous")

    total_frames = 0
    total_videos = 0
    episode_lengths: dict[int, int] = {}
    expected_fps = (
        info.get("fps") if isinstance(info.get("fps"), (int, float)) else None
    )
    if expected_fps is None or expected_fps <= 0:
        structural_errors.append("info.json fps must be positive")
        expected_fps = None
    features = info.get("features", {}) if info else {}
    if not isinstance(features, dict):
        structural_errors.append("info.json features must be an object")
        features = {}

    for episode_index, episode in sorted(episode_rows.items()):
        expected_length = _nonnegative_integer_field(
            episode,
            "length",
            f"episode {episode_index}",
            structural_errors,
        )
        parquet_path = root / DATA_PATH_TEMPLATE.format(
            episode_chunk=episode_index // 1000,
            episode_index=episode_index,
        )
        if not parquet_path.is_file():
            structural_errors.append(
                f"missing episode parquet: {parquet_path.relative_to(root)}"
            )
            continue
        try:
            table = pq.read_table(parquet_path)
        except Exception as exc:
            structural_errors.append(
                f"cannot read episode parquet {parquet_path.relative_to(root)}: {exc}"
            )
            continue
        total_frames += table.num_rows
        episode_lengths[episode_index] = table.num_rows
        if expected_length is not None and table.num_rows != expected_length:
            structural_errors.append(
                f"episode {episode_index} Parquet row count is incorrect"
            )
        _validate_episode_table(
            table,
            episode_index,
            structural_errors,
            expected_fps,
            features,
            global_start=total_frames - table.num_rows,
            task_indexes=set(task_rows),
        )
        _validate_episode_stats(
            table,
            episode_index,
            stats_rows.get(episode_index),
            structural_errors,
        )

    video_keys = [
        key
        for key, value in features.items()
        if isinstance(value, dict) and value.get("dtype") == "video"
    ]
    for episode_index in sorted(episode_rows):
        for video_key in video_keys:
            video_path = root / VIDEO_PATH_TEMPLATE.format(
                episode_chunk=episode_index // 1000,
                episode_index=episode_index,
                video_key=video_key,
            )
            if not video_path.is_file():
                structural_errors.append(
                    f"missing episode video: {video_path.relative_to(root)}"
                )
            else:
                total_videos += 1
                try:
                    frame_count, video_fps, width, height = _video_timing(video_path)
                except (OSError, RuntimeError, ValueError) as exc:
                    structural_errors.append(
                        f"cannot read episode video {video_path.relative_to(root)}: "
                        f"{exc}"
                    )
                    continue
                if frame_count != episode_lengths.get(episode_index):
                    structural_errors.append(
                        f"episode video frame count does not match Parquet: "
                        f"{video_path.relative_to(root)}"
                    )
                if expected_fps is not None and not np.isclose(video_fps, expected_fps):
                    structural_errors.append(
                        f"episode video FPS does not match info.json: "
                        f"{video_path.relative_to(root)}"
                    )
                _validate_video_feature(
                    video_key,
                    features[video_key],
                    width,
                    height,
                    video_fps,
                    structural_errors,
                )

    _validate_info_contract(
        info,
        episode_count=len(episodes),
        frame_count=total_frames,
        task_count=len(tasks),
        video_count=total_videos,
        errors=structural_errors,
    )
    reference_profile_valid = _validate_reference_profile(info, reference_errors)
    structural_errors = _unique_errors(structural_errors)
    reference_errors = _unique_errors(reference_errors)
    errors = [*structural_errors, *reference_errors]
    return LeRobotV21ValidationReport(
        valid=not errors,
        episode_count=len(episodes),
        frame_count=total_frames,
        video_count=total_videos,
        structurally_valid=not structural_errors,
        reference_profile_valid=reference_profile_valid,
        errors=errors,
    )


def _aligned_state(
    mapping: TimelineMapping,
    rows: list[dict[str, Any]],
    features: list[FeatureSpec],
) -> np.ndarray:
    before_index = mapping.state_before_index
    after_index = mapping.state_after_index
    weight = mapping.state_interpolation_weight
    if before_index is None or after_index is None or weight is None:
        raise ValueError(
            f"source frame {mapping.source_frame_index} has no state alignment"
        )
    before = rows[before_index]
    after = rows[after_index]
    values = [
        _interpolate_feature(
            rows,
            before_index,
            after_index,
            before,
            after,
            feature,
            weight,
        )
        for feature in features
    ]
    return np.asarray(values, dtype=np.float32)


def _interpolate_feature(
    rows: list[dict[str, Any]],
    before_index: int,
    after_index: int,
    before: dict[str, Any],
    after: dict[str, Any],
    feature: FeatureSpec,
    weight: float,
) -> float:
    before_value = _feature_value_with_sparse_ee(rows, before_index, before, feature)
    after_value = _feature_value_with_sparse_ee(rows, after_index, after, feature)
    if feature.unit == "binary":
        return before_value if weight < 0.5 else after_value
    return before_value + (after_value - before_value) * weight


def _feature_value_with_sparse_ee(
    rows: list[dict[str, Any]],
    row_index: int,
    row: dict[str, Any],
    feature: FeatureSpec,
) -> float:
    try:
        return _feature_value(row, feature)
    except ValueError:
        if feature.source not in {"left_ee_states[4]", "right_ee_states[4]"}:
            raise

    before_index = row_index - 1
    before_value: float | None = None
    while before_index >= 0:
        try:
            before_value = _feature_value(rows[before_index], feature)
            break
        except ValueError:
            before_index -= 1
    after_index = row_index + 1
    after_value: float | None = None
    while after_index < len(rows):
        try:
            after_value = _feature_value(rows[after_index], feature)
            break
        except ValueError:
            after_index += 1
    if before_value is None and after_value is None:
        raise ValueError(f"state feature {feature.name!r} has no recorded values")
    if before_value is None:
        assert after_value is not None
        return after_value
    if after_value is None:
        return before_value

    before_time = sensor_time_s(rows[before_index])
    current_time = sensor_time_s(row)
    after_time = sensor_time_s(rows[after_index])
    if (
        before_time is not None
        and current_time is not None
        and after_time is not None
        and after_time > before_time
    ):
        weight = (current_time - before_time) / (after_time - before_time)
    else:
        weight = (row_index - before_index) / (after_index - before_index)
    return before_value + (after_value - before_value) * weight


def _feature_value(row: dict[str, Any], feature: FeatureSpec) -> float:
    if feature.source.startswith("joints."):
        joints = row.get("joints")
        if not isinstance(joints, dict):
            raise ValueError("state record has no joints object")
        value = joints.get(feature.source.removeprefix("joints."))
    elif feature.source.endswith("]") and "[" in feature.source:
        field, _, raw_index = feature.source[:-1].partition("[")
        values = row.get(field)
        try:
            index = int(raw_index)
        except ValueError as exc:
            raise ValueError(
                f"invalid indexed feature source: {feature.source}"
            ) from exc
        if not isinstance(values, list) or not 0 <= index < len(values):
            raise ValueError(f"state feature source is unavailable: {feature.source}")
        value = values[index]
    else:
        value = row.get(feature.source)
    if not isinstance(value, (int, float)):
        raise ValueError(f"state feature {feature.name!r} is missing or non-numeric")
    return float(value)


def _write_parquet(
    root: Path,
    states: list[np.ndarray],
    actions: list[np.ndarray],
    fps: int,
) -> None:
    frame_count = len(states)
    table = pa.table(
        {
            "observation.state": pa.array(states, type=pa.list_(pa.float32())),
            "action": pa.array(actions, type=pa.list_(pa.float32())),
            "timestamp": pa.array(
                np.arange(frame_count, dtype=np.float32) / fps,
                type=pa.float32(),
            ),
            "frame_index": pa.array(range(frame_count), type=pa.int64()),
            "episode_index": pa.array([0] * frame_count, type=pa.int64()),
            "index": pa.array(range(frame_count), type=pa.int64()),
            "task_index": pa.array([0] * frame_count, type=pa.int64()),
        }
    )
    path = root / DATA_PATH_TEMPLATE.format(episode_chunk=0, episode_index=0)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


def _write_videos(
    episode_dir: Path,
    plan: EpisodePlan,
    root: Path,
    fps: int,
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    features: dict[str, dict[str, Any]] = {}
    omitted: list[str] = []
    selected_by_stream = _selected_video_indexes(episode_dir, plan)
    for name, stream in plan.streams.items():
        if stream.kind == "depth" and stream.status != StreamStatus.AVAILABLE:
            omitted.append(name)
            continue
        if stream.kind not in {"rgb", "depth"}:
            continue
        source_path = episode_dir / stream.path
        selected = selected_by_stream[name]
        video_key = f"observation.images.{name}"
        destination = root / VIDEO_PATH_TEMPLATE.format(
            episode_chunk=0,
            episode_index=0,
            video_key=video_key,
        )
        if source_path.suffix.lower() == ".mp4":
            width, height = _filter_video(source_path, destination, selected, fps)
        elif stream.kind == "depth" and source_path.suffix.lower() == ".u16":
            width, height = _filter_depth_u16(
                source_path,
                episode_dir / f"{name}_meta.json",
                destination,
                selected,
                fps,
                stream.frame_count,
            )
        else:
            omitted.append(name)
            continue
        features[video_key] = _video_feature(height, width, fps)
    if not features:
        raise ValueError("at least one usable MP4 camera stream is required")
    return features, omitted


def _selected_video_indexes(
    episode_dir: Path,
    plan: EpisodePlan,
    *,
    max_gap_s: float = 0.1,
) -> dict[str, list[int]]:
    primary_name = _primary_rgb_stream_name(plan)
    camera_rows = load_self_collected_camera_rows(episode_dir)
    primary_rows = camera_rows.get(primary_name, [])
    primary_stream = plan.streams[primary_name]
    if primary_stream.frame_count != len(primary_rows):
        raise ValueError(
            f"{primary_name}.csv row count does not match its source video"
        )
    selected_primary = [item.source_frame_index for item in plan.timeline_mapping]
    selected: dict[str, list[int]] = {}
    for name, stream in plan.streams.items():
        if stream.kind not in {"rgb", "depth"}:
            continue
        if stream.kind == "depth" and stream.status != StreamStatus.AVAILABLE:
            continue
        rows = camera_rows.get(name, [])
        if stream.frame_count != len(rows):
            raise ValueError(f"{name}.csv row count does not match its source payload")
        if name == primary_name:
            indexes = selected_primary
        else:
            indexes = nearest_timestamp_indexes(
                primary_rows,
                rows,
                selected_primary,
                max_gap_s=max_gap_s,
            )
        if any(current < previous for previous, current in zip(indexes, indexes[1:])):
            raise ValueError(f"{name} timestamp alignment is not monotonic")
        selected[name] = indexes
    return selected


def _primary_rgb_stream_name(plan: EpisodePlan) -> str:
    head = plan.streams.get("head")
    if (
        head is not None
        and head.kind == "rgb"
        and head.status == StreamStatus.AVAILABLE
    ):
        return "head"
    for name, stream in plan.streams.items():
        if stream.kind == "rgb" and stream.status == StreamStatus.AVAILABLE:
            return name
    raise ValueError("at least one available RGB stream is required")


def _filter_video(
    source: Path,
    destination: Path,
    selected_indexes: list[int],
    fps: int,
) -> tuple[int, int]:
    if any(
        current < previous
        for previous, current in zip(selected_indexes, selected_indexes[1:])
    ):
        raise ValueError("selected MP4 indexes must be non-decreasing")
    reader = imageio_ffmpeg.read_frames(str(source), pix_fmt="rgb24")
    metadata = next(reader)
    width, height = metadata["size"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio_ffmpeg.write_frames(
        str(destination),
        (width, height),
        fps=fps,
        codec="mpeg4",
        pix_fmt_in="rgb24",
        pix_fmt_out="yuv420p",
    )
    writer.send(None)
    selected = iter(selected_indexes)
    wanted = next(selected, None)
    written = 0
    try:
        for index, frame in enumerate(reader):
            if wanted is None:
                break
            if index < wanted:
                continue
            while wanted == index:
                writer.send(frame)
                written += 1
                wanted = next(selected, None)
    finally:
        reader.close()
        writer.close()
    if wanted is not None or written != len(selected_indexes):
        raise ValueError("source MP4 does not contain every retained frame")
    return width, height


def _filter_depth_u16(
    source: Path,
    metadata_path: Path,
    destination: Path,
    selected_indexes: list[int],
    fps: int,
    declared_frame_count: int | None,
) -> tuple[int, int]:
    metadata = _read_depth_metadata(metadata_path)
    width = metadata["width"]
    height = metadata["height"]
    depth_scale = metadata["depth_scale"]
    frame_bytes = width * height * np.dtype("<u2").itemsize
    source_size = source.stat().st_size
    if source_size % frame_bytes:
        raise ValueError(
            f"{source.name} size is not divisible by one {width}x{height} u16 frame"
        )
    frame_count = source_size // frame_bytes
    if declared_frame_count is not None and frame_count != declared_frame_count:
        raise ValueError(
            f"{source.name} contains {frame_count} frames but its timestamp CSV "
            f"contains {declared_frame_count} rows"
        )
    if selected_indexes and selected_indexes[-1] >= frame_count:
        raise ValueError("raw depth payload does not contain every retained frame")

    depth = np.memmap(
        source,
        dtype="<u2",
        mode="r",
        shape=(frame_count, height, width),
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio_ffmpeg.write_frames(
        str(destination),
        (width, height),
        fps=fps,
        codec="mpeg4",
        pix_fmt_in="rgb24",
        pix_fmt_out="yuv420p",
    )
    writer.send(None)
    try:
        for source_index in selected_indexes:
            rgb = _jet_depth_frame(depth[source_index], depth_scale)
            writer.send(rgb.tobytes())
    finally:
        writer.close()
    return width, height


def _read_depth_metadata(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"raw depth metadata is missing: {path.name}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"raw depth metadata must be an object: {path.name}")
    width = value.get("width")
    height = value.get("height")
    encoding = value.get("encoding")
    depth_scale = value.get("depth_scale")
    if not isinstance(width, int) or width <= 0:
        raise ValueError("raw depth metadata width must be a positive integer")
    if not isinstance(height, int) or height <= 0:
        raise ValueError("raw depth metadata height must be a positive integer")
    if not isinstance(encoding, str) or encoding.lower() not in {"16uc1", "16uc12"}:
        raise ValueError("raw depth encoding must be 16uc1 or 16uc12")
    if not isinstance(depth_scale, (int, float)) or depth_scale <= 0:
        raise ValueError("raw depth metadata depth_scale must be positive")
    return {
        "width": width,
        "height": height,
        "encoding": encoding.lower(),
        "depth_scale": float(depth_scale),
    }


def _jet_depth_frame(depth: np.ndarray, depth_scale: float) -> np.ndarray:
    normalized = np.clip(
        depth.astype(np.float32) * depth_scale / 3.0,
        0.0,
        1.0,
    )
    red = np.clip(1.5 - np.abs(4.0 * normalized - 3.0), 0.0, 1.0)
    green = np.clip(1.5 - np.abs(4.0 * normalized - 2.0), 0.0, 1.0)
    blue = np.clip(1.5 - np.abs(4.0 * normalized - 1.0), 0.0, 1.0)
    return np.ascontiguousarray(
        np.rint(np.stack((red, green, blue), axis=-1) * 255.0).astype(np.uint8)
    )


def _write_metadata(
    root: Path,
    plan: EpisodePlan,
    states: list[np.ndarray],
    actions: list[np.ndarray],
    fps: int,
    robot_type: str,
    video_features: dict[str, dict[str, Any]],
) -> None:
    frame_count = len(states)
    features: dict[str, dict[str, Any]] = {
        "observation.state": _vector_feature(plan, "observation_state"),
        "action": _vector_feature(plan, "action"),
        **video_features,
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
    }
    info = {
        "codebase_version": LEROBOT_CODEBASE_VERSION,
        "robot_type": robot_type,
        "total_episodes": 1,
        "total_frames": frame_count,
        "total_tasks": 1,
        "total_videos": len(video_features),
        "total_chunks": 1,
        "chunks_size": 1000,
        "fps": fps,
        "splits": {"train": "0:1"},
        "data_path": DATA_PATH_TEMPLATE,
        "video_path": VIDEO_PATH_TEMPLATE,
        "features": features,
    }
    if "observation.images.head_depth" in video_features:
        info["s2_depth_video_colormap"] = "jet"
    meta = root / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    _write_json(meta / "info.json", info)
    _write_jsonl(
        meta / "episodes.jsonl",
        [{"episode_index": 0, "tasks": [plan.task.text], "length": frame_count}],
    )
    _write_jsonl(meta / "tasks.jsonl", [{"task_index": 0, "task": plan.task.text}])
    timestamps = np.arange(frame_count, dtype=np.float32) / fps
    stats = {
        "observation.state": _stats(np.stack(states)),
        "action": _stats(np.stack(actions)),
        "timestamp": _stats(timestamps),
        "frame_index": _stats(np.arange(frame_count, dtype=np.int64)),
        "episode_index": _stats(np.zeros(frame_count, dtype=np.int64)),
        "index": _stats(np.arange(frame_count, dtype=np.int64)),
        "task_index": _stats(np.zeros(frame_count, dtype=np.int64)),
    }
    _write_jsonl(
        meta / "episodes_stats.jsonl",
        [{"episode_index": 0, "stats": stats}],
    )


def _vector_feature(plan: EpisodePlan, field: str) -> dict[str, Any]:
    feature_list = getattr(plan.feature_schema, field)
    return {
        "dtype": "float32",
        "shape": [len(feature_list)],
        "names": [feature.name for feature in feature_list],
    }


def _video_feature(height: int, width: int, fps: int) -> dict[str, Any]:
    video_info = {
        "video.is_depth_map": False,
        "video.fps": float(fps),
        "video.codec": "mp4v",
        "video.pix_fmt": "yuv420p",
        "has_audio": False,
    }
    return {
        "dtype": "video",
        "video_info": video_info,
        "shape": [height, width, 3],
        "names": ["height", "width", "channel"],
        "info": {
            "video.height": height,
            "video.width": width,
            "video.codec": "mp4v",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "video.fps": fps,
            "video.channels": 3,
            "has_audio": False,
        },
    }


def _stats(values: np.ndarray) -> dict[str, list[Any]]:
    array = np.asarray(values)
    if array.ndim == 1:
        array = array[:, None]
    return {
        "min": array.min(axis=0).tolist(),
        "max": array.max(axis=0).tolist(),
        "mean": array.mean(axis=0).tolist(),
        "std": array.std(axis=0).tolist(),
        "count": [len(array)],
    }


def _validate_episode_table(
    table: pa.Table,
    episode_index: int,
    errors: list[str],
    expected_fps: float | None,
    features: dict[str, Any],
    *,
    global_start: int,
    task_indexes: set[int],
) -> None:
    expected_columns = [
        "observation.state",
        "action",
        "timestamp",
        "frame_index",
        "episode_index",
        "index",
        "task_index",
    ]
    if table.column_names != expected_columns:
        errors.append(f"episode {episode_index} Parquet columns do not match v2.1")
        return
    expected_types = {
        "observation.state": pa.list_(pa.float32()),
        "action": pa.list_(pa.float32()),
        "timestamp": pa.float32(),
        "frame_index": pa.int64(),
        "episode_index": pa.int64(),
        "index": pa.int64(),
        "task_index": pa.int64(),
    }
    for key, expected_type in expected_types.items():
        if table.schema.field(key).type != expected_type:
            errors.append(
                f"episode {episode_index} {key} has an incorrect Parquet type"
            )
    frame_indexes = table["frame_index"].to_pylist()
    if frame_indexes != list(range(table.num_rows)):
        errors.append(f"episode {episode_index} frame_index is not contiguous")
    states = table["observation.state"].to_pylist()
    actions = table["action"].to_pylist()
    state_array = _numeric_matrix(states, episode_index, "states", errors)
    action_array = _numeric_matrix(actions, episode_index, "actions", errors)
    for key, values in (
        ("observation.state", state_array),
        ("action", action_array),
    ):
        dimension = _declared_vector_dimension(features, key, errors)
        if (
            values is not None
            and dimension is not None
            and values.shape[1] != dimension
        ):
            errors.append(
                f"episode {episode_index} {key} dimension does not match info.json"
            )
    expected_actions = [*states[1:], states[-1]] if states else []
    if actions != expected_actions:
        errors.append(f"episode {episode_index} actions are not next-state targets")
    if table["episode_index"].to_pylist() != [episode_index] * table.num_rows:
        errors.append(f"episode {episode_index} episode_index values are incorrect")
    if table["index"].to_pylist() != list(
        range(global_start, global_start + table.num_rows)
    ):
        errors.append(f"episode {episode_index} global indexes are not contiguous")
    if any(value not in task_indexes for value in table["task_index"].to_pylist()):
        errors.append(f"episode {episode_index} contains an unknown task_index")
    if expected_fps is not None:
        timestamps = np.asarray(table["timestamp"].to_pylist(), dtype=np.float64)
        expected = (
            np.arange(table.num_rows, dtype=np.float32) / np.float32(expected_fps)
        ).astype(np.float64)
        if not np.allclose(timestamps, expected, rtol=0, atol=1e-6):
            errors.append(f"episode {episode_index} timestamps do not match FPS")


def _numeric_matrix(
    values: list[Any],
    episode_index: int,
    label: str,
    errors: list[str],
) -> np.ndarray | None:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        errors.append(f"episode {episode_index} {label} have inconsistent dimensions")
        return None
    if array.ndim != 2:
        errors.append(f"episode {episode_index} {label} must be vectors")
        return None
    if not np.isfinite(array).all():
        errors.append(f"episode {episode_index} {label} contain NaN or infinity")
    return array


def _declared_vector_dimension(
    features: dict[str, Any],
    key: str,
    errors: list[str],
) -> int | None:
    feature = features.get(key)
    if not isinstance(feature, dict):
        errors.append(f"info.json is missing feature {key}")
        return None
    shape = feature.get("shape")
    if (
        not isinstance(shape, list)
        or len(shape) != 1
        or not isinstance(shape[0], int)
        or shape[0] <= 0
    ):
        errors.append(f"info.json feature {key} has an invalid vector shape")
        return None
    return shape[0]


def _validate_episode_stats(
    table: pa.Table,
    episode_index: int,
    stats_row: dict[str, Any] | None,
    errors: list[str],
) -> None:
    if stats_row is None:
        errors.append(f"episode {episode_index} has no statistics row")
        return
    stats = stats_row.get("stats")
    if not isinstance(stats, dict):
        errors.append(f"episode {episode_index} statistics must be an object")
        return
    for key in table.column_names:
        actual = stats.get(key)
        if not isinstance(actual, dict):
            errors.append(f"episode {episode_index} statistics are missing {key}")
            continue
        dtype = (
            np.float32
            if key in {"observation.state", "action", "timestamp"}
            else np.int64
        )
        expected = _stats(np.asarray(table[key].to_pylist(), dtype=dtype))
        for metric, expected_values in expected.items():
            actual_values = actual.get(metric)
            if not isinstance(actual_values, list):
                errors.append(
                    f"episode {episode_index} {key} statistics are missing {metric}"
                )
                continue
            if metric == "count":
                matches = actual_values == expected_values
            else:
                try:
                    matches = np.allclose(
                        np.asarray(actual_values, dtype=np.float64),
                        np.asarray(expected_values, dtype=np.float64),
                        rtol=1e-4,
                        atol=1e-5,
                    )
                except (TypeError, ValueError):
                    matches = False
            if not matches:
                errors.append(
                    f"episode {episode_index} {key} statistic {metric} is incorrect"
                )


def _validate_video_feature(
    video_key: str,
    feature: Any,
    width: int,
    height: int,
    fps: float,
    errors: list[str],
) -> None:
    if not isinstance(feature, dict):
        errors.append(f"video feature {video_key} must be an object")
        return
    if feature.get("shape") != [height, width, 3]:
        errors.append(f"video feature {video_key} shape does not match the MP4")
    video_info = feature.get("video_info")
    if not isinstance(video_info, dict):
        errors.append(f"video feature {video_key} has no video_info")
        return
    declared_fps = video_info.get("video.fps")
    if not isinstance(declared_fps, (int, float)) or not np.isclose(
        declared_fps,
        fps,
    ):
        errors.append(f"video feature {video_key} FPS does not match the MP4")


def _validate_reference_profile(info: dict[str, Any], errors: list[str]) -> bool:
    if info.get("robot_type") != "s2":
        return True
    features = info.get("features")
    if not isinstance(features, dict):
        errors.append("S2 dataset has no feature declarations")
        return False
    expected_names = list(S2_REFERENCE_STATE_ORDER)
    valid = True
    for key in ("observation.state", "action"):
        feature = features.get(key)
        if not isinstance(feature, dict):
            errors.append(f"S2 reference profile is missing {key}")
            valid = False
            continue
        if feature.get("shape") != [len(expected_names)]:
            errors.append(f"S2 reference profile {key} must have 24 dimensions")
            valid = False
        if feature.get("names") != expected_names:
            errors.append(f"S2 reference profile {key} names or order are incorrect")
            valid = False
    depth_key = "observation.images.head_depth"
    if depth_key in features and info.get("s2_depth_video_colormap") != "jet":
        errors.append("S2 depth video requires s2_depth_video_colormap='jet'")
        valid = False
    return valid


def _validate_info_contract(
    info: dict[str, Any],
    *,
    episode_count: int,
    frame_count: int,
    task_count: int,
    video_count: int,
    errors: list[str],
) -> None:
    if not info:
        return
    expected_totals = {
        "total_episodes": episode_count,
        "total_frames": frame_count,
        "total_tasks": task_count,
        "total_videos": video_count,
    }
    for key, expected in expected_totals.items():
        if info.get(key) != expected:
            errors.append(f"info.json {key} is incorrect")
    if info.get("data_path") != DATA_PATH_TEMPLATE:
        errors.append("info.json data_path is incorrect")
    if info.get("video_path") != VIDEO_PATH_TEMPLATE:
        errors.append("info.json video_path is incorrect")
    chunks_size = info.get("chunks_size")
    if not isinstance(chunks_size, int) or chunks_size <= 0:
        errors.append("info.json chunks_size must be a positive integer")
    else:
        expected_chunks = (episode_count + chunks_size - 1) // chunks_size
        if info.get("total_chunks") != expected_chunks:
            errors.append("info.json total_chunks is incorrect")
    if info.get("splits") != {"train": f"0:{episode_count}"}:
        errors.append("info.json train split is incorrect")


def _indexed_rows(
    rows: list[dict[str, Any]],
    field: str,
    label: str,
    errors: list[str],
) -> dict[int, dict[str, Any]]:
    indexed: dict[int, dict[str, Any]] = {}
    for row_number, row in enumerate(rows, start=1):
        value = row.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            errors.append(f"{label} row {row_number} has an invalid {field}")
            continue
        if value in indexed:
            errors.append(f"{label} contains duplicate {field} {value}")
            continue
        indexed[value] = row
    return indexed


def _nonnegative_integer_field(
    row: dict[str, Any],
    field: str,
    label: str,
    errors: list[str],
) -> int | None:
    value = row.get(field)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        errors.append(f"{label} has an invalid {field}")
        return None
    return value


def _unique_errors(errors: list[str]) -> list[str]:
    return list(dict.fromkeys(errors))


def _video_timing(path: Path) -> tuple[int, float, int, int]:
    reader = imageio_ffmpeg.read_frames(str(path), pix_fmt="rgb24")
    try:
        metadata = next(reader)
    finally:
        reader.close()
    frame_count, _ = imageio_ffmpeg.count_frames_and_secs(str(path))
    if frame_count is None:
        raise ValueError("MP4 frame count is unavailable")
    width, height = metadata["size"]
    return frame_count, float(metadata["fps"]), int(width), int(height)


def _read_json(path: Path, errors: list[str]) -> dict[str, Any]:
    if not path.is_file():
        errors.append(f"missing metadata file: {path.name}")
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"cannot parse metadata file {path.name}: {exc}")
        return {}
    if not isinstance(value, dict):
        errors.append(f"metadata file must contain an object: {path.name}")
        return {}
    return value


def _read_jsonl(path: Path, errors: list[str]) -> list[dict[str, Any]]:
    if not path.is_file():
        errors.append(f"missing metadata file: {path.name}")
        return []
    values: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        errors.append(f"cannot read metadata file {path.name}: {exc}")
        return []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"cannot parse {path.name} line {line_number}: {exc.msg}")
            continue
        if not isinstance(value, dict):
            errors.append(
                f"metadata JSONL line must contain an object: {path.name}:{line_number}"
            )
            continue
        values.append(value)
    return values


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(value, ensure_ascii=False) + "\n" for value in values),
        encoding="utf-8",
    )
