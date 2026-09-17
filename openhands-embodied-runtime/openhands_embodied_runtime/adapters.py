"""Source adapters that normalize embodied datasets into episode plans."""

from __future__ import annotations

import bisect
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import imageio_ffmpeg
import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict, Field

from openhands_embodied_runtime.models import (
    ActionDerivationMode,
    ActionProvenance,
    ActionProvenanceMode,
    DatasetInspection,
    EnvironmentSpec,
    EpisodePlan,
    EpisodeSummary,
    FeatureSpec,
    QualityReport,
    QualityStatus,
    SourceType,
    StreamDescriptor,
    StreamStatus,
    SubtaskSegment,
    TaskSpec,
    TimelineBasis,
    TimelineSpec,
    TrainingFeatureSchema,
)
from openhands_embodied_runtime.timestamps import (
    common_alignment_times,
    sensor_time_s,
    sensor_timestamp_ns,
    wall_time_s,
)


S2_REFERENCE_JOINT_ORDER = (
    "L_shoulder_pitch_joint",
    "L_shoulder_roll_joint",
    "L_shoulder_yaw_joint",
    "L_elbow_roll_joint",
    "L_elbow_yaw_joint",
    "L_wrist_pitch_joint",
    "L_wrist_roll_joint",
    "R_shoulder_pitch_joint",
    "R_shoulder_roll_joint",
    "R_shoulder_yaw_joint",
    "R_elbow_roll_joint",
    "R_elbow_yaw_joint",
    "R_wrist_pitch_joint",
    "R_wrist_roll_joint",
    "waist_yaw_joint",
    "lifter_pitch_1_joint",
    "lifter_pitch_2_joint",
    "lifter_pitch_3_joint",
    "head_pitch_joint",
    "head_yaw_joint",
    "driving_wheel_left_joint",
    "driving_wheel_right_joint",
)
S2_REFERENCE_STATE_ORDER = S2_REFERENCE_JOINT_ORDER + (
    "left_grip_pos",
    "right_grip_pos",
)
LEROBOT_V21_DATA_PATH = (
    "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
)
LEROBOT_V21_VIDEO_PATH = (
    "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
)


class OnlineActionLabel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    start_frame: int = Field(ge=0)
    end_frame: int = Field(gt=0)
    action_text: str
    skill: str


class OnlineLabelInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action_config: list[OnlineActionLabel]


class OnlineEpisode(BaseModel):
    model_config = ConfigDict(extra="forbid")

    episode_id: str | int
    label_info: OnlineLabelInfo
    task_name: str
    init_scene_text: str | None = None


@dataclass(frozen=True)
class LeRobotV21Episode:
    episode_id: str
    episode_index: int
    metadata: dict[str, Any]
    parquet_path: Path


class LeRobotV21Adapter:
    """Read a mounted Hugging Face LeRobot v2.1 dataset and label sidecar."""

    source_type = SourceType.LEROBOT_V21

    def __init__(self, source: Path):
        self.source = source.resolve()
        if not _is_lerobot_v21_dataset(self.source):
            raise ValueError(f"LeRobot v2.1 metadata was not found under {source}")
        self.info = _load_optional_object(self.source / "meta" / "info.json")
        if self.info.get("codebase_version") != "v2.1":
            raise ValueError("meta/info.json codebase_version must be 'v2.1'")
        self.tasks = _lerobot_tasks(self.source)
        self.label_path = _find_optional_online_label_file(self.source)
        self.labels = (
            {
                str(episode.episode_id): episode
                for episode in _load_online_episodes(self.label_path)
            }
            if self.label_path is not None
            else None
        )
        self.episodes = self._load_episodes()

    @property
    def episode_ids(self) -> list[str]:
        return [episode.episode_id for episode in self.episodes]

    def inspect(self, sample_limit: int) -> DatasetInspection:
        plans = [self.build_plan(episode) for episode in self.episodes[:sample_limit]]
        dataset_ids = set(self.episode_ids)
        label_ids = set(self.labels) if self.labels is not None else set()
        warnings: list[str] = []
        if self.labels is not None:
            missing = sorted(dataset_ids - label_ids)
            extra = sorted(label_ids - dataset_ids)
            if missing:
                warnings.append(
                    f"online labels are missing {len(missing)} dataset episode(s)"
                )
            if extra:
                warnings.append(
                    f"online labels contain {len(extra)} unmatched episode(s)"
                )
        return DatasetInspection(
            source_type=self.source_type,
            source_path=str(self.source),
            episode_count=len(self.episodes),
            sampled_episodes=[_summary(plan) for plan in plans],
            warnings=warnings,
        )

    def select(self, episode_id: str | None) -> LeRobotV21Episode:
        if episode_id is None:
            if len(self.episodes) != 1:
                raise ValueError(
                    "episode_id is required when the source has multiple episodes"
                )
            return self.episodes[0]
        for episode in self.episodes:
            if episode.episode_id == episode_id:
                return episode
        raise ValueError(f"episode_id not found under source: {episode_id}")

    def load_state_vectors(self, episode: LeRobotV21Episode) -> list[list[float]]:
        table = pq.read_table(episode.parquet_path, columns=["observation.state"])
        values = table["observation.state"].to_pylist()
        return [[float(item) for item in row] for row in values]

    def build_plan(self, episode: LeRobotV21Episode) -> EpisodePlan:
        errors: list[str] = []
        warnings: list[str] = []
        table = _read_lerobot_episode_table(episode.parquet_path, errors)
        frame_count = table.num_rows if table is not None else None
        expected_length = episode.metadata.get("length")
        if (
            frame_count is not None
            and isinstance(expected_length, int)
            and expected_length != frame_count
        ):
            errors.append("episode metadata length does not match Parquet rows")

        state_features = _lerobot_feature_specs(
            self.info,
            "observation.state",
            source="observation.state",
        )
        action_features = [
            feature.model_copy(update={"source": f"next_state.{feature.source}"})
            for feature in state_features
        ]
        state_status = _lerobot_column_status(
            table,
            "observation.state",
            state_features,
            errors,
        )
        source_action_status = _lerobot_column_status(
            table,
            "action",
            _lerobot_feature_specs(self.info, "action", source="action"),
            errors,
        )
        relative_parquet = _relative_path(self.source, episode.parquet_path)
        streams: dict[str, StreamDescriptor] = {
            "state": StreamDescriptor(
                kind="state",
                path=relative_parquet,
                status=state_status,
                frame_count=frame_count,
            ),
            "action": StreamDescriptor(
                kind="action",
                path=relative_parquet,
                status=source_action_status,
                frame_count=frame_count,
                message=(
                    "source action is validated, but reference output action is "
                    "derived from the next retained observation state"
                ),
            ),
        }

        video_streams = self._video_streams(episode, errors)
        streams.update(video_streams)
        primary = _primary_video_stream(video_streams)
        if primary is None:
            streams["rgb"] = StreamDescriptor(
                kind="rgb",
                path="",
                status=StreamStatus.MISSING,
                message="LeRobot source declares no usable RGB MP4",
            )
            warnings.append("matching RGB video is missing")
        elif frame_count is not None and primary.frame_count != frame_count:
            errors.append("primary MP4 frame count does not match Parquet rows")

        task = TaskSpec(
            text=_lerobot_episode_task(episode.metadata, self.tasks),
            origin="meta/episodes.jsonl",
        )
        segments: list[SubtaskSegment] = []
        label = self.labels.get(episode.episode_id) if self.labels is not None else None
        if label is not None:
            task = TaskSpec(text=label.task_name, origin="online_label.task_name")
            segments, label_errors, label_warnings = _online_segments(label)
            errors.extend(label_errors)
            warnings.extend(label_warnings)
            assert self.label_path is not None
            streams["labels"] = StreamDescriptor(
                kind="labels",
                path=_relative_path(self.source, self.label_path),
                status=StreamStatus.AVAILABLE,
            )
        elif self.labels is not None:
            assert self.label_path is not None
            streams["labels"] = StreamDescriptor(
                kind="labels",
                path=_relative_path(self.source, self.label_path),
                status=StreamStatus.MISSING,
                message="no online label matched this episode_id",
            )
            warnings.append("online label sidecar has no matching episode_id")

        if frame_count is not None:
            for index, segment in enumerate(segments):
                if segment.source_end_frame > frame_count:
                    errors.append(
                        f"action_config[{index}] exceeds the source frame count"
                    )

        if not state_features:
            warnings.append("observation state feature names are missing")
        elif any(feature.unit is None for feature in state_features):
            warnings.append("observation state feature units are missing")
        warnings.append(
            "action will use the reference-dataset convention: each action is the "
            "next cleaned observation state and the final action repeats the final "
            "state"
        )
        status = QualityStatus.REJECTED if errors else QualityStatus.NEEDS_REVIEW
        return EpisodePlan(
            episode_id=episode.episode_id,
            source_type=self.source_type,
            task=task,
            environment=EnvironmentSpec(
                initial_scene=label.init_scene_text if label is not None else None
            ),
            streams=streams,
            feature_schema=TrainingFeatureSchema(
                observation_state=state_features,
                action=action_features,
            ),
            action_provenance=ActionProvenance(
                mode=ActionProvenanceMode.DERIVED,
                source_path=relative_parquet,
                derivation=ActionDerivationMode.NEXT_STATE,
                description=(
                    "action[t] is observation.state[t+1]; the final action repeats "
                    "the final state"
                ),
            ),
            source_timeline=TimelineSpec(
                basis=TimelineBasis.FRAME_INDEX,
                frame_count=frame_count,
                fps=primary.fps if primary is not None else None,
            ),
            segments=segments,
            quality=QualityReport(status=status, errors=errors, warnings=warnings),
        )

    def _load_episodes(self) -> list[LeRobotV21Episode]:
        rows = _read_jsonl(self.source / "meta" / "episodes.jsonl")
        chunks_size = self.info.get("chunks_size", 1000)
        if not isinstance(chunks_size, int) or chunks_size <= 0:
            raise ValueError("meta/info.json chunks_size must be a positive integer")
        template = self.info.get("data_path", LEROBOT_V21_DATA_PATH)
        if not isinstance(template, str):
            raise ValueError("meta/info.json data_path must be a string")
        episodes: list[LeRobotV21Episode] = []
        for row in rows:
            episode_index = row.get("episode_index")
            if not isinstance(episode_index, int) or episode_index < 0:
                raise ValueError("episodes.jsonl episode_index must be non-negative")
            parquet_path = _dataset_template_path(
                self.source,
                template,
                episode_chunk=episode_index // chunks_size,
                episode_index=episode_index,
            )
            episode_id = _lerobot_source_episode_id(row, parquet_path, episode_index)
            episodes.append(
                LeRobotV21Episode(
                    episode_id=episode_id,
                    episode_index=episode_index,
                    metadata=row,
                    parquet_path=parquet_path,
                )
            )
        episodes.sort(key=lambda item: item.episode_index)
        episode_indexes = [episode.episode_index for episode in episodes]
        if len(episode_indexes) != len(set(episode_indexes)):
            raise ValueError("LeRobot source contains duplicate episode_index values")
        episode_ids = [episode.episode_id for episode in episodes]
        if len(episode_ids) != len(set(episode_ids)):
            raise ValueError("LeRobot source contains duplicate source episode IDs")
        return episodes

    def _video_streams(
        self,
        episode: LeRobotV21Episode,
        errors: list[str],
    ) -> dict[str, StreamDescriptor]:
        features = self.info.get("features")
        if not isinstance(features, dict):
            errors.append("meta/info.json features must be an object")
            return {}
        chunks_size = int(self.info.get("chunks_size", 1000))
        template = self.info.get("video_path", LEROBOT_V21_VIDEO_PATH)
        if not isinstance(template, str):
            errors.append("meta/info.json video_path must be a string")
            return {}
        streams: dict[str, StreamDescriptor] = {}
        for video_key, feature in features.items():
            if not isinstance(feature, dict) or feature.get("dtype") != "video":
                continue
            name = video_key.removeprefix("observation.images.")
            path = _dataset_template_path(
                self.source,
                template,
                episode_chunk=episode.episode_index // chunks_size,
                episode_index=episode.episode_index,
                video_key=video_key,
            )
            kind = "depth" if "depth" in name.lower() else "rgb"
            descriptor = _stream_descriptor(kind, path)
            streams[name] = descriptor.model_copy(
                update={"path": _relative_path(self.source, path)}
            )
            if descriptor.status != StreamStatus.AVAILABLE:
                errors.append(f"video stream {video_key} is not usable")
        return streams


class OnlineActionLabelAdapter:
    """Normalize frame-range annotations distributed with online datasets."""

    source_type = SourceType.ONLINE_LABELS

    def __init__(self, source: Path):
        self.source = _resolve_online_label_file(source)
        self.episodes = _load_online_episodes(self.source)

    def inspect(self, sample_limit: int) -> DatasetInspection:
        plans = [self.build_plan(item) for item in self.episodes[:sample_limit]]
        warnings: list[str] = []
        if not self.episodes:
            warnings.append("label file contains no episodes")
        return DatasetInspection(
            source_type=self.source_type,
            source_path=str(self.source),
            episode_count=len(self.episodes),
            sampled_episodes=[_summary(plan) for plan in plans],
            warnings=warnings,
        )

    def select(self, episode_id: str | None) -> OnlineEpisode:
        if episode_id is None:
            if len(self.episodes) != 1:
                raise ValueError(
                    "episode_id is required when labels contain multiple episodes"
                )
            return self.episodes[0]
        for episode in self.episodes:
            if str(episode.episode_id) == episode_id:
                return episode
        raise ValueError(f"episode_id not found in labels: {episode_id}")

    def build_plan(self, episode: OnlineEpisode) -> EpisodePlan:
        segments, errors, warnings = _online_segments(episode)

        episode_id = str(episode.episode_id)
        video_path = _find_online_video(self.source.parent, episode_id)
        streams = {
            "labels": StreamDescriptor(
                kind="labels",
                path=str(self.source),
                status=StreamStatus.AVAILABLE,
            )
        }
        if video_path is None:
            streams["rgb"] = StreamDescriptor(
                kind="rgb",
                path="",
                status=StreamStatus.MISSING,
                message="no video matched the episode_id",
            )
            warnings.append("matching RGB video is missing")
        else:
            streams["rgb"] = _stream_descriptor("rgb", video_path)
        warnings.append("source frame count and FPS are not declared by the labels")

        quality = QualityReport(
            status=QualityStatus.REJECTED if errors else QualityStatus.NEEDS_REVIEW,
            errors=errors,
            warnings=warnings,
        )
        return EpisodePlan(
            episode_id=episode_id,
            source_type=self.source_type,
            task=TaskSpec(
                text=episode.task_name,
                origin="online_label.task_name",
            ),
            environment=EnvironmentSpec(initial_scene=episode.init_scene_text),
            streams=streams,
            action_provenance=ActionProvenance(
                mode=ActionProvenanceMode.MISSING,
                description="online labels do not contain robot control commands",
            ),
            source_timeline=TimelineSpec(basis=TimelineBasis.FRAME_INDEX),
            segments=segments,
            quality=quality,
        )


class SelfCollectedAdapter:
    """Normalize timestamped robot recordings with joints and camera CSV files."""

    source_type = SourceType.SELF_COLLECTED

    def __init__(self, source: Path):
        self.source = source.resolve()
        self.episode_dirs = _find_self_collected_episodes(self.source)
        if not self.episode_dirs:
            raise ValueError(f"no self-collected episodes found under {source}")

    def inspect(self, sample_limit: int) -> DatasetInspection:
        plans = [self.build_plan(path) for path in self.episode_dirs[:sample_limit]]
        return DatasetInspection(
            source_type=self.source_type,
            source_path=str(self.source),
            episode_count=len(self.episode_dirs),
            sampled_episodes=[_summary(plan) for plan in plans],
        )

    def select(self, episode_id: str | None) -> Path:
        if episode_id is None:
            if len(self.episode_dirs) != 1:
                raise ValueError(
                    "episode_id is required when the source has multiple episodes"
                )
            return self.episode_dirs[0]
        for episode_dir in self.episode_dirs:
            if episode_dir.name == episode_id:
                return episode_dir
        raise ValueError(f"episode_id not found under source: {episode_id}")

    def build_plan(self, episode_dir: Path) -> EpisodePlan:
        manifest = _load_optional_object(episode_dir / "manifest.json")
        joints = _read_jsonl(episode_dir / "joints.jsonl")
        camera_streams, primary_rows = _camera_streams(episode_dir)
        camera_streams = {
            name: descriptor.model_copy(update={"path": Path(descriptor.path).name})
            for name, descriptor in camera_streams.items()
        }
        errors: list[str] = []
        warnings: list[str] = []

        if not joints:
            errors.append("joints.jsonl contains no records")
        if not primary_rows:
            errors.append("no usable camera timestamp stream was found")

        streams = {
            "state": StreamDescriptor(
                kind="state",
                path="joints.jsonl",
                status=StreamStatus.AVAILABLE if joints else StreamStatus.INVALID,
                frame_count=len(joints),
                message=None if joints else "file has no usable JSONL records",
            ),
            "action": _self_collected_action_stream(Path("joints.jsonl"), joints),
            **camera_streams,
        }
        invalid_streams = [
            name
            for name, descriptor in streams.items()
            if descriptor.status == StreamStatus.INVALID
        ]
        blocking_invalid_streams = [
            name for name in invalid_streams if streams[name].kind != "depth"
        ]
        if invalid_streams:
            warnings.append(f"invalid streams: {', '.join(sorted(invalid_streams))}")
        optional_invalid_streams = [
            name for name in invalid_streams if streams[name].kind == "depth"
        ]
        if optional_invalid_streams:
            warnings.append(
                "invalid optional depth streams will be omitted from RGB-only "
                "output: " + ", ".join(sorted(optional_invalid_streams))
            )
        warnings.append(
            "action will use the reference-dataset convention: each action is the "
            "next cleaned observation state and the final action repeats the final "
            "state"
        )

        primary_stream = _primary_video_stream(camera_streams)
        frame_count = primary_stream.frame_count if primary_stream else None
        fps = primary_stream.fps if primary_stream else None
        start_stamp_ns = sensor_timestamp_ns(primary_rows[0]) if primary_rows else None
        end_stamp_ns = sensor_timestamp_ns(primary_rows[-1]) if primary_rows else None
        segments = _joint_subtask_segments(joints, primary_rows, warnings)
        if not segments:
            segments = _manifest_segments(manifest, primary_rows, warnings)
        if not segments:
            warnings.append("no trusted subtask labels are available")

        task = _self_collected_task_spec(manifest, episode_dir.name, warnings)
        feature_schema = _self_collected_feature_schema(joints)
        errors.extend(_s2_reference_profile_errors(joints))
        sparse_ee_rows = _sparse_reference_ee_rows(joints)
        if [feature.name for feature in feature_schema.observation_state] == list(
            S2_REFERENCE_STATE_ORDER
        ) and sparse_ee_rows:
            warnings.append(
                f"{sparse_ee_rows} sparse end-effector row(s) will be "
                "interpolated on the sensor clock"
            )

        if errors:
            status = QualityStatus.REJECTED
        elif (
            blocking_invalid_streams
            or optional_invalid_streams
            or not segments
            or any(segment.origin == "manifest_weak_label" for segment in segments)
            or (
                task.origin
                in {
                    "manifest.composition_id",
                    "episode_directory",
                    "generated",
                }
                and not task.user_confirmed
            )
        ):
            status = QualityStatus.NEEDS_REVIEW
        else:
            status = QualityStatus.ACCEPTED
        return EpisodePlan(
            episode_id=episode_dir.name,
            source_type=self.source_type,
            task=task,
            streams=streams,
            feature_schema=feature_schema,
            action_provenance=_self_collected_action_provenance(streams["action"]),
            source_timeline=TimelineSpec(
                basis=TimelineBasis.STAMP_NS,
                frame_count=frame_count,
                fps=fps,
                start_stamp_ns=start_stamp_ns,
                end_stamp_ns=end_stamp_ns,
            ),
            segments=segments,
            quality=QualityReport(status=status, errors=errors, warnings=warnings),
        )


def detect_source_type(path: Path) -> SourceType:
    path = path.resolve()
    if path.is_dir() and _find_self_collected_episodes(path):
        return SourceType.SELF_COLLECTED
    if _is_lerobot_v21_dataset(path):
        return SourceType.LEROBOT_V21
    if path.is_file() or path.is_dir():
        try:
            _resolve_online_label_file(path)
        except ValueError:
            pass
        else:
            return SourceType.ONLINE_LABELS
    raise ValueError(f"unsupported embodied dataset source: {path}")


def inspect_dataset(path: Path, *, sample_limit: int = 3) -> DatasetInspection:
    if sample_limit <= 0:
        raise ValueError("sample_limit must be positive")
    source_type = detect_source_type(path)
    if source_type == SourceType.SELF_COLLECTED:
        return SelfCollectedAdapter(path).inspect(sample_limit)
    if source_type == SourceType.LEROBOT_V21:
        return LeRobotV21Adapter(path).inspect(sample_limit)
    return OnlineActionLabelAdapter(path).inspect(sample_limit)


def build_episode_plan(path: Path, *, episode_id: str | None = None) -> EpisodePlan:
    source_type = detect_source_type(path)
    if source_type == SourceType.SELF_COLLECTED:
        adapter = SelfCollectedAdapter(path)
        return adapter.build_plan(adapter.select(episode_id))
    if source_type == SourceType.LEROBOT_V21:
        adapter = LeRobotV21Adapter(path)
        return adapter.build_plan(adapter.select(episode_id))
    adapter = OnlineActionLabelAdapter(path)
    return adapter.build_plan(adapter.select(episode_id))


def load_lerobot_state_vectors(
    source: Path,
    episode_id: str,
) -> list[list[float]]:
    """Load ordered state vectors for one LeRobot v2.1 source episode."""
    adapter = LeRobotV21Adapter(source)
    episode = adapter.select(episode_id)
    return adapter.load_state_vectors(episode)


def load_self_collected_signals(
    episode_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Load joints and the primary camera index for idle planning."""
    joints = _read_jsonl(episode_dir / "joints.jsonl")
    _, primary_rows = _camera_streams(episode_dir)
    return joints, primary_rows


def load_self_collected_camera_rows(
    episode_dir: Path,
) -> dict[str, list[dict[str, str]]]:
    """Load every timestamp CSV by stream name."""
    return {
        csv_path.stem: _read_camera_csv(csv_path)
        for csv_path in sorted(episode_dir.glob("*.csv"))
    }


def _self_collected_action_stream(
    path: Path, joints: list[dict[str, Any]]
) -> StreamDescriptor:
    if not joints:
        return StreamDescriptor(
            kind="action",
            path=str(path),
            status=StreamStatus.MISSING,
            message="next-state actions require state observations",
        )
    return StreamDescriptor(
        kind="action",
        path=str(path),
        status=StreamStatus.AVAILABLE,
        frame_count=len(joints),
        message="derived during materialization from the next cleaned state",
    )


def _self_collected_feature_schema(
    joints: list[dict[str, Any]],
) -> TrainingFeatureSchema:
    if not joints:
        return TrainingFeatureSchema()
    if _has_s2_reference_features(joints):
        state_features = [
            FeatureSpec(name=name, unit="rad", source=f"joints.{name}")
            for name in S2_REFERENCE_JOINT_ORDER
        ]
        state_features.extend(
            [
                FeatureSpec(
                    name="left_grip_pos",
                    unit="m",
                    source="left_ee_states[4]",
                ),
                FeatureSpec(
                    name="right_grip_pos",
                    unit="m",
                    source="right_ee_states[4]",
                ),
            ]
        )
        return TrainingFeatureSchema(
            observation_state=state_features,
            action=[
                feature.model_copy(update={"source": f"next_state.{feature.source}"})
                for feature in state_features
            ],
        )
    joint_values = joints[0].get("joints")
    state_features = (
        [
            FeatureSpec(name=name, unit="rad", source=f"joints.{name}")
            for name in joint_values
            if isinstance(name, str)
        ]
        if isinstance(joint_values, dict)
        else []
    )
    if isinstance(joints[0].get("suction_state"), (int, float)):
        state_features.append(
            FeatureSpec(
                name="suction_state",
                unit="binary",
                source="suction_state",
            )
        )
    action_features = [
        feature.model_copy(update={"source": f"next_state.{feature.source}"})
        for feature in state_features
    ]
    return TrainingFeatureSchema(
        observation_state=state_features,
        action=action_features,
    )


def _has_s2_reference_features(rows: list[dict[str, Any]]) -> bool:
    for row in rows:
        values = row.get("joints")
        if not isinstance(values, dict) or any(
            not _is_finite_number(values.get(name)) for name in S2_REFERENCE_JOINT_ORDER
        ):
            return False
    return all(
        any(_grip_position(row.get(field)) is not None for row in rows)
        for field in ("left_ee_states", "right_ee_states")
    )


def _sparse_reference_ee_rows(rows: list[dict[str, Any]]) -> int:
    return sum(
        1
        for row in rows
        if any(
            _grip_position(row.get(field)) is None
            for field in ("left_ee_states", "right_ee_states")
        )
    )


def _grip_position(value: Any) -> float | None:
    if isinstance(value, list) and len(value) > 4 and _is_finite_number(value[4]):
        return float(value[4])
    return None


def _s2_reference_profile_errors(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return []
    invalid_joints = [
        name
        for name in S2_REFERENCE_JOINT_ORDER
        if any(
            not isinstance(row.get("joints"), dict)
            or not _is_finite_number(row["joints"].get(name))
            for row in rows
        )
    ]
    errors: list[str] = []
    if invalid_joints:
        errors.append(
            "S2 reference profile has missing, non-numeric, or non-finite joints: "
            + ", ".join(invalid_joints)
        )
    for field in ("left_ee_states", "right_ee_states"):
        if not any(_grip_position(row.get(field)) is not None for row in rows):
            errors.append(f"S2 reference profile has no recorded {field}[4] values")
    return errors


def _is_finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _self_collected_action_provenance(
    stream: StreamDescriptor,
) -> ActionProvenance:
    if stream.status == StreamStatus.AVAILABLE:
        return ActionProvenance(
            mode=ActionProvenanceMode.DERIVED,
            source_path="joints.jsonl",
            derivation=ActionDerivationMode.NEXT_STATE,
            description=(
                "action[t] is observation.state[t+1]; the final action repeats "
                "the final state"
            ),
            user_confirmed=False,
        )
    return ActionProvenance(
        mode=ActionProvenanceMode.MISSING,
        source_path="joints.jsonl",
        description=stream.message,
    )


def _online_segments(
    episode: OnlineEpisode,
) -> tuple[list[SubtaskSegment], list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    segments: list[SubtaskSegment] = []
    previous_end: int | None = None
    for index, action in enumerate(episode.label_info.action_config):
        if action.end_frame <= action.start_frame:
            errors.append(f"action_config[{index}] has an invalid frame range")
            continue
        if previous_end is not None:
            if action.start_frame < previous_end:
                errors.append(f"action_config[{index}] overlaps the previous action")
            elif action.start_frame > previous_end:
                warnings.append(f"action_config[{index}] starts after an unlabeled gap")
        previous_end = max(previous_end or 0, action.end_frame)
        segments.append(
            SubtaskSegment(
                source_start_frame=action.start_frame,
                source_end_frame=action.end_frame,
                instruction=action.action_text,
                event_type=action.skill.strip().lower(),
                origin="online_label",
            )
        )
    if segments and segments[0].source_start_frame > 0:
        warnings.append(f"frames [0, {segments[0].source_start_frame}) are unlabeled")
    return segments, errors, warnings


def _is_lerobot_v21_dataset(path: Path) -> bool:
    return (
        path.is_dir()
        and (path / "meta" / "info.json").is_file()
        and (path / "meta" / "episodes.jsonl").is_file()
        and (path / "meta" / "tasks.jsonl").is_file()
    )


def _find_optional_online_label_file(source: Path) -> Path | None:
    candidates = [
        source / name
        for name in ("labels.json", "annotations.json", "labels.jsonl")
        if (source / name).is_file()
    ]
    if len(candidates) > 1:
        raise ValueError("multiple online label files found under LeRobot source")
    return candidates[0] if candidates else None


def _lerobot_tasks(source: Path) -> dict[int, str]:
    tasks: dict[int, str] = {}
    for row in _read_jsonl(source / "meta" / "tasks.jsonl"):
        task_index = row.get("task_index")
        task = row.get("task")
        if not isinstance(task_index, int) or not isinstance(task, str):
            raise ValueError("tasks.jsonl rows require integer task_index and task")
        if task_index in tasks:
            raise ValueError("tasks.jsonl contains duplicate task_index values")
        tasks[task_index] = task
    return tasks


def _lerobot_episode_task(
    episode: dict[str, Any],
    tasks: dict[int, str],
) -> str:
    values = episode.get("tasks")
    if isinstance(values, list):
        task = next((item for item in values if isinstance(item, str) and item), None)
        if task is not None:
            return task
    if len(tasks) == 1:
        return next(iter(tasks.values()))
    return ""


def _lerobot_feature_specs(
    info: dict[str, Any],
    key: str,
    *,
    source: str,
) -> list[FeatureSpec]:
    features = info.get("features")
    feature = features.get(key) if isinstance(features, dict) else None
    if not isinstance(feature, dict):
        return []
    names = feature.get("names")
    if not isinstance(names, list) or not all(
        isinstance(name, str) and name for name in names
    ):
        return []
    units = feature.get("units")
    resolved_units: list[str | None]
    if (
        isinstance(units, list)
        and len(units) == len(names)
        and all(isinstance(unit, str) and unit for unit in units)
    ):
        resolved_units = units
    elif names == list(S2_REFERENCE_STATE_ORDER):
        resolved_units = [
            "rad" if name in S2_REFERENCE_JOINT_ORDER else "m" for name in names
        ]
    else:
        resolved_units = [None] * len(names)
    return [
        FeatureSpec(name=name, unit=unit, source=f"{source}[{index}]")
        for index, (name, unit) in enumerate(zip(names, resolved_units))
    ]


def _read_lerobot_episode_table(path: Path, errors: list[str]) -> Any | None:
    if not path.is_file():
        errors.append(f"missing episode Parquet: {path.name}")
        return None
    try:
        return pq.read_table(path)
    except Exception as exc:
        errors.append(f"cannot read episode Parquet: {exc}")
        return None


def _lerobot_column_status(
    table: Any | None,
    column: str,
    features: list[FeatureSpec],
    errors: list[str],
) -> StreamStatus:
    if table is None or column not in table.column_names:
        errors.append(f"episode Parquet is missing {column}")
        return StreamStatus.MISSING
    values = table[column].to_pylist()
    if not values:
        errors.append(f"episode Parquet {column} contains no rows")
        return StreamStatus.INVALID
    expected_dimension = len(features)
    for index, row in enumerate(values):
        if not isinstance(row, list) or (
            expected_dimension and len(row) != expected_dimension
        ):
            errors.append(f"episode Parquet {column}[{index}] has invalid dimensions")
            return StreamStatus.INVALID
        if any(
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            for value in row
        ):
            errors.append(f"episode Parquet {column}[{index}] is not finite numeric")
            return StreamStatus.INVALID
    return StreamStatus.AVAILABLE


def _lerobot_source_episode_id(
    metadata: dict[str, Any],
    parquet_path: Path,
    episode_index: int,
) -> str:
    for key in ("source_episode_id", "episode_id"):
        value = metadata.get(key)
        if isinstance(value, (str, int)) and not isinstance(value, bool):
            return str(value)
    if parquet_path.is_file():
        try:
            schema = pq.read_schema(parquet_path)
            if "episode_id" in schema.names:
                values = pq.read_table(parquet_path, columns=["episode_id"])[
                    "episode_id"
                ].to_pylist()
                unique = {str(value) for value in values if value is not None}
                if len(unique) == 1:
                    return unique.pop()
        except Exception:
            pass
    return str(episode_index)


def _dataset_template_path(root: Path, template: str, **values: Any) -> Path:
    try:
        relative = Path(template.format(**values))
    except (KeyError, ValueError) as exc:
        raise ValueError(f"invalid LeRobot path template: {template}") from exc
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("LeRobot path templates must stay within the dataset root")
    path = (root / relative).resolve()
    if path != root and root not in path.parents:
        raise ValueError("LeRobot path template escaped the dataset root")
    return path


def _relative_path(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _summary(plan: EpisodePlan) -> EpisodeSummary:
    return EpisodeSummary(
        episode_id=plan.episode_id,
        segment_count=len(plan.segments),
        frame_count=plan.source_timeline.frame_count,
        quality=plan.quality,
    )


def _resolve_online_label_file(path: Path) -> Path:
    path = path.resolve()
    if path.is_file() and path.suffix.lower() in {".json", ".jsonl"}:
        return path
    if path.is_dir():
        candidates = [
            path / name
            for name in (
                "labels.json",
                "annotations.json",
                "episodes.json",
                "labels.jsonl",
            )
            if (path / name).is_file()
        ]
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1:
            raise ValueError("multiple online label files found; pass one explicitly")
    raise ValueError(f"online label JSON or JSONL not found: {path}")


def _load_online_episodes(path: Path) -> list[OnlineEpisode]:
    if path.suffix.lower() == ".jsonl":
        payloads = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and isinstance(payload.get("episodes"), list):
            payloads = payload["episodes"]
        elif isinstance(payload, list):
            payloads = payload
        else:
            raise ValueError(
                "online label JSON must be an array or contain an episodes array"
            )
    episodes = [OnlineEpisode.model_validate(item) for item in payloads]
    ids = [str(item.episode_id) for item in episodes]
    if len(ids) != len(set(ids)):
        raise ValueError("online labels contain duplicate episode_id values")
    return episodes


def _find_online_video(root: Path, episode_id: str) -> Path | None:
    candidates = (
        root / f"{episode_id}.mp4",
        root / f"episode_{episode_id}.mp4",
        root / "videos" / f"{episode_id}.mp4",
        root / "videos" / f"episode_{episode_id}.mp4",
    )
    return next((candidate for candidate in candidates if candidate.is_file()), None)


def _find_self_collected_episodes(path: Path) -> list[Path]:
    if path.is_dir() and (path / "joints.jsonl").is_file():
        return [path]
    if not path.is_dir():
        return []
    episodes = {
        joints.parent.resolve()
        for joints in path.rglob("joints.jsonl")
        if not any(
            part.startswith("aligned_") for part in joints.relative_to(path).parts
        )
    }
    return sorted(episodes, key=lambda item: (item.name, str(item)))


def _load_optional_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path.name} line {line_number} must contain an object")
        rows.append(value)
    return rows


def _read_camera_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def _camera_streams(
    episode_dir: Path,
) -> tuple[dict[str, StreamDescriptor], list[dict[str, str]]]:
    streams: dict[str, StreamDescriptor] = {}
    storage_files = _storage_file_metadata(episode_dir)
    primary_rows: list[dict[str, str]] = []
    primary_priority = -1
    for csv_path in sorted(episode_dir.glob("*.csv")):
        rows = _read_camera_csv(csv_path)
        name = csv_path.stem
        mp4_path = episode_dir / f"{name}.mp4"
        depth_path = episode_dir / f"{name}.u16"
        mp4_storage = storage_files.get(mp4_path.name)
        depth_storage = storage_files.get(depth_path.name)
        if mp4_path.is_file():
            descriptor = _stream_descriptor("rgb", mp4_path, rows)
            priority = 2 if name == "head" else 1
        elif mp4_storage is not None and not mp4_storage.get("materialized", False):
            descriptor = StreamDescriptor(
                kind="rgb",
                path=str(mp4_path),
                status=StreamStatus.NOT_MATERIALIZED,
                frame_count=len(rows),
                message="payload exists in Storage but was not materialized",
            )
            priority = 2 if name == "head" else 1
        elif depth_path.is_file():
            descriptor = _stream_descriptor("depth", depth_path, rows)
            priority = 0
        elif depth_storage is not None and not depth_storage.get("materialized", False):
            descriptor = StreamDescriptor(
                kind="depth",
                path=str(depth_path),
                status=StreamStatus.NOT_MATERIALIZED,
                frame_count=len(rows),
                message="payload exists in Storage but was not materialized",
            )
            priority = 0
        elif "depth" in name.lower():
            descriptor = StreamDescriptor(
                kind="depth",
                path=str(depth_path),
                status=StreamStatus.INVALID,
                frame_count=len(rows),
                message="depth payload file is missing",
            )
            priority = 0
        else:
            descriptor = StreamDescriptor(
                kind="camera_timestamps",
                path=str(csv_path),
                status=StreamStatus.INVALID if not rows else StreamStatus.AVAILABLE,
                frame_count=len(rows),
                message="camera payload file is missing",
            )
            priority = 0
        streams[name] = descriptor
        if rows and priority > primary_priority:
            primary_rows = rows
            primary_priority = priority
    for depth_path in sorted(episode_dir.glob("*.u16")):
        if depth_path.stem not in streams:
            streams[depth_path.stem] = _stream_descriptor("depth", depth_path)
    return streams, primary_rows


def _storage_file_metadata(episode_dir: Path) -> dict[str, dict[str, Any]]:
    path = episode_dir / ".pyromind_storage.json"
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("files"), list):
        raise ValueError(f"{path.name} must contain a files list")
    metadata: dict[str, dict[str, Any]] = {}
    for item in value["files"]:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        if isinstance(name, str) and name:
            metadata[name] = item
    return metadata


def _stream_descriptor(
    kind: str, path: Path, rows: list[dict[str, str]] | None = None
) -> StreamDescriptor:
    frame_count = len(rows) if rows is not None else None
    if not path.is_file():
        return StreamDescriptor(kind=kind, path=str(path), status=StreamStatus.MISSING)
    if path.stat().st_size == 0 or (rows is not None and not rows):
        return StreamDescriptor(
            kind=kind,
            path=str(path),
            status=StreamStatus.INVALID,
            frame_count=frame_count,
            message="stream file or timestamp index is empty",
        )
    media_frame_count = frame_count
    media_fps = _infer_fps(rows or [])
    if kind in {"rgb", "depth"} and path.suffix.lower() == ".mp4":
        try:
            media_frame_count, media_fps = _video_timing(path)
        except (OSError, RuntimeError, ValueError) as exc:
            return StreamDescriptor(
                kind=kind,
                path=str(path),
                status=StreamStatus.INVALID,
                frame_count=frame_count,
                message=f"cannot read MP4 metadata: {exc}",
            )
    message = None
    if rows is not None and media_frame_count != len(rows):
        message = (
            f"MP4 contains {media_frame_count} frames but timestamp CSV contains "
            f"{len(rows)} rows"
        )
    return StreamDescriptor(
        kind=kind,
        path=str(path),
        status=StreamStatus.AVAILABLE,
        frame_count=media_frame_count,
        fps=media_fps,
        message=message,
    )


def _video_timing(path: Path) -> tuple[int, float]:
    reader = imageio_ffmpeg.read_frames(str(path), pix_fmt="rgb24")
    try:
        metadata = next(reader)
    finally:
        reader.close()
    fps = float(metadata["fps"])
    frame_count, _ = imageio_ffmpeg.count_frames_and_secs(str(path))
    if frame_count is None:
        raise ValueError("MP4 frame count is unavailable")
    return frame_count, fps


def _primary_video_stream(
    streams: dict[str, StreamDescriptor],
) -> StreamDescriptor | None:
    head = streams.get("head")
    if (
        head is not None
        and head.kind == "rgb"
        and head.status == StreamStatus.AVAILABLE
    ):
        return head
    return next(
        (
            stream
            for stream in streams.values()
            if stream.kind == "rgb" and stream.status == StreamStatus.AVAILABLE
        ),
        None,
    )


def _infer_fps(rows: list[dict[str, str]]) -> float | None:
    if len(rows) < 2:
        return None
    first = sensor_time_s(rows[0])
    last = sensor_time_s(rows[-1])
    if first is None or last is None:
        first = wall_time_s(rows[0])
        last = wall_time_s(rows[-1])
    if first is None or last is None or last <= first:
        return None
    return (len(rows) - 1) / (last - first)


def _joint_subtask_segments(
    joints: list[dict[str, Any]],
    camera_rows: list[dict[str, str]],
    warnings: list[str],
) -> list[SubtaskSegment]:
    if not joints or not camera_rows:
        return []
    try:
        state_times, camera_times, _ = common_alignment_times(joints, camera_rows)
    except ValueError as exc:
        warnings.append(f"joints subtask timestamps could not be aligned: {exc}")
        return []

    segments: list[SubtaskSegment] = []
    run_start = 0
    while run_start < len(joints):
        label = joints[run_start].get("subtask")
        normalized = label.strip() if isinstance(label, str) else ""
        run_end = run_start + 1
        while run_end < len(joints) and joints[run_end].get("subtask") == label:
            run_end += 1
        if normalized:
            start_frame = min(
                bisect.bisect_left(camera_times, state_times[run_start]),
                len(camera_rows) - 1,
            )
            end_time = (
                state_times[run_end] if run_end < len(state_times) else state_times[-1]
            )
            end_frame = (
                bisect.bisect_left(camera_times, end_time)
                if run_end < len(state_times)
                else len(camera_rows)
            )
            end_frame = min(max(start_frame + 1, end_frame), len(camera_rows))
            end_stamp_row = min(end_frame, len(camera_rows) - 1)
            source_start_ns = sensor_timestamp_ns(camera_rows[start_frame])
            source_end_ns = sensor_timestamp_ns(camera_rows[end_stamp_row])
            if (
                source_start_ns is not None
                and source_end_ns is not None
                and source_end_ns <= source_start_ns
            ):
                source_end_ns = None
            segments.append(
                SubtaskSegment(
                    source_start_frame=start_frame,
                    source_end_frame=end_frame,
                    instruction=normalized,
                    event_type=normalized,
                    origin="joints.subtask",
                    source_start_ns=source_start_ns,
                    source_end_ns=source_end_ns,
                )
            )
        run_start = run_end
    if segments:
        warnings.append("subtask ranges were imported from joints.jsonl")
    return segments


def _manifest_segments(
    manifest: dict[str, Any],
    camera_rows: list[dict[str, str]],
    warnings: list[str],
) -> list[SubtaskSegment]:
    if not camera_rows:
        return []
    wall_times = [wall_time_s(row) for row in camera_rows]
    if any(value is None for value in wall_times):
        warnings.append("camera CSV has missing wall_time values")
        return []
    resolved_times = [float(value) for value in wall_times if value is not None]
    segments: list[SubtaskSegment] = []
    steps = manifest.get("steps")
    if not isinstance(steps, list):
        return segments
    for step in steps:
        if not isinstance(step, dict):
            continue
        started_at = _float_or_none(step.get("started_at"))
        completed_at = _float_or_none(step.get("completed_at"))
        if started_at is None or completed_at is None or completed_at <= started_at:
            continue
        start_frame = min(
            bisect.bisect_left(resolved_times, started_at), len(resolved_times) - 1
        )
        end_frame = min(
            bisect.bisect_right(resolved_times, completed_at), len(resolved_times)
        )
        if end_frame <= start_frame:
            end_frame = min(start_frame + 1, len(resolved_times))
        if end_frame <= start_frame:
            continue
        instruction = _first_text(step, "subtask", "type", "step_id") or "robot action"
        event_type = _first_text(step, "type", "subtask") or "unknown"
        source_start_ns = sensor_timestamp_ns(camera_rows[start_frame])
        source_end_ns = sensor_timestamp_ns(
            camera_rows[end_frame] if end_frame < len(camera_rows) else camera_rows[-1]
        )
        if (
            source_start_ns is not None
            and source_end_ns is not None
            and source_end_ns <= source_start_ns
        ):
            source_end_ns = None
        segments.append(
            SubtaskSegment(
                source_start_frame=start_frame,
                source_end_frame=end_frame,
                instruction=instruction,
                event_type=event_type,
                origin="manifest_weak_label",
                source_start_ns=source_start_ns,
                source_end_ns=source_end_ns,
            )
        )
    if segments:
        warnings.append(
            "manifest steps were imported as weak labels and require validation"
        )
    return segments


def _first_text(value: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        candidate = value.get(key)
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return None


def _self_collected_task_spec(
    manifest: dict[str, Any],
    episode_id: str,
    warnings: list[str],
) -> TaskSpec:
    for key in ("task", "task_name"):
        text = _first_text(manifest, key)
        if text is not None:
            return TaskSpec(text=text, origin=f"manifest.{key}")
    composition_id = _first_text(manifest, "composition_id")
    if composition_id is not None:
        warnings.append(
            "natural-language task is missing; composition_id requires confirmation"
        )
        return TaskSpec(
            text=composition_id,
            origin="manifest.composition_id",
        )
    warnings.append("task metadata is missing; using the episode directory name")
    return TaskSpec(text=episode_id, origin="episode_directory")


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
