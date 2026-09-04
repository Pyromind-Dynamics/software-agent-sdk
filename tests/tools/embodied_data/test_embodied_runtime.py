"""Tests for self-collected cleaning and LeRobotDataset v2.1 output."""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import imageio_ffmpeg
import numpy as np
import pyarrow.parquet as pq
import pytest
from openhands_embodied_runtime import batch as batch_module, lerobot_v21, planning
from openhands_embodied_runtime.adapters import (
    S2_REFERENCE_JOINT_ORDER,
    S2_REFERENCE_STATE_ORDER,
    build_episode_plan,
    detect_source_type,
    inspect_dataset,
)
from openhands_embodied_runtime.batch import batch_clean_lerobot_v21_dataset
from openhands_embodied_runtime.lerobot_v21 import (
    materialize_lerobot_v21_episode,
    merge_lerobot_v21_datasets,
    validate_lerobot_v21_dataset,
)
from openhands_embodied_runtime.models import (
    ActionDerivationMode,
    ActionProvenanceMode,
    FrameInterval,
    QualityStatus,
    SourceType,
    StreamStatus,
)
from openhands_embodied_runtime.planning import (
    finalize_episode_plan,
    plan_episode_cleaning,
)
from pytest import MonkeyPatch


def _add_shifted_secondary_camera(episode: Path, shift_ns: int) -> None:
    shutil.copy2(episode / "head.mp4", episode / "wrist.mp4")
    with (episode / "wrist.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["frame_index", "wall_time", "stamp_ns"],
        )
        writer.writeheader()
        for index in range(21):
            writer.writerow(
                {
                    "frame_index": index,
                    "wall_time": 100.0 + index / 10,
                    "stamp_ns": 100_000_000_000 + index * 100_000_000 + shift_ns,
                }
            )


def test_self_collected_inspection_uses_mp4_timing(
    self_collected_path: Path,
) -> None:
    plan = build_episode_plan(self_collected_path)

    assert detect_source_type(self_collected_path) == SourceType.SELF_COLLECTED
    assert plan.source_timeline.frame_count == 21
    assert plan.source_timeline.fps == 30
    assert plan.source_timeline.start_stamp_ns == 100_000_000_000
    assert plan.source_timeline.end_stamp_ns == 102_000_000_000
    assert plan.streams["head"].status == StreamStatus.AVAILABLE
    assert plan.streams["head"].fps == 30
    assert plan.streams["head_depth"].status == StreamStatus.INVALID
    assert plan.action_provenance is not None
    assert plan.action_provenance.mode == ActionProvenanceMode.DERIVED
    assert plan.action_provenance.derivation == ActionDerivationMode.NEXT_STATE
    assert not plan.action_provenance.user_confirmed
    assert [feature.name for feature in plan.feature_schema.observation_state] == list(
        S2_REFERENCE_STATE_ORDER
    )
    assert [feature.name for feature in plan.feature_schema.action] == list(
        S2_REFERENCE_STATE_ORDER
    )
    assert plan.quality.status == QualityStatus.NEEDS_REVIEW


def test_lerobot_v21_source_joins_online_labels_by_episode_id(
    lerobot_v21_online_path: Path,
) -> None:
    inspection = inspect_dataset(lerobot_v21_online_path)
    plan = plan_episode_cleaning(
        lerobot_v21_online_path,
        episode_id="648649",
        idle_min_duration_s=10,
    )

    assert detect_source_type(lerobot_v21_online_path) == SourceType.LEROBOT_V21
    assert inspection.episode_count == 1
    assert inspection.sampled_episodes[0].episode_id == "648649"
    assert plan.source_timeline.frame_count == 21
    assert plan.source_timeline.fps == 30
    assert [segment.event_type for segment in plan.segments] == ["pick", "place"]
    assert plan.task.text == "Pickup items in the supermarket"
    assert plan.environment.initial_scene == ("The robot is in front of a fruit stand.")
    assert plan.streams["labels"].status == StreamStatus.AVAILABLE
    assert plan.quality.status == QualityStatus.NEEDS_REVIEW


def test_batch_cleans_lerobot_v21_source_with_online_labels(
    lerobot_v21_online_path: Path,
    tmp_path: Path,
) -> None:
    output = tmp_path / "cleaned_huggingface_lerobot_v21"

    result = batch_clean_lerobot_v21_dataset(
        lerobot_v21_online_path,
        output,
        task_text="Pickup items in the supermarket",
        confirm_subtasks=True,
        confirm_derived_action=True,
        idle_min_duration_s=10,
    )

    assert result.complete
    assert result.discovered_episode_count == 1
    assert result.accepted_episode_ids == ["648649"]
    assert result.failed_episode_count == 0
    assert validate_lerobot_v21_dataset(output).valid
    assert not (output / "annotations.json").exists()
    table = pq.read_table(output / "data/chunk-000/episode_000000.parquet")
    states = table["observation.state"].to_pylist()
    assert table["action"].to_pylist() == [*states[1:], states[-1]]


def test_lerobot_v21_unmatched_online_label_stays_in_review(
    lerobot_v21_online_path: Path,
) -> None:
    annotations_path = lerobot_v21_online_path / "annotations.json"
    annotations = json.loads(annotations_path.read_text(encoding="utf-8"))
    annotations[0]["episode_id"] = 999999
    annotations_path.write_text(json.dumps(annotations) + "\n", encoding="utf-8")

    candidate = plan_episode_cleaning(
        lerobot_v21_online_path,
        episode_id="648649",
        idle_min_duration_s=10,
    )
    finalized = finalize_episode_plan(
        lerobot_v21_online_path,
        candidate,
        task_text="Pickup items in the supermarket",
        confirm_subtasks=True,
        confirm_derived_action=True,
    )

    assert finalized.quality.status == QualityStatus.NEEDS_REVIEW
    assert any("did not match" in warning for warning in finalized.quality.warnings)


def test_storage_sidecar_distinguishes_unmaterialized_depth_from_invalid(
    self_collected_path: Path,
) -> None:
    (self_collected_path / "head_depth.u16").unlink()
    (self_collected_path / ".pyromind_storage.json").write_text(
        json.dumps(
            {
                "files": [
                    {
                        "name": "head_depth.u16",
                        "source_path": "/robot/episode/head_depth.u16",
                        "size": 410_419_200,
                        "materialized": False,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    plan = plan_episode_cleaning(self_collected_path, idle_min_duration_s=10)

    assert plan.streams["head_depth"].status == StreamStatus.NOT_MATERIALIZED
    assert plan.quality.status == QualityStatus.NEEDS_REVIEW
    assert any("must be materialized" in item for item in plan.quality.warnings)


def test_cleaning_plan_aligns_multirate_state_and_keeps_mp4_clock(
    self_collected_path: Path,
) -> None:
    plan = plan_episode_cleaning(
        self_collected_path,
        idle_min_duration_s=0.5,
        context_s=0.1,
    )

    assert plan.source_timeline.fps == 30
    assert plan.drop_intervals
    assert len(plan.timeline_mapping) < 21
    assert [item.clean_frame_index for item in plan.timeline_mapping] == list(
        range(len(plan.timeline_mapping))
    )
    assert all(
        item.state_before_index is not None
        and item.state_after_index is not None
        and item.state_interpolation_weight is not None
        for item in plan.timeline_mapping
    )


def test_sensor_clock_ignores_wall_time_spike_and_allows_camera_duplicates(
    self_collected_path: Path,
) -> None:
    joints_path = self_collected_path / "joints.jsonl"
    joint_rows = [
        json.loads(line)
        for line in joints_path.read_text(encoding="utf-8").splitlines()
    ]
    joint_rows[10]["wall_time"] += 0.108
    joints_path.write_text(
        "".join(json.dumps(row) + "\n" for row in joint_rows),
        encoding="utf-8",
    )

    camera_path = self_collected_path / "head.csv"
    with camera_path.open(encoding="utf-8", newline="") as file:
        camera_rows = list(csv.DictReader(file))
        fieldnames = list(camera_rows[0])
    camera_rows[11]["stamp_sec"] = camera_rows[10]["stamp_sec"]
    camera_rows[11]["stamp_nsec"] = camera_rows[10]["stamp_nsec"]
    with camera_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(camera_rows)

    plan = plan_episode_cleaning(self_collected_path, idle_min_duration_s=10)

    assert plan.quality.status == QualityStatus.NEEDS_REVIEW
    assert plan.quality.errors == []
    assert len(plan.timeline_mapping) == 21
    assert [item.source_frame_index for item in plan.timeline_mapping[10:12]] == [
        10,
        11,
    ]


def test_v21_materialization_matches_reference_layout(
    self_collected_path: Path,
    tmp_path: Path,
) -> None:
    plan = plan_episode_cleaning(
        self_collected_path,
        idle_min_duration_s=10,
    )
    output = tmp_path / "lerobot_v21"

    result = materialize_lerobot_v21_episode(
        self_collected_path,
        plan,
        output,
        task_text="Pick and place the item on the table",
        confirm_subtasks=True,
        confirm_derived_action=True,
    )

    assert result.episode_count == 1
    assert result.frame_count == 21
    assert result.video_count == 1
    assert result.omitted_streams == ["head_depth"]
    expected_files = {
        "meta/info.json",
        "meta/episodes.jsonl",
        "meta/episodes_stats.jsonl",
        "meta/tasks.jsonl",
        "data/chunk-000/episode_000000.parquet",
        "videos/chunk-000/observation.images.head/episode_000000.mp4",
    }
    assert {
        str(path.relative_to(output)) for path in output.rglob("*") if path.is_file()
    } == expected_files

    info = json.loads((output / "meta/info.json").read_text(encoding="utf-8"))
    assert info["codebase_version"] == "v2.1"
    assert info["fps"] == 30
    assert info["data_path"] == (
        "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
    )
    assert info["video_path"] == (
        "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
    )
    assert list(info["features"]) == [
        "observation.state",
        "action",
        "observation.images.head",
        "timestamp",
        "frame_index",
        "episode_index",
        "index",
        "task_index",
    ]

    table = pq.read_table(output / "data/chunk-000/episode_000000.parquet")
    assert table.column_names == [
        "observation.state",
        "action",
        "timestamp",
        "frame_index",
        "episode_index",
        "index",
        "task_index",
    ]
    states = table["observation.state"].to_pylist()
    actions = table["action"].to_pylist()
    assert actions[:-1] == states[1:]
    assert actions[-1] == states[-1]

    report = validate_lerobot_v21_dataset(output)
    assert report.valid
    assert report.errors == []


def test_v21_merge_uses_one_confirmed_task_for_all_episodes(
    self_collected_path: Path,
    tmp_path: Path,
) -> None:
    plan = plan_episode_cleaning(self_collected_path, idle_min_duration_s=10)
    inputs = [tmp_path / "episode_a", tmp_path / "episode_b"]
    for output, task_text in zip(
        inputs,
        ["golden rectangle", "golden rectangle or blue rectangle"],
        strict=True,
    ):
        materialize_lerobot_v21_episode(
            self_collected_path,
            plan,
            output,
            task_text=task_text,
            confirm_subtasks=True,
            confirm_derived_action=True,
        )

    output = tmp_path / "merged"
    result = merge_lerobot_v21_datasets(
        inputs,
        output,
        task_text="Pick and place the item on the table",
    )

    assert result.episode_count == 2
    assert result.frame_count == 42
    assert result.video_count == 2
    info = json.loads((output / "meta/info.json").read_text(encoding="utf-8"))
    assert info["total_tasks"] == 1
    assert info["total_episodes"] == 2
    tasks = [
        json.loads(line)
        for line in (output / "meta/tasks.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert tasks == [{"task_index": 0, "task": "Pick and place the item on the table"}]
    second = pq.read_table(output / "data/chunk-000/episode_000001.parquet")
    assert second["episode_index"].to_pylist() == [1] * 21
    assert second["index"].to_pylist() == list(range(21, 42))
    assert second["task_index"].to_pylist() == [0] * 21
    assert validate_lerobot_v21_dataset(output).valid


def test_batch_clean_merges_and_resumes_from_checkpoint(
    self_collected_path: Path,
    tmp_path: Path,
) -> None:
    source = tmp_path / "batch_source"
    shutil.copytree(self_collected_path, source / "episode_a")
    shutil.copytree(self_collected_path, source / "episode_b")
    output = tmp_path / "batch_lerobot_v21"

    result = batch_clean_lerobot_v21_dataset(
        source,
        output,
        task_text="Pick and place the item on the table",
        confirm_subtasks=True,
        confirm_derived_action=True,
        idle_min_duration_s=10,
    )

    assert result.complete
    assert result.discovered_episode_count == 2
    assert result.accepted_episode_count == 2
    assert result.rejected_episode_count == 0
    assert result.failed_episode_count == 0
    assert result.frame_count == 42
    assert result.video_count == 2
    checkpoint_mtime = Path(result.checkpoint_path).stat().st_mtime_ns
    report = json.loads(Path(result.report_path).read_text(encoding="utf-8"))
    assert set(report["episodes"]) == {"episode_a", "episode_b"}
    assert validate_lerobot_v21_dataset(output).valid

    resumed = batch_clean_lerobot_v21_dataset(
        source,
        output,
        task_text="Pick and place the item on the table",
        confirm_subtasks=True,
        confirm_derived_action=True,
        idle_min_duration_s=10,
        resume=True,
    )

    assert resumed == result
    assert Path(result.checkpoint_path).stat().st_mtime_ns == checkpoint_mtime


def test_batch_clean_isolates_quality_rejections(
    self_collected_path: Path,
    tmp_path: Path,
) -> None:
    source = tmp_path / "batch_with_rejection"
    shutil.copytree(self_collected_path, source / "accepted_episode")
    rejected = source / "rejected_episode"
    shutil.copytree(self_collected_path, rejected)
    rows = [
        json.loads(line)
        for line in (rejected / "joints.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    for row in rows:
        row["stamp_ns"] += 501_000_000
    (rejected / "joints.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    result = batch_clean_lerobot_v21_dataset(
        source,
        tmp_path / "batch_with_rejection_lerobot_v21",
        task_text="Pick and place the item on the table",
        confirm_subtasks=True,
        confirm_derived_action=True,
        idle_min_duration_s=10,
    )

    assert result.complete
    assert result.accepted_episode_ids == ["accepted_episode"]
    assert result.rejected_episode_ids == ["rejected_episode"]
    assert result.failed_episode_count == 0
    report = json.loads(Path(result.report_path).read_text(encoding="utf-8"))
    assert "precedes state coverage" in " ".join(
        report["episodes"]["rejected_episode"]["errors"]
    )
    diagnostic = result.rejected_episode_reports[0]
    assert diagnostic.error_code == "CAMERA_LEADS_STATE_OVER_LIMIT"
    assert diagnostic.details["observed_gap_s"] == 0.501
    assert diagnostic.details["allowed_gap_s"] == 0.5


def test_alignment_accepts_boundary_and_rejects_both_directions_over_limit() -> None:
    camera_at_zero = [{"stamp_ns": "100000000000"}]
    camera_at_one = [{"stamp_ns": "101000000000"}]

    for gap_ns in (499_000_000, 500_000_000):
        leading_state = [{"stamp_ns": 100_000_000_000 + gap_ns}]
        lagging_state = [{"stamp_ns": 101_000_000_000 - gap_ns}]
        assert planning._build_state_alignment(
            leading_state,
            camera_at_zero,
            max_gap_s=0.5,
        )
        assert planning._build_state_alignment(
            lagging_state,
            camera_at_one,
            max_gap_s=0.5,
        )

    with pytest.raises(ValueError, match="precedes state coverage by 0.501000s"):
        planning._build_state_alignment(
            [{"stamp_ns": 100_501_000_000}],
            camera_at_zero,
            max_gap_s=0.5,
        )
    with pytest.raises(ValueError, match="exceeds state coverage by 0.501000s"):
        planning._build_state_alignment(
            [{"stamp_ns": 100_499_000_000}],
            camera_at_one,
            max_gap_s=0.5,
        )


def test_plan_warns_for_500ms_alignment_gap(
    self_collected_path: Path,
    tmp_path: Path,
) -> None:
    source = tmp_path / "alignment-warning"
    shutil.copytree(self_collected_path, source)
    rows = [
        json.loads(line)
        for line in (source / "joints.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    for row in rows:
        row["stamp_ns"] += 500_000_000
    (source / "joints.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    plan = plan_episode_cleaning(source, idle_min_duration_s=10)

    assert plan.quality.status != QualityStatus.REJECTED
    alignment_warnings = [
        warning
        for warning in plan.quality.warnings
        if warning.startswith("large alignment gap:")
    ]
    assert alignment_warnings == [
        "large alignment gap: type=camera_leads_state, frame=0, "
        "gap_s=0.500000, reject_threshold_s=0.500000"
    ]


def test_secondary_camera_uses_500ms_quality_gate(
    self_collected_path: Path,
    tmp_path: Path,
) -> None:
    warning_source = tmp_path / "secondary-warning"
    shutil.copytree(self_collected_path, warning_source)
    _add_shifted_secondary_camera(warning_source, 500_000_000)

    warning_plan = plan_episode_cleaning(warning_source, idle_min_duration_s=10)

    assert warning_plan.quality.status != QualityStatus.REJECTED
    assert any(
        "stream=wrist" in warning and "gap_s=0.500000" in warning
        for warning in warning_plan.quality.warnings
    )
    warning_result = materialize_lerobot_v21_episode(
        warning_source,
        warning_plan,
        tmp_path / "secondary-warning-output",
        task_text="Pick and place the item on the table",
        confirm_subtasks=True,
        confirm_derived_action=True,
    )
    assert warning_result.video_count == 2
    assert validate_lerobot_v21_dataset(
        Path(warning_result.root)
    ).reference_profile_valid

    rejected_root = tmp_path / "secondary-rejected-root"
    rejected_source = rejected_root / "secondary-rejected"
    shutil.copytree(self_collected_path, rejected_source)
    _add_shifted_secondary_camera(rejected_source, 501_000_000)

    result = batch_clean_lerobot_v21_dataset(
        rejected_root,
        tmp_path / "secondary-output",
        task_text="Pick and place the item on the table",
        confirm_subtasks=True,
        confirm_derived_action=True,
        idle_min_duration_s=10,
    )

    assert result.processing_complete
    assert result.rejected_episode_ids == ["secondary-rejected"]
    diagnostic = result.rejected_episode_reports[0]
    assert diagnostic.error_code == "SECONDARY_STREAM_ALIGNMENT_OVER_LIMIT"
    assert diagnostic.details["stream"] == "wrist"
    assert diagnostic.details["direction"] == "primary_leads_secondary"
    assert diagnostic.details["observed_gap_s"] == 0.501


def test_alignment_rejects_internal_state_gap_over_limit() -> None:
    with pytest.raises(ValueError, match="inside a 0.501000s state gap"):
        planning._build_state_alignment(
            [
                {"stamp_ns": 100_000_000_000},
                {"stamp_ns": 100_501_000_000},
            ],
            [{"stamp_ns": "100250000000"}],
            max_gap_s=0.5,
        )


def test_batch_clean_reports_when_every_episode_is_rejected(
    self_collected_path: Path,
    tmp_path: Path,
) -> None:
    source = tmp_path / "batch_all_rejected"
    rejected = source / "rejected_episode"
    shutil.copytree(self_collected_path, rejected)
    rows = [
        json.loads(line)
        for line in (rejected / "joints.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    for row in rows:
        row["stamp_ns"] += 501_000_000
    (rejected / "joints.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    output = tmp_path / "batch_all_rejected_lerobot_v21"
    result = batch_clean_lerobot_v21_dataset(
        source,
        output,
        task_text="Pick and place the item on the table",
        confirm_subtasks=True,
        confirm_derived_action=True,
        idle_min_duration_s=10,
    )

    assert not result.complete
    assert result.processing_complete
    assert result.accepted_episode_count == 0
    assert result.rejected_episode_ids == ["rejected_episode"]
    assert result.failed_episode_count == 0
    assert not output.exists()
    report = json.loads(Path(result.report_path).read_text(encoding="utf-8"))
    assert report["episodes"]["rejected_episode"]["status"] == "rejected"


def test_batch_clean_resume_retries_only_runtime_failures(
    self_collected_path: Path,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    source = tmp_path / "batch_resume_source"
    shutil.copytree(self_collected_path, source / "episode_a")
    shutil.copytree(self_collected_path, source / "episode_b")
    output = tmp_path / "batch_resume_lerobot_v21"
    original_materialize = batch_module.materialize_finalized_lerobot_v21_episode

    def fail_episode_b_once(source, prepared, output_root, *, robot_type):
        if prepared.episode_id == "episode_b":
            raise RuntimeError("simulated transient materialization failure")
        return original_materialize(
            source,
            prepared,
            output_root,
            robot_type=robot_type,
        )

    monkeypatch.setattr(
        batch_module,
        "materialize_finalized_lerobot_v21_episode",
        fail_episode_b_once,
    )
    first = batch_clean_lerobot_v21_dataset(
        source,
        output,
        task_text="Pick and place the item on the table",
        confirm_subtasks=True,
        confirm_derived_action=True,
        idle_min_duration_s=10,
    )

    assert not first.complete
    assert first.accepted_episode_ids == ["episode_a"]
    assert first.failed_episode_ids == ["episode_b"]
    episode_a_parquet = (
        Path(first.work_path)
        / "episodes/episode_a/data/chunk-000/episode_000000.parquet"
    )
    episode_a_mtime = episode_a_parquet.stat().st_mtime_ns

    monkeypatch.setattr(
        batch_module,
        "materialize_finalized_lerobot_v21_episode",
        original_materialize,
    )
    resumed = batch_clean_lerobot_v21_dataset(
        source,
        output,
        task_text="Pick and place the item on the table",
        confirm_subtasks=True,
        confirm_derived_action=True,
        idle_min_duration_s=10,
        resume=True,
    )

    assert resumed.complete
    assert resumed.accepted_episode_ids == ["episode_a", "episode_b"]
    assert resumed.failed_episode_count == 0
    assert episode_a_parquet.stat().st_mtime_ns == episode_a_mtime


def test_v21_materialization_converts_valid_raw_depth(
    self_collected_path: Path,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    depth = np.arange(21 * 16 * 16, dtype=np.uint16).reshape(21, 16, 16) % 4096
    (self_collected_path / "head_depth.u16").write_bytes(depth.astype("<u2").tobytes())
    (self_collected_path / "head_depth_meta.json").write_text(
        json.dumps(
            {
                "width": 16,
                "height": 16,
                "encoding": "16uc12",
                "depth_scale": 0.001,
            }
        ),
        encoding="utf-8",
    )
    with (self_collected_path / "head_depth.csv").open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["frame_index", "wall_time", "stamp_ns"],
        )
        writer.writeheader()
        for index in range(21):
            if index <= 1:
                offset_ns = 0
            elif index == 20:
                offset_ns = 2_000_000_000
            else:
                offset_ns = (index - 1) * 100_000_000
            writer.writerow(
                {
                    "frame_index": index,
                    "wall_time": 100.0 + index / 10,
                    "stamp_ns": 100_000_000_000 + offset_ns,
                }
            )

    selected_depth_indexes: list[int] = []
    original_filter = lerobot_v21._filter_depth_u16

    def capture_depth_indexes(
        source: Path,
        metadata_path: Path,
        destination: Path,
        selected_indexes: list[int],
        fps: int,
        declared_frame_count: int | None,
    ) -> tuple[int, int]:
        selected_depth_indexes.extend(selected_indexes)
        return original_filter(
            source,
            metadata_path,
            destination,
            selected_indexes,
            fps,
            declared_frame_count,
        )

    monkeypatch.setattr(lerobot_v21, "_filter_depth_u16", capture_depth_indexes)
    plan = plan_episode_cleaning(self_collected_path, idle_min_duration_s=10)
    output = tmp_path / "lerobot_with_depth"
    result = materialize_lerobot_v21_episode(
        self_collected_path,
        plan,
        output,
        task_text="Pick and place the item on the table",
        confirm_subtasks=True,
        confirm_derived_action=True,
    )

    assert result.video_count == 2
    assert result.omitted_streams == []
    info = json.loads((output / "meta/info.json").read_text(encoding="utf-8"))
    assert info["s2_depth_video_colormap"] == "jet"
    assert "observation.images.head_depth" in info["features"]
    depth_video = (
        output / "videos/chunk-000/observation.images.head_depth/episode_000000.mp4"
    )
    frame_count, _ = imageio_ffmpeg.count_frames_and_secs(str(depth_video))
    assert frame_count == 21
    assert selected_depth_indexes[:5] == [0, 2, 3, 4, 5]
    assert selected_depth_indexes[-1] == 20
    assert selected_depth_indexes == sorted(selected_depth_indexes)


def test_s2_reference_profile_uses_ordered_joints_and_gripper_positions(
    self_collected_path: Path,
    tmp_path: Path,
) -> None:
    joints_path = self_collected_path / "joints.jsonl"
    rows = [
        json.loads(line)
        for line in joints_path.read_text(encoding="utf-8").splitlines()
    ]
    for frame_index, row in enumerate(rows):
        row["joints"] = {
            name: frame_index + joint_index / 100
            for joint_index, name in enumerate(S2_REFERENCE_JOINT_ORDER)
        }
        row["left_ee_states"] = [0.0, 0.0, 0.0, 0.0, 0.05, 0.0, 0.0]
        row["right_ee_states"] = [0.0, 0.0, 0.0, 0.0, 0.02, 0.0, 0.0]
    joints_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    plan = plan_episode_cleaning(self_collected_path, idle_min_duration_s=10)
    assert [feature.name for feature in plan.feature_schema.observation_state] == list(
        S2_REFERENCE_STATE_ORDER
    )

    output = tmp_path / "reference_profile"
    materialize_lerobot_v21_episode(
        self_collected_path,
        plan,
        output,
        task_text="Pick and place the item on the table",
        confirm_subtasks=True,
        confirm_derived_action=True,
    )
    info = json.loads((output / "meta/info.json").read_text(encoding="utf-8"))
    assert info["features"]["observation.state"]["shape"] == [24]
    assert info["features"]["observation.state"]["names"] == list(
        S2_REFERENCE_STATE_ORDER
    )
    table = pq.read_table(output / "data/chunk-000/episode_000000.parquet")
    first_state = table["observation.state"][0].as_py()
    np.testing.assert_allclose(first_state[-2:], [0.05, 0.02])


def test_s2_reference_profile_interpolates_sparse_gripper_rows(
    self_collected_path: Path,
    tmp_path: Path,
) -> None:
    joints_path = self_collected_path / "joints.jsonl"
    rows = [
        json.loads(line)
        for line in joints_path.read_text(encoding="utf-8").splitlines()
    ]
    rows[10]["left_ee_states"] = []
    rows[10]["right_ee_states"] = []
    joints_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    plan = plan_episode_cleaning(self_collected_path, idle_min_duration_s=10)
    assert len(plan.feature_schema.observation_state) == 24
    assert any("sparse end-effector" in warning for warning in plan.quality.warnings)
    output = tmp_path / "sparse_gripper"
    materialize_lerobot_v21_episode(
        self_collected_path,
        plan,
        output,
        task_text="Pick and place the item on the table",
        confirm_subtasks=True,
        confirm_derived_action=True,
    )
    table = pq.read_table(output / "data/chunk-000/episode_000000.parquet")
    state = table["observation.state"][10].as_py()
    np.testing.assert_allclose(state[-2:], [0.05, 0.02])


def test_v21_materialization_requires_subtask_confirmation(
    self_collected_path: Path,
    tmp_path: Path,
) -> None:
    plan = plan_episode_cleaning(self_collected_path)

    try:
        materialize_lerobot_v21_episode(
            self_collected_path,
            plan,
            tmp_path / "unconfirmed",
            task_text="Pick and place the item on the table",
            confirm_subtasks=False,
            confirm_derived_action=True,
        )
    except ValueError as exc:
        assert "subtask ranges require explicit confirmation" in str(exc)
    else:
        raise AssertionError("unconfirmed subtask ranges should be rejected")


def test_v21_materialization_requires_derived_action_confirmation(
    self_collected_path: Path,
    tmp_path: Path,
) -> None:
    plan = plan_episode_cleaning(self_collected_path, idle_min_duration_s=10)

    try:
        materialize_lerobot_v21_episode(
            self_collected_path,
            plan,
            tmp_path / "unconfirmed_action",
            task_text="Pick and place the item on the table",
            confirm_subtasks=True,
        )
    except ValueError as exc:
        assert "derived action semantics require explicit user confirmation" in str(exc)
    else:
        raise AssertionError("unconfirmed derived action semantics should be rejected")


def test_s2_profile_errors_are_reported_before_materialization(
    self_collected_path: Path,
) -> None:
    joints_path = self_collected_path / "joints.jsonl"
    rows = [
        json.loads(line)
        for line in joints_path.read_text(encoding="utf-8").splitlines()
    ]
    rows[0]["joints"].pop(S2_REFERENCE_JOINT_ORDER[0])
    joints_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    plan = plan_episode_cleaning(self_collected_path, idle_min_duration_s=10)

    assert plan.quality.status == QualityStatus.REJECTED
    assert any(S2_REFERENCE_JOINT_ORDER[0] in error for error in plan.quality.errors)


def test_finalization_records_confirmations_and_removes_resolved_warnings(
    self_collected_path: Path,
) -> None:
    candidate = plan_episode_cleaning(self_collected_path, idle_min_duration_s=10)

    finalized = finalize_episode_plan(
        self_collected_path,
        candidate,
        task_text="Pick and place the item on the table",
        confirm_subtasks=True,
        confirm_derived_action=True,
    )

    assert finalized.quality.status == QualityStatus.ACCEPTED
    assert finalized.action_provenance is not None
    assert finalized.action_provenance.user_confirmed
    assert all(segment.user_confirmed for segment in finalized.segments)
    assert not any(
        warning.startswith("task metadata is missing;")
        or warning.startswith("subtask ranges were imported from")
        or warning.startswith("action will use the reference-dataset convention:")
        for warning in finalized.quality.warnings
    )


def test_v21_validator_checks_stats_video_shape_and_final_structural_status(
    self_collected_path: Path,
    tmp_path: Path,
) -> None:
    plan = plan_episode_cleaning(self_collected_path, idle_min_duration_s=10)
    output = tmp_path / "validator_checks"
    materialize_lerobot_v21_episode(
        self_collected_path,
        plan,
        output,
        task_text="Pick and place the item on the table",
        confirm_subtasks=True,
        confirm_derived_action=True,
    )
    stats_path = output / "meta/episodes_stats.jsonl"
    stats_row = json.loads(stats_path.read_text(encoding="utf-8"))
    stats_row["stats"]["observation.state"]["mean"][0] += 1.0
    stats_path.write_text(json.dumps(stats_row) + "\n", encoding="utf-8")
    info_path = output / "meta/info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info["total_frames"] += 1
    info["features"]["observation.images.head"]["shape"] = [1, 1, 3]
    info_path.write_text(json.dumps(info), encoding="utf-8")

    report = validate_lerobot_v21_dataset(output)

    assert not report.valid
    assert not report.structurally_valid
    assert any("statistic mean is incorrect" in error for error in report.errors)
    assert any("shape does not match" in error for error in report.errors)
    assert "info.json total_frames is incorrect" in report.errors


def test_materialization_rebuilds_source_state_instead_of_trusting_accepted_plan(
    self_collected_path: Path,
    tmp_path: Path,
) -> None:
    plan = plan_episode_cleaning(self_collected_path, idle_min_duration_s=10)
    tampered = plan.model_copy(
        update={
            "quality": plan.quality.model_copy(
                update={"status": QualityStatus.ACCEPTED, "errors": []}
            )
        }
    )
    (self_collected_path / "joints.jsonl").write_text("", encoding="utf-8")

    try:
        materialize_lerobot_v21_episode(
            self_collected_path,
            tampered,
            tmp_path / "tampered",
            task_text="Pick and place the item on the table",
            confirm_subtasks=True,
            confirm_derived_action=True,
        )
    except ValueError as exc:
        assert "episode plan is not accepted" in str(exc)
        assert "joints.jsonl contains no records" in str(exc)
    else:
        raise AssertionError("a hand-edited accepted status must not bypass validation")


def test_finalization_rejects_drop_interval_over_suction_transition(
    self_collected_path: Path,
) -> None:
    joints_path = self_collected_path / "joints.jsonl"
    rows = [
        json.loads(line)
        for line in joints_path.read_text(encoding="utf-8").splitlines()
    ]
    for row in rows[10:]:
        row["suction_state"] = 1
    joints_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    candidate = plan_episode_cleaning(self_collected_path, idle_min_duration_s=10)
    candidate = candidate.model_copy(
        update={
            "drop_intervals": [
                FrameInterval(
                    start_frame=10,
                    end_frame=11,
                    reason="manual_drop",
                )
            ]
        }
    )

    finalized = finalize_episode_plan(
        self_collected_path,
        candidate,
        task_text="Pick and place the item on the table",
        confirm_subtasks=True,
        confirm_derived_action=True,
    )

    assert finalized.quality.status == QualityStatus.REJECTED
    assert any("suction transition" in error for error in finalized.quality.errors)
