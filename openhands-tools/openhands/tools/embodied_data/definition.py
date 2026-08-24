"""Agent-facing tools for embodied-data inspection and cleaning plans."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

from pydantic import Field

from openhands.sdk.tool import (
    Action,
    Observation,
    ToolAnnotations,
    ToolDefinition,
    ToolExecutor,
    register_tool,
)
from openhands.tools.embodied_data.adapters import inspect_dataset
from openhands.tools.embodied_data.batch import batch_clean_lerobot_v21_dataset
from openhands.tools.embodied_data.lerobot_v21 import (
    LeRobotV21ValidationReport,
    materialize_lerobot_v21_episode,
    merge_lerobot_v21_datasets,
    validate_lerobot_v21_dataset,
)
from openhands.tools.embodied_data.models import EpisodePlan, EpisodeSummary
from openhands.tools.embodied_data.planning import (
    finalize_episode_plan,
    plan_episode_cleaning,
)
from openhands.tools.embodied_data.reporting import render_episode_plan_summary
from openhands.tools.pyromind_dataset.definition import (
    UploadFileToPyromindAction,
    UploadFileToPyromindExecutor,
)


if TYPE_CHECKING:
    from openhands.sdk.conversation.state import ConversationState


class InspectEmbodiedDatasetAction(Action):
    path: str = Field(description="Dataset path inside the conversation workspace")
    sample_limit: int = Field(
        default=3,
        ge=1,
        le=3,
        description="Maximum number of episode summaries to inspect",
    )


class InspectEmbodiedDatasetObservation(Observation):
    source_type: str
    source_path: str
    episode_count: int
    sampled_episodes: list[EpisodeSummary] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class BuildEmbodiedEpisodePlanAction(Action):
    path: str = Field(description="Dataset path inside the conversation workspace")
    episode_id: str | None = Field(
        default=None,
        description=(
            "Episode identifier; required when the source contains many episodes"
        ),
    )
    motion_speed_threshold: float = Field(
        default=0.02,
        gt=0,
        description="Maximum joint speed in radians/second considered static",
    )
    idle_min_duration_s: float = Field(
        default=1.5,
        gt=0,
        description="Minimum sustained idle duration eligible for compression",
    )
    context_s: float = Field(
        default=0.5,
        ge=0,
        description="Context retained at idle and subtask boundaries",
    )
    output_path: str | None = Field(
        default=None,
        description=(
            "Optional JSON output path inside the workspace. Defaults under "
            ".agent_tmp/embodied-data-cleaning/."
        ),
    )


class BuildEmbodiedEpisodePlanObservation(Observation):
    episode_id: str
    source_type: str
    quality_status: str
    segment_count: int
    drop_interval_count: int
    retained_frame_count: int | None = None
    output_path: str
    summary_path: str = ""
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class MaterializeLeRobotV21Action(Action):
    path: str = Field(description="Self-collected dataset path in the workspace")
    plan_path: str = Field(description="Reviewed episode_plan.json path")
    output_path: str = Field(description="New LeRobot v2.1 dataset directory")
    task_text: str = Field(description="Confirmed natural-language episode task")
    confirm_subtasks: bool = Field(
        description="Whether source-derived subtask ranges were reviewed"
    )
    confirm_derived_action: bool = Field(
        default=False,
        description=(
            "Whether next-state action semantics were explicitly confirmed by the user"
        ),
    )
    robot_type: str = Field(default="s2", description="LeRobot robot_type value")


class FinalizeEmbodiedEpisodePlanAction(Action):
    path: str = Field(description="Self-collected dataset path in the workspace")
    plan_path: str = Field(description="Candidate episode_plan.json path")
    task_text: str = Field(description="Confirmed natural-language episode task")
    confirm_subtasks: bool = Field(
        description="Whether source-derived subtask ranges were reviewed"
    )
    confirm_derived_action: bool = Field(
        default=False,
        description=(
            "Whether next-state action semantics were explicitly confirmed by the user"
        ),
    )
    output_path: str | None = Field(
        default=None,
        description="Accepted plan output path; defaults beside the candidate plan",
    )


class MaterializeLeRobotV21Observation(Observation):
    output_path: str
    episode_count: int = 0
    frame_count: int = 0
    video_count: int = 0
    omitted_streams: list[str] = Field(default_factory=list)
    validation_errors: list[str] = Field(default_factory=list)


class BatchCleanLeRobotV21Action(Action):
    path: str = Field(description="Self-collected dataset path in the workspace")
    output_path: str = Field(description="New merged LeRobot v2.1 dataset directory")
    task_text: str = Field(description="Confirmed task shared by every episode")
    confirm_subtasks: bool = Field(
        description="Dataset-level confirmation of source-derived subtask ranges"
    )
    confirm_derived_action: bool = Field(
        description="Dataset-level confirmation of next-state action semantics"
    )
    robot_type: str = Field(default="s2", description="LeRobot robot_type value")
    motion_speed_threshold: float = Field(default=0.02, gt=0)
    idle_min_duration_s: float = Field(default=1.5, gt=0)
    context_s: float = Field(default=0.5, ge=0)
    resume: bool = Field(
        default=False,
        description="Resume the matching checkpoint instead of starting a new batch",
    )


class BatchCleanLeRobotV21Observation(Observation):
    output_path: str
    work_path: str = ""
    checkpoint_path: str = ""
    report_path: str = ""
    complete: bool = False
    discovered_episode_count: int = 0
    accepted_episode_count: int = 0
    needs_review_episode_count: int = 0
    rejected_episode_count: int = 0
    failed_episode_count: int = 0
    frame_count: int = 0
    video_count: int = 0
    accepted_episode_ids: list[str] = Field(default_factory=list)
    needs_review_episode_ids: list[str] = Field(default_factory=list)
    rejected_episode_ids: list[str] = Field(default_factory=list)
    failed_episode_ids: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class MergeLeRobotV21Action(Action):
    input_paths: list[str] = Field(
        min_length=1,
        description="Validated single-episode LeRobot v2.1 dataset directories",
    )
    output_path: str = Field(description="New merged LeRobot v2.1 dataset directory")
    task_text: str = Field(description="Confirmed task shared by every episode")


class ValidateLeRobotV21Action(Action):
    path: str = Field(description="LeRobot v2.1 dataset directory in the workspace")


class ValidateLeRobotV21Observation(Observation):
    valid: bool
    structurally_valid: bool = False
    reference_profile_valid: bool = False
    episode_count: int = 0
    frame_count: int = 0
    video_count: int = 0
    errors: list[str] = Field(default_factory=list)


class PublishLeRobotV21Action(Action):
    path: str = Field(
        description="Validated LeRobot v2.1 dataset directory in the workspace"
    )
    storage_path: str = Field(
        description=(
            "Destination dataset directory in Pyromind Storage, for example "
            "'/workspace/robot/episode_lerobot_v21'. The platform workspace "
            "prefix is accepted as an alias and normalized to '/robot/...'."
        ),
    )


class PublishLeRobotV21Observation(Observation):
    local_path: str
    storage_path: str
    platform_path: str = ""
    complete: bool = False
    file_count: int = 0
    uploaded_file_count: int = 0
    total_bytes: int = 0
    uploaded_paths: list[str] = Field(default_factory=list)
    failed_file: str | None = None
    validation_errors: list[str] = Field(default_factory=list)


class InspectEmbodiedDatasetExecutor(
    ToolExecutor[InspectEmbodiedDatasetAction, InspectEmbodiedDatasetObservation]
):
    def __init__(self, working_dir: str):
        self.working_dir = Path(working_dir).resolve()

    def __call__(
        self,
        action: InspectEmbodiedDatasetAction,
        _conversation=None,
    ) -> InspectEmbodiedDatasetObservation:
        try:
            source = _resolve_workspace_path(self.working_dir, action.path)
            inspection = inspect_dataset(source, sample_limit=action.sample_limit)
            summary = (
                f"Detected {inspection.source_type.value} data with "
                f"{inspection.episode_count} episode(s)."
            )
            return InspectEmbodiedDatasetObservation.from_text(
                text=summary,
                source_type=inspection.source_type.value,
                source_path=inspection.source_path,
                episode_count=inspection.episode_count,
                sampled_episodes=inspection.sampled_episodes,
                warnings=inspection.warnings,
            )
        except Exception as exc:
            return InspectEmbodiedDatasetObservation.from_text(
                text=str(exc),
                is_error=True,
                source_type="unknown",
                source_path=action.path,
                episode_count=0,
            )


class BuildEmbodiedEpisodePlanExecutor(
    ToolExecutor[
        BuildEmbodiedEpisodePlanAction,
        BuildEmbodiedEpisodePlanObservation,
    ]
):
    def __init__(self, working_dir: str):
        self.working_dir = Path(working_dir).resolve()

    def __call__(
        self,
        action: BuildEmbodiedEpisodePlanAction,
        _conversation=None,
    ) -> BuildEmbodiedEpisodePlanObservation:
        try:
            source = _resolve_workspace_path(self.working_dir, action.path)
            plan = plan_episode_cleaning(
                source,
                episode_id=action.episode_id,
                motion_speed_threshold=action.motion_speed_threshold,
                idle_min_duration_s=action.idle_min_duration_s,
                context_s=action.context_s,
            )
            output_path = _resolve_output_path(
                self.working_dir,
                action.output_path,
                plan.episode_id,
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                json.dumps(plan.model_dump(mode="json"), ensure_ascii=False, indent=2)
                + "\n",
                encoding="utf-8",
            )
            summary_path = output_path.with_suffix(".summary.md")
            summary_path.write_text(
                render_episode_plan_summary(plan),
                encoding="utf-8",
            )
            retained = (
                len(plan.timeline_mapping)
                if plan.source_timeline.frame_count is not None
                else None
            )
            summary = (
                f"Wrote episode plan for {plan.episode_id} with quality "
                f"{plan.quality.status.value} to {output_path}."
            )
            return BuildEmbodiedEpisodePlanObservation.from_text(
                text=summary,
                episode_id=plan.episode_id,
                source_type=plan.source_type.value,
                quality_status=plan.quality.status.value,
                segment_count=len(plan.segments),
                drop_interval_count=len(plan.drop_intervals),
                retained_frame_count=retained,
                output_path=str(output_path),
                summary_path=str(summary_path),
                errors=plan.quality.errors,
                warnings=plan.quality.warnings,
            )
        except Exception as exc:
            return BuildEmbodiedEpisodePlanObservation.from_text(
                text=str(exc),
                is_error=True,
                episode_id=action.episode_id or "unknown",
                source_type="unknown",
                quality_status="rejected",
                segment_count=0,
                drop_interval_count=0,
                output_path="",
                errors=[str(exc)],
            )


class MaterializeLeRobotV21Executor(
    ToolExecutor[MaterializeLeRobotV21Action, MaterializeLeRobotV21Observation]
):
    def __init__(self, working_dir: str):
        self.working_dir = Path(working_dir).resolve()

    def __call__(
        self,
        action: MaterializeLeRobotV21Action,
        _conversation=None,
    ) -> MaterializeLeRobotV21Observation:
        try:
            source = _resolve_workspace_path(self.working_dir, action.path)
            plan_path = _resolve_workspace_path(self.working_dir, action.plan_path)
            plan = EpisodePlan.model_validate_json(
                plan_path.read_text(encoding="utf-8")
            )
            output = _resolve_workspace_output_dir(self.working_dir, action.output_path)
            result = materialize_lerobot_v21_episode(
                source,
                plan,
                output,
                task_text=action.task_text,
                confirm_subtasks=action.confirm_subtasks,
                confirm_derived_action=action.confirm_derived_action,
                robot_type=action.robot_type,
            )
            return MaterializeLeRobotV21Observation.from_text(
                text=(
                    f"Wrote and validated LeRobot v2.1 dataset with "
                    f"{result.frame_count} frames to {result.root}."
                ),
                output_path=result.root,
                episode_count=result.episode_count,
                frame_count=result.frame_count,
                video_count=result.video_count,
                omitted_streams=result.omitted_streams,
            )
        except Exception as exc:
            return MaterializeLeRobotV21Observation.from_text(
                text=str(exc),
                is_error=True,
                output_path=action.output_path,
                validation_errors=[str(exc)],
            )


class FinalizeEmbodiedEpisodePlanExecutor(
    ToolExecutor[
        FinalizeEmbodiedEpisodePlanAction,
        BuildEmbodiedEpisodePlanObservation,
    ]
):
    def __init__(self, working_dir: str):
        self.working_dir = Path(working_dir).resolve()

    def __call__(
        self,
        action: FinalizeEmbodiedEpisodePlanAction,
        _conversation=None,
    ) -> BuildEmbodiedEpisodePlanObservation:
        try:
            source = _resolve_workspace_path(self.working_dir, action.path)
            plan_path = _resolve_workspace_path(self.working_dir, action.plan_path)
            candidate = EpisodePlan.model_validate_json(
                plan_path.read_text(encoding="utf-8")
            )
            plan = finalize_episode_plan(
                source,
                candidate,
                task_text=action.task_text,
                confirm_subtasks=action.confirm_subtasks,
                confirm_derived_action=action.confirm_derived_action,
            )
            output_path = _resolve_finalized_output_path(
                self.working_dir,
                plan_path,
                action.output_path,
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                json.dumps(plan.model_dump(mode="json"), ensure_ascii=False, indent=2)
                + "\n",
                encoding="utf-8",
            )
            summary_path = output_path.with_suffix(".summary.md")
            summary_path.write_text(
                render_episode_plan_summary(plan),
                encoding="utf-8",
            )
            return BuildEmbodiedEpisodePlanObservation.from_text(
                text=(
                    f"Rebuilt and finalized episode plan with quality "
                    f"{plan.quality.status.value} to {output_path}."
                ),
                is_error=plan.quality.status.value != "accepted",
                episode_id=plan.episode_id,
                source_type=plan.source_type.value,
                quality_status=plan.quality.status.value,
                segment_count=len(plan.segments),
                drop_interval_count=len(plan.drop_intervals),
                retained_frame_count=len(plan.timeline_mapping),
                output_path=str(output_path),
                summary_path=str(summary_path),
                errors=plan.quality.errors,
                warnings=plan.quality.warnings,
            )
        except Exception as exc:
            return BuildEmbodiedEpisodePlanObservation.from_text(
                text=str(exc),
                is_error=True,
                episode_id="unknown",
                source_type="unknown",
                quality_status="rejected",
                segment_count=0,
                drop_interval_count=0,
                output_path="",
                errors=[str(exc)],
            )


class MergeLeRobotV21Executor(
    ToolExecutor[MergeLeRobotV21Action, MaterializeLeRobotV21Observation]
):
    def __init__(self, working_dir: str):
        self.working_dir = Path(working_dir).resolve()

    def __call__(
        self,
        action: MergeLeRobotV21Action,
        _conversation=None,
    ) -> MaterializeLeRobotV21Observation:
        try:
            sources = [
                _resolve_workspace_path(self.working_dir, path)
                for path in action.input_paths
            ]
            output = _resolve_workspace_output_dir(self.working_dir, action.output_path)
            result = merge_lerobot_v21_datasets(
                sources,
                output,
                task_text=action.task_text,
            )
            return MaterializeLeRobotV21Observation.from_text(
                text=(
                    f"Merged and validated {result.episode_count} LeRobot v2.1 "
                    f"episodes with {result.frame_count} frames at {result.root}."
                ),
                output_path=result.root,
                episode_count=result.episode_count,
                frame_count=result.frame_count,
                video_count=result.video_count,
                omitted_streams=result.omitted_streams,
            )
        except Exception as exc:
            return MaterializeLeRobotV21Observation.from_text(
                text=str(exc),
                is_error=True,
                output_path=action.output_path,
                validation_errors=[str(exc)],
            )


class BatchCleanLeRobotV21Executor(
    ToolExecutor[BatchCleanLeRobotV21Action, BatchCleanLeRobotV21Observation]
):
    def __init__(self, working_dir: str):
        self.working_dir = Path(working_dir).resolve()

    def __call__(
        self,
        action: BatchCleanLeRobotV21Action,
        _conversation=None,
    ) -> BatchCleanLeRobotV21Observation:
        try:
            source = _resolve_workspace_path(self.working_dir, action.path)
            output = _resolve_workspace_output_dir(
                self.working_dir,
                action.output_path,
            )
            result = batch_clean_lerobot_v21_dataset(
                source,
                output,
                task_text=action.task_text,
                confirm_subtasks=action.confirm_subtasks,
                confirm_derived_action=action.confirm_derived_action,
                robot_type=action.robot_type,
                motion_speed_threshold=action.motion_speed_threshold,
                idle_min_duration_s=action.idle_min_duration_s,
                context_s=action.context_s,
                resume=action.resume,
            )
            text = (
                f"Processed {result.discovered_episode_count} episode(s): "
                f"accepted={result.accepted_episode_count}, "
                f"needs_review={result.needs_review_episode_count}, "
                f"rejected={result.rejected_episode_count}, "
                f"failed={result.failed_episode_count}."
            )
            if result.complete:
                text += (
                    f" Merged {result.frame_count} frames and "
                    f"{result.video_count} videos at {result.output_path}."
                )
            elif result.failed_episode_count:
                text += f" Resume from {result.checkpoint_path}."
            else:
                text += (
                    f" No publishable dataset was produced; see {result.report_path}."
                )
            return BatchCleanLeRobotV21Observation.from_text(
                text=text,
                **result.model_dump(),
            )
        except Exception as exc:
            return BatchCleanLeRobotV21Observation.from_text(
                text=str(exc),
                is_error=True,
                output_path=action.output_path,
                errors=[str(exc)],
            )


class ValidateLeRobotV21Executor(
    ToolExecutor[ValidateLeRobotV21Action, ValidateLeRobotV21Observation]
):
    def __init__(self, working_dir: str):
        self.working_dir = Path(working_dir).resolve()

    def __call__(
        self,
        action: ValidateLeRobotV21Action,
        _conversation=None,
    ) -> ValidateLeRobotV21Observation:
        try:
            root = _resolve_workspace_path(self.working_dir, action.path)
            report = validate_lerobot_v21_dataset(root)
            return _validation_observation(report, root)
        except Exception as exc:
            return ValidateLeRobotV21Observation.from_text(
                text=str(exc),
                is_error=True,
                valid=False,
                errors=[str(exc)],
            )


class PublishLeRobotV21Executor(
    ToolExecutor[PublishLeRobotV21Action, PublishLeRobotV21Observation]
):
    def __init__(
        self,
        working_dir: str,
        *,
        storage_base_url: str | None = None,
        headers: dict[str, str] | None = None,
        secret_headers: dict[str, str] | None = None,
        timeout: float = 300.0,
    ) -> None:
        self.working_dir = Path(working_dir).resolve()
        self._uploader = UploadFileToPyromindExecutor(
            storage_base_url=storage_base_url,
            headers=headers,
            secret_headers=secret_headers,
            timeout=timeout,
        )

    def __call__(
        self,
        action: PublishLeRobotV21Action,
        conversation=None,
    ) -> PublishLeRobotV21Observation:
        storage_path = action.storage_path
        try:
            root = _resolve_workspace_path(self.working_dir, action.path)
            if not root.is_dir():
                raise ValueError("path must be a LeRobot v2.1 dataset directory")
            storage_path = _normalize_storage_dataset_path(action.storage_path)
            report = validate_lerobot_v21_dataset(root)
            if not report.valid:
                return PublishLeRobotV21Observation.from_text(
                    text=(
                        "Refusing to publish an invalid LeRobot v2.1 dataset: "
                        + "; ".join(report.errors)
                    ),
                    is_error=True,
                    local_path=str(root),
                    storage_path=storage_path,
                    platform_path=_platform_workspace_path(storage_path),
                    validation_errors=report.errors,
                )
            files = _lerobot_v21_publish_files(root)
        except Exception as exc:
            return PublishLeRobotV21Observation.from_text(
                text=str(exc),
                is_error=True,
                local_path=action.path,
                storage_path=storage_path,
                platform_path=_platform_workspace_path(storage_path),
                validation_errors=[str(exc)],
            )

        total_bytes = sum(path.stat().st_size for path in files)
        uploaded_paths: list[str] = []
        for source in files:
            relative = source.relative_to(root)
            workspace_relative = source.relative_to(self.working_dir)
            target_dir = _storage_parent_path(storage_path, relative)
            uploaded = self._uploader(
                UploadFileToPyromindAction(
                    file_path=workspace_relative.as_posix(),
                    target_dir=target_dir,
                ),
                conversation,
            )
            if uploaded.is_error or uploaded.storage_path is None:
                return PublishLeRobotV21Observation.from_text(
                    text=(
                        f"LeRobot v2.1 publish stopped at {relative.as_posix()}: "
                        f"{uploaded.text}"
                    ),
                    is_error=True,
                    local_path=str(root),
                    storage_path=storage_path,
                    platform_path=_platform_workspace_path(storage_path),
                    file_count=len(files),
                    uploaded_file_count=len(uploaded_paths),
                    total_bytes=total_bytes,
                    uploaded_paths=uploaded_paths,
                    failed_file=relative.as_posix(),
                )
            uploaded_paths.append(uploaded.storage_path)

        return PublishLeRobotV21Observation.from_text(
            text=(
                f"Published {len(files)} validated LeRobot v2.1 files "
                f"({total_bytes} bytes) to Pyromind Storage at {storage_path}."
            ),
            local_path=str(root),
            storage_path=storage_path,
            platform_path=_platform_workspace_path(storage_path),
            complete=True,
            file_count=len(files),
            uploaded_file_count=len(uploaded_paths),
            total_bytes=total_bytes,
            uploaded_paths=uploaded_paths,
        )


class InspectEmbodiedDatasetTool(
    ToolDefinition[
        InspectEmbodiedDatasetAction,
        InspectEmbodiedDatasetObservation,
    ]
):
    @classmethod
    def create(
        cls,
        conv_state: ConversationState,
    ) -> Sequence[InspectEmbodiedDatasetTool]:
        return [
            cls(
                description=(
                    "Inspect a self-collected or online embodied dataset, detect its "
                    "source format, and return at most three episode summaries. The "
                    "tool never modifies source data."
                ),
                action_type=InspectEmbodiedDatasetAction,
                observation_type=InspectEmbodiedDatasetObservation,
                annotations=ToolAnnotations(
                    title="inspect embodied dataset",
                    readOnlyHint=True,
                    destructiveHint=False,
                    idempotentHint=True,
                    openWorldHint=False,
                ),
                executor=InspectEmbodiedDatasetExecutor(
                    conv_state.workspace.working_dir
                ),
            )
        ]


class BuildEmbodiedEpisodePlanTool(
    ToolDefinition[
        BuildEmbodiedEpisodePlanAction,
        BuildEmbodiedEpisodePlanObservation,
    ]
):
    @classmethod
    def create(
        cls,
        conv_state: ConversationState,
    ) -> Sequence[BuildEmbodiedEpisodePlanTool]:
        return [
            cls(
                description=(
                    "Build and validate a reversible EpisodePlan for one embodied "
                    "episode. It proposes idle ranges and source-to-clean frame "
                    "mapping, writes a JSON audit plan plus a human-readable Markdown "
                    "summary, and never edits raw data."
                ),
                action_type=BuildEmbodiedEpisodePlanAction,
                observation_type=BuildEmbodiedEpisodePlanObservation,
                annotations=ToolAnnotations(
                    title="build embodied episode plan",
                    readOnlyHint=False,
                    destructiveHint=False,
                    idempotentHint=True,
                    openWorldHint=False,
                ),
                executor=BuildEmbodiedEpisodePlanExecutor(
                    conv_state.workspace.working_dir
                ),
            )
        ]


class MaterializeLeRobotV21Tool(
    ToolDefinition[MaterializeLeRobotV21Action, MaterializeLeRobotV21Observation]
):
    @classmethod
    def create(
        cls,
        conv_state: ConversationState,
    ) -> Sequence[MaterializeLeRobotV21Tool]:
        return [
            cls(
                description=(
                    "Materialize one reviewed self-collected episode as a complete "
                    "LeRobotDataset v2.1 directory. It writes per-episode Parquet "
                    "and MP4 files plus info, episodes, episode stats, and tasks "
                    "metadata, then validates the result. Actions use the confirmed "
                    "next-clean-state convention."
                ),
                action_type=MaterializeLeRobotV21Action,
                observation_type=MaterializeLeRobotV21Observation,
                annotations=ToolAnnotations(
                    title="materialize LeRobot v2.1 dataset",
                    readOnlyHint=False,
                    destructiveHint=False,
                    idempotentHint=False,
                    openWorldHint=False,
                ),
                executor=MaterializeLeRobotV21Executor(
                    conv_state.workspace.working_dir
                ),
            )
        ]


class FinalizeEmbodiedEpisodePlanTool(
    ToolDefinition[
        FinalizeEmbodiedEpisodePlanAction,
        BuildEmbodiedEpisodePlanObservation,
    ]
):
    @classmethod
    def create(
        cls,
        conv_state: ConversationState,
    ) -> Sequence[FinalizeEmbodiedEpisodePlanTool]:
        return [
            cls(
                description=(
                    "Rebuild source-derived fields for a candidate embodied episode "
                    "plan, apply confirmed task and subtask review decisions, rerun "
                    "all quality gates, and write an accepted audit plan only when "
                    "the source remains valid."
                ),
                action_type=FinalizeEmbodiedEpisodePlanAction,
                observation_type=BuildEmbodiedEpisodePlanObservation,
                annotations=ToolAnnotations(
                    title="finalize embodied episode plan",
                    readOnlyHint=False,
                    destructiveHint=False,
                    idempotentHint=True,
                    openWorldHint=False,
                ),
                executor=FinalizeEmbodiedEpisodePlanExecutor(
                    conv_state.workspace.working_dir
                ),
            )
        ]


class MergeLeRobotV21Tool(
    ToolDefinition[MergeLeRobotV21Action, MaterializeLeRobotV21Observation]
):
    @classmethod
    def create(
        cls,
        conv_state: ConversationState,
    ) -> Sequence[MergeLeRobotV21Tool]:
        return [
            cls(
                description=(
                    "Merge validated single-episode LeRobotDataset v2.1 "
                    "directories into one reference-compatible dataset. Reindex "
                    "episodes and frames deterministically and apply one confirmed "
                    "task shared by the complete dataset."
                ),
                action_type=MergeLeRobotV21Action,
                observation_type=MaterializeLeRobotV21Observation,
                annotations=ToolAnnotations(
                    title="merge LeRobot v2.1 episodes",
                    readOnlyHint=False,
                    destructiveHint=False,
                    idempotentHint=True,
                    openWorldHint=False,
                ),
                executor=MergeLeRobotV21Executor(conv_state.workspace.working_dir),
            )
        ]


class BatchCleanLeRobotV21Tool(
    ToolDefinition[
        BatchCleanLeRobotV21Action,
        BatchCleanLeRobotV21Observation,
    ]
):
    @classmethod
    def create(
        cls,
        conv_state: ConversationState,
    ) -> Sequence[BatchCleanLeRobotV21Tool]:
        return [
            cls(
                description=(
                    "Clean every self-collected S2 episode in one deterministic, "
                    "resumable call. Build and finalize reversible plans, isolate "
                    "quality rejections, materialize accepted episodes, checkpoint "
                    "after each episode, and merge them with one confirmed task. "
                    "Detailed episode results are written to a report; the tool "
                    "returns compact batch counts."
                ),
                action_type=BatchCleanLeRobotV21Action,
                observation_type=BatchCleanLeRobotV21Observation,
                annotations=ToolAnnotations(
                    title="batch clean LeRobot v2.1 dataset",
                    readOnlyHint=False,
                    destructiveHint=False,
                    idempotentHint=True,
                    openWorldHint=False,
                ),
                executor=BatchCleanLeRobotV21Executor(conv_state.workspace.working_dir),
            )
        ]


class ValidateLeRobotV21Tool(
    ToolDefinition[ValidateLeRobotV21Action, ValidateLeRobotV21Observation]
):
    @classmethod
    def create(
        cls,
        conv_state: ConversationState,
    ) -> Sequence[ValidateLeRobotV21Tool]:
        return [
            cls(
                description=(
                    "Validate a LeRobotDataset v2.1 directory, including required "
                    "metadata, Parquet columns and counts, video paths, contiguous "
                    "indexes, and next-state action semantics."
                ),
                action_type=ValidateLeRobotV21Action,
                observation_type=ValidateLeRobotV21Observation,
                annotations=ToolAnnotations(
                    title="validate LeRobot v2.1 dataset",
                    readOnlyHint=True,
                    destructiveHint=False,
                    idempotentHint=True,
                    openWorldHint=False,
                ),
                executor=ValidateLeRobotV21Executor(conv_state.workspace.working_dir),
            )
        ]


class PublishLeRobotV21Tool(
    ToolDefinition[PublishLeRobotV21Action, PublishLeRobotV21Observation]
):
    @classmethod
    def create(
        cls,
        conv_state: ConversationState,
        **params: Any,
    ) -> Sequence[PublishLeRobotV21Tool]:
        storage_base_url = params.pop("storage_base_url", None)
        headers = _optional_string_mapping(params.pop("headers", None), "headers")
        secret_headers = _optional_string_mapping(
            params.pop("secret_headers", None),
            "secret_headers",
        )
        timeout = float(params.pop("timeout", 300.0))
        if timeout <= 0:
            raise ValueError("timeout must be greater than 0")
        if params:
            names = ", ".join(sorted(params))
            raise ValueError(f"PublishLeRobotV21Tool got unknown params: {names}")
        return [
            cls(
                description=(
                    "Publish a locally validated LeRobotDataset v2.1 directory to "
                    "Pyromind Storage while preserving its meta, data, and videos "
                    "layout. Only standard v2.1 files are uploaded; info.json is "
                    "uploaded last. A partial upload is reported as incomplete."
                ),
                action_type=PublishLeRobotV21Action,
                observation_type=PublishLeRobotV21Observation,
                annotations=ToolAnnotations(
                    title="publish LeRobot v2.1 dataset",
                    readOnlyHint=False,
                    destructiveHint=False,
                    idempotentHint=True,
                    openWorldHint=True,
                ),
                executor=PublishLeRobotV21Executor(
                    conv_state.workspace.working_dir,
                    storage_base_url=(
                        str(storage_base_url) if storage_base_url is not None else None
                    ),
                    headers=headers,
                    secret_headers=secret_headers,
                    timeout=timeout,
                ),
            )
        ]


def _resolve_workspace_path(working_dir: Path, value: str) -> Path:
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = working_dir / candidate
    resolved = candidate.resolve()
    if not resolved.is_relative_to(working_dir):
        raise ValueError(f"path is outside the conversation workspace: {value}")
    if not resolved.exists():
        raise ValueError(f"path does not exist: {value}")
    return resolved


def _normalize_storage_dataset_path(value: str) -> str:
    normalized = value.strip().replace("\\", "/")
    if not normalized:
        raise ValueError("storage_path must name a dataset directory")
    if any(part == ".." for part in normalized.split("/")):
        raise ValueError("storage_path must not contain '..'")
    parts = [part for part in normalized.split("/") if part not in {"", "."}]
    if parts and parts[0] == "workspace":
        parts = parts[1:]
    if not parts:
        raise ValueError("storage_path must not be the Storage root")
    return "/" + "/".join(parts)


def _platform_workspace_path(storage_path: str) -> str:
    if storage_path == "/workspace" or storage_path.startswith("/workspace/"):
        return storage_path
    if not storage_path.startswith("/"):
        return storage_path
    return f"/workspace{storage_path}"


def _lerobot_v21_publish_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError(
                "LeRobot v2.1 publish does not allow symbolic links: "
                f"{path.relative_to(root)}"
            )
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if not _is_lerobot_v21_output_file(relative):
            raise ValueError(
                f"unexpected file in LeRobot v2.1 output: {relative.as_posix()}"
            )
        files.append(path)
    info_path = root / "meta" / "info.json"
    return sorted(
        files,
        key=lambda path: (
            path == info_path,
            path.relative_to(root).as_posix(),
        ),
    )


def _is_lerobot_v21_output_file(relative: Path) -> bool:
    parts = relative.parts
    if len(parts) == 2 and parts[0] == "meta":
        return parts[1] in {
            "info.json",
            "episodes.jsonl",
            "episodes_stats.jsonl",
            "tasks.jsonl",
        }
    if len(parts) == 3 and parts[0] == "data":
        return parts[1].startswith("chunk-") and relative.suffix == ".parquet"
    if len(parts) == 4 and parts[0] == "videos":
        return parts[1].startswith("chunk-") and relative.suffix == ".mp4"
    return False


def _storage_parent_path(storage_root: str, relative: Path) -> str:
    parent = PurePosixPath(storage_root)
    if relative.parent != Path("."):
        parent /= PurePosixPath(relative.parent.as_posix())
    return str(parent)


def _optional_string_mapping(value: Any, label: str) -> dict[str, str] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a string mapping")
    return {str(key): str(item) for key, item in value.items()}


def _resolve_output_path(
    working_dir: Path,
    value: str | None,
    episode_id: str,
) -> Path:
    if value is None:
        candidate = (
            working_dir
            / ".agent_tmp"
            / "embodied-data-cleaning"
            / episode_id
            / "episode_plan.json"
        )
    else:
        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = working_dir / candidate
    resolved = candidate.resolve()
    if not resolved.is_relative_to(working_dir):
        raise ValueError("output_path must stay inside the conversation workspace")
    if resolved.suffix.lower() != ".json":
        raise ValueError("output_path must use a .json extension")
    return resolved


def _resolve_workspace_output_dir(working_dir: Path, value: str) -> Path:
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = working_dir / candidate
    resolved = candidate.resolve()
    if not resolved.is_relative_to(working_dir):
        raise ValueError("output_path must stay inside the conversation workspace")
    return resolved


def _resolve_finalized_output_path(
    working_dir: Path,
    candidate_path: Path,
    value: str | None,
) -> Path:
    candidate = (
        Path(value)
        if value is not None
        else candidate_path.with_name(f"{candidate_path.stem}.accepted.json")
    )
    if not candidate.is_absolute():
        candidate = working_dir / candidate
    resolved = candidate.resolve()
    if not resolved.is_relative_to(working_dir):
        raise ValueError("output_path must stay inside the conversation workspace")
    if resolved.suffix.lower() != ".json":
        raise ValueError("output_path must use a .json extension")
    return resolved


def _validation_observation(
    report: LeRobotV21ValidationReport,
    root: Path,
) -> ValidateLeRobotV21Observation:
    detail = ""
    if report.errors:
        detail = " Errors: " + "; ".join(report.errors)
    return ValidateLeRobotV21Observation.from_text(
        text=(
            f"LeRobot v2.1 dataset at {root} is "
            f"{'valid' if report.valid else 'invalid'}.{detail}"
        ),
        is_error=not report.valid,
        valid=report.valid,
        structurally_valid=report.structurally_valid,
        reference_profile_valid=report.reference_profile_valid,
        episode_count=report.episode_count,
        frame_count=report.frame_count,
        video_count=report.video_count,
        errors=report.errors,
    )


register_tool(InspectEmbodiedDatasetTool.name, InspectEmbodiedDatasetTool)
register_tool(BuildEmbodiedEpisodePlanTool.name, BuildEmbodiedEpisodePlanTool)
register_tool(FinalizeEmbodiedEpisodePlanTool.name, FinalizeEmbodiedEpisodePlanTool)
register_tool(MaterializeLeRobotV21Tool.name, MaterializeLeRobotV21Tool)
register_tool(BatchCleanLeRobotV21Tool.name, BatchCleanLeRobotV21Tool)
register_tool(MergeLeRobotV21Tool.name, MergeLeRobotV21Tool)
register_tool(ValidateLeRobotV21Tool.name, ValidateLeRobotV21Tool)
register_tool(PublishLeRobotV21Tool.name, PublishLeRobotV21Tool)
