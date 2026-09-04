"""Canonical models for embodied-data inspection and cleaning plans."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _StrEnum(str, Enum):  # noqa: UP042 - sandbox runtime supports Python 3.10
    def __str__(self) -> str:
        return self.value


class SourceType(_StrEnum):
    SELF_COLLECTED = "self_collected"
    LEROBOT_V21 = "lerobot_v21"
    ONLINE_LABELS = "online_labels"


class TimelineBasis(_StrEnum):
    FRAME_INDEX = "frame_index"
    STAMP_NS = "stamp_ns"


class StreamStatus(_StrEnum):
    AVAILABLE = "available"
    NOT_MATERIALIZED = "not_materialized"
    MISSING = "missing"
    INVALID = "invalid"


class QualityStatus(_StrEnum):
    PENDING = "pending"
    ACCEPTED = "accepted"
    NEEDS_REVIEW = "needs_review"
    REJECTED = "rejected"


class ActionProvenanceMode(_StrEnum):
    RECORDED = "recorded"
    DERIVED = "derived"
    MISSING = "missing"


class ActionDerivationMode(_StrEnum):
    NEXT_STATE = "next_state"


class StreamDescriptor(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: str
    path: str
    status: StreamStatus
    frame_count: int | None = Field(default=None, ge=0)
    fps: float | None = Field(default=None, gt=0)
    message: str | None = None


class FeatureSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    unit: str | None = None
    source: str = Field(min_length=1)


class TrainingFeatureSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    observation_state: list[FeatureSpec] = Field(default_factory=list)
    action: list[FeatureSpec] = Field(default_factory=list)


class ActionProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: ActionProvenanceMode
    source_path: str | None = None
    derivation: ActionDerivationMode | None = None
    description: str | None = None
    user_confirmed: bool = False

    @model_validator(mode="after")
    def validate_derived_action(self) -> ActionProvenance:
        if self.mode == ActionProvenanceMode.DERIVED and (
            self.source_path is None
            or self.derivation is None
            or self.description is None
        ):
            raise ValueError(
                "derived action provenance requires source_path, derivation, and "
                "description"
            )
        return self


class TaskSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str
    origin: str = "unspecified"
    user_confirmed: bool = False


class EnvironmentSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    initial_scene: str | None = None


class TimelineSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    basis: TimelineBasis
    frame_count: int | None = Field(default=None, ge=0)
    fps: float | None = Field(default=None, gt=0)
    start_stamp_ns: int | None = None
    end_stamp_ns: int | None = None


class SubtaskSegment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_start_frame: int = Field(ge=0)
    source_end_frame: int = Field(gt=0)
    instruction: str
    event_type: str
    origin: str
    source_start_ns: int | None = None
    source_end_ns: int | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)
    user_confirmed: bool = False

    @model_validator(mode="after")
    def validate_interval(self) -> SubtaskSegment:
        if self.source_end_frame <= self.source_start_frame:
            raise ValueError("segment end frame must be greater than start frame")
        if (
            self.source_start_ns is not None
            and self.source_end_ns is not None
            and self.source_end_ns <= self.source_start_ns
        ):
            raise ValueError(
                "segment end timestamp must be greater than start timestamp"
            )
        return self


class FrameInterval(BaseModel):
    model_config = ConfigDict(extra="forbid")

    start_frame: int = Field(ge=0)
    end_frame: int = Field(gt=0)
    reason: str

    @model_validator(mode="after")
    def validate_interval(self) -> FrameInterval:
        if self.end_frame <= self.start_frame:
            raise ValueError("interval end frame must be greater than start frame")
        return self


class TimelineMapping(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_frame_index: int = Field(ge=0)
    clean_frame_index: int = Field(ge=0)
    source_time_s: float | None = Field(default=None, ge=0)
    clean_time_s: float | None = Field(default=None, ge=0)
    state_before_index: int | None = Field(default=None, ge=0)
    state_after_index: int | None = Field(default=None, ge=0)
    state_interpolation_weight: float | None = Field(default=None, ge=0, le=1)


class QualityReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: QualityStatus = QualityStatus.PENDING
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class EpisodePlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    episode_id: str
    source_type: SourceType
    task: TaskSpec
    environment: EnvironmentSpec = Field(default_factory=EnvironmentSpec)
    streams: dict[str, StreamDescriptor] = Field(default_factory=dict)
    feature_schema: TrainingFeatureSchema = Field(default_factory=TrainingFeatureSchema)
    action_provenance: ActionProvenance | None = None
    source_timeline: TimelineSpec
    segments: list[SubtaskSegment] = Field(default_factory=list)
    drop_intervals: list[FrameInterval] = Field(default_factory=list)
    timeline_mapping: list[TimelineMapping] = Field(default_factory=list)
    quality: QualityReport = Field(default_factory=QualityReport)


class EpisodeSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    episode_id: str
    segment_count: int = Field(ge=0)
    frame_count: int | None = Field(default=None, ge=0)
    quality: QualityReport


class DatasetInspection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_type: SourceType
    source_path: str
    episode_count: int = Field(ge=0)
    sampled_episodes: list[EpisodeSummary] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
