"""Validated plain Python inputs shared with platform business helpers."""

from __future__ import annotations

import uuid
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


BatchLabelSource = Literal["dataset", "collection"]

OutputSchema = Literal[
    "text",
    "dpo",
    "vision",
    "structured",
    "multiturn",
    "function_call",
    "quality_evaluation",
    "text2sql",
    "artifacts",
]


class ReuseAssessment(BaseModel):
    """Agent-authored evidence that committed output remains reusable."""

    decision: Literal["compatible_resume"] = "compatible_resume"
    changed_dimensions: list[
        Literal["pipeline", "prompt", "model", "schema", "runtime"]
    ] = Field(default_factory=list)
    change_summary: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    verification_samples: list[str] = Field(min_length=1)
    verification_result: Literal["passed"]


class PipelineOptions(BaseModel):
    """Submit a DataFlow-compatible Python pipeline to Pyromind Studio."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    script_path: str | None = Field(
        default=None,
        description=(
            "Workspace-relative pipeline script for a new run or compatible resume,"
            " e.g. 'public_data/data-preparation/pipeline.py'."
            " A strict resume may omit it and reuse the prior frozen script."
        ),
    )
    input_path: str = Field(
        description=(
            "Source data path in user Storage, e.g. '/workspace/datasets/data.jsonl'."
        )
    )
    prepare_script_path: str | None = Field(
        default=None,
        description=(
            "Optional Python adapter: SOURCE_BATCH OUTPUT_MANIFEST. "
            "Use with sft_export_script_path."
        ),
    )
    sft_export_script_path: str | None = Field(
        default=None,
        description=(
            "Python exporter: PROCESSED_JSONL OUTPUT_DIR [FIXED_VALIDATION]. "
            "Writes train_messages.jsonl and val_messages.jsonl for Train."
        ),
    )
    batch_label_source: BatchLabelSource | None = Field(
        default=None,
        description=(
            "Confirmed source of batch supervision: dataset if this step labels "
            "raw samples; collection if the sealed inputs already carry the labels."
        ),
    )
    fixed_validation_path: str | None = Field(
        default=None,
        description="Optional fixed training-validation JSONL path in Storage.",
    )
    replay_of_task_id: str | None = Field(
        default=None,
        pattern=r"^[0-9]{1,64}$",
        description=(
            "Original successful Dataset task when verifying this recipe "
            "on another small input."
        ),
    )
    support_file_path: str | None = Field(
        default=None,
        description=(
            "Optional workspace JSON contract staged beside the frozen script and "
            "passed as its third positional argument, for example taxonomy.json, "
            "analysis_spec.json, or augmentation_plan.json."
        ),
    )
    mode: Literal["full", "resume"] = Field(default="full")
    resume_run_id: uuid.UUID | None = Field(
        default=None,
        description="Existing run UUID. Required when mode='resume'.",
    )
    reuse_assessment: ReuseAssessment | None = Field(
        default=None,
        description=(
            "Required when a resume changes pipeline, prompt, model, schema, or "
            "runtime but the agent judges committed records reusable."
        ),
    )
    prompt_fingerprint: str | None = Field(
        default=None,
        description=(
            "Optional stable hash/version for prompts loaded outside the script."
        ),
    )
    model_profile: Literal["none", "text", "vision"] | None = Field(
        default=None,
        description=(
            "Use none for pure Python/AVI work without model credentials, text "
            "for the conversation model, or vision for managed image work. "
            "Resume inherits the prior profile when omitted."
        ),
    )
    output_schema: OutputSchema | None = Field(
        default=None,
        description=(
            "Canonical output schema: text, dpo, vision, structured, multiturn, "
            "function_call, "
            "quality_evaluation, text2sql, or artifacts. New standard runs should "
            "always set this; None is retained only for legacy pipelines. "
            "Use structured for direct image annotations, vision for training "
            "messages; "
            "match ImagePipelineConfig.output_format."
        ),
    )
    convert_format: Literal["messages", "preference", "none"] = Field(
        default="none",
        description=(
            "Optional format conversion after pipeline completes."
            " 'none' skips conversion."
        ),
    )
    cpu: int = Field(
        default=4,
        ge=1,
        le=64,
        description=("CPU cores for the pipeline pod (1-64, default 4)."),
    )
    memory: int = Field(
        default=32,
        ge=1,
        le=256,
        description=(
            "Memory for the pipeline pod in GiB (1-256, default 32). "
            "32 is enough for typical text pipelines; use 64 for image/"
            "multimodal data. Memory is per-pod, not per-dataset, so it does "
            "not scale with dataset size."
        ),
    )

    @model_validator(mode="after")
    def execution_options(self) -> PipelineOptions:
        if (
            self.prepare_script_path
            or self.batch_label_source
            or self.fixed_validation_path
            or self.replay_of_task_id
        ) and not self.sft_export_script_path:
            raise ValueError("Reusable Dataset steps require sft_export_script_path")
        if self.sft_export_script_path and self.mode == "resume":
            raise ValueError(
                "A Dataset recipe replay uses full mode and a fresh output directory"
            )
        if self.mode == "full" and self.script_path is None:
            raise ValueError(
                "df_submit_pipeline full requires script_path; "
                "use df_submit_evaluation for model evaluation"
            )
        if self.mode == "resume" and self.resume_run_id is None:
            raise ValueError("resume requires resume_run_id")
        if self.mode == "full" and self.resume_run_id is not None:
            raise ValueError("resume_run_id is only valid when mode='resume'")
        return self


class CleaningOptions(BaseModel):
    """Submit a dataset cleaning script as a Studio workflow."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    input_path: str = Field(
        description="Source dataset path under /workspace in the user's Storage.",
    )
    script_path: str | None = Field(
        default=None,
        description=(
            "Uploaded Python cleaning script path for a new run. Resume uses the "
            "frozen script and may omit this field."
        ),
    )
    limit: int | None = Field(
        default=None,
        ge=1,
        description="Optional maximum number of source records for this run.",
    )
    resume_run_id: uuid.UUID | None = Field(
        default=None,
        description=("Existing cleaning run UUID to resume. Omit to create a new run."),
    )
    cpu: int = Field(default=4, ge=1, le=64)
    memory: int = Field(default=32, ge=1, le=256)


class RenderOptions(BaseModel):
    """Submit a template-driven manifest render to Pyromind Studio."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    template_path: str = Field(
        description=(
            "Local render template JSON (field mapping + shard_size). Pass "
            "the same workspace-relative path you gave file_editor/apply_patch "
            "when writing the template (e.g. 'public_data/render_template.json'); "
            "absolute paths work too. The tool stages it into the task and the "
            "node renders against it."
        )
    )
    data_source: str = Field(
        description=(
            "User Storage path of the source data (e.g. "
            "'/workspace/datasets/tmax/data/train-*.parquet'). The node reads it from "
            "its Storage mount."
        )
    )
    output_root: str | None = Field(
        default=None,
        description=(
            "User Storage root for batch-XXX/ shards and shards.json, e.g. "
            "'/workspace/datasets/rendered'. Defaults to the conversation's "
            "edp_render directory."
        ),
    )
    shard_size: int = Field(
        default=500, ge=1, description="Records per shard manifest."
    )
    limit: int | None = Field(
        default=None, ge=1, description="Optional caps on rendered records."
    )
    cpu: int = Field(default=4, ge=1, le=64)
    memory: int = Field(default=16, ge=1, le=256)


class AggregateOptions(BaseModel):
    """Submit the merge + training-format conversion task to Studio."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    run_dirs: list[str] = Field(
        min_length=1,
        description=(
            "Per-shard output_dir values returned by edp_submit "
            "('/workspace/...'). Each must contain "
            "run/verdicts.jsonl next to its shard manifest."
        ),
    )
    out_dir: str = Field(
        description=(
            "User Storage output directory for the merged dataset + training "
            "files, e.g. '/workspace/datasets/merged'. "
            "It contains verdicts.jsonl / slime.jsonl / sft.jsonl / report.json. "
            "Reuse it to resume: already-aggregated task_ids are skipped."
        ),
    )
    protocol: str = Field(
        default="tmax",
        description="Protocol tag embedded in slime record metadata.",
    )
    min_reward: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description=(
            "Minimum reward for the SFT set (default 1.0: solved "
            "trajectories only; slime keeps every usable record regardless)."
        ),
    )
    system_prompt: str | None = Field(
        default=None,
        description=(
            "Optional system message prepended to every SFT sample "
            "(coding-agent persona)."
        ),
    )
    limit: int | None = Field(
        default=None,
        ge=1,
        description="Optional cap on new records this pass (smoke runs).",
    )
    cpu: int = Field(default=4, ge=1, le=64)
    memory: int = Field(default=16, ge=1, le=256)
