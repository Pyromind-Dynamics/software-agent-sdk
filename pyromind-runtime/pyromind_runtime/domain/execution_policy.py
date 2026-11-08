"""Server-owned task execution capabilities, independent of roles and scenarios."""

from typing import Literal

from pydantic import Field, model_validator

from pyromind_runtime.domain.base import ContractModel
from pyromind_runtime.domain.content import JsonObject


class TaskExecutionPolicy(ContractModel):
    submission_tool: Literal[
        "df_submit_evaluation", "df_submit_pipeline", "submit_workflow"
    ]
    artifact_contract: Literal["evaluation", "data", "model"]
    require_review: bool = False
    sample_limit: int = Field(default=3, ge=1, le=3)


class CheckpointRef(ContractModel):
    run_id: str = Field(min_length=1)
    checkpoint_id: str = Field(min_length=1)


class ExecutionCheckpoint(ContractModel):
    run_id: str = Field(min_length=1)
    summary: str = Field(min_length=1, max_length=16000)
    sample_count: int = Field(ge=1, le=3)
    source_count: int = Field(ge=1)
    retained_count: int = Field(ge=0)
    source_path: str = Field(
        min_length=1,
        description="Exact full source file used by configuration.input_path. "
        "For a directory dataset, use its generated full manifest, not the directory.",
    )
    sample_input: str = Field(min_length=1)
    sample_output: str = Field(min_length=1)
    script_path: str = Field(min_length=1)
    implementation_files: tuple[str, ...] = Field(
        default=(),
        max_length=16,
        exclude_if=lambda value: not value,
        description="Task-local helper scripts used to produce the reviewed output.",
    )
    configuration: JsonObject
    preview: tuple[JsonObject, ...] = Field(min_length=1, max_length=3)
    validation: JsonObject

    @model_validator(mode="after")
    def bounded_document(self):
        if len(self.model_dump_json().encode()) > 65536:
            raise ValueError("checkpoint document exceeds the preview budget")
        return self


class CheckpointView(ContractModel):
    checkpoint_id: str
    status: Literal["pending", "approved", "revise"] = "pending"
    document: ExecutionCheckpoint
    evidence: dict[str, str]
