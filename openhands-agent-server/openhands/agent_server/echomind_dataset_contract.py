from __future__ import annotations

from pathlib import PurePosixPath
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


DatasetOutputSchema = Literal[
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


class DatasetSubmissionSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    pipeline_path: str = Field(min_length=1, max_length=4096)
    input_path: str = Field(min_length=1, max_length=4096)
    output_schema: DatasetOutputSchema
    model_profile: Literal["none", "text", "vision"]
    convert_format: Literal["messages", "preference", "none"] = "none"

    @field_validator("pipeline_path")
    @classmethod
    def require_workspace_python_file(cls, value: str) -> str:
        path = PurePosixPath(value)
        if (
            path.is_absolute()
            or ".." in path.parts
            or "\\" in value
            or path.suffix != ".py"
        ):
            raise ValueError("pipeline_path must be a workspace-relative Python file")
        return path.as_posix()

    @field_validator("input_path")
    @classmethod
    def require_storage_path(cls, value: str) -> str:
        path = PurePosixPath(value)
        if (
            not path.is_absolute()
            or ".." in path.parts
            or "\\" in value
            or any(ord(character) < 32 for character in value)
        ):
            raise ValueError("input_path must be an absolute Storage path")
        return path.as_posix()


def require_dataset_submission_spec(
    details: dict[str, Any],
    *,
    task_run_id: str,
) -> DatasetSubmissionSpec:
    raw = details.get("submission_spec")
    if not isinstance(raw, dict):
        raise ValueError(
            "waiting_user requires details.submission_spec for a Dataset sample"
        )
    spec = DatasetSubmissionSpec.model_validate(raw)
    expected_root = PurePosixPath("public_data/experiment-runs") / task_run_id
    pipeline_path = PurePosixPath(spec.pipeline_path)
    if pipeline_path == expected_root or not pipeline_path.is_relative_to(
        expected_root
    ):
        raise ValueError(
            f"submission_spec.pipeline_path must stay under {expected_root.as_posix()}/"
        )
    return spec
