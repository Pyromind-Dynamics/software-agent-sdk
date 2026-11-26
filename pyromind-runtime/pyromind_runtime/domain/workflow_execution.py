from __future__ import annotations

from typing import Literal

from pydantic import Field

from pyromind_runtime.domain.base import ContractModel
from pyromind_runtime.domain.content import JsonObject
from pyromind_runtime.domain.task_correlation import TaskCorrelation


DATASET_COMMAND_TEMPLATE_MAX_LENGTH = 20_000

type BatchLabelSource = Literal["dataset", "collection"]


class DatasetRecipeAsset(ContractModel):
    role: Literal["prepare", "pipeline", "export", "support_file", "verifier"]
    path: str
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")


class DatasetRecipeReceipt(ContractModel):
    schema_version: Literal[1] = 1
    source_path: str
    output_directory: str
    command_template: str = Field(
        min_length=1, max_length=DATASET_COMMAND_TEMPLATE_MAX_LENGTH
    )
    assets: tuple[DatasetRecipeAsset, ...]
    batch_label_source: BatchLabelSource | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    fixed_validation_path: str | None = None
    replay_of_task_id: str | None = Field(default=None, pattern=r"^[0-9]{1,64}$")
    uses_labeling_gateway: bool


class WorkflowExecutionRequest(ContractModel):
    request_id: str = Field(min_length=1)
    conversation_id: str = Field(min_length=1)
    workflow_path: str = Field(min_length=1)
    fingerprint: str = Field(min_length=1)
    workflow: JsonObject
    dataset_recipe: DatasetRecipeReceipt | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    summary: JsonObject
    status: Literal["submitting", "submitted", "uncertain"] = "submitting"
    correlation: TaskCorrelation | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    continue_after_completion: bool = False
    task_id: str | None = None
    error: str | None = None
    revision: int = Field(default=1, ge=1)
