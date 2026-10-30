"""Bind the existing evaluation schemas to the experiment proposal boundary."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field
from pyromind_runtime.domain.content import JsonObject

from openhands.tools.data_preparation.inference import (
    DatasetMapping,
    EvaluationConfig,
    storage_path,
)


class EvaluationConfiguration(BaseModel):
    model_config = ConfigDict(extra="forbid")
    input_path: str
    model_path: str = Field(min_length=1)
    model_version: str = Field(min_length=1)
    dataset_config: DatasetMapping
    evaluation_config: EvaluationConfig
    gpu_product: str = Field(min_length=1)
    gpu_count: int = Field(ge=1)
    cpu: int = Field(default=4, ge=1)
    memory: int = Field(default=32, ge=1)


def validate_evaluation_configuration(value: JsonObject) -> JsonObject:
    config = EvaluationConfiguration.model_validate(value)
    storage_path(config.input_path)
    if config.evaluation_config.model_source.type in {"storage", "checkpoint"}:
        storage_path(config.model_path)
    return config.model_dump(mode="json")
