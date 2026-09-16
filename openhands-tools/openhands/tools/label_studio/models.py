"""Pydantic models for Label Studio project state and manifest tracking."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ManifestBatch(BaseModel):
    """One batch file inside a task manifest."""

    path: str = Field(description="Batch filename, e.g. tasks-00001.json.")
    task_count: int = Field(description="Number of tasks in this batch.")
    sha256: str = Field(description="SHA-256 hash of the batch file content.")
    index: int = Field(description="Zero-based batch index.")


class ManifestData(BaseModel):
    """Index file describing all batches for one project import."""

    project_ref: str = Field(description="UUID that identifies this project import.")
    dataset_path: str = Field(description="User storage path of the source dataset.")
    converter: str = Field(description="Converter adapter name, e.g. avi_train.")
    converter_version: int = Field(default=1)
    config_hash: str = Field(description="SHA-256 hash of label_config.xml.")
    total_tasks: int = Field(description="Total number of tasks across all batches.")
    batches: list[ManifestBatch] = Field(default_factory=list)
    unmapped_quality: list[str] = Field(
        default_factory=list,
        description=(
            "Quality values that matched no known synonym and were written to "
            "predictions verbatim. Non-empty means those pre-annotations may not "
            "render, because the value is absent from the config's <Choice> list."
        ),
    )


class ProjectState(BaseModel):
    """Persisted project state for idempotent retry and status tracking."""

    project_ref: str
    project_id: int
    dataset_path: str
    adapter: str
    config_version: int = 1
    status: Literal["CREATED", "IMPORTING", "READY", "ERROR"] = "CREATED"
    next_batch: int = 0
    imported_count: int = 0
    total_tasks: int = 0
    total_batches: int = 0
    config_hash: str = ""
    last_error: str | None = None
    idempotency_key: str | None = None
    created_at: str = ""
    media_expires_at: str | None = None
