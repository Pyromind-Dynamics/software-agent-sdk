from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints


RuntimeResourceId = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9_-]{1,128}$")]


class RuntimeResourceProposalArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    model_path: str | None = Field(
        default=None,
        min_length=1,
        max_length=2000,
        description="Storage path of the model to deploy if creating Inference.",
    )
    model_name: str = Field(
        min_length=1, max_length=128, description="Model identifier clients will use."
    )
    gpu_card: str | None = Field(
        default=None,
        min_length=1,
        max_length=128,
        description="Requested GPU type for new Inference.",
    )
    inference_id: RuntimeResourceId | None = Field(
        default=None, description="Existing Serving Inference ID to bind."
    )
    tracer_id: RuntimeResourceId | None = Field(
        default=None, description="Existing Tracer ID to bind."
    )
    batch_size: int = Field(
        default=20, ge=3, le=10000, description="Samples per training batch."
    )
