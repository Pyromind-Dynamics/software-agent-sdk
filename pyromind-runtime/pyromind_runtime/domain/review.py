"""Independent, harness-neutral review and distribution artifact contracts."""

from typing import Literal

from pydantic import Field

from pyromind_runtime.domain.base import ContractModel
from pyromind_runtime.domain.content import JsonObject


class DatasetProfile(ContractModel):
    schema_version: Literal[1] = 1
    status: Literal["ready", "incomplete"]
    datasets: JsonObject
    distribution: JsonObject
    extraction_sha256: str
    label_mapping: JsonObject = Field(default_factory=dict)
    issues: list[str] = Field(default_factory=list)


class ReviewTask(ContractModel):
    schema_version: Literal[1] = 1
    review_id: str
    source_kind: Literal["scoring", "import"]
    source_path: str
    source_sha256: str
    scoring_run_id: str | None = None
    scoring_spec_sha256: str
    profile_sha256: str
    case_count: int = Field(ge=0)
    status: Literal["ready"]
    # Cases and human saves live in immutable Storage objects. This object is
    # the task manifest; its owning PipelineRun supplies conversation isolation.
