from typing import Literal

from pydantic import Field, FiniteFloat

from pyromind_runtime.domain.base import AwareDatetime, ContractModel


class EvaluationRubric(ContractModel):
    name: str
    criterion: str
    weight: FiniteFloat
    required: bool


class ModelEvaluationRecord(ContractModel):
    candidate_id: str
    project_id: str
    run_id: str
    release_id: str
    release_digest: str
    round_no: int
    execution_mode: Literal["debug", "published"]
    source_batch_id: str | None = None
    model_uri: str
    deployment_model_uri: str | None = None
    training_base_uri: str | None = None
    report_uri: str
    report_artifact_id: str
    evaluation_dataset_uri: str | None = None
    created_at: AwareDatetime
    report_status: Literal["pending", "ready", "invalid", "unavailable"] = "pending"
    evaluation_status: str | None = None
    metrics: dict[str, FiniteFloat] = Field(default_factory=dict)
    dataset_sha256: str | None = None
    evaluation_config_sha256: str | None = None
    evaluation_config_uri: str | None = None
    rubrics: tuple[EvaluationRubric, ...] = ()
    pass_threshold: FiniteFloat | None = None
    comparison_group: str | None = None
    report_html_uri: str | None = None
    issues: tuple[str, ...] = ()


class ModelReportCatalog(ContractModel):
    project_id: str
    candidates: tuple[ModelEvaluationRecord, ...] = ()
