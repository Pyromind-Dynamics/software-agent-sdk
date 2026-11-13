"""Read models for experiment lineage, separate from artifact version identity."""

from typing import Literal

from pydantic import Field

from pyromind_runtime.domain.base import ContractModel
from pyromind_runtime.domain.content import JsonObject
from pyromind_runtime.domain.training_evaluation import StepEvaluation
from pyromind_runtime.domain.versions import ArtifactKind, VersionParent, VersionState


class ExperimentArtifact(ContractModel):
    id: str
    kind: ArtifactKind
    state: VersionState
    storage_ref: str | None = None
    digest: str | None = None
    parents: tuple[VersionParent, ...] = ()
    output_role: str | None = None
    producer_run_id: str | None = None
    prediction: JsonObject | None = None


class ExperimentExecution(ContractModel):
    run_id: str
    task_id: str | None = None
    status: str = "unknown"
    output_dir: str | None = None
    archive_ref: str | None = None


class ExperimentEvaluation(ExperimentExecution):
    operation: str
    training_run_id: str | None = None
    report_path: str | None = None
    prediction_path: str | None = None
    metrics_path: str | None = None
    evaluation_status: str | None = None
    error_summary: str | None = None
    model_path: str | None = None
    dataset_role: str = "unknown"
    steps: tuple[StepEvaluation, ...] = ()
    selection: JsonObject = Field(default_factory=dict)


class ExperimentVersion(ContractModel):
    schema_version: Literal[1] = 1
    experiment_id: str
    version: str
    project_id: str | None = None
    artifact_project_id: str
    conversation_id: str
    iteration_id: str
    created_at: int
    directory_ref: str
    parent_experiment_ids: tuple[str, ...] = ()
    execution: ExperimentExecution
    attempts: tuple[ExperimentExecution, ...] = ()
    inputs: dict[str, ExperimentArtifact] = Field(default_factory=dict)
    outputs: tuple[ExperimentArtifact, ...] = ()
    evaluations: tuple[ExperimentEvaluation, ...] = ()
    issues: tuple[str, ...] = ()


class ExperimentMetric(ContractModel):
    experiment_id: str
    version: str
    evaluation_run_id: str
    step_run_id: str
    step: int
    role: Literal["checkpoint", "baseline"]
    dataset_role: Literal["validation", "test", "train"]
    comparison_group: str
    metric: str
    value: float = Field(allow_inf_nan=False)
    selected: bool = False


class ExperimentCatalog(ContractModel):
    schema_version: Literal[1] = 1
    conversation_id: str
    project_id: str | None = None
    artifact_project_id: str | None = None
    versions: tuple[ExperimentVersion, ...] = ()
    unassigned_evaluations: tuple[ExperimentEvaluation, ...] = ()
