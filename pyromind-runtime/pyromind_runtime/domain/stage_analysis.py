from typing import Annotated, Literal

from pydantic import Field, StringConstraints

from pyromind_runtime.domain.base import ContractModel


MetricUnit = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=32,
        pattern=r"^[^\x00-\x1f\x7f]+$",
    ),
]


class ReportIssue(ContractModel):
    field: str = Field(min_length=1, max_length=240)
    code: Literal[
        "invalid_metric",
        "invalid_count",
        "invalid_file",
        "invalid_check",
        "truncated",
        "unreadable_report",
        "invalid_report",
    ]


class StageAnalysisMetric(ContractModel):
    label: str = Field(min_length=1, max_length=100)
    value: Annotated[float, Field(allow_inf_nan=False, strict=True)]
    unit: MetricUnit = "number"


class StageAnalysisArtifact(ContractModel):
    artifact_id: str
    name: str
    kind: str
    storage_ref: str


class StageAnalysisResult(ContractModel):
    summary: str = Field(min_length=1, max_length=900)
    findings: tuple[Annotated[str, Field(min_length=1, max_length=300)], ...] = Field(
        max_length=8
    )
    metrics: tuple[StageAnalysisMetric, ...] = ()
    artifacts: tuple[StageAnalysisArtifact, ...] = ()


class StageAnalysisJob(ContractModel):
    job_id: str
    project_id: str
    plan_sequence: int
    stage_id: str
    role: str
    run_ids: tuple[str, ...]
    artifact_ids: tuple[str, ...]
    report_ids: tuple[str, ...]
    status: Literal["queued", "ready", "failed"] = "queued"
    report_version_id: str | None = None
    result: StageAnalysisResult | None = None
