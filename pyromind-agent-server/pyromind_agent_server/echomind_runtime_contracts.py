from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pyromind_agent_server.echomind_graph_runs import (
    FailureCategory,
    GraphRun,
    GraphRunOutcome,
    TaskRun,
)
from pyromind_agent_server.echomind_projects import ProjectNode
from pyromind_agent_server.echomind_stage_executors import StageArtifactPayload
from pyromind_agent_server.echomind_stage_runs import StageRun


PreparationStatus = Literal["completed", "waiting_external", "failed"]
SubmissionStatus = Literal["submitted", "failed"]
PollStatus = Literal["running", "completed", "failed"]
CancellationStatus = Literal["canceled", "failed"]


class RuntimeCredentialUnavailableError(RuntimeError):
    pass


class RuntimeSubmissionNotAttemptedError(RuntimeError):
    """The executor failed before it could send a create request."""


def utc_now() -> datetime:
    return datetime.now(UTC)


class RuntimeArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    port: str = Field(min_length=1, max_length=128)
    payload: StageArtifactPayload
    metadata: dict[str, Any] = Field(default_factory=dict)


class ExternalTaskIntent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    provider: str = Field(min_length=1, max_length=128)
    metadata: dict[str, Any] = Field(default_factory=dict)


class StagePreparationResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: PreparationStatus
    message: str = Field(min_length=1, max_length=4000)
    artifacts: list[RuntimeArtifact] = Field(default_factory=list)
    external_task: ExternalTaskIntent | None = None
    failure_category: FailureCategory = "external"
    error: str | None = Field(default=None, max_length=4000)

    @model_validator(mode="after")
    def validate_preparation(self) -> StagePreparationResult:
        if self.status == "waiting_external" and self.external_task is None:
            raise ValueError("waiting_external preparation requires an external task")
        if self.status != "waiting_external" and self.external_task is not None:
            raise ValueError("only waiting_external preparation may include a task")
        if self.status == "failed" and not self.error:
            raise ValueError("failed preparation requires an error")
        return self


class StageSubmissionResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: SubmissionStatus
    message: str = Field(min_length=1, max_length=4000)
    provider_run_id: str | None = Field(default=None, max_length=256)
    failure_category: FailureCategory = "external"

    @model_validator(mode="after")
    def validate_submission(self) -> StageSubmissionResult:
        if self.status == "submitted" and not self.provider_run_id:
            raise ValueError("submitted task requires a provider_run_id")
        if self.status == "failed" and self.provider_run_id:
            raise ValueError("failed submission must not include provider_run_id")
        return self


class StagePollResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: PollStatus
    message: str = Field(min_length=1, max_length=4000)
    artifacts: list[RuntimeArtifact] = Field(default_factory=list)
    failure_category: FailureCategory = "external"

    @model_validator(mode="after")
    def validate_poll(self) -> StagePollResult:
        if self.status != "completed" and self.artifacts:
            raise ValueError("only a completed poll may include artifacts")
        return self


class StageCancellationResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: CancellationStatus
    message: str = Field(min_length=1, max_length=4000)


class RuntimeStageContext(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    project_id: str
    release_id: str
    graph_run_id: str
    stage_run_id: str
    node: ProjectNode
    inputs: dict[str, Any] = Field(default_factory=dict)


class RuntimeNodeExecutor(Protocol):
    def prepare(self, context: RuntimeStageContext) -> StagePreparationResult: ...

    def submit(
        self,
        context: RuntimeStageContext,
        task_run: TaskRun,
    ) -> StageSubmissionResult: ...

    def poll(
        self,
        context: RuntimeStageContext,
        task_run: TaskRun,
    ) -> StagePollResult: ...

    def cancel(
        self,
        context: RuntimeStageContext,
        task_run: TaskRun,
    ) -> StageCancellationResult: ...


class ContractProbeRuntimeExecutor:
    def prepare(self, context: RuntimeStageContext) -> StagePreparationResult:
        artifacts = [
            RuntimeArtifact(
                port=port,
                payload=StageArtifactPayload(
                    kind=output.artifact_type,
                    content={
                        "probe_version": 1,
                        "project_id": context.project_id,
                        "release_id": context.release_id,
                        "graph_run_id": context.graph_run_id,
                        "stage_run_id": context.stage_run_id,
                        "node_id": context.node.id,
                        "node_type": context.node.type,
                        "port": port,
                        "inputs": context.inputs,
                    },
                ),
            )
            for port, output in context.node.outputs.items()
        ]
        return StagePreparationResult(
            status="completed",
            message=f"{context.node.title} contract probe completed.",
            artifacts=artifacts,
        )

    def submit(
        self,
        context: RuntimeStageContext,
        task_run: TaskRun,
    ) -> StageSubmissionResult:
        del context, task_run
        raise RuntimeError("contract probe does not submit external tasks")

    def poll(
        self,
        context: RuntimeStageContext,
        task_run: TaskRun,
    ) -> StagePollResult:
        del context, task_run
        raise RuntimeError("contract probe does not poll external tasks")

    def cancel(
        self,
        context: RuntimeStageContext,
        task_run: TaskRun,
    ) -> StageCancellationResult:
        del task_run
        return StageCancellationResult(
            status="canceled",
            message=f"{context.node.title} canceled.",
        )


class ArtifactReplayRuntimeExecutor:
    def prepare(self, context: RuntimeStageContext) -> StagePreparationResult:
        raw_artifacts = context.node.config.get("replay_artifacts")
        if not isinstance(raw_artifacts, dict):
            return StagePreparationResult(
                status="failed",
                message="Replay artifacts are not configured.",
                error=f"Node {context.node.id} has no replay_artifacts mapping.",
                failure_category="validation",
            )
        artifacts: list[RuntimeArtifact] = []
        for port, output in context.node.outputs.items():
            spec = raw_artifacts.get(port)
            if not isinstance(spec, dict) or not isinstance(spec.get("uri"), str):
                if not output.required:
                    continue
                return StagePreparationResult(
                    status="failed",
                    message="Replay artifact mapping is incomplete.",
                    error=f"Node {context.node.id}.{port} has no replay URI.",
                    failure_category="validation",
                )
            uri = spec["uri"]
            content: dict[str, Any] = {
                "uri": uri,
                "project_id": context.project_id,
                "release_id": context.release_id,
                "graph_run_id": context.graph_run_id,
                "stage_run_id": context.stage_run_id,
                "node_id": context.node.id,
                "port": port,
            }
            content_sha256 = spec.get("content_sha256")
            if isinstance(content_sha256, str):
                content["content_sha256"] = content_sha256
            metadata = spec.get("metadata")
            if isinstance(metadata, dict):
                content["metadata"] = metadata
            artifacts.append(
                RuntimeArtifact(
                    port=port,
                    payload=StageArtifactPayload(
                        kind=output.artifact_type,
                        content=content,
                    ),
                    metadata={"source_uri": uri},
                )
            )
        return StagePreparationResult(
            status="completed",
            message=f"{context.node.title} replay completed.",
            artifacts=artifacts,
        )

    def submit(
        self,
        context: RuntimeStageContext,
        task_run: TaskRun,
    ) -> StageSubmissionResult:
        del context, task_run
        raise RuntimeError("artifact replay does not submit external tasks")

    def poll(
        self,
        context: RuntimeStageContext,
        task_run: TaskRun,
    ) -> StagePollResult:
        del context, task_run
        raise RuntimeError("artifact replay does not poll external tasks")

    def cancel(
        self,
        context: RuntimeStageContext,
        task_run: TaskRun,
    ) -> StageCancellationResult:
        del task_run
        return StageCancellationResult(
            status="canceled",
            message=f"{context.node.title} canceled.",
        )


class RuntimeFinalization(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    outcome: GraphRunOutcome
    promotion_result: dict[str, Any] = Field(default_factory=dict)


class RuntimeFinalizer(Protocol):
    def finalize(
        self,
        *,
        project_id: str,
        graph_run: GraphRun,
        stages: list[StageRun],
    ) -> RuntimeFinalization: ...


class CandidateRecordingFinalizer:
    def finalize(
        self,
        *,
        project_id: str,
        graph_run: GraphRun,
        stages: list[StageRun],
    ) -> RuntimeFinalization:
        return RuntimeFinalization(
            outcome="candidate_recorded",
            promotion_result={
                "decision": "awaiting_user_selection",
                "reason": "manual_deployment",
                "candidate_id": graph_run.run_id,
                "project_id": project_id,
                "round_no": graph_run.round_no,
                "stage_count": len(stages),
            },
        )


class BaselineRetainingFinalizer:
    def finalize(
        self,
        *,
        project_id: str,
        graph_run: GraphRun,
        stages: list[StageRun],
    ) -> RuntimeFinalization:
        return RuntimeFinalization(
            outcome="baseline_retained",
            promotion_result={
                "decision": "baseline_retained",
                "reason": "finalizer_not_configured",
                "project_id": project_id,
                "round_no": graph_run.round_no,
                "stage_count": len(stages),
            },
        )
