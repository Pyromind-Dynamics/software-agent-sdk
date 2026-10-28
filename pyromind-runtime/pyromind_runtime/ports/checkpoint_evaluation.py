"""Platform effects available to the checkpoint evaluation coordinator."""

from typing import Protocol

from pyromind_runtime.domain.checkpoint_evaluation import (
    CheckpointCandidate,
    CheckpointEvaluationRequest,
    EvaluationExecution,
    TrainingObservation,
)
from pyromind_runtime.domain.content import JsonObject
from pyromind_runtime.domain.context import RequestContext
from pyromind_runtime.domain.pipeline import PipelineRun


class EvaluationCredentialsRequired(PermissionError):
    """The caller must supply fresh ephemeral platform credentials."""


class CheckpointEvaluationPlatform(Protocol):
    async def freeze(
        self,
        request: CheckpointEvaluationRequest,
        output_dir: str,
        context: RequestContext,
    ) -> JsonObject: ...

    async def observe(
        self,
        definition: JsonObject,
        context: RequestContext,
    ) -> TrainingObservation: ...

    async def prepare(
        self,
        parent: PipelineRun,
        child: PipelineRun,
        context: RequestContext,
    ) -> JsonObject:
        """Validate and stage immutable inputs; must not create a platform task."""
        ...

    async def submit(
        self,
        child: PipelineRun,
        workflow: JsonObject,
        context: RequestContext,
    ) -> str:
        """Create at most one task. Exceptions mean acceptance is uncertain."""
        ...

    async def inspect(
        self,
        parent: PipelineRun,
        candidate: CheckpointCandidate,
        context: RequestContext,
    ) -> EvaluationExecution: ...

    async def verify_submission(
        self,
        child: PipelineRun,
        task_id: str,
        context: RequestContext,
    ) -> None:
        """Reject a task unless its workflow identifies this exact child run."""
        ...

    async def stop(self, task_id: str, context: RequestContext) -> None: ...

    async def publish(self, parent: PipelineRun, context: RequestContext) -> None: ...
