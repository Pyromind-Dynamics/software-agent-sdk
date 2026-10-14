from __future__ import annotations

from typing import Protocol

from pyromind_runtime.domain.commands import CommandReceipt, ProductCommand
from pyromind_runtime.domain.evaluation import EvaluationCommitState
from pyromind_runtime.domain.events import ProductEvent, WorkflowRunState
from pyromind_runtime.domain.pipeline import PipelineRun
from pyromind_runtime.domain.snapshot import ConversationSnapshot
from pyromind_runtime.domain.workflow_execution import WorkflowExecutionRequest


class ProductStore(Protocol):
    def load_workflow_execution_requests(
        self,
    ) -> dict[str, WorkflowExecutionRequest]: ...

    def save_workflow_execution_request(
        self, state: WorkflowExecutionRequest, *, expected_revision: int | None
    ) -> None: ...

    def load_pipeline_runs(self) -> dict[str, PipelineRun]: ...

    def save_pipeline_run(
        self, state: PipelineRun, *, expected_revision: int | None
    ) -> None: ...

    def load_workflow_runs(self) -> dict[str, WorkflowRunState]: ...

    def save_workflow_run(self, state: WorkflowRunState) -> None: ...

    def create(self, snapshot: ConversationSnapshot, *, user_id: str) -> None: ...

    def load_snapshot(self) -> ConversationSnapshot: ...

    def append(
        self, event: ProductEvent
    ) -> tuple[ProductEvent, ConversationSnapshot]: ...

    def replay(self, after_seq: int = 0) -> tuple[ProductEvent, ...]: ...

    def claim_command(
        self,
        command: ProductCommand,
    ) -> tuple[CommandReceipt, bool]: ...

    def complete_command(self, receipt: CommandReceipt) -> CommandReceipt: ...


class EvaluationCommitStore(Protocol):
    """Narrow Product Store view holding annotation save commit state.

    Implementations (the conversation Product Store) guarantee the atomic
    single-pointer semantics the save protocol relies on.
    """

    def load_evaluation_commit(self, run_id: str) -> EvaluationCommitState | None: ...

    def save_evaluation_commit(self, state: EvaluationCommitState) -> None: ...
