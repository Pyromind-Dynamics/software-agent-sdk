from __future__ import annotations

from typing import Protocol

from pyromind_runtime.domain.commands import CommandReceipt, ProductCommand
from pyromind_runtime.domain.events import ProductEvent, WorkflowRunState
from pyromind_runtime.domain.pipeline import PipelineRun
from pyromind_runtime.domain.snapshot import ConversationSnapshot


class ProductStore(Protocol):
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
