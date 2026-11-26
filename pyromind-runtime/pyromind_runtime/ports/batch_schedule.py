from contextlib import AbstractContextManager
from typing import Protocol

from pyromind_runtime.domain.batch_schedule import (
    BatchReceipt,
    BatchRunState,
    BatchScheduleConfig,
    BatchScheduleState,
    ReadyProjectBatch,
)
from pyromind_runtime.domain.content import JsonObject


class BatchScheduleBlocked(RuntimeError):
    """A safe, credential-free explanation that can be shown to the owner."""


class BatchScheduleStore(Protocol):
    def load(self, project_id: str) -> BatchScheduleState | None: ...

    def save(self, state: BatchScheduleState) -> None: ...

    def list_projects(self) -> list[tuple[str, str]]: ...

    def lease(self, project_id: str) -> AbstractContextManager[None]: ...


class BatchScheduleRuntime(Protocol):
    @property
    def tracer_resource_id(self) -> str: ...

    @property
    def release_id(self) -> str: ...

    def preflight(self, config: BatchScheduleConfig) -> None:
        """Require published resources, durable Storage auth, GT and batch control."""
        ...

    def list_ready(self) -> list[ReadyProjectBatch]: ...

    def claim(self, receipt: BatchReceipt) -> None:
        """Claim the batch; reject a claim already held by another consumer."""
        ...

    def materialize(
        self, receipt: BatchReceipt, config: BatchScheduleConfig
    ) -> JsonObject: ...

    def find_run(self, receipt: BatchReceipt) -> BatchRunState | None: ...

    def start_run(self, receipt: BatchReceipt, inputs: JsonObject) -> BatchRunState:
        """Deduplicate by Project, Tracer resource and batch identity."""
        ...

    def release(self, receipt: BatchReceipt) -> None:
        """Release only the specified batch; retrying must be idempotent."""
        ...
