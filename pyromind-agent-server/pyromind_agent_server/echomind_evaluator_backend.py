from __future__ import annotations

from pathlib import Path

from pyromind_agent_server.echomind_runtime_credentials import RuntimeCredentialStore
from pyromind_agent_server.echomind_train_backend import (
    PlatformTrainWorkflowBackend,
)
from pyromind_runtime.ports.storage import WorkflowStorageResolver


class PlatformEvaluatorWorkflowBackend(PlatformTrainWorkflowBackend):
    def __init__(
        self,
        *,
        recorded_storage: WorkflowStorageResolver | None = None,
        credential_store: RuntimeCredentialStore | None = None,
        projects_root: Path | str | None = None,
    ) -> None:
        super().__init__(
            task_kind="echomind_evaluator",
            credential_store=credential_store,
            recorded_storage=recorded_storage,
            projects_root=projects_root,
        )
