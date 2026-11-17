from __future__ import annotations

from pyromind_agent_server.echomind_runtime_credentials import RuntimeCredentialStore
from pyromind_agent_server.echomind_train_backend import (
    PlatformTrainWorkflowBackend,
)


class PlatformEvaluatorWorkflowBackend(PlatformTrainWorkflowBackend):
    def __init__(
        self, *, credential_store: RuntimeCredentialStore | None = None
    ) -> None:
        super().__init__(
            task_kind="echomind_evaluator", credential_store=credential_store
        )
