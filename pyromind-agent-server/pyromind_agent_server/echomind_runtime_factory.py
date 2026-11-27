from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from pyromind_agent_server.echomind_dataset_backend import (
    PlatformDatasetWorkflowBackend,
)
from pyromind_agent_server.echomind_dataset_runtime import (
    DatasetRuntimeExecutor,
    PlatformDatasetRuntimeExecutor,
)
from pyromind_agent_server.echomind_evaluator_backend import (
    PlatformEvaluatorWorkflowBackend,
)
from pyromind_agent_server.echomind_evaluator_runtime import (
    EvaluatorRuntimeExecutor,
)
from pyromind_agent_server.echomind_projects import ProjectStore
from pyromind_agent_server.echomind_runtime_credentials import RuntimeCredentialStore
from pyromind_agent_server.echomind_runtime_scheduler import RuntimeGraphScheduler
from pyromind_agent_server.echomind_train_backend import (
    PlatformTrainWorkflowBackend,
)
from pyromind_agent_server.echomind_train_runtime import TrainRuntimeExecutor
from pyromind_runtime.domain.context import RequestContext
from pyromind_runtime.ports.storage import WorkflowStorageResolver


logger = logging.getLogger(__name__)


def build_echomind_runtime_scheduler(
    *,
    projects_root: Path | str,
    conversations_dir: Path | str | None = None,
    request_context: RequestContext | None = None,
    recorded_storage: WorkflowStorageResolver | None = None,
    credential_store: RuntimeCredentialStore | None = None,
) -> RuntimeGraphScheduler:
    return RuntimeGraphScheduler(
        projects_root=projects_root,
        executors={
            "dataset_runtime": DatasetRuntimeExecutor(
                workflow_executor=PlatformDatasetRuntimeExecutor(
                    projects_root=projects_root,
                    backend=PlatformDatasetWorkflowBackend(
                        conversations_dir=conversations_dir,
                        credential_store=credential_store,
                        recorded_storage=recorded_storage,
                        projects_root=projects_root,
                    ),
                    request_context=request_context,
                ),
            ),
            "evaluator_runtime": EvaluatorRuntimeExecutor(
                projects_root=str(projects_root),
                backend=PlatformEvaluatorWorkflowBackend(
                    credential_store=credential_store,
                    recorded_storage=recorded_storage,
                    projects_root=projects_root,
                ),
                request_context=request_context,
            ),
            "train_runtime": TrainRuntimeExecutor(
                projects_root=str(projects_root),
                backend=PlatformTrainWorkflowBackend(
                    credential_store=credential_store,
                    recorded_storage=recorded_storage,
                    conversations_dir=conversations_dir,
                    projects_root=projects_root,
                ),
                request_context=request_context,
            ),
        },
    )


class GraphRuntimeCallbackService:
    def __init__(
        self,
        projects_root: Path | str,
        *,
        conversations_dir: Path | str | None = None,
        recorded_storage: WorkflowStorageResolver | None = None,
        credential_store: RuntimeCredentialStore | None = None,
    ) -> None:
        self.projects_root = Path(projects_root)
        self.conversations_dir = conversations_dir
        self.credential_store = credential_store
        self.recorded_storage = recorded_storage

    async def handle(
        self,
        *,
        task_id: str,
        status: str,
        conversation_id: str,
        task_kind: str | None,
    ) -> str | None:
        try:
            return await asyncio.to_thread(
                self._handle, task_id, status, conversation_id, task_kind
            )
        except Exception:
            logger.exception(
                "Published Graph callback could not advance its TaskRun: %s",
                task_id,
            )
            return None

    def _handle(
        self, task_id: str, status: str, conversation_id: str, task_kind: str | None
    ) -> str | None:
        scheduler = build_echomind_runtime_scheduler(
            projects_root=self.projects_root,
            conversations_dir=self.conversations_dir,
            credential_store=self.credential_store,
            recorded_storage=self.recorded_storage,
        )
        run = scheduler.callback_by_provider_task(
            provider_run_id=task_id,
            status=status,
            conversation_id=conversation_id,
            task_kind=task_kind,
        )
        if run is None:
            return None
        project = ProjectStore(self.projects_root).load(
            run.project_id,
            owner_user_id=None,
        )
        return project.project.conversation_id
