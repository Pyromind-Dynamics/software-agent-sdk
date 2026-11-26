from typing import Protocol

from pyromind_runtime.domain.batch_schedule import (
    BatchScheduleConfig,
    BatchScheduleState,
)
from pyromind_runtime.domain.content import JsonObject
from pyromind_runtime.domain.context import RequestContext
from pyromind_runtime.domain.echomind_deployments import (
    DeploymentRecord,
    DeploymentRequest,
)
from pyromind_runtime.domain.echomind_model_reports import (
    ModelEvaluationRecord,
    ModelReportCatalog,
)
from pyromind_runtime.domain.project_storage import (
    ProjectStorageBinding,
    StorageProxyConfig,
)


class ProjectFlywheel(Protocol):
    def storage_binding(
        self, project_id: str, context: RequestContext
    ) -> ProjectStorageBinding | None: ...

    def bind_storage(
        self, project_id: str, config: StorageProxyConfig, context: RequestContext
    ) -> ProjectStorageBinding: ...

    def refresh_storage_binding(
        self, project_id: str, context: RequestContext
    ) -> ProjectStorageBinding: ...

    def reports(
        self, project_id: str, context: RequestContext
    ) -> ModelReportCatalog: ...

    def refresh_report(
        self, project_id: str, candidate_id: str, context: RequestContext
    ) -> ModelEvaluationRecord: ...

    def report_document(
        self, project_id: str, candidate_id: str, context: RequestContext
    ) -> str: ...

    def schedule(
        self, project_id: str, context: RequestContext
    ) -> BatchScheduleState: ...

    def configure_schedule(
        self, project_id: str, config: BatchScheduleConfig, context: RequestContext
    ) -> BatchScheduleState: ...

    def deployments(
        self, project_id: str, context: RequestContext
    ) -> list[DeploymentRecord]: ...

    def deploy(
        self, project_id: str, request: DeploymentRequest, context: RequestContext
    ) -> DeploymentRecord: ...

    def advance_deployment(
        self, project_id: str, deployment_id: str, context: RequestContext
    ) -> DeploymentRecord: ...

    def report_cases(
        self,
        project_id: str,
        candidate_id: str,
        context: RequestContext,
        cursor: str | None,
        limit: int,
        query: str,
    ) -> JsonObject: ...

    def report_media(
        self,
        project_id: str,
        candidate_id: str,
        media_id: str,
        context: RequestContext,
    ) -> tuple[bytes, str]: ...

    def poll(self) -> None: ...
