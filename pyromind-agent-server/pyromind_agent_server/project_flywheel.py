from __future__ import annotations

import logging
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from uuid import UUID

from filelock import FileLock, Timeout
from pydantic import BaseModel, TypeAdapter

from pyromind_agent_server.echomind_acquisition import AcquisitionSimulationStore
from pyromind_agent_server.echomind_batch_archive import archive_tracer_batch
from pyromind_agent_server.echomind_batch_schedule import PlatformBatchScheduleRuntime
from pyromind_agent_server.echomind_deployments import build_manual_deployment_service
from pyromind_agent_server.echomind_graph_runs import GraphRunStore
from pyromind_agent_server.echomind_model_reports import GraphModelReportStore
from pyromind_agent_server.echomind_platform_tracer import is_platform_tracer_script
from pyromind_agent_server.echomind_projects import ProjectStore
from pyromind_agent_server.echomind_publication import PublicationStore
from pyromind_agent_server.echomind_runtime_access import platform_request_token
from pyromind_agent_server.echomind_runtime_credentials import RuntimeCredentialStore
from pyromind_agent_server.echomind_runtime_factory import (
    build_echomind_runtime_scheduler,
)
from pyromind_agent_server.echomind_runtime_slots import RuntimeSlotStore
from pyromind_agent_server.echomind_sandbox_storage import (
    SandboxEvaluationStorage,
    build_bound_evaluation_storage,
    build_project_storage_bindings,
)
from pyromind_agent_server.echomind_storage import (
    project_directory,
    read_model,
    write_model,
)
from pyromind_agent_server.echomind_tracer_admin import load_tracer_admin_key
from pyromind_agent_server.evaluation_storage import build_evaluation_storage
from pyromind_common.sdk.utils.cipher import Cipher
from pyromind_runtime.application.batch_schedules import BatchSchedules
from pyromind_runtime.application.evaluation_cases import EvaluationCaseService
from pyromind_runtime.application.model_reports import ProjectModelReports
from pyromind_runtime.domain.batch_schedule import (
    BatchReceipt,
    BatchScheduleConfig,
    BatchScheduleState,
)
from pyromind_runtime.domain.content import JsonObject
from pyromind_runtime.domain.context import RequestContext
from pyromind_runtime.domain.echomind_deployments import (
    TERMINAL_DEPLOYMENT_STATUSES,
    DeploymentCandidate,
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
from pyromind_runtime.domain.storage_paths import to_storage_api_path
from pyromind_runtime.infrastructure.file_batch_schedule_store import (
    FileBatchScheduleStore,
)
from pyromind_runtime.infrastructure.file_deployment_store import FileDeploymentStore
from pyromind_runtime.ports.storage import (
    EvaluationStorage,
    EvaluationStorageUnavailableError,
    WorkflowStorageResolver,
)


logger = logging.getLogger(__name__)
_REPORT_MAX_BYTES = 8 * 1024 * 1024


class _CachedDocument(BaseModel):
    source_uri: str
    html: str


class ProjectFlywheelServices:
    def __init__(
        self,
        *,
        projects_root: Path,
        conversations_dir: Path,
        credential_store: RuntimeCredentialStore | None,
        cipher: Cipher | None,
        recorded_storage: WorkflowStorageResolver | None = None,
    ) -> None:
        self.root = projects_root
        self.conversations_dir = conversations_dir
        self.credential_store = credential_store
        self.cipher = cipher
        self.recorded_storage = recorded_storage
        self.projects = ProjectStore(projects_root)
        self.report_service = ProjectModelReports(GraphModelReportStore(projects_root))
        self.schedule_store = FileBatchScheduleStore(projects_root)
        self.deployment_store = FileDeploymentStore(projects_root)

    def _context(self, project_id: str, context: RequestContext) -> RequestContext:
        project = self.projects.load(project_id, owner_user_id=context.user_id)
        if context.x_cluster:
            self.projects.require_cluster(project, context.x_cluster)
        return replace(context, x_cluster=project.project.cluster)

    def reports(self, project_id: str, context: RequestContext) -> ModelReportCatalog:
        self._context(project_id, context)
        return self.report_service.catalog(project_id)

    def storage_binding(
        self, project_id: str, context: RequestContext
    ) -> ProjectStorageBinding | None:
        context = self._context(project_id, context)
        return build_project_storage_bindings(
            projects_root=self.root,
            project_id=project_id,
            context=context,
            credential_store=self.credential_store,
        ).get(project_id, context.user_id)

    def bind_storage(
        self, project_id: str, config: StorageProxyConfig, context: RequestContext
    ) -> ProjectStorageBinding:
        context = self._context(project_id, context)
        return build_project_storage_bindings(
            projects_root=self.root,
            project_id=project_id,
            context=context,
            credential_store=self.credential_store,
        ).bind(project_id, context.user_id, config)

    def refresh_storage_binding(
        self, project_id: str, context: RequestContext
    ) -> ProjectStorageBinding:
        context = self._context(project_id, context)
        return build_project_storage_bindings(
            projects_root=self.root,
            project_id=project_id,
            context=context,
            credential_store=self.credential_store,
        ).refresh(project_id, context.user_id)

    def _storage(
        self,
        project_id: str,
        context: RequestContext,
        *,
        max_read_bytes: int = _REPORT_MAX_BYTES,
        write_prefixes: tuple[str, ...] = (),
    ) -> EvaluationStorage:
        if platform_request_token(context):
            return build_evaluation_storage(context, max_read_bytes=max_read_bytes)
        storage = build_bound_evaluation_storage(
            projects_root=self.root,
            project_id=project_id,
            context=context,
            credential_store=self.credential_store,
            max_read_bytes=max_read_bytes,
            write_prefixes=write_prefixes,
        )
        if storage is None:
            raise EvaluationStorageUnavailableError(
                "Bind an existing Storage Sandbox or refresh the owner's login"
            )
        return storage

    def refresh_report(
        self, project_id: str, candidate_id: str, context: RequestContext
    ) -> ModelEvaluationRecord:
        context = self._context(project_id, context)
        storage = self._storage(project_id, context)
        record = self.report_service.refresh(project_id, candidate_id, storage)
        if record.report_html_uri:
            uri = self._html_uri(record)
            payload = storage.read(uri)
            if len(payload) > _REPORT_MAX_BYTES:
                raise ValueError("Report document exceeds 8 MiB")
            write_model(
                self._document_path(project_id, candidate_id),
                _CachedDocument(
                    source_uri=uri,
                    html=payload.decode("utf-8"),
                ),
            )
        return record

    def report_document(
        self, project_id: str, candidate_id: str, context: RequestContext
    ) -> str:
        self._context(project_id, context)
        record = self.report_service.get(project_id, candidate_id)
        uri = self._html_uri(record)
        try:
            document = read_model(
                self._document_path(project_id, candidate_id), _CachedDocument
            )
        except FileNotFoundError:
            raise EvaluationStorageUnavailableError(
                "Refresh this report before opening its document"
            ) from None
        if (
            document.source_uri != uri
            or len(document.html.encode()) > _REPORT_MAX_BYTES
        ):
            raise ValueError("Cached document does not match the selected report")
        return document.html

    def _document_path(self, project_id: str, candidate_id: str) -> Path:
        return (
            project_directory(self.root, project_id)
            / "model_reports"
            / f"{UUID(candidate_id)}.html.json"
        )

    @staticmethod
    def _html_uri(record: ModelEvaluationRecord) -> str:
        if not record.report_html_uri:
            raise FileNotFoundError("This candidate has no HTML report")
        uri = to_storage_api_path(record.report_html_uri)
        if (
            PurePosixPath(uri).parent
            != PurePosixPath(to_storage_api_path(record.report_uri)).parent
            or PurePosixPath(uri).suffix != ".html"
        ):
            raise ValueError(
                "HTML report must belong to the candidate's evaluation directory"
            )
        return uri

    def _case_storage(
        self, project_id: str, candidate_id: str, context: RequestContext
    ) -> tuple[str, EvaluationStorage]:
        context = self._context(project_id, context)
        record = self.report_service.get(project_id, candidate_id)
        if record.report_status not in {"ready", "invalid"}:
            raise ValueError("Refresh the report before reading its cases")
        prefix = str(PurePosixPath(to_storage_api_path(record.report_uri)).parent)
        return prefix, _ReportCaseStorage(
            self._storage(project_id, context, max_read_bytes=64 * 1024 * 1024), prefix
        )

    def report_cases(
        self,
        project_id: str,
        candidate_id: str,
        context: RequestContext,
        cursor: str | None,
        limit: int,
        query: str,
    ) -> JsonObject:
        if not 1 <= limit <= 100 or len(query) > 256:
            raise ValueError("Invalid report page request")
        prefix, storage = self._case_storage(project_id, candidate_id, context)
        page = EvaluationCaseService(storage).list_page(
            output_dir=prefix,
            cursor=cursor,
            limit=limit,
            query=query,
            selection_key=f"{project_id}:{candidate_id}",
        )
        return TypeAdapter(JsonObject).validate_python(asdict(page))

    def report_media(
        self,
        project_id: str,
        candidate_id: str,
        media_id: str,
        context: RequestContext,
    ) -> tuple[bytes, str]:
        prefix, storage = self._case_storage(project_id, candidate_id, context)
        media = EvaluationCaseService(storage).read_media(
            output_dir=prefix, media_id=media_id
        )
        if media.mime_type not in {
            "image/png",
            "image/jpeg",
            "image/webp",
            "image/gif",
            "image/bmp",
        }:
            raise ValueError("Unsupported evaluation image format")
        return media.payload, media.mime_type

    def schedule(self, project_id: str, context: RequestContext) -> BatchScheduleState:
        context = self._context(project_id, context)
        return self._schedules(context).get(project_id, context.user_id)

    def configure_schedule(
        self, project_id: str, config: BatchScheduleConfig, context: RequestContext
    ) -> BatchScheduleState:
        context = self._context(project_id, context)
        return self._schedules(context).configure(project_id, context.user_id, config)

    def _schedules(self, context: RequestContext) -> BatchSchedules:
        return BatchSchedules(
            self.schedule_store,
            lambda project_id, owner: self._batch_runtime(
                project_id, replace(context, user_id=owner)
            ),
        )

    def _batch_runtime(
        self, project_id: str, context: RequestContext
    ) -> PlatformBatchScheduleRuntime:
        context = self._context(project_id, context)
        project = self.projects.load(project_id, owner_user_id=context.user_id)
        published = PublicationStore(self.root).load_active(project_id)
        slot = RuntimeSlotStore(self.root).load(project_id)
        admin_key = None
        if slot.tracer is not None and slot.tracer.resource_id:
            admin_key = load_tracer_admin_key(
                projects_root=self.root,
                project_id=project_id,
                resource_id=slot.tracer.resource_id,
                cipher=self.cipher,
            )
        conversation_id = UUID(project.project.conversation_id).hex
        try:
            storage = self._storage(
                project_id,
                context,
                write_prefixes=(
                    f"/workspace/.pyromind-agent/{conversation_id}/project-batches",
                ),
            )
        except EvaluationStorageUnavailableError:
            storage = None
        return PlatformBatchScheduleRuntime(
            projects_root=self.root,
            project=project,
            published=published,
            slot=slot,
            scheduler=build_echomind_runtime_scheduler(
                projects_root=self.root,
                conversations_dir=self.conversations_dir,
                request_context=context,
                credential_store=self.credential_store,
                recorded_storage=self.recorded_storage,
            ),
            storage=storage,
            admin_key=admin_key,
            report_archived=lambda run_id: self._report_archived(
                project_id, run_id, storage
            ),
        )

    def _report_archived(
        self, project_id: str, run_id: str, storage: EvaluationStorage | None
    ) -> bool:
        reports = self.report_service.catalog(project_id).candidates
        candidate = next((item for item in reports if item.run_id == run_id), None)
        if candidate is None:
            return False
        if (
            candidate.report_status in {"pending", "unavailable"}
            and storage is not None
        ):
            candidate = self.report_service.refresh(
                project_id, candidate.candidate_id, storage
            )
        return (
            candidate.report_status == "ready"
            and candidate.evaluation_status == "validated"
        )

    def _reconcile_acquisition_batches(
        self, project_id: str, context: RequestContext
    ) -> None:
        slot = RuntimeSlotStore(self.root).load(project_id)
        tracer = slot.tracer
        if (
            slot.status != "ready"
            or tracer is None
            or tracer.resource_id is None
            or not is_platform_tracer_script(
                project_id, tracer.custom_runtime_script_path
            )
        ):
            return
        directory = project_directory(self.root, project_id) / "acquisition_archives"
        acquisitions = {
            run_id: simulation
            for simulation in AcquisitionSimulationStore(self.root).list(project_id)
            if simulation.auto_start_graph_run
            and simulation.tracer_resource_id == tracer.resource_id
            for run_id in simulation.graph_run_ids
        }
        if not acquisitions:
            return
        admin_key = load_tracer_admin_key(
            projects_root=self.root,
            project_id=project_id,
            resource_id=tracer.resource_id,
            cipher=self.cipher,
        )
        runtime = None
        for run in reversed(GraphRunStore(self.root).list(project_id)):
            if (
                run.status != "succeeded"
                or run.execution_mode != "published"
                or run.trigger != "tracer_batch"
                or run.source_tracer_resource_id != tracer.resource_id
                or run.source_batch_id is None
            ):
                continue
            simulation = acquisitions.get(run.run_id)
            if simulation is None:
                continue
            batch = simulation.batches.get(run.source_batch_id)
            if (
                batch is None
                or not batch.input_directory
                or not set(batch.sample_ids).issubset(simulation.completed_sample_ids)
            ):
                continue
            source = run.inputs.get("tracer_batch")
            if (
                not isinstance(source, dict)
                or source.get("simulation_id") != simulation.simulation_id
                or source.get("batch_id") != run.source_batch_id
                or source.get("resource_id") != tracer.resource_id
                or source.get("batch_index") != batch.index
            ):
                continue
            path = directory / f"{UUID(run.run_id)}.json"
            try:
                saved = read_model(path, BatchReceipt)
            except FileNotFoundError:
                saved = None
            if saved is not None and (saved.status == "released" or not admin_key):
                continue
            try:
                candidate = self.report_service.get(project_id, run.run_id)
            except LookupError:
                continue
            if runtime is None:
                runtime = self._batch_runtime(project_id, context)
            if (
                not isinstance(runtime.storage, SandboxEvaluationStorage)
                or runtime.tracer_resource_id != tracer.resource_id
                or runtime.published.publication_mode != "production"
            ):
                return
            if candidate.report_status in {"pending", "unavailable"}:
                candidate = self.report_service.refresh(
                    project_id, candidate.candidate_id, runtime.storage
                )
            if (
                candidate.report_status not in {"ready", "invalid"}
                or not candidate.metrics
            ):
                continue
            receipt = BatchReceipt(
                batch_id=run.source_batch_id,
                batch_index=batch.index,
                tracer_resource_id=tracer.resource_id,
                release_id=run.release_id,
                run_id=run.run_id,
                claimed=True,
                status="releasing",
                created_at=run.created_at,
            )
            try:
                self._archive_acquisition_batch(path, receipt, runtime, runtime.storage)
            except Exception as exc:
                logger.warning(
                    "Tracer batch %s / %d could not be archived or released (%s)",
                    tracer.resource_id,
                    batch.index,
                    type(exc).__name__,
                )

    def _archive_acquisition_batch(
        self,
        path: Path,
        receipt: BatchReceipt,
        runtime: PlatformBatchScheduleRuntime,
        storage: SandboxEvaluationStorage,
    ) -> None:
        conversation = UUID(runtime.project.project.conversation_id).hex
        archive_directory = (
            f"/workspace/.pyromind-agent/{conversation}/project-batches/archives/"
            f"{receipt.tracer_resource_id}/{receipt.batch_index}"
        )

        def verify_archive() -> None:
            archive_tracer_batch(
                storage,
                tracer_resource_id=receipt.tracer_resource_id,
                batch_index=receipt.batch_index,
                archive_directory=archive_directory,
                sealed_before=receipt.created_at,
            )
            write_model(path, receipt)

        if not runtime.admin_key:
            verify_archive()
        elif runtime.release_archived(receipt, verify_archive):
            write_model(
                path,
                receipt.model_copy(
                    update={"status": "released", "released_at": datetime.now(UTC)}
                ),
            )
        elif not path.exists():
            verify_archive()

    def deployments(
        self, project_id: str, context: RequestContext
    ) -> list[DeploymentRecord]:
        self._context(project_id, context)
        return self.deployment_store.list(project_id)

    def _candidate(self, project_id: str, candidate_id: str) -> DeploymentCandidate:
        report = self.report_service.get(project_id, candidate_id)
        if (
            report.execution_mode != "published"
            or report.report_status != "ready"
            or report.evaluation_status != "validated"
            or not report.deployment_model_uri
        ):
            raise ValueError(
                "Deployment requires a validated published report "
                "and merged model weights"
            )
        return DeploymentCandidate(
            candidate_id=candidate_id, model_uri=report.deployment_model_uri
        )

    def deploy(
        self, project_id: str, request: DeploymentRequest, context: RequestContext
    ) -> DeploymentRecord:
        context = self._context(project_id, context)
        with build_manual_deployment_service(
            projects_root=self.root,
            project_id=project_id,
            context=context,
            credential_store=self.credential_store,
            resolve_candidate=self._candidate,
        ) as service:
            return service.start(project_id, request)

    def advance_deployment(
        self, project_id: str, deployment_id: str, context: RequestContext
    ) -> DeploymentRecord:
        context = self._context(project_id, context)
        with build_manual_deployment_service(
            projects_root=self.root,
            project_id=project_id,
            context=context,
            credential_store=self.credential_store,
            resolve_candidate=self._candidate,
        ) as service:
            return service.advance(project_id, deployment_id)

    def poll(self) -> None:
        for path in sorted(self.root.glob("*/deployments.json")):
            project_id = path.parent.name
            try:
                project = self.projects.load(project_id, owner_user_id=None)
                owner = project.project.owner_user_id
                if not owner:
                    continue
                context = RequestContext(
                    user_id=owner, x_cluster=project.project.cluster
                )
                for record in self.deployment_store.list(project_id):
                    if (
                        record.status not in TERMINAL_DEPLOYMENT_STATUSES
                        or record.cleanup_status == "pending"
                    ):
                        self.advance_deployment(
                            project_id, record.deployment_id, context
                        )
            except Exception as exc:
                logger.warning(
                    "Deployment recovery paused for %s (%s)",
                    project_id,
                    type(exc).__name__,
                )
        for project_id, owner in self.schedule_store.list_projects():
            try:
                context = self._context(project_id, RequestContext(user_id=owner))
                self._schedules(context).tick(project_id, owner)
            except Exception as exc:
                logger.warning(
                    "Batch scheduling paused for %s (%s)",
                    project_id,
                    type(exc).__name__,
                )
        for path in sorted(self.root.glob("*/runtime_slot.json")):
            project_id = path.parent.name
            try:
                project = self.projects.load(project_id, owner_user_id=None).project
                if project.lifecycle != "frozen" or not project.owner_user_id:
                    continue
                context = RequestContext(
                    user_id=project.owner_user_id, x_cluster=project.cluster
                )
                with FileLock(path.parent / "acquisition_archives.lock", timeout=0):
                    self._reconcile_acquisition_batches(project_id, context)
            except Timeout:
                continue
            except Exception as exc:
                logger.warning(
                    "Tracer archive recovery paused for %s (%s)",
                    project_id,
                    type(exc).__name__,
                )


class _ReportCaseStorage:
    def __init__(self, storage: EvaluationStorage, prefix: str) -> None:
        self.storage = storage
        self.prefix = prefix.rstrip("/") + "/"

    def read(self, path: str) -> bytes:
        normalized = to_storage_api_path(path)
        if not normalized.startswith(self.prefix):
            raise PermissionError("Case artifact is outside the selected report")
        return self.storage.read(normalized)

    def list(self, prefix: str) -> set[str]:
        del prefix
        raise PermissionError("Report case access is read-only")

    def write(self, path: str, payload: bytes) -> None:
        del path, payload
        raise PermissionError("Report case access is read-only")
