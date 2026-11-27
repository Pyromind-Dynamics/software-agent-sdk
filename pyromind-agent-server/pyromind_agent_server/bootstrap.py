from __future__ import annotations

import asyncio
import logging
import os
import re
import secrets
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request

from harness_adapter.artifact_snapshots import PlatformArtifactSnapshots
from harness_adapter.pi_adapter import PiAdapter, resolve_pi_terminal_backend
from harness_adapter.task_status_query import PlatformTaskStatusQuery
from harness_adapter.workflow_submission import PlatformWorkflowSubmission
from pyromind_agent_server.accepted_workflow import read_accepted_workflow
from pyromind_agent_server.config import (
    Config,
)
from pyromind_agent_server.echomind_acquisition_tasks import (
    EchoMindAcquisitionTaskManager,
)
from pyromind_agent_server.echomind_projects import ProjectStore
from pyromind_agent_server.echomind_publication import (
    ProjectLifecycleError,
    ProjectPublicationService,
    PublicationValidationError,
    RuntimeImplementationAdoptionError,
)
from pyromind_agent_server.echomind_repair_proposals import GraphRepairProposals
from pyromind_agent_server.echomind_resource_proposals import (
    prepare_runtime_resources,
)
from pyromind_agent_server.echomind_runtime_access import build_runtime_credential_store
from pyromind_agent_server.echomind_runtime_factory import (
    GraphRuntimeCallbackService,
    build_echomind_runtime_scheduler,
)
from pyromind_agent_server.echomind_runtime_scheduler import (
    RuntimeGraphPoller,
)
from pyromind_agent_server.echomind_tracer_admin import build_tracer_admin_cipher
from pyromind_agent_server.evaluation_storage import (
    build_evaluation_storage,
    build_observation_evaluation_storage,
)
from pyromind_agent_server.experiment_plan_projects import (
    EchoMindExperimentPlanProjectBinder,
)
from pyromind_agent_server.external_task_registry import WorkflowExternalTaskRegistry
from pyromind_agent_server.observability import build_execution_observer
from pyromind_agent_server.plan_database import build_experiment_plan_store
from pyromind_agent_server.profiles import ECHOMIND_PRODUCT_PROFILE
from pyromind_agent_server.project_flywheel import ProjectFlywheelServices
from pyromind_agent_server.resource_inspector import StorageResourceInspector
from pyromind_agent_server.storage_quota import ensure_conversation_quota
from pyromind_agent_server.task_monitor_worker import TOKEN_ENV
from pyromind_agent_server.training_metric_source import PlatformTrainingMetricSource
from pyromind_agent_server.workflow_status import set_workflow_status_dispatcher
from pyromind_agent_server.workflow_status_dispatcher import WorkflowStatusDispatcher
from pyromind_runtime.application.conversation_runtime import ConversationRuntime
from pyromind_runtime.application.pipeline_runs import PipelineRuns
from pyromind_runtime.application.task_monitor import TaskMonitor
from pyromind_runtime.application.train_observer import ObservationCursor, TrainObserver
from pyromind_runtime.application.versions import Versions
from pyromind_runtime.domain.capabilities import ResourceLimits
from pyromind_runtime.domain.content import JsonObject
from pyromind_runtime.domain.context import RequestContext
from pyromind_runtime.domain.echomind_resources import RuntimeResourceProposalArguments
from pyromind_runtime.domain.task_correlation import TaskCorrelation
from pyromind_runtime.domain.versions import VersionToolRequest
from pyromind_runtime.domain.workflow_execution import DatasetRecipeReceipt
from pyromind_runtime.infrastructure.file_product_store import FileProductStore
from pyromind_runtime.infrastructure.sql_experiment_index import SQLExperimentIndex
from pyromind_runtime.infrastructure.sqlite_task_monitor_store import (
    SQLiteTaskMonitorStore,
)
from pyromind_runtime.infrastructure.sqlite_version_store import SQLiteVersionStore
from pyromind_runtime.ports.harness import HarnessAdapter


logger = logging.getLogger(__name__)


def ensure_product_runtime(app: FastAPI) -> ConversationRuntime | None:
    runtime = getattr(app.state, "product_runtime", None)
    if isinstance(runtime, ConversationRuntime):
        return runtime
    service = getattr(app.state, "conversation_service", None)
    if service is None:
        return None
    raw_backend = os.getenv("PYROMIND_HARNESS_BACKEND")
    backend = (raw_backend or "pi").strip().lower()
    if backend != "pi":
        raise RuntimeError(f"Unsupported PYROMIND_HARNESS_BACKEND: {backend}")
    version_path = Path(
        os.getenv("PYROMIND_VERSION_DB_PATH")
        or service.conversations_dir.parent / "product-versions" / "versions.sqlite3"
    )
    versions = Versions(
        SQLiteVersionStore(version_path),
        PlatformArtifactSnapshots(version_path.parent / "snapshot-staging"),
        archive_runs=os.getenv("PYROMIND_RUN_ARCHIVES_ENABLED", "0") == "1",
    )

    async def version_call(
        scope: str,
        request: VersionToolRequest,
        command_id: str,
        context: RequestContext,
    ):
        assert isinstance(runtime, ConversationRuntime)
        return await runtime.version_call(scope, request, command_id, context)

    async def execution_call(
        scope: str,
        request_id: str,
        workflow: JsonObject,
        code: str | None,
        context: RequestContext,
        graph_run_id: str | None,
        correlation: TaskCorrelation | None = None,
        continue_after_completion: bool = False,
        dataset_recipe: DatasetRecipeReceipt | None = None,
    ):
        assert isinstance(runtime, ConversationRuntime)
        return await runtime.submit_versioned_workflow(
            scope,
            request_id,
            workflow,
            code,
            context,
            graph_run_id=graph_run_id,
            correlation=correlation,
            continue_after_completion=continue_after_completion,
            dataset_recipe=dataset_recipe,
        )

    external_tasks = WorkflowExternalTaskRegistry(service.conversations_dir)

    def pipeline_runs(conversation_id: str) -> PipelineRuns:
        return PipelineRuns(
            conversation_id,
            FileProductStore(service.conversations_dir / conversation_id),
            external_tasks,
            versions=versions,
        )

    terminal_backend = resolve_pi_terminal_backend()
    config = getattr(app.state, "config", None)
    pi_adapter = PiAdapter(
        service.conversations_dir,
        terminal_backend=terminal_backend,
        platform_sdk_enabled=(
            config.platform_sdk_enabled if isinstance(config, Config) else True
        ),
        apply_workspace_quota=ensure_conversation_quota,
        pipeline_runs=pipeline_runs,
    )
    pi_adapter.set_version_handler(version_call, execution_call)

    def recorded_storage(scope: str, context: RequestContext):
        assert isinstance(runtime, ConversationRuntime)
        return runtime.recorded_workflow_storage(scope, context)

    async def record_training_artifacts(scope, task_id, nodes, context):
        assert isinstance(runtime, ConversationRuntime)
        return await runtime.record_training_artifacts(scope, task_id, nodes, context)

    pi_adapter.set_training_artifact_hook(record_training_artifacts)
    adapters: dict[str, HarnessAdapter] = {"pi": pi_adapter}
    idle_eviction_seconds = int(
        os.getenv("PYROMIND_IDLE_CONVERSATION_EVICTION_SECONDS", "300")
    )
    release_grace_seconds = int(
        os.getenv("PYROMIND_CONVERSATION_RELEASE_GRACE_SECONDS", "300")
    )
    resource_retention_seconds = int(
        os.getenv("PYROMIND_SANDBOX_IDLE_DELETE_SECONDS", "1800")
    )
    max_active_conversations = int(os.getenv("PYROMIND_MAX_ACTIVE_CONVERSATIONS", "0"))
    # The harness reclaims idle conversations on its own timer. When the product
    # timer is the slower of the two, live product sessions go cold underneath
    # the product layer and every command in that window has to self-heal on
    # re-attach, so make the mismatch visible at boot.
    harness_eviction_seconds = int(getattr(service, "idle_eviction_timeout", 0) or 0)
    if 0 < harness_eviction_seconds < idle_eviction_seconds:
        logger.warning(
            "Pyromind idle eviction (%ds) is slower than the harness idle "
            "eviction (%ds): sessions are reclaimed by the harness first and "
            "re-activated lazily on next access. Align them with "
            "PYROMIND_IDLE_CONVERSATION_EVICTION_SECONDS<=%d or "
            "OH_IDLE_CONVERSATION_EVICTION_SECONDS>=%d to avoid the gap.",
            idle_eviction_seconds,
            harness_eviction_seconds,
            harness_eviction_seconds,
            idle_eviction_seconds,
        )
    preparation_store = build_experiment_plan_store()
    projects_root = Path(service.conversations_dir).parent / "projects"
    runtime = ConversationRuntime(
        service.conversations_dir,
        adapters,
        default_harness_id=backend,
        preparation_store=preparation_store,
        graph_repair_submitter=GraphRepairProposals(projects_root),
        experiment_index=SQLExperimentIndex(preparation_store.engine)
        if preparation_store is not None
        else None,
        experiment_plan_project_binder=EchoMindExperimentPlanProjectBinder(
            service.conversations_dir
        )
        if preparation_store
        else None,
        resource_media_reader=StorageResourceInspector(),
        external_task_query=PlatformWorkflowSubmission(),
        accepted_workflow_reader=read_accepted_workflow,
        observer=build_execution_observer(service.conversations_dir),
        external_tasks=external_tasks,
        idle_eviction_seconds=idle_eviction_seconds,
        release_grace_seconds=release_grace_seconds,
        resource_retention_seconds=resource_retention_seconds,
        max_active_conversations=max_active_conversations,
        resource_limits=resource_limits_from_environment(),
        workflow_submission=PlatformWorkflowSubmission(),
        versions=versions,
        evaluation_storage_factory=build_evaluation_storage,
        observation_storage_factory=build_observation_evaluation_storage,
        train_observer=TrainObserver(
            PlatformTrainingMetricSource(), ObservationCursor(secrets.token_bytes(32))
        ),
        project_flywheel=ProjectFlywheelServices(
            recorded_storage=recorded_storage,
            projects_root=projects_root,
            conversations_dir=service.conversations_dir,
            credential_store=build_runtime_credential_store(projects_root, config),
            cipher=build_tracer_admin_cipher(config),
        ),
    )
    if os.getenv("PYROMIND_TASK_MONITOR_ENABLED", "1") == "1":
        runtime.task_monitor = TaskMonitor(
            SQLiteTaskMonitorStore(service.conversations_dir / ".task-monitor.sqlite3"),
            PlatformTaskStatusQuery(),
            runtime,
            first_delay=float(os.getenv("PYROMIND_TASK_MONITOR_FIRST_DELAY", "60")),
            interval=float(os.getenv("PYROMIND_TASK_MONITOR_INTERVAL", "180")),
        )
        runtime.task_monitor.recover(service.conversations_dir)
    pi_adapter.set_workflow_preparer(runtime.prepare_workflow_execution)
    pi_adapter.set_plan_submission_handler(runtime.bind_plan_submission)
    pi_adapter.set_graph_repair_handler(runtime.graph_repair_proposal)
    pi_adapter.set_external_wait_provider(runtime.pending_external_tasks)
    pi_adapter.set_sdk_context_provider(runtime.platform_sdk_context)
    pi_adapter.set_echo_chat_handler(
        runtime.echomind_chat.call, runtime.echomind_chat.model_history
    )
    pi_adapter.set_project_reward_reader(runtime.project_reward)
    pi_adapter.set_evaluation_reward_reader(runtime.evaluation_reward)
    pi_adapter.set_preparation_handler(
        runtime.preparation_call,
        runtime.preparation_revision,
        runtime.preparation_review,
    )

    async def propose_runtime_resources(
        conversation_id: str, arguments: JsonObject, context: RequestContext
    ) -> JsonObject:
        if context.profile_id != ECHOMIND_PRODUCT_PROFILE.profile_id:
            raise PermissionError("Runtime resources belong to an EchoMind Project.")
        view = await runtime.get_project(conversation_id, context)
        if view is None:
            raise ValueError("Create an EchoMind Project first.")
        projects = ProjectStore.for_conversations_dir(service.conversations_dir)
        project = projects.load(view.project_id, owner_user_id=context.user_id)
        projects.require_cluster(project, context.x_cluster)
        try:
            release = await asyncio.to_thread(
                ProjectPublicationService(projects.root).resource_setup_release,
                project,
            )
        except (
            FileNotFoundError,
            ProjectLifecycleError,
            PublicationValidationError,
            RuntimeImplementationAdoptionError,
        ) as exc:
            raise ValueError(
                "Adopt a verified execution plan before configuring runtime resources."
            ) from exc
        proposal = await asyncio.to_thread(
            prepare_runtime_resources,
            projects.root,
            project,
            release,
            RuntimeResourceProposalArguments.model_validate(arguments),
        )
        return {
            "status": "awaiting_user_confirmation",
            "project_id": proposal.project_id,
            "release_id": proposal.release_id,
            "proposal_digest": proposal.digest,
        }

    pi_adapter.set_runtime_resource_proposer(propose_runtime_resources)
    logger.info(
        "Pyromind product runtime ready: default_harness=%s "
        "PYROMIND_HARNESS_BACKEND=%s registered_harnesses=%s "
        "pi_terminal_backend=%s idle_eviction_seconds=%d "
        "release_grace_seconds=%d resource_retention_seconds=%d "
        "max_active_conversations=%d conversations_dir=%s",
        backend,
        raw_backend if raw_backend is not None else "<unset, defaulting to pi>",
        sorted(adapters),
        terminal_backend,
        idle_eviction_seconds,
        release_grace_seconds,
        resource_retention_seconds,
        max_active_conversations,
        service.conversations_dir,
    )
    projects_root = Path(service.conversations_dir).parent / "projects"
    graph_callbacks = GraphRuntimeCallbackService(
        projects_root,
        recorded_storage=runtime.recorded_workflow_storage,
        conversations_dir=service.conversations_dir,
        credential_store=build_runtime_credential_store(
            projects_root, getattr(app.state, "config", None)
        ),
    )
    set_workflow_status_dispatcher(
        WorkflowStatusDispatcher(
            runtime,
            graph_task_handler=graph_callbacks.handle,
        ).dispatch
    )
    app.state.product_runtime = runtime
    return runtime


def resource_limits_from_environment() -> ResourceLimits:
    memory_raw = os.getenv("OH_SANDBOX_VMEM_LIMIT", "500M")
    match = re.fullmatch(
        r"(?P<number>\d+)\s*(?P<unit>[kmg]?)", memory_raw, re.IGNORECASE
    )
    if match is None:
        raise RuntimeError(f"invalid OH_SANDBOX_VMEM_LIMIT={memory_raw!r}")
    multipliers = {"": 1, "k": 1024, "m": 1024**2, "g": 1024**3}
    memory_limit_bytes = int(match["number"]) * multipliers[match["unit"].lower()]
    nproc_raw = os.getenv("OH_SANDBOX_NPROC_LIMIT", "2")
    try:
        nproc_limit = int(nproc_raw)
    except ValueError as exc:
        raise RuntimeError(f"invalid OH_SANDBOX_NPROC_LIMIT={nproc_raw!r}") from exc
    try:
        return ResourceLimits(
            memory_limit_bytes=memory_limit_bytes, nproc_limit=nproc_limit
        )
    except ValueError as exc:
        raise RuntimeError(
            "invalid sandbox resource limits: "
            f"OH_SANDBOX_VMEM_LIMIT={memory_raw!r}, "
            f"OH_SANDBOX_NPROC_LIMIT={nproc_raw!r}"
        ) from exc


def install_product_api(app: FastAPI) -> FastAPI:
    """Install the product API and own its runtime lifecycle."""
    from pyromind_agent_server.api.demo_collection import (
        router as demo_collection_router,
    )
    from pyromind_agent_server.api.echomind_chat import router as echomind_chat_router
    from pyromind_agent_server.api.echomind_observations import (
        router as observations_router,
    )
    from pyromind_agent_server.api.echomind_projects import (
        router as echomind_projects_router,
    )
    from pyromind_agent_server.api.evaluation_runs import (
        create_evaluation_runs_router,
    )
    from pyromind_agent_server.api.project_flywheel import router as flywheel_router
    from pyromind_agent_server.api.router import create_product_router

    app.state.product_runtime = None
    app.state.task_monitor_token = os.getenv(TOKEN_ENV) or secrets.token_urlsafe(32)
    original_lifespan = app.router.lifespan_context

    @app.post("/internal/task-monitor/check", include_in_schema=False)
    async def check_background_tasks(request: Request) -> dict[str, int]:
        supplied = request.headers.get("x-task-monitor-token", "")
        if not secrets.compare_digest(supplied, app.state.task_monitor_token):
            raise HTTPException(403, "Invalid task monitor credential")
        runtime = ensure_product_runtime(app)
        if runtime is None or runtime.task_monitor is None:
            raise HTTPException(503, "Task monitoring is unavailable")
        checked = await runtime.task_monitor.check_due(limit=1)
        return {"checked": checked}

    @asynccontextmanager
    async def product_lifespan(current_app: FastAPI):
        async with original_lifespan(current_app):
            try:
                await start_product_runtime(current_app)
                yield
            finally:
                await stop_product_runtime(current_app)

    app.router.lifespan_context = product_lifespan
    app.include_router(create_product_router())
    app.include_router(create_product_router(ECHOMIND_PRODUCT_PROFILE))
    app.include_router(echomind_projects_router)
    app.include_router(demo_collection_router)
    app.include_router(flywheel_router)
    app.include_router(echomind_chat_router)
    app.include_router(observations_router)
    app.include_router(create_evaluation_runs_router())
    return app


async def _poll_project_flywheels(
    runtime: ConversationRuntime, stop: asyncio.Event
) -> None:
    while not stop.is_set():
        try:
            await runtime.project_flywheel_poll()
        except Exception as exc:
            logger.warning("Project flywheel recovery paused (%s)", type(exc).__name__)
        try:
            await asyncio.wait_for(
                stop.wait(), timeout=_echomind_runtime_poll_seconds()
            )
        except TimeoutError:
            pass


def _echomind_runtime_poll_seconds() -> float:
    raw = os.getenv("PYROMIND_ECHOMIND_RUNTIME_POLL_SECONDS", "15")
    try:
        value = float(raw)
    except ValueError:
        return 15
    return value if value > 0 else 15


async def _supervise_monitor(app: FastAPI) -> None:
    while True:
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "pyromind_agent_server.task_monitor_worker",
            "--url",
            os.getenv("OH_INTERNAL_SERVER_URL", "http://127.0.0.1:8000"),
            env={**os.environ, TOKEN_ENV: app.state.task_monitor_token},
        )
        try:
            await process.wait()
            logger.warning("Task monitor worker exited; restarting")
        finally:
            if process.returncode is None:
                process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), timeout=10)
                except TimeoutError:
                    process.kill()
                    await process.wait()
        await asyncio.sleep(5)


async def start_product_runtime(app: FastAPI) -> None:
    runtime = ensure_product_runtime(app)
    if runtime is not None:
        await runtime.recover_version_metadata()
        await runtime.start_execution_events()
        stop = asyncio.Event()
        app.state.echomind_flywheel_stop = stop
        app.state.echomind_flywheel_task = asyncio.create_task(
            _poll_project_flywheels(runtime, stop)
        )
        conversation_service = getattr(
            app.state,
            "conversation_service",
            None,
        )
        if conversation_service is not None:
            projects_root = (
                Path(conversation_service.conversations_dir).parent / "projects"
            )
            graph_poller = RuntimeGraphPoller(
                build_echomind_runtime_scheduler(
                    projects_root=projects_root,
                    recorded_storage=runtime.recorded_workflow_storage,
                    conversations_dir=conversation_service.conversations_dir,
                    credential_store=build_runtime_credential_store(
                        projects_root,
                        getattr(app.state, "config", None),
                    ),
                ),
                interval_seconds=_echomind_runtime_poll_seconds(),
            )
            app.state.echomind_graph_poller = graph_poller
            await graph_poller.start()

        if (
            runtime.task_monitor is not None
            and os.getenv("PYROMIND_TASK_MONITOR_AUTOSTART", "1") == "1"
            and getattr(app.state, "task_monitor_worker", None) is None
        ):
            app.state.task_monitor_worker = asyncio.create_task(_supervise_monitor(app))


async def stop_product_runtime(app: FastAPI) -> None:
    worker = getattr(app.state, "task_monitor_worker", None)
    if worker is not None:
        worker.cancel()
        await asyncio.gather(worker, return_exceptions=True)
        app.state.task_monitor_worker = None
    flywheel_stop = getattr(app.state, "echomind_flywheel_stop", None)
    if flywheel_stop is not None:
        flywheel_stop.set()
        await app.state.echomind_flywheel_task
        app.state.echomind_flywheel_stop = None
        app.state.echomind_flywheel_task = None
    graph_poller = getattr(app.state, "echomind_graph_poller", None)
    if graph_poller is not None:
        await graph_poller.stop()
        app.state.echomind_graph_poller = None
    acquisition_tasks = getattr(
        app.state,
        "echomind_acquisition_tasks",
        None,
    )
    if isinstance(acquisition_tasks, EchoMindAcquisitionTaskManager):
        await acquisition_tasks.stop()
    runtime = getattr(app.state, "product_runtime", None)
    if isinstance(runtime, ConversationRuntime):
        await runtime.close()
    set_workflow_status_dispatcher(None)
    app.state.echomind_acquisition_tasks = None
    app.state.product_runtime = None
