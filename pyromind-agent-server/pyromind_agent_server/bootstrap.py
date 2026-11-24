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
from pyromind_agent_server.config import WORKSPACE_DIR_ENV, WORKSPACE_DIR_ENV_UPPER
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
from pyromind_runtime.domain.task_correlation import TaskCorrelation
from pyromind_runtime.domain.versions import VersionToolRequest
from pyromind_runtime.infrastructure.file_product_store import FileProductStore
from pyromind_runtime.infrastructure.sql_experiment_index import SQLExperimentIndex
from pyromind_runtime.infrastructure.sqlite_task_monitor_store import (
    SQLiteTaskMonitorStore,
)
from pyromind_runtime.infrastructure.sqlite_version_store import SQLiteVersionStore
from pyromind_runtime.ports.harness import HarnessAdapter


logger = logging.getLogger(__name__)

AGENTGENOME_HOME_ENV = "AGENTGENOME_HOME"
# Agent workspaces live under this directory inside the mounted user Storage, so
# the sandbox sees the same tree at ``/target-workspace/.pyromind-agent``.
AGENTGENOME_STORAGE_DIRNAME = ".pyromind-agent"
AGENTGENOME_DIRNAME = "agentgenome"


def _move_legacy_agentgenome_home(home: Path, legacy: Path) -> None:
    """Adopt a registry written before the Storage-root layout."""
    if home.exists() or not legacy.is_dir():
        return
    try:
        home.parent.mkdir(parents=True, exist_ok=True)
        legacy.rename(home)
    except OSError:
        logger.warning(
            "AgentGenome registry at %s could not move to %s; set %s to keep using it",
            legacy,
            home,
            AGENTGENOME_HOME_ENV,
            exc_info=True,
        )
        return
    logger.info("Moved AgentGenome registry %s -> %s", legacy, home)


def agentgenome_home(conversations_dir: Path) -> Path:
    """Registry root shared by every conversation of one user.

    The default sits beside the conversation workspaces inside the mounted
    Storage, which is also what the sandbox mounts, so a frozen package is
    readable from both sides. ``AGENTGENOME_HOME`` still wins when set.
    """
    configured = os.getenv(AGENTGENOME_HOME_ENV)
    if configured:
        return Path(configured)
    root = _workspace_root(conversations_dir)
    home = root / AGENTGENOME_STORAGE_DIRNAME / AGENTGENOME_DIRNAME
    _move_legacy_agentgenome_home(home, root / AGENTGENOME_DIRNAME)
    return home


def _workspace_root(conversations_dir: Path) -> Path:
    """Mounted Storage root, which stays fixed when conversations move."""
    configured = os.getenv(WORKSPACE_DIR_ENV) or os.getenv(WORKSPACE_DIR_ENV_UPPER)
    return Path(configured) if configured else conversations_dir.parent


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
    pi_adapter = PiAdapter(
        service.conversations_dir,
        terminal_backend=terminal_backend,
        apply_workspace_quota=ensure_conversation_quota,
        pipeline_runs=pipeline_runs,
    )
    pi_adapter.set_version_handler(version_call, execution_call)

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
    runtime = ConversationRuntime(
        service.conversations_dir,
        adapters,
        default_harness_id=backend,
        preparation_store=preparation_store,
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
    set_workflow_status_dispatcher(WorkflowStatusDispatcher(runtime).dispatch)
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
    app.include_router(echomind_chat_router)
    app.include_router(observations_router)
    app.include_router(create_evaluation_runs_router())
    return app


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
    runtime = getattr(app.state, "product_runtime", None)
    if isinstance(runtime, ConversationRuntime):
        await runtime.close()
    set_workflow_status_dispatcher(None)
    app.state.product_runtime = None
