from __future__ import annotations

import logging
import os
import re
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from harness_adapter.agentgenome_bridge import AgentGenomeBackend, ExecutionHost
from harness_adapter.artifact_snapshots import PlatformArtifactSnapshots
from harness_adapter.openhands_adapter import OpenHandsAdapter
from harness_adapter.pi_adapter import PiAdapter, resolve_pi_terminal_backend
from harness_adapter.workflow_submission import PlatformWorkflowSubmission
from pyromind_runtime.application.conversation_runtime import ConversationRuntime
from pyromind_runtime.application.pipeline_runs import PipelineRuns
from pyromind_runtime.application.versions import Versions
from pyromind_runtime.domain.capabilities import ResourceLimits
from pyromind_runtime.domain.content import JsonObject
from pyromind_runtime.domain.context import RequestContext
from pyromind_runtime.domain.versions import VersionToolRequest
from pyromind_runtime.infrastructure.file_product_store import FileProductStore
from pyromind_runtime.infrastructure.sqlite_version_store import SQLiteVersionStore
from pyromind_runtime.ports.harness import HarnessAdapter
from pyromind_runtime.ports.workflows import WorkflowEventSink

from openhands.agent_server.config import WORKSPACE_DIR_ENV, WORKSPACE_DIR_ENV_UPPER
from openhands.agent_server.run_workflow_callback import set_workflow_status_dispatcher
from openhands.agent_server.storage_quota import ensure_conversation_quota
from pyromind_agent_server.external_task_registry import WorkflowExternalTaskRegistry
from pyromind_agent_server.workflow_status_dispatcher import WorkflowStatusDispatcher


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
    backend = (raw_backend or "openhands").strip().lower()
    if backend not in {"openhands", "pi"}:
        raise RuntimeError(f"Unsupported PYROMIND_HARNESS_BACKEND: {backend}")
    openhands_adapter = OpenHandsAdapter(lambda: app.state.conversation_service)
    adapters: dict[str, HarnessAdapter] = {"openhands": openhands_adapter}
    version_path = Path(
        os.getenv("PYROMIND_VERSION_DB_PATH")
        or service.conversations_dir.parent / "product-versions" / "versions.sqlite3"
    )
    versions = Versions(
        SQLiteVersionStore(version_path),
        PlatformArtifactSnapshots(version_path.parent / "snapshot-staging"),
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
    ):
        assert isinstance(runtime, ConversationRuntime)
        return await runtime.submit_versioned_workflow(
            scope, request_id, workflow, code, context
        )

    openhands_adapter.set_version_handler(version_call, execution_call)
    terminal_backend: str | None = None
    external_tasks = WorkflowExternalTaskRegistry(service.conversations_dir)

    def pipeline_runs(conversation_id: str) -> PipelineRuns:
        return PipelineRuns(
            conversation_id,
            FileProductStore(service.conversations_dir / conversation_id),
            external_tasks,
            versions=versions,
        )

    workflows = None
    workflow_enabled = os.getenv("PYROMIND_AGENTGENOME_ENABLED", "0") == "1"

    async def workflow_call(
        scope: str,
        action: str,
        arguments: JsonObject,
        request_id: str,
        context: RequestContext,
        origin_run_id: str | None,
    ) -> JsonObject:
        assert isinstance(runtime, ConversationRuntime)
        return await runtime.workflow_call(
            scope, action, arguments, request_id, context, origin_run_id
        )

    if backend == "pi":
        terminal_backend = resolve_pi_terminal_backend()
        pi_adapter = PiAdapter(
            service.conversations_dir,
            terminal_backend=terminal_backend,
            apply_workspace_quota=ensure_conversation_quota,
            pipeline_runs=pipeline_runs,
            workflow_call=workflow_call if workflow_enabled else None,
        )
        pi_adapter.set_version_handler(version_call, execution_call)
        adapters["pi"] = pi_adapter
        if workflow_enabled:

            def execution_host(scope: str) -> ExecutionHost:
                def subscribe(
                    execution_id: str, sink: WorkflowEventSink | None
                ) -> None:
                    pi_adapter.execution_subscribe(scope, execution_id, sink)

                return ExecutionHost(
                    pi_adapter.execution_request(scope),
                    subscribe,
                    service.conversations_dir / scope,
                )

            workflows = AgentGenomeBackend(
                agentgenome_home(service.conversations_dir),
                execution_host,
            )
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
    runtime = ConversationRuntime(
        service.conversations_dir,
        adapters,
        default_harness_id=backend,
        external_tasks=external_tasks,
        idle_eviction_seconds=idle_eviction_seconds,
        release_grace_seconds=release_grace_seconds,
        resource_retention_seconds=resource_retention_seconds,
        max_active_conversations=max_active_conversations,
        resource_limits=resource_limits_from_environment(),
        workflows=workflows,
        workflow_submission=PlatformWorkflowSubmission(),
        versions=versions,
    )
    if backend == "pi":
        adapter = adapters["pi"]
        assert isinstance(adapter, PiAdapter)
        adapter.set_workflow_preparer(runtime.prepare_workflow_execution)
    logger.info(
        "Pyromind product runtime ready: default_harness=%s "
        "PYROMIND_HARNESS_BACKEND=%s registered_harnesses=%s "
        "pi_terminal_backend=%s idle_eviction_seconds=%d "
        "release_grace_seconds=%d resource_retention_seconds=%d "
        "max_active_conversations=%d conversations_dir=%s",
        backend,
        raw_backend if raw_backend is not None else "<unset, defaulting to openhands>",
        sorted(adapters),
        terminal_backend or "-",
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
    """Compose Product API around the existing OpenHands application."""
    from pyromind_agent_server.api.router import create_product_router

    app.state.product_runtime = None
    original_lifespan = app.router.lifespan_context

    @asynccontextmanager
    async def product_lifespan(current_app: FastAPI):
        async with original_lifespan(current_app):
            runtime = ensure_product_runtime(current_app)
            if runtime is not None:
                await runtime.recover_version_metadata()
            try:
                yield
            finally:
                runtime = getattr(current_app.state, "product_runtime", None)
                if isinstance(runtime, ConversationRuntime):
                    await runtime.close()
                set_workflow_status_dispatcher(None)
                current_app.state.product_runtime = None

    app.router.lifespan_context = product_lifespan
    app.include_router(create_product_router())
    return app
