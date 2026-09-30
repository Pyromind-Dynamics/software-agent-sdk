from __future__ import annotations

import logging
import os
import re
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from harness_adapter.agentgenome_bridge import AgentGenomeBackend, ExecutionHost
from harness_adapter.openhands_adapter import OpenHandsAdapter
from harness_adapter.pi_adapter import PiAdapter, resolve_pi_terminal_backend
from pyromind_runtime.application.conversation_runtime import ConversationRuntime
from pyromind_runtime.application.pipeline_runs import PipelineRuns
from pyromind_runtime.domain.capabilities import ResourceLimits
from pyromind_runtime.domain.content import JsonObject
from pyromind_runtime.domain.context import RequestContext
from pyromind_runtime.infrastructure.file_product_store import FileProductStore
from pyromind_runtime.ports.harness import HarnessAdapter
from pyromind_runtime.ports.workflows import WorkflowEventSink

from openhands.agent_server.run_workflow_callback import set_workflow_status_dispatcher
from openhands.agent_server.storage_quota import ensure_conversation_quota
from pyromind_agent_server.external_task_registry import WorkflowExternalTaskRegistry
from pyromind_agent_server.workflow_status_dispatcher import WorkflowStatusDispatcher


logger = logging.getLogger(__name__)


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
    adapters: dict[str, HarnessAdapter] = {
        "openhands": OpenHandsAdapter(lambda: app.state.conversation_service),
    }
    terminal_backend: str | None = None
    external_tasks = WorkflowExternalTaskRegistry(service.conversations_dir)

    def pipeline_runs(conversation_id: str) -> PipelineRuns:
        return PipelineRuns(
            conversation_id,
            FileProductStore(service.conversations_dir / conversation_id),
            external_tasks,
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
        adapters["pi"] = pi_adapter
        if workflow_enabled:

            def execution_host(scope: str) -> ExecutionHost:
                def subscribe(
                    execution_id: str, sink: WorkflowEventSink | None
                ) -> None:
                    pi_adapter.execution_subscribe(scope, execution_id, sink)

                return ExecutionHost(pi_adapter.execution_request(scope), subscribe)

            workflows = AgentGenomeBackend(
                Path(
                    os.getenv("AGENTGENOME_HOME")
                    or service.conversations_dir.parent / "agentgenome"
                ),
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
    )
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
            ensure_product_runtime(current_app)
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
