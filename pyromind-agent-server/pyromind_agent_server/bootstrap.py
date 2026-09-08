from __future__ import annotations

import os
import re
from contextlib import asynccontextmanager

from fastapi import FastAPI
from harness_adapter.openhands_adapter import OpenHandsAdapter
from harness_adapter.pi_adapter import PiAdapter, resolve_pi_terminal_backend
from pyromind_runtime.application.conversation_runtime import ConversationRuntime
from pyromind_runtime.domain.capabilities import ResourceLimits
from pyromind_runtime.ports.harness import HarnessAdapter

from openhands.agent_server.run_workflow_callback import set_workflow_status_dispatcher
from openhands.agent_server.storage_quota import ensure_conversation_quota
from pyromind_agent_server.external_task_registry import WorkflowExternalTaskRegistry
from pyromind_agent_server.workflow_status_dispatcher import WorkflowStatusDispatcher


def ensure_product_runtime(app: FastAPI) -> ConversationRuntime | None:
    runtime = getattr(app.state, "product_runtime", None)
    if isinstance(runtime, ConversationRuntime):
        return runtime
    service = getattr(app.state, "conversation_service", None)
    if service is None:
        return None
    backend = os.getenv("PYROMIND_HARNESS_BACKEND", "openhands").strip().lower()
    if backend not in {"openhands", "pi"}:
        raise RuntimeError(f"Unsupported PYROMIND_HARNESS_BACKEND: {backend}")
    adapters: dict[str, HarnessAdapter] = {
        "openhands": OpenHandsAdapter(lambda: app.state.conversation_service),
    }
    if backend == "pi":
        terminal_backend = resolve_pi_terminal_backend()
        adapters["pi"] = PiAdapter(
            service.conversations_dir,
            terminal_backend=terminal_backend,
            apply_workspace_quota=ensure_conversation_quota,
        )
    runtime = ConversationRuntime(
        service.conversations_dir,
        adapters,
        default_harness_id=backend,
        external_tasks=WorkflowExternalTaskRegistry(service.conversations_dir),
        idle_eviction_seconds=int(
            os.getenv("PYROMIND_IDLE_CONVERSATION_EVICTION_SECONDS", "1800")
        ),
        resource_limits=resource_limits_from_environment(),
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
