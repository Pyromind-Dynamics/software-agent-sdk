from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from harness_adapter.openhands_adapter import OpenHandsAdapter
from harness_adapter.pi_adapter import PiAdapter
from pyromind_runtime.application.conversation_runtime import ConversationRuntime
from pyromind_runtime.ports.harness import HarnessAdapter

from openhands.agent_server.run_workflow_callback import set_workflow_status_dispatcher
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
    app_env = os.getenv("APP_ENV", "dev").strip().lower()
    if app_env not in {"prod", "production", "online"}:
        adapters["pi"] = PiAdapter(service.conversations_dir)
    elif backend == "pi":
        raise RuntimeError(
            "Pi local execution is disabled in production until sk-sandbox "
            "is integrated"
        )
    runtime = ConversationRuntime(
        service.conversations_dir,
        adapters,
        default_harness_id=backend,
        external_tasks=WorkflowExternalTaskRegistry(service.conversations_dir),
    )
    set_workflow_status_dispatcher(WorkflowStatusDispatcher(runtime).dispatch)
    app.state.product_runtime = runtime
    return runtime


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
