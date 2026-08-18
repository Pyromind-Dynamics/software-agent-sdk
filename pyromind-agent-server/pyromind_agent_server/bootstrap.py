from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from harness_adapter.openhands_adapter import OpenHandsAdapter
from pyromind_runtime.application.conversation_runtime import ConversationRuntime


def ensure_product_runtime(app: FastAPI) -> ConversationRuntime | None:
    runtime = getattr(app.state, "product_runtime", None)
    if isinstance(runtime, ConversationRuntime):
        return runtime
    service = getattr(app.state, "conversation_service", None)
    if service is None:
        return None
    runtime = ConversationRuntime(
        service.conversations_dir,
        OpenHandsAdapter(lambda: app.state.conversation_service),
    )
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
                current_app.state.product_runtime = None

    app.router.lifespan_context = product_lifespan
    app.include_router(create_product_router())
    return app
