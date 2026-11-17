import traceback
import uuid
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from typing import Any
from urllib.parse import urlparse

from fastapi import APIRouter, Depends, FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pyromind_common.sdk.logger import DEBUG, get_logger
from pyromind_common.sdk.utils.redact import sanitize_dict
from starlette.requests import Request

from pyromind_agent_server.auth_router import auth_router
from pyromind_agent_server.config import Config, get_default_config
from pyromind_agent_server.dependencies import (
    check_session_api_key,
    check_workspace_session,
)
from pyromind_agent_server.init_router import (
    InitService,
    init_router,
    require_initialized,
)
from pyromind_agent_server.kafka_bus.kafka_bus import kafka_message_bus
from pyromind_agent_server.middleware import CORSDispatcher
from pyromind_agent_server.pyromind_router import (
    pyromind_debug_webhook_router,
    pyromind_router,
)
from pyromind_agent_server.server_details_router import (
    get_server_info,
    mark_initialization_complete,
    server_details_router,
)
from pyromind_agent_server.service_context import ProductServiceContext
from pyromind_agent_server.workspace_router import workspace_router


logger = get_logger(__name__)

_PYROMIND_PORTAL_CORS_ORIGINS = (
    "https://pyromind.ai",
    "https://www.pyromind.ai",
    "https://pre.pyromind.ai",
    "https://pre2.pyromind.ai",
    "https://api.pyromind.ai",
    "https://pre-api.pyromind.ai",
    "https://pre2-api.pyromind.ai",
    "https://pre2-studio.pyromind.ai",
    "https://pre-studio.pyromind.ai",
    "https://studio.pyromind.ai",
    "https://console.pyromind.ai",
    "https://pre-console.pyromind.ai",
    "https://pre2-console.pyromind.ai",
    "https://api-portal.pyromind.ai",
    "https://pre-api-portal.pyromind.ai",
    "https://pre2-api-portal.pyromind.ai",
    "https://agent.pyromind.ai",
    "https://pre-agent.pyromind.ai",
    "https://pre2-agent.pyromind.ai",
    "http://localhost:5173",
    "http://localhost:5174",
    "http://localhost:5275",
    "http://localhost:8000",
)


def _cors_allow_origins(config: Config) -> list[str]:
    return list(
        dict.fromkeys((*_PYROMIND_PORTAL_CORS_ORIGINS, *config.allow_cors_origins))
    )


@asynccontextmanager
async def api_lifespan(api: FastAPI) -> AsyncIterator[None]:
    config = api.state.config
    if config.deferred_init:
        api.state.init_service = InitService(api, config)
    else:
        api.state.conversation_service = ProductServiceContext.from_config(config)
    try:
        await kafka_message_bus.start_consumer()
    except Exception:
        logger.exception("Kafka consumer could not start")
    mark_initialization_complete()
    try:
        yield
    finally:
        if config.deferred_init:
            await api.state.init_service.teardown()
        await kafka_message_bus.stop()


def _get_root_path(config: Config) -> str:
    root_path = ""
    if config.web_url:
        web_url = urlparse(config.web_url)
        root_path = web_url.path.rstrip("/")
    return root_path


def _create_fastapi_instance(config: Config) -> FastAPI:
    """Create the basic FastAPI application instance.

    Returns:
        Basic FastAPI application with title, description, and lifespan.
    """
    return FastAPI(
        title="Pyromind Agent Server",
        description=("Pyromind product API for configured harnesses"),
        lifespan=api_lifespan,
        root_path=_get_root_path(config),
    )


def _find_http_exception(exc: BaseExceptionGroup) -> HTTPException | None:
    """Helper function to find HTTPException in ExceptionGroup.

    Args:
        exc: BaseExceptionGroup to search for HTTPException.

    Returns:
        HTTPException if found, None otherwise.
    """
    for inner_exc in exc.exceptions:
        if isinstance(inner_exc, HTTPException):
            return inner_exc
        # Recursively search nested ExceptionGroups
        if isinstance(inner_exc, BaseExceptionGroup):
            found = _find_http_exception(inner_exc)
            if found:
                return found
    return None


def _add_api_routes(app: FastAPI) -> None:
    app.include_router(server_details_router)
    app.include_router(init_router, prefix="/api")
    api_router = APIRouter(
        prefix="/api",
        dependencies=[Depends(check_session_api_key), Depends(require_initialized)],
    )
    api_router.include_router(pyromind_router)
    api_router.include_router(auth_router)
    app.include_router(api_router)
    app.include_router(pyromind_debug_webhook_router)
    app.include_router(
        workspace_router,
        prefix="/api",
        dependencies=[Depends(check_workspace_session), Depends(require_initialized)],
    )


def _setup_static_files(app: FastAPI, config: Config) -> None:
    """Set up static file serving and root redirect if configured.

    Args:
        app: FastAPI application instance.
        config: Configuration object containing static files settings.
    """
    # Only proceed if static files are configured and directory exists
    if not (
        config.static_files_path
        and config.static_files_path.exists()
        and config.static_files_path.is_dir()
    ):
        # Map the root path to server info if there are no static files
        app.get("/", tags=["Server Details"])(get_server_info)
        return

    # Mount static files directory
    app.mount(
        "/static",
        StaticFiles(directory=str(config.static_files_path)),
        name="static",
    )

    # Add root redirect to static files
    @app.get("/", tags=["Server Details"])
    async def root_redirect():
        """Redirect root endpoint to static files directory."""
        # Check if index.html exists in the static directory
        # We know static_files_path is not None here due to the outer condition
        assert config.static_files_path is not None
        index_path = config.static_files_path / "index.html"
        if index_path.exists():
            return RedirectResponse(url="/static/index.html", status_code=302)
        else:
            return RedirectResponse(url="/static/", status_code=302)


def _sanitize_validation_errors(errors: Sequence[Any]) -> list[dict]:
    """Sanitize validation error details to remove sensitive input values.

    FastAPI's default 422 response includes the raw request ``input`` in each
    validation error dict.  If the request contained secret-bearing fields
    (e.g. ``agent.llm.api_key``, MCP server ``env``), those values would be
    echoed back to the caller.  This helper redacts them.

    Args:
        errors: The list of error dicts produced by ``exc.errors()``.

    Returns:
        A new list with ``input`` values sanitized through ``sanitize_dict``.
    """
    sanitized: list[dict] = []
    for error in errors:
        error = dict(error)  # shallow copy so we don't mutate the original
        if "input" in error:
            error["input"] = sanitize_dict(error["input"])
        if isinstance(error.get("ctx"), dict) and isinstance(
            error["ctx"].get("error"), Exception
        ):
            error["ctx"] = {**error["ctx"], "error": str(error["ctx"]["error"])}
        sanitized.append(error)
    return sanitized


def _add_exception_handlers(api: FastAPI) -> None:
    """Add exception handlers to the FastAPI application."""

    @api.exception_handler(RequestValidationError)
    async def _validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        """Handle request validation errors, sanitizing sensitive input.

        FastAPI's default 422 handler echoes the raw request body inside the
        ``detail[].input`` field.  When the request contains secrets (e.g.
        ``agent.llm.api_key``, MCP server ``env``), this would leak credentials
        in the error response.  We intercept the error, redact secret-bearing
        fields, and return a safe 422 response.

        Refs: OpenHands/evaluation#385
        """
        logger.info(
            "Validation error on %s %s: %d error(s)",
            request.method,
            request.url.path,
            len(exc.errors()),
        )
        return JSONResponse(
            status_code=422,
            content={"detail": _sanitize_validation_errors(exc.errors())},
        )

    @api.exception_handler(Exception)
    async def _unhandled_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        """Handle unhandled exceptions."""
        # Correlation id that ties the 500 a caller receives to the server-side
        # log line (with full traceback) for this failure, so an otherwise
        # opaque 500 can be matched to its traceback in the server logs.
        error_id = uuid.uuid4().hex
        # Always log that we're in the exception handler for debugging
        logger.debug(
            "Exception handler called for %s %s with %s: %s [error_id=%s]",
            request.method,
            request.url.path,
            type(exc).__name__,
            str(exc),
            error_id,
        )

        content = {
            "detail": "Internal Server Error",
            "exception": str(exc),
            "error_id": error_id,
        }
        # In DEBUG mode, include stack trace in response
        if DEBUG:
            content["traceback"] = traceback.format_exc()
        # Check if this is an HTTPException that should be handled directly
        if isinstance(exc, HTTPException):
            return await _http_exception_handler(request, exc)

        # Check if this is a BaseExceptionGroup with HTTPExceptions
        if isinstance(exc, BaseExceptionGroup):
            http_exc = _find_http_exception(exc)
            if http_exc:
                return await _http_exception_handler(request, http_exc)
            # If no HTTPException found, treat as unhandled exception
            logger.error(
                "Unhandled ExceptionGroup on %s %s [error_id=%s]",
                request.method,
                request.url.path,
                error_id,
                exc_info=(type(exc), exc, exc.__traceback__),
            )
            return JSONResponse(status_code=500, content=content)

        # Logs full stack trace for any unhandled error that FastAPI would
        # turn into a 500
        logger.error(
            "Unhandled exception on %s %s [error_id=%s]",
            request.method,
            request.url.path,
            error_id,
            exc_info=(type(exc), exc, exc.__traceback__),
        )
        return JSONResponse(status_code=500, content=content)

    @api.exception_handler(HTTPException)
    async def _http_exception_handler(
        request: Request, exc: HTTPException
    ) -> JSONResponse:
        """Handle HTTPExceptions with appropriate logging."""
        # Log 4xx errors at info level (expected client errors like auth failures)
        if 400 <= exc.status_code < 500:
            logger.info(
                "HTTPException %d on %s %s: %s",
                exc.status_code,
                request.method,
                request.url.path,
                exc.detail,
            )
        # Log 5xx errors at error level. HTTPException is intentionally
        # raised flow control — the route picked this status and detail
        # on purpose — so a stack trace adds no information beyond
        # `exc.detail` and makes routine upstream blips look
        # indistinguishable from a process crash. Unhandled exceptions
        # still get a full traceback via _unhandled_exception_handler
        # above. Include the traceback only when DEBUG is on, as an
        # opt-in debugging aid.
        elif exc.status_code >= 500:
            logger.error(
                "HTTPException %d on %s %s: %s",
                exc.status_code,
                request.method,
                request.url.path,
                exc.detail,
                exc_info=(type(exc), exc, exc.__traceback__) if DEBUG else None,
            )
            content = {
                "detail": "Internal Server Error",
                "exception": str(exc),
            }
            if DEBUG:
                content["traceback"] = traceback.format_exc()
            # Don't leak internal details to clients for 5xx errors in production
            return JSONResponse(
                status_code=exc.status_code,
                content=content,
            )

        # Return clean JSON response for all non-5xx HTTP exceptions
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


def create_app(config: Config | None = None) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        config: Configuration object. If None, uses default config.
        include_conversation_api: 是否挂载 SDK 会话执行接口；Pi 服务关闭此项。

    Returns:
        Configured FastAPI application.
    """
    if config is None:
        config = get_default_config()
    app = _create_fastapi_instance(config)
    app.state.config = config

    _add_api_routes(app)
    _setup_static_files(app, config)
    app.add_middleware(
        CORSDispatcher,
        allow_origins=_cors_allow_origins(config),
        allow_origin_regex=config.allow_cors_origin_regex,
    )
    _add_exception_handlers(app)

    return app
