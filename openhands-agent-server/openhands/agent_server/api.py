import asyncio
import os
import tempfile
import traceback
import uuid
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import libtmux
from fastapi import APIRouter, Depends, FastAPI, HTTPException, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pyromind_runtime.adapters.openhands import (
    OpenHandsAdapter,
    PersistedSettingsOpenHandsSessionFactory,
)
from pyromind_runtime.adapters.pi import (
    LocalPiRunnerLauncher,
    PiAdapter,
    SandboxGatewayError,
    StaticPiModelConfigResolver,
    safe_runner_environment,
)
from pyromind_runtime.product.router import create_product_router
from pyromind_runtime.product.runtime import (
    ProductRuntimeService,
    ProductRuntimeSettings,
)
from pyromind_runtime.tool_host import (
    PythonToolHost,
    SessionToolContextStore,
    first_version_tool_specs,
)
from starlette.requests import Request

from openhands.agent_server.agent_profiles_router import agent_profiles_router
from openhands.agent_server.auth_router import auth_router
from openhands.agent_server.bash_router import bash_router
from openhands.agent_server.bash_service import get_default_bash_event_service
from openhands.agent_server.config import (
    Config,
    get_default_config,
)
from openhands.agent_server.conversation_router import conversation_router
from openhands.agent_server.conversation_service import (
    get_default_conversation_service,
)
from openhands.agent_server.dependencies import (
    check_session_api_key,
    check_workspace_session,
    get_current_user_id,
)
from openhands.agent_server.desktop_router import desktop_router
from openhands.agent_server.desktop_service import get_desktop_service
from openhands.agent_server.event_router import event_router
from openhands.agent_server.file_router import file_router
from openhands.agent_server.git_router import git_router
from openhands.agent_server.hooks_router import hooks_router
from openhands.agent_server.init_router import (
    InitService,
    init_router,
    require_initialized,
)
from openhands.agent_server.kafka_bus.kafka_bus import kafka_message_bus
from openhands.agent_server.llm_router import llm_router
from openhands.agent_server.mcp_router import mcp_router
from openhands.agent_server.middleware import CORSDispatcher
from openhands.agent_server.openai.router import (
    check_openai_api_key,
    openai_router,
)
from openhands.agent_server.plugins_router import plugins_router
from openhands.agent_server.profiles_router import profiles_router
from openhands.agent_server.pyromind_router import (
    build_product_tool_host,
    get_product_tool_request_context,
    pyromind_debug_webhook_router,
    pyromind_router,
)
from openhands.agent_server.server_details_router import (
    get_server_info,
    mark_initialization_complete,
    server_details_router,
)
from openhands.agent_server.settings_router import settings_router
from openhands.agent_server.skills_router import skills_router
from openhands.agent_server.sockets import sockets_router
from openhands.agent_server.tool_preload_service import get_tool_preload_service
from openhands.agent_server.tool_router import tool_router
from openhands.agent_server.vscode_router import vscode_router
from openhands.agent_server.vscode_service import get_vscode_service
from openhands.agent_server.workflow_canvas_router import workflow_canvas_router
from openhands.agent_server.workspace_router import workspace_router
from openhands.agent_server.workspaces_router import workspaces_router
from openhands.sdk.logger import DEBUG, get_logger
from openhands.sdk.utils.redact import sanitize_dict
from openhands.tools.terminal.constants import TMUX_SOCKET_NAME


logger = get_logger(__name__)

_DEFAULT_PI_SKILL_NAMES = ("generate-workflow-dsl",)
_DEFAULT_PI_SYSTEM_PROMPT = """\
You are the Pyromind workflow and dataset assistant.

Work only inside the assigned workspace. Use preview_dataset for
bounded dataset inspection and validate_workflow_dsl for platform validation.
Create or edit the canonical workflow at
public_data/workflow_canvas/workflow.py. Never expose credentials or attempt to
access files outside the assigned workspace.
"""

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
    "https://api-aws-west-2.pyromind.ai",
    "https://pre-api-aws-west-2.pyromind.ai",
    "https://pre2-api-aws-west-2.pyromind.ai",
    "https://console-aws-west-2.pyromind.ai",
    "https://pre-console-aws-west-2.pyromind.ai",
    "https://pre2-console-aws-west-2.pyromind.ai",
    "https://console-us-west-2.pyromind.ai",
    "https://console-us-west-1.pyromind.ai",
    "https://pre-console-us-west-2.pyromind.ai",
    "https://pre-console-us-west-1.pyromind.ai",
    "https://pre2-console-us-west-2.pyromind.ai",
    "https://pre2-console-us-west-1.pyromind.ai",
    "https://api-us-west-1.pyromind.ai",
    "https://api-us-west-2.pyromind.ai",
    "https://pre-api-us-west-1.pyromind.ai",
    "https://pre-api-us-west-2.pyromind.ai",
    "https://pre2-api-us-west-1.pyromind.ai",
    "https://pre2-api-us-west-2.pyromind.ai",
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


def _default_server_tmux_tmpdir() -> Path:
    return Path(tempfile.gettempdir()) / f"openhands-agent-server-{os.getpid()}"


def _ensure_server_tmux_tmpdir() -> tuple[Path, bool]:
    existing = os.getenv("TMUX_TMPDIR")
    if existing:
        return Path(existing), False

    tmux_tmpdir = _default_server_tmux_tmpdir()
    tmux_tmpdir.mkdir(parents=True, exist_ok=True)
    os.environ["TMUX_TMPDIR"] = str(tmux_tmpdir)
    logger.info(
        "TMUX_TMPDIR not set; defaulting to per-server tmux directory %s",
        tmux_tmpdir,
    )
    return tmux_tmpdir, True


def _cleanup_stale_tmux_sessions() -> None:
    """Clean up any stale tmux sessions on server startup.

    Tmux sessions live in a separate process that survives agent-server restarts.
    This function kills all existing sessions on the shared OpenHands tmux socket
    to prevent accumulation of orphaned sessions.
    """
    try:
        server = libtmux.Server(socket_name=TMUX_SOCKET_NAME)
        sessions = server.sessions
        if not sessions:
            logger.debug("No tmux sessions found on %s socket", TMUX_SOCKET_NAME)
            return

        logger.info("Cleaning up %d stale tmux session(s) on startup", len(sessions))

        for session in sessions:
            try:
                logger.debug("Killing tmux session: %s", session.name)
                session.kill()
            except Exception as e:
                logger.warning("Failed to kill tmux session %s: %s", session.name, e)

        logger.info("Tmux cleanup completed")

    except Exception as e:
        # Don't let tmux cleanup failures prevent server startup
        logger.warning("Failed to cleanup tmux sessions: %s", e)


@asynccontextmanager
async def api_lifespan(api: FastAPI) -> AsyncIterator[None]:
    tmux_tmpdir, tmux_tmpdir_was_defaulted = _ensure_server_tmux_tmpdir()
    try:
        # Clean up stale tmux sessions from previous server runs
        _cleanup_stale_tmux_sessions()

        config: Config = api.state.config
        deferred = config.deferred_init
        vscode_service = get_vscode_service()
        desktop_service = get_desktop_service()
        tool_preload_service = get_tool_preload_service()

        # Define async functions for starting each service
        async def start_vscode_service():
            if vscode_service is not None:
                vscode_started = await vscode_service.start()
                if vscode_started:
                    logger.info("VSCode service started successfully")
                else:
                    logger.warning(
                        "VSCode service failed to start, continuing without VSCode"
                    )
            else:
                logger.info("VSCode service is disabled")

        async def start_desktop_service():
            if desktop_service is not None:
                desktop_started = await desktop_service.start()
                if desktop_started:
                    logger.info("Desktop service started successfully")
                else:
                    logger.warning(
                        "Desktop service failed to start, continuing without desktop"
                    )
            else:
                logger.info("Desktop service is disabled")

        async def start_tool_preload_service():
            if tool_preload_service is not None:
                tool_preload_started = await tool_preload_service.start()
                if tool_preload_started:
                    logger.info("Tool preload service started successfully")
                else:
                    logger.warning("Tool preload service failed to start - skipping")
            else:
                logger.info("Tool preload service is disabled")

        # Start all services concurrently
        results = await asyncio.gather(
            start_vscode_service(),
            start_desktop_service(),
            start_tool_preload_service(),
            return_exceptions=True,
        )

        # Check for any exceptions during initialization
        exceptions = [r for r in results if isinstance(r, Exception)]
        if exceptions:
            logger.error(
                "Service initialization failed with %d exception(s): %s",
                len(exceptions),
                exceptions,
            )
            # Re-raise the first exception to prevent server from starting
            raise RuntimeError(
                f"Server initialization failed with {len(exceptions)} exception(s)"
            ) from exceptions[0]

        # Kafka consumer starts in the background and must not block server boot.
        try:
            await kafka_message_bus.start_consumer()
            logger.info("Kafka message bus consumer start requested")
        except Exception:
            logger.exception(
                "Kafka message bus consumer failed to start; continuing without Kafka"
            )

        async def stop_stateless_services():
            async def stop_vscode_service():
                if vscode_service is not None:
                    await vscode_service.stop()

            async def stop_desktop_service():
                if desktop_service is not None:
                    await desktop_service.stop()

            async def stop_tool_preload_service():
                if tool_preload_service is not None:
                    await tool_preload_service.stop()

            async def stop_kafka_message_bus():
                try:
                    await kafka_message_bus.stop()
                except Exception:
                    logger.exception("Kafka message bus stop failed")

            await asyncio.gather(
                stop_vscode_service(),
                stop_desktop_service(),
                stop_tool_preload_service(),
                stop_kafka_message_bus(),
                return_exceptions=True,
            )

        # In deferred-init mode the conversation service is *not* entered
        # here — that happens later, when POST /api/init delivers the runtime
        # config. We still mark the /ready endpoint as ready so a warm-pool
        # orchestrator can tell the pod has finished booting and is
        # available to receive its /api/init payload.
        if deferred:
            init_service = InitService(api, base_config=config)
            api.state.init_service = init_service
            mark_initialization_complete()
            logger.info("Server started in deferred-init mode; awaiting POST /api/init")
            try:
                yield
            finally:
                await init_service.teardown()
                await stop_stateless_services()
            return

        # Non-deferred (legacy) path: build and enter the conversation
        # service as part of the lifespan, exactly as before.
        service = get_default_conversation_service()
        mark_initialization_complete()
        logger.info("Server initialization complete - ready to serve requests")

        bash_svc = get_default_bash_event_service()
        api.state.bash_event_service = bash_svc

        async with service:
            api.state.conversation_service = service

            config = api.state.config
            retention_task: asyncio.Task | None = None
            if config.bash_events_retention_seconds is not None:
                retention_task = asyncio.create_task(
                    bash_svc.run_retention_cleanup_loop(
                        config.bash_events_retention_seconds
                    )
                )
                logger.info(
                    "Bash events retention cleanup started (retention: %ds)",
                    config.bash_events_retention_seconds,
                )

            try:
                yield
            finally:
                if retention_task is not None:
                    retention_task.cancel()
                    with suppress(asyncio.CancelledError):
                        await retention_task

                await stop_stateless_services()
    finally:
        product_runtime = getattr(api.state, "product_runtime", None)
        if product_runtime is not None:
            await product_runtime.close()
        if tmux_tmpdir_was_defaulted and os.environ.get("TMUX_TMPDIR") == str(
            tmux_tmpdir
        ):
            os.environ.pop("TMUX_TMPDIR", None)


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
        title="OpenHands Agent Server",
        description=(
            "OpenHands Agent Server - REST/WebSocket interface for OpenHands AI Agent"
        ),
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
    """Add all API routes to the FastAPI application."""
    app.include_router(server_details_router)

    # The /api/init endpoint bypasses both the session-key auth and the
    # dormant gate. It has its own X-Init-API-Key auth. When
    # ``deferred_init`` is False the endpoints are still mounted but return
    # 404 because no InitService is registered on app.state — see
    # ``get_init_service``.
    init_api_router = APIRouter(prefix="/api")
    init_api_router.include_router(init_router)
    app.include_router(init_api_router)

    # Header-only auth: applied to every /api/* route EXCEPT the workspace
    # static-file routes (handled separately below). Cookies are NOT honored
    # here so that we don't expand the CSRF surface across the whole API.
    # check_session_api_key reads config from request.app.state at request time,
    # so keys delivered via POST /api/init are honoured without re-registering routes.
    dependencies = [
        Depends(check_session_api_key),
        # Dormant gate: 503s every /api/* route until POST /api/init completes.
        # No-op for non-deferred deployments.
        Depends(require_initialized),
    ]

    api_router = APIRouter(prefix="/api", dependencies=dependencies)
    api_router.include_router(event_router)
    api_router.include_router(conversation_router)
    api_router.include_router(tool_router)
    api_router.include_router(bash_router)
    api_router.include_router(git_router)
    api_router.include_router(file_router)
    api_router.include_router(vscode_router)
    api_router.include_router(desktop_router)
    api_router.include_router(skills_router)
    api_router.include_router(plugins_router)
    api_router.include_router(hooks_router)
    api_router.include_router(llm_router)
    api_router.include_router(mcp_router)
    api_router.include_router(settings_router)
    api_router.include_router(workspaces_router)
    api_router.include_router(workflow_canvas_router)
    api_router.include_router(profiles_router)
    api_router.include_router(agent_profiles_router)
    api_router.include_router(pyromind_router)
    api_router.include_router(
        create_product_router(
            get_current_user_id,
            get_product_tool_request_context,
        )
    )
    # /api/auth/* mints workspace cookies and requires the header to bootstrap,
    # so it lives under the header-only auth group.
    api_router.include_router(auth_router)
    app.include_router(api_router)

    app.include_router(openai_router, dependencies=[Depends(check_openai_api_key)])

    # Debug-platform webhook: intentionally NOT behind check_session_api_key
    # (see pyromind_debug_webhook_router's docstring for why) or the dormant
    # gate -- a callback for a run that was submitted before a restart should
    # still be able to land.
    app.include_router(pyromind_debug_webhook_router)

    # Workspace static-file routes get their own auth group that accepts
    # EITHER the X-Session-API-Key header OR the workspace session cookie.
    # The cookie is required so that <iframe src> / <img src> embeds of
    # workspace artifacts work — browsers cannot attach custom headers to
    # those requests.
    workspace_api_router = APIRouter(
        prefix="/api", dependencies=[Depends(check_workspace_session)]
    )
    workspace_api_router.include_router(workspace_router)
    app.include_router(workspace_api_router)

    app.include_router(sockets_router)


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

    @api.exception_handler(SandboxGatewayError)
    async def _sandbox_gateway_exception_handler(
        request: Request, exc: SandboxGatewayError
    ) -> JSONResponse:
        status_code = {
            "invalid": status.HTTP_400_BAD_REQUEST,
            "permission_denied": status.HTTP_401_UNAUTHORIZED,
            "not_found": status.HTTP_404_NOT_FOUND,
        }.get(exc.code, status.HTTP_502_BAD_GATEWAY)
        logger.warning(
            "Sandbox gateway error [%s] on %s %s: %s",
            exc.code,
            request.method,
            request.url.path,
            exc,
        )
        return JSONResponse(status_code=status_code, content={"detail": str(exc)})


def _pi_enabled(default_harness_id: str) -> bool:
    configured = os.getenv("PYROMIND_ENABLE_PI", "").strip().lower()
    return default_harness_id == "pi" or configured in {"1", "true", "yes", "on"}


def _default_product_harness_id() -> str:
    configured = os.getenv("PYROMIND_DEFAULT_HARNESS")
    if configured is not None and configured.strip():
        return configured.strip()
    enabled = os.getenv("PYROMIND_ENABLE_PI", "").strip().lower()
    return "pi" if enabled in {"1", "true", "yes", "on"} else "openhands"


def _pi_skill_directories() -> tuple[Path, ...]:
    skills_root = Path(
        os.getenv(
            "PYROMIND_SKILLS_PATH",
            str(Path(__file__).resolve().parents[3] / ".agents" / "skills"),
        )
    ).resolve()
    configured = os.getenv("PYROMIND_PI_SKILLS")
    names = (
        tuple(name.strip() for name in configured.split(",") if name.strip())
        if configured is not None
        else _DEFAULT_PI_SKILL_NAMES
    )
    skill_directories: list[Path] = []
    for name in names:
        if "/" in name or "\\" in name or name in {".", ".."}:
            raise RuntimeError(f"Invalid PYROMIND_PI_SKILLS entry: {name}")
        directory = (skills_root / name).resolve()
        if directory.parent != skills_root or not (directory / "SKILL.md").is_file():
            raise RuntimeError(f"Configured Pi skill is missing: {directory}")
        skill_directories.append(directory)
    return tuple(skill_directories)


def _register_pi_adapter(
    app: FastAPI,
    *,
    default_harness_id: str,
    tool_host: PythonToolHost,
    model_profile_id: str,
    workspace_root: Path,
) -> None:
    if not _pi_enabled(default_harness_id):
        return

    llm_model = os.getenv("LLM_MODEL", "gpt-5.5").strip()
    fallback_provider, fallback_model_id = (
        llm_model.split("/", 1) if "/" in llm_model else ("openai", llm_model)
    )
    model_provider = os.getenv("PYROMIND_PI_MODEL_PROVIDER", fallback_provider).strip()
    model_id = os.getenv("PYROMIND_PI_MODEL_ID", fallback_model_id).strip()
    model_api_key = (
        os.getenv("PYROMIND_PI_MODEL_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or os.getenv("LLM_API_KEY")
        or ""
    )
    model_base_url = os.getenv("PYROMIND_PI_MODEL_BASE_URL") or os.getenv(
        "LLM_BASE_URL"
    )
    thinking_level = os.getenv("PYROMIND_PI_THINKING_LEVEL", "high").strip()
    pi_runtime_dir = Path(__file__).resolve().parents[2] / "pi-runtime"
    runner_entry = pi_runtime_dir / "src" / "index.ts"
    if not runner_entry.is_file():
        raise RuntimeError(f"Pi runner entrypoint is missing: {runner_entry}")

    launcher = LocalPiRunnerLauncher(
        command=(
            os.getenv("PYROMIND_PI_NODE_BINARY", "node"),
            str(runner_entry),
        ),
        workspace_root=workspace_root,
        cwd=str(pi_runtime_dir),
        environment=safe_runner_environment(),
        model_resolver=StaticPiModelConfigResolver(
            profile_id=model_profile_id,
            provider=model_provider,
            model_id=model_id,
            api_key=model_api_key,
            base_url=model_base_url,
            thinking_level=thinking_level,
        ),
        tool_host=tool_host,
        skill_directories=_pi_skill_directories(),
        system_prompt=os.getenv(
            "PYROMIND_PI_SYSTEM_PROMPT",
            _DEFAULT_PI_SYSTEM_PROMPT,
        ),
    )
    app.state.product_runtime.registry.register("pi", PiAdapter(launcher))


def create_app(config: Config | None = None) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        config: Configuration object. If None, uses default config.

    Returns:
        Configured FastAPI application.
    """
    if config is None:
        config = get_default_config()
    app = _create_fastapi_instance(config)
    app.state.config = config
    product_storage_root = Path(
        os.getenv(
            "PYROMIND_PRODUCT_RUNTIME_DIR",
            str(config.conversations_path.parent / "product_conversations"),
        )
    )
    default_harness_id = _default_product_harness_id()
    model_profile_id = os.getenv("PYROMIND_MODEL_PROFILE", "default")
    tool_context_store = SessionToolContextStore(
        allowed_header_names={"authorization", "cookie", "x-cluster"}
    )
    product_tool_host = build_product_tool_host(tool_context_store)
    app.state.product_runtime = ProductRuntimeService(
        ProductRuntimeSettings(
            storage_root=product_storage_root,
            default_harness_id=default_harness_id,
            default_workspace_root=str(config.workspace_path),
            default_model_profile_id=model_profile_id,
        ),
        default_tools_by_harness={"pi": first_version_tool_specs()},
        tool_context_store=tool_context_store,
    )
    app.state.product_runtime.registry.register(
        "openhands",
        OpenHandsAdapter(
            lambda: app.state.conversation_service,
            PersistedSettingsOpenHandsSessionFactory(lambda: app.state.config),
        ),
    )
    _register_pi_adapter(
        app,
        default_harness_id=default_harness_id,
        tool_host=product_tool_host,
        model_profile_id=model_profile_id,
        workspace_root=config.workspace_path,
    )

    _add_api_routes(app)
    _setup_static_files(app, config)
    app.add_middleware(
        CORSDispatcher,
        allow_origins=_cors_allow_origins(config),
        allow_origin_regex=config.allow_cors_origin_regex,
    )
    _add_exception_handlers(app)

    return app


# Create the default app instance
api = create_app()
