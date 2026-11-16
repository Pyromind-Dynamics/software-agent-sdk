"""供 Pi 与通用服务壳复用的平台辅助接口、鉴权上下文及状态回调。"""

from __future__ import annotations

from typing import Any, Literal
from uuid import UUID

import httpx
from fastapi import APIRouter, HTTPException, Request, Response, status
from pydantic import BaseModel, Field

from pyromind_agent_server.workflow_status import (
    WorkflowCallbackResult,
    dispatch_workflow_status as deliver_workflow_status,
)
from pyromind_common.platform.auth import (
    CurrentLoginUser,
    get_debug_current_login_user_by_conversation,
    is_dev,
)
from pyromind_common.tools.data_preparation import (
    DfCheckProgressAction,
)
from pyromind_common.tools.data_preparation.progress import DfCheckProgressExecutor
from pyromind_common.tools.pyromind_debug import get_debug_result_broker


class Success(BaseModel):
    success: bool = True


PYROMIND_AUTH_TOKEN_SECRET = "auth_token"


PYROMIND_X_CLUSTER_SECRET = "PYROMIND_X_CLUSTER"


_PYROMIND_DEBUG_URL_TIMEOUT_SECONDS = 30.0


_PYROMIND_DEBUG_RESPONSE_BODY_LIMIT = 20000


class PyromindDebugCallbackRequest(BaseModel):
    """Webhook payload the debug platform posts when an async run finishes."""

    task_id: str = Field(description="The task id returned when the run was submitted.")
    status: Literal["passed", "failed"] = Field(description="Outcome of the run.")
    error_log: str | None = Field(
        default=None, description="Runtime error output when status='failed'."
    )


class PyromindWorkflowCallbackRequest(BaseModel):
    """发送给 Product Runtime 的工作流状态 Webhook 请求体。"""

    task_id: str = Field(description="Platform task id from studio.create().")
    status: str = Field(
        description="Raw workflow status from the platform or Kafka message."
    )
    conversation_id: str | None = Field(
        default=None,
        description=(
            "Conversation to resume. Dataset cleaning callbacks may omit this "
            "because task association is persisted at submission time."
        ),
    )
    error_log: str | None = Field(
        default=None,
        description="Runtime error log when status is Failed or Error.",
    )
    auto_run: bool = Field(
        default=True,
        description="Restart the agent on the target conversation after delivery.",
    )
    from_workflow_debug: bool = Field(
        default=False,
        description=(
            "True when simulating a workflow_debug (test) terminal callback. "
            "Kafka derives this from out_id agent1#debug#..."
        ),
    )


class PyromindWorkflowCallbackResponse(BaseModel):
    """Product Runtime 工作流状态投递结果。"""

    success: bool = True
    outcome: str
    task_id: str
    normalized_status: str | None = None
    conversation_id: str | None = None


class PyromindDebugUrlRequest(BaseModel):
    url: str = Field(description="HTTP or HTTPS URL to call with dev auth context.")
    conversation_id: UUID = Field(
        description=(
            "Conversation to scope the debug request to. The server verifies the "
            "conversation belongs to the current user before forwarding the "
            "current request's login context."
        ),
    )


class PyromindDebugUrlResponse(BaseModel):
    url: str
    context_source: str
    conversation_id: UUID | None = None
    status_code: int
    forwarded_headers: dict[str, str]
    response_headers: dict[str, str]
    body: str


pyromind_router = APIRouter(prefix="/pyromind", tags=["Pyromind"])


def _get_current_login_user(http_request: Request) -> CurrentLoginUser:
    current_user = getattr(http_request.state, "current_user", None)
    if not isinstance(current_user, CurrentLoginUser):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Pyromind user context is not available.",
        )
    return current_user


def _validate_debug_url(url: str) -> str:
    try:
        parsed = httpx.URL(url)
    except httpx.InvalidURL as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid URL: {exc}",
        ) from exc

    if parsed.scheme not in {"http", "https"} or not parsed.host:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="URL must be an absolute HTTP or HTTPS URL.",
        )
    return str(parsed)


def _build_debug_context_headers(current_user: CurrentLoginUser) -> dict[str, str]:
    headers: dict[str, str] = {}
    if current_user.cookie:
        headers["cookie"] = current_user.cookie
    if current_user.x_cluster:
        headers["x-cluster"] = current_user.x_cluster
    return headers


@pyromind_router.post(
    "/debug/request-url",
    response_model=PyromindDebugUrlResponse,
)
async def request_debug_url(
    request: PyromindDebugUrlRequest,
) -> PyromindDebugUrlResponse:
    if not is_dev():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)

    url = _validate_debug_url(request.url)
    conversation_user = get_debug_current_login_user_by_conversation(
        request.conversation_id
    )
    if conversation_user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Pyromind conversation login context is not available.",
        )
    headers = _build_debug_context_headers(conversation_user)

    try:
        async with httpx.AsyncClient(
            timeout=_PYROMIND_DEBUG_URL_TIMEOUT_SECONDS,
            follow_redirects=False,
        ) as client:
            response = await client.get(url, headers=headers)
    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Failed to request debug URL: {type(exc).__name__}: {exc}",
        ) from exc

    return PyromindDebugUrlResponse(
        url=url,
        context_source="conversation",
        conversation_id=request.conversation_id,
        status_code=response.status_code,
        forwarded_headers=headers,
        response_headers=dict(response.headers),
        body=response.text[:_PYROMIND_DEBUG_RESPONSE_BODY_LIMIT],
    )


@pyromind_router.get("/data-preparation/progress")
def get_data_preparation_progress(
    http_request: Request,
    response: Response,
    output_dir: str,
    tail_lines: int = 5,
) -> dict[str, Any]:
    """Poll the live progress of a submitted DataFlow pipeline run.

    Reads ``progress.json`` and the tail of ``processed.jsonl`` from the
    run's Storage output directory so the frontend can render a progress
    bar, ETA, and recently processed records while the job is running.
    """
    # The frontend polls this URL repeatedly with the same query string; make
    # sure browsers/proxies never serve a cached (stale) copy.
    response.headers["Cache-Control"] = "no-store, max-age=0"
    current_user = _get_current_login_user(http_request)
    headers = _build_debug_context_headers(current_user)
    executor = DfCheckProgressExecutor(headers=headers)
    action = DfCheckProgressAction(output_dir=output_dir, tail_lines=tail_lines)
    observation = executor(action, conversation=None)
    return {
        "output_dir": observation.output_dir,
        "progress_found": observation.progress_found,
        "total": observation.total,
        "processed": observation.processed,
        "succeeded": observation.succeeded,
        "failed": observation.failed,
        "elapsed_ms": observation.elapsed_ms,
        "records_per_second": observation.records_per_second,
        "eta_ms": observation.eta_ms,
        "updated_at": observation.updated_at,
        "percent": observation.percent,
        "latest_records": observation.latest_records,
    }


pyromind_debug_webhook_router = APIRouter(prefix="/api/pyromind", tags=["Pyromind"])


@pyromind_debug_webhook_router.post("/debug/callback", response_model=Success)
async def pyromind_debug_callback(request: PyromindDebugCallbackRequest) -> Success:
    """Webhook the debug platform calls when an async debug run finishes.

    Resolves the in-process :class:`DebugResultBroker`, waking the
    ``debug_workflow`` tool-executor thread that is blocked waiting for this
    task's result. See ``pyromind_common.tools.pyromind_debug`` for the tool side
    and ``MockDebugPlatform`` for the local stand-in used until the real
    platform integration is wired up.
    """
    broker = get_debug_result_broker()
    resolved = broker.resolve(
        request.task_id, status=request.status, error_log=request.error_log
    )
    if not resolved:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Unknown or already-resolved debug task: {request.task_id}",
        )
    return Success()


@pyromind_debug_webhook_router.post(
    "/workflow/callback",
    response_model=PyromindWorkflowCallbackResponse,
)
async def pyromind_workflow_callback(
    request: PyromindWorkflowCallbackRequest,
) -> PyromindWorkflowCallbackResponse:
    """将工作流状态分发给 Product Runtime。

    调用 :func:`deliver_workflow_status`，便于手动 HTTP 请求在没有 Kafka
    消费者时测试异步恢复。与 ``/debug/callback`` 共用无需鉴权的 Webhook
    路由，仅供内部可信网络调用。
    """
    result: WorkflowCallbackResult = await deliver_workflow_status(
        task_id=request.task_id,
        status=request.status,
        error_log=request.error_log,
        conversation_id=request.conversation_id,
        auto_run=request.auto_run,
        from_workflow_debug=request.from_workflow_debug,
    )
    if result.outcome in {"unknown_task", "unknown_conversation"}:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"run_workflow callback failed for task_id={request.task_id}: "
                f"{result.outcome}"
            ),
        )
    return PyromindWorkflowCallbackResponse(
        outcome=result.outcome,
        task_id=result.task_id,
        normalized_status=result.normalized_status,
        conversation_id=result.conversation_id,
    )
