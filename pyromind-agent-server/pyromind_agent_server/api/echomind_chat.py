from __future__ import annotations

from typing import Annotated

from fastapi import (
    APIRouter,
    Depends,
    Header,
    HTTPException,
    Path,
    Query,
    Request,
    Response,
)
from fastapi.responses import StreamingResponse

from pyromind_agent_server.api.router import (
    _not_found,
    _product_error,
    _resolve_cursor,
    _sse_stream,
    configure_session,
    get_product_runtime,
)
from pyromind_agent_server.api.schemas import AgentLLMConfig
from pyromind_agent_server.auth_context import request_context
from pyromind_agent_server.dependencies import check_session_api_key
from pyromind_agent_server.profiles import ECHOMIND_PRODUCT_PROFILE
from pyromind_runtime.domain.commands import (
    CancelCommand,
    CommandReceipt,
    UserMessageCommand,
)
from pyromind_runtime.domain.echomind_chat import CHAT_PREFIX
from pyromind_runtime.domain.errors import ProductRuntimeError
from pyromind_runtime.domain.snapshot import ConversationSnapshot
from pyromind_runtime.infrastructure.file_product_store import (
    CommandConflictError,
    ProductStoreError,
)


router = APIRouter(
    prefix="/api/v2/echomind/conversations/{manager_id}/chat",
    dependencies=[Depends(check_session_api_key)],
    tags=["EchoMind Chat Agent"],
)


@router.post("", response_model=ConversationSnapshot)
async def open_chat(
    manager_id: str, model: AgentLLMConfig, request: Request
) -> ConversationSnapshot:
    runtime = get_product_runtime(request)
    try:
        await configure_session(request, manager_id, ECHOMIND_PRODUCT_PROFILE)
        return await runtime.echomind_chat.open(
            manager_id, model.model_dump(mode="json"), request_context(request)
        )
    except ProductRuntimeError as exc:
        raise _product_error(exc) from exc
    except (FileNotFoundError, PermissionError, ProductStoreError, ValueError) as exc:
        raise _not_found(manager_id) from exc


@router.get("/snapshot", response_model=ConversationSnapshot | None)
async def chat_snapshot(
    manager_id: str, request: Request
) -> ConversationSnapshot | None:
    runtime = get_product_runtime(request)
    try:
        return await runtime.echomind_chat.snapshot(
            manager_id, request_context(request)
        )
    except ProductRuntimeError as exc:
        raise _product_error(exc) from exc
    except (FileNotFoundError, PermissionError, ProductStoreError, ValueError) as exc:
        raise _not_found(manager_id) from exc


@router.post("/commands", response_model=CommandReceipt, status_code=202)
async def chat_command(
    manager_id: str, command: UserMessageCommand | CancelCommand, request: Request
) -> CommandReceipt:
    if isinstance(command, UserMessageCommand) and command.workflow_xyflow is not None:
        raise HTTPException(422, detail="Chat Agent messages cannot supply a workflow")
    runtime = get_product_runtime(request)
    chat_id = CHAT_PREFIX + manager_id
    try:
        context = await runtime.echomind_chat.route(
            manager_id, request_context(request)
        )
        if isinstance(command, UserMessageCommand):
            await configure_session(request, manager_id, ECHOMIND_PRODUCT_PROFILE)
        return await runtime.submit_command(chat_id, command, context)
    except CommandConflictError as exc:
        raise HTTPException(409, detail=str(exc)) from exc
    except ProductRuntimeError as exc:
        raise _product_error(exc) from exc
    except (FileNotFoundError, PermissionError, ProductStoreError, ValueError) as exc:
        raise _not_found(manager_id) from exc


@router.get("/reports/{artifact_id}")
async def read_report(
    manager_id: str,
    artifact_id: Annotated[str, Path(min_length=1, max_length=256)],
    request: Request,
    response: Response,
) -> dict[str, str]:
    runtime = get_product_runtime(request)
    try:
        text = await runtime.echomind_chat.read_report(
            manager_id, artifact_id, request_context(request)
        )
    except ProductRuntimeError as exc:
        raise _product_error(exc) from exc
    except (FileNotFoundError, PermissionError, ProductStoreError, ValueError) as exc:
        raise _not_found(manager_id) from exc
    response.headers["Cache-Control"] = "no-store"
    return {"artifact_id": artifact_id, "text": text}


@router.get("/events")
async def chat_events(
    manager_id: str,
    request: Request,
    after_seq: Annotated[int | None, Query(alias="afterSeq", ge=0)] = None,
    last_event_id: Annotated[str | None, Header(alias="Last-Event-ID")] = None,
) -> StreamingResponse:
    runtime = get_product_runtime(request)
    chat_id = CHAT_PREFIX + manager_id
    cursor = _resolve_cursor(after_seq, last_event_id)
    try:
        context = await runtime.echomind_chat.route(
            manager_id, request_context(request)
        )
        await runtime.authorize_event_stream(chat_id, context)
    except ProductRuntimeError as exc:
        raise _product_error(exc) from exc
    except (FileNotFoundError, PermissionError, ProductStoreError, ValueError) as exc:
        raise _not_found(manager_id) from exc
    return StreamingResponse(
        _sse_stream(runtime, chat_id, cursor, context, None),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
