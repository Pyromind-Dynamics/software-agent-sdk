from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from dataclasses import replace
from typing import Annotated
from uuid import uuid4

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from pyromind_runtime.application.conversation_runtime import ConversationRuntime
from pyromind_runtime.domain.commands import CommandReceipt, ProductCommand
from pyromind_runtime.domain.content import TextContent
from pyromind_runtime.domain.errors import ProductRuntimeError, UserInputValidationError
from pyromind_runtime.domain.snapshot import ConversationSnapshot
from pyromind_runtime.infrastructure.file_product_store import (
    CommandConflictError,
    ProductStoreError,
)
from pyromind_runtime.ports.harness import SessionSpec

from openhands.agent_server.conversation_service import ConversationService
from openhands.agent_server.dependencies import check_session_api_key
from pyromind_agent_server.api.schemas import (
    CreateConversationRequest,
    ForkConversationRequest,
)
from pyromind_agent_server.auth_context import request_context
from pyromind_agent_server.bootstrap import ensure_product_runtime
from pyromind_agent_server.profiles import DEFAULT_PRODUCT_PROFILE, ConversationProfile


_SSE_HEARTBEAT_SECONDS = 15.0


class PrepareWorkflowExecutionBody(BaseModel):
    workflow_path: str = Field(min_length=1)
    request_id: str | None = None


def get_product_runtime(request: Request) -> ConversationRuntime:
    runtime = ensure_product_runtime(request.app)
    if runtime is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Product runtime is not available",
        )
    return runtime


def get_conversation_service(request: Request) -> ConversationService:
    service = getattr(request.app.state, "conversation_service", None)
    if not isinstance(service, ConversationService):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Conversation service is not available",
        )
    return service


def create_product_router(
    profile: ConversationProfile = DEFAULT_PRODUCT_PROFILE,
) -> APIRouter:
    def context_for(request: Request):
        return replace(request_context(request), profile_id=profile.profile_id)

    router = APIRouter(
        prefix=profile.route_prefix,
        tags=["Product API"],
        dependencies=[Depends(check_session_api_key)],
    )

    @router.post("/{conversation_id}/workflow-executions")
    async def prepare_workflow_execution(
        conversation_id: str,
        body: PrepareWorkflowExecutionBody,
        request: Request,
    ) -> dict:
        try:
            return await get_product_runtime(request).prepare_workflow_execution(
                conversation_id,
                body.workflow_path,
                context_for(request),
                request_id=body.request_id,
            )
        except (FileNotFoundError, PermissionError, ProductStoreError) as exc:
            raise _not_found(conversation_id) from exc
        except ValueError as exc:
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
            ) from exc

    @router.post("", response_model=ConversationSnapshot, status_code=201)
    async def create_conversation(
        body: CreateConversationRequest,
        request: Request,
    ) -> ConversationSnapshot:
        context = context_for(request)
        runtime = get_product_runtime(request)
        if profile.tool_allowlist and "get_experiment_plan" in profile.tool_allowlist:
            if not runtime.experiment_plans_available:
                raise HTTPException(503, detail="Plan database is not configured")
            if body.workflow_xyflow is not None:
                raise HTTPException(422, detail="Workflow execution is not enabled")
        conversation_id = uuid4().hex
        conversations_dir = get_conversation_service(request).conversations_dir
        spec = SessionSpec(
            conversation_id=conversation_id,
            user_id=context.user_id,
            workspace_root=str(conversations_dir / conversation_id),
            initial_message=((TextContent(text=body.message),) if body.message else ()),
            workflow_xyflow=body.workflow_xyflow,
            model_configuration=body.llm.model_dump(mode="json"),
            user_input_enabled=body.user_input_enabled,
            extra=body.extra,
            profile_id=profile.profile_id,
            harness_id=profile.harness_id,
            product_instructions=profile.instructions,
            tool_allowlist=profile.tool_allowlist,
            skill_allowlist=profile.skill_allowlist,
            command_allowlist=profile.command_allowlist,
        )
        try:
            return await get_product_runtime(request).create_conversation(spec, context)
        except ProductRuntimeError as exc:
            raise _product_error(exc) from exc

    @router.get("", response_model=tuple[ConversationSnapshot, ...])
    async def list_conversations(request: Request) -> tuple[ConversationSnapshot, ...]:
        return await get_product_runtime(request).list_snapshots(context_for(request))

    @router.get("/{conversation_id}/snapshot", response_model=ConversationSnapshot)
    async def get_snapshot(
        conversation_id: str,
        request: Request,
    ) -> ConversationSnapshot:
        try:
            return await get_product_runtime(request).get_snapshot(
                conversation_id, context_for(request)
            )
        except ProductRuntimeError as exc:
            raise _product_error(exc) from exc
        except (
            FileNotFoundError,
            PermissionError,
            ProductStoreError,
            ValueError,
        ) as exc:
            raise _not_found(conversation_id) from exc

    @router.post(
        "/{conversation_id}/commands",
        response_model=CommandReceipt,
        status_code=status.HTTP_202_ACCEPTED,
    )
    async def submit_command(
        conversation_id: str,
        command: ProductCommand,
        request: Request,
    ) -> CommandReceipt:
        try:
            return await get_product_runtime(request).submit_command(
                conversation_id, command, context_for(request)
            )
        except CommandConflictError as exc:
            raise HTTPException(
                status.HTTP_409_CONFLICT,
                detail={"code": "command_conflict", "message": str(exc)},
            ) from exc
        except ProductRuntimeError as exc:
            raise _product_error(exc) from exc
        except UserInputValidationError as exc:
            # Invalid answers return 422 and leave the question pending.
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={"code": "invalid_input", "message": str(exc)},
            ) from exc
        except (
            FileNotFoundError,
            PermissionError,
            ProductStoreError,
        ) as exc:
            raise _not_found(conversation_id) from exc
        except ValueError as exc:
            raise HTTPException(status.HTTP_409_CONFLICT, detail=str(exc)) from exc

    @router.get("/{conversation_id}/events")
    async def stream_events(
        conversation_id: str,
        request: Request,
        after_seq: Annotated[int | None, Query(alias="afterSeq", ge=0)] = None,
        last_event_id: Annotated[str | None, Header(alias="Last-Event-ID")] = None,
    ) -> StreamingResponse:
        cursor = _resolve_cursor(after_seq, last_event_id)
        context = context_for(request)
        runtime = get_product_runtime(request)
        try:
            await runtime.get_snapshot(conversation_id, context)
        except ProductRuntimeError as exc:
            raise _product_error(exc) from exc
        except (
            FileNotFoundError,
            PermissionError,
            ProductStoreError,
            ValueError,
        ) as exc:
            raise _not_found(conversation_id) from exc
        return StreamingResponse(
            _sse_stream(runtime, conversation_id, cursor, context),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @router.get("/{conversation_id}/experiment-plans")
    async def get_experiment_plan(
        conversation_id: str,
        request: Request,
        revision: Annotated[int | None, Query(ge=1)] = None,
    ):
        try:
            return await get_product_runtime(request).get_experiment_plan(
                conversation_id, context_for(request), revision
            )
        except ProductRuntimeError as exc:
            raise _product_error(exc) from exc
        except (FileNotFoundError, PermissionError, ProductStoreError) as exc:
            raise _not_found(conversation_id) from exc
        except ValueError as exc:
            raise HTTPException(409, detail=str(exc)) from exc

    @router.post(
        "/{conversation_id}/forks",
        response_model=ConversationSnapshot,
        status_code=status.HTTP_201_CREATED,
    )
    async def fork_conversation(
        conversation_id: str,
        body: ForkConversationRequest,
        request: Request,
    ) -> ConversationSnapshot:
        context = context_for(request)
        try:
            return await get_product_runtime(request).fork_conversation(
                conversation_id,
                event_id=body.event_id,
                title=body.title,
                context=context,
            )
        except ProductRuntimeError as exc:
            raise _product_error(exc) from exc
        except (
            FileNotFoundError,
            PermissionError,
            ProductStoreError,
        ) as exc:
            raise _not_found(conversation_id) from exc

    return router


def _not_found(conversation_id: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Conversation not found: {conversation_id}",
    )


def _product_error(error: ProductRuntimeError) -> HTTPException:
    status_code = {
        "plan_store_unavailable": status.HTTP_503_SERVICE_UNAVAILABLE,
        "conversation_not_found": status.HTTP_404_NOT_FOUND,
        "checkpoint_not_found": status.HTTP_404_NOT_FOUND,
        "capacity_exceeded": status.HTTP_429_TOO_MANY_REQUESTS,
        "harness_operation_failed": status.HTTP_502_BAD_GATEWAY,
        "input_not_pending": status.HTTP_409_CONFLICT,
        "input_pending": status.HTTP_409_CONFLICT,
    }.get(error.code, status.HTTP_409_CONFLICT)
    return HTTPException(
        status_code=status_code,
        detail={"code": error.code, "message": error.message},
    )


def _resolve_cursor(after_seq: int | None, last_event_id: str | None) -> int:
    if after_seq is not None:
        return after_seq
    if last_event_id is None or not last_event_id.strip():
        return 0
    try:
        cursor = int(last_event_id)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Last-Event-ID must be a non-negative integer",
        ) from exc
    if cursor < 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Last-Event-ID must be a non-negative integer",
        )
    return cursor


async def _sse_stream(
    runtime: ConversationRuntime,
    conversation_id: str,
    after_seq: int,
    context,
) -> AsyncGenerator[str]:
    events = runtime.stream_events(conversation_id, after_seq, context)
    # Flush the HTTP response immediately, even when the cursor is already at
    # the latest persisted event. This lets clients establish the SSE channel
    # before submitting the first command without waiting for a heartbeat.
    yield ": connected\n\n"
    pending = asyncio.ensure_future(anext(events))
    try:
        while True:
            done, _ = await asyncio.wait({pending}, timeout=_SSE_HEARTBEAT_SECONDS)
            if not done:
                yield ": heartbeat\n\n"
                continue
            try:
                event = pending.result()
            except StopAsyncIteration:
                return
            yield (
                f"id: {event.seq}\n"
                f"event: {event.type}\n"
                f"data: {event.model_dump_json()}\n\n"
            )
            pending = asyncio.ensure_future(anext(events))
    finally:
        pending.cancel()
        await asyncio.gather(pending, return_exceptions=True)
        await events.aclose()
