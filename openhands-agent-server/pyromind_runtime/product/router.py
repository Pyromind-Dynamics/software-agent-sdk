from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Callable
from typing import Annotated

from fastapi import APIRouter, Header, HTTPException, Query, Request, status
from fastapi.responses import StreamingResponse
from pydantic import Field

from pyromind_runtime.contracts.base import ContractModel
from pyromind_runtime.contracts.harness import CapabilityName
from pyromind_runtime.contracts.sandbox import ModelProfile, SandboxRef, WorkspaceRef
from pyromind_runtime.product.commands import ProductCommand
from pyromind_runtime.product.event_store import (
    CommandReceiptConflictError,
    ConversationNotFoundError,
)
from pyromind_runtime.product.models import (
    CommandReceipt,
    ConversationSnapshot,
)
from pyromind_runtime.product.registry import HarnessNotRegisteredError
from pyromind_runtime.product.runtime import (
    CapabilityNotSupportedError,
    ProductConversationNotActiveError,
    ProductRuntimeError,
    ProductRuntimeService,
)
from pyromind_runtime.tool_host import ToolRequestContext


_SSE_HEARTBEAT_SECONDS = 15.0


class CreateConversationRequest(ContractModel):
    workspace: WorkspaceRef | None = None
    sandbox: SandboxRef | None = None
    model_profile: ModelProfile | None = None
    required_capabilities: frozenset[CapabilityName] = frozenset()


class ForkConversationRequest(ContractModel):
    after_seq: int | None = Field(default=None, ge=0)


def get_product_runtime(request: Request) -> ProductRuntimeService:
    runtime: ProductRuntimeService = request.app.state.product_runtime
    return runtime


def create_product_router(
    resolve_user_id: Callable[[Request], str | None],
    resolve_tool_context: Callable[[Request], ToolRequestContext | None] | None = None,
) -> APIRouter:
    router = APIRouter(prefix="/v2/pyromind/conversations", tags=["Pyromind v2"])

    @router.post(
        "",
        response_model=ConversationSnapshot,
        status_code=status.HTTP_201_CREATED,
    )
    async def create_conversation(
        body: CreateConversationRequest,
        request: Request,
    ) -> ConversationSnapshot:
        runtime = get_product_runtime(request)
        try:
            return await runtime.create_conversation(
                user_id=resolve_user_id(request) or "anonymous",
                workspace=body.workspace,
                sandbox=body.sandbox,
                model_profile=body.model_profile,
                required_capabilities=body.required_capabilities,
                tool_context=(
                    resolve_tool_context(request)
                    if resolve_tool_context is not None
                    else None
                ),
            )
        except HarnessNotRegisteredError as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Harness is not registered: {exc}",
            ) from exc
        except CapabilityNotSupportedError as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=str(exc),
            ) from exc

    @router.get("", response_model=tuple[ConversationSnapshot, ...])
    async def list_conversations(request: Request) -> tuple[ConversationSnapshot, ...]:
        return await get_product_runtime(request).list_conversations(
            resolve_user_id(request) or "anonymous"
        )

    @router.get("/{conversation_id}/snapshot", response_model=ConversationSnapshot)
    async def get_snapshot(
        conversation_id: str,
        request: Request,
    ) -> ConversationSnapshot:
        try:
            return await get_product_runtime(request).get_snapshot(
                conversation_id,
                resolve_user_id(request) or "anonymous",
            )
        except ConversationNotFoundError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Conversation not found: {conversation_id}",
            ) from exc

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
                conversation_id,
                command,
                resolve_user_id(request) or "anonymous",
                tool_context=(
                    resolve_tool_context(request)
                    if resolve_tool_context is not None
                    else None
                ),
            )
        except ConversationNotFoundError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Conversation not found: {conversation_id}",
            ) from exc
        except (
            CapabilityNotSupportedError,
            CommandReceiptConflictError,
            ProductConversationNotActiveError,
        ) as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=str(exc),
            ) from exc

    @router.get("/{conversation_id}/events")
    async def stream_events(
        conversation_id: str,
        request: Request,
        after_seq: Annotated[int | None, Query(alias="afterSeq", ge=0)] = None,
        last_event_id: Annotated[
            str | None,
            Header(alias="Last-Event-ID"),
        ] = None,
    ) -> StreamingResponse:
        cursor = _resolve_cursor(after_seq, last_event_id)
        user_id = resolve_user_id(request) or "anonymous"
        try:
            await get_product_runtime(request).get_snapshot(conversation_id, user_id)
        except ConversationNotFoundError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Conversation not found: {conversation_id}",
            ) from exc
        return StreamingResponse(
            _sse_stream(
                get_product_runtime(request),
                conversation_id,
                cursor,
                user_id,
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

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
        try:
            return await get_product_runtime(request).fork_conversation(
                conversation_id,
                resolve_user_id(request) or "anonymous",
                after_seq=body.after_seq,
            )
        except ConversationNotFoundError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Conversation not found: {conversation_id}",
            ) from exc
        except (CapabilityNotSupportedError, ProductRuntimeError) as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=str(exc),
            ) from exc

    return router


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
    runtime: ProductRuntimeService,
    conversation_id: str,
    after_seq: int,
    user_id: str,
) -> AsyncGenerator[str]:
    events = runtime.stream_events(conversation_id, after_seq, user_id)
    pending = asyncio.ensure_future(anext(events))
    try:
        while True:
            done, _ = await asyncio.wait(
                {pending},
                timeout=_SSE_HEARTBEAT_SECONDS,
            )
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
