from __future__ import annotations

import logging
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Annotated, Literal
from uuid import UUID

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Query,
    Request,
    Response,
    status,
)
from pydantic import BaseModel, Field, SecretStr

from openhands.agent_server.agent_profiles import ECHOMIND_PROFILE
from openhands.agent_server.conversation_service import ConversationService
from openhands.agent_server.dependencies import (
    get_conversation_service,
    get_current_user_id,
    get_event_service,
)
from openhands.agent_server.echomind_projects import (
    ProjectRequest,
    ProjectRequestInvalidError,
    ProjectRequestNotFoundError,
    read_project_request,
)
from openhands.agent_server.event_service import EventService
from openhands.agent_server.models import (
    ConversationInfo,
    ConversationSortOrder,
    EventPage,
    EventSortOrder,
    Success,
)
from openhands.agent_server.pyromind_auth import (
    CurrentLoginUser,
    bind_debug_current_login_user_to_conversation,
    is_dev,
)
from openhands.agent_server.pyromind_constants import (
    PYROMIND_AGENT_PROFILE_TAG_KEY,
    PYROMIND_APP_TAG_KEY,
    PYROMIND_APP_TAG_VALUE,
)
from openhands.agent_server.pyromind_router import (
    _DEFAULT_KNOWLEDGE_BASE_PATH,
    PYROMIND_PRODUCT_USER_ID_SECRET,
    PyromindCreateConversationRequest,
    PyromindSendMessageRequest,
    _build_agent_llm,
    _build_agent_start_request,
    _build_agent_with_profile,
    _sync_workflow_with_canvas,
    _workflow_dsl_from_xyflow,
    _workflow_state_reminder,
    apply_pyromind_validation_context,
)
from openhands.sdk import LLM, Agent, TextContent
from openhands.sdk.conversation.state import ConversationExecutionStatus
from openhands.sdk.llm.message import Message
from openhands.sdk.secret import SecretSource, StaticSecret
from openhands.sdk.workspace import LocalWorkspace
from openhands.tools.preset.default import register_default_tools
from openhands.tools.workflow.definition import WORKFLOW_RELATIVE_PATH


logger = logging.getLogger(__name__)

echomind_router = APIRouter(prefix="/echomind", tags=["EchoMind"])


class EchoMindCreateConversationRequest(PyromindCreateConversationRequest):
    """Create EchoMind Chat with the full development capability contract."""


class EchoMindConversationInfo(BaseModel):
    conversation_id: UUID = Field(alias="conversationId")
    agent_profile: Literal["echomind"] = Field(default="echomind", alias="agentProfile")
    conversation: ConversationInfo

    model_config = {"populate_by_name": True}


class EchoMindConversationSummary(BaseModel):
    conversation_id: UUID = Field(alias="conversationId")
    agent_profile: Literal["echomind"] = Field(default="echomind", alias="agentProfile")
    title: str | None = None
    execution_status: ConversationExecutionStatus = Field(alias="executionStatus")
    created_at: datetime = Field(alias="createdAt")
    updated_at: datetime = Field(alias="updatedAt")

    model_config = {"populate_by_name": True}


class EchoMindSendMessageRequest(PyromindSendMessageRequest):
    """Send a message only through the independent EchoMind Chat contract."""


def _build_echomind_agent(
    http_request: Request,
    *,
    request: PyromindCreateConversationRequest,
    llm: LLM,
    conversation_dir: Path,
    knowledge_base_path: str,
) -> tuple[Agent, dict[str, SecretSource]]:
    return _build_agent_with_profile(
        http_request,
        request=request,
        profile=ECHOMIND_PROFILE,
        instructions_prefix=ECHOMIND_PROFILE.system_prompt,
        llm=llm,
        conversation_dir=conversation_dir,
        knowledge_base_path=knowledge_base_path,
    )


def _require_echomind_conversation(event_service: EventService) -> None:
    profile = event_service.stored.tags.get(PYROMIND_AGENT_PROFILE_TAG_KEY)
    if profile != ECHOMIND_PROFILE.tag_value:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="EchoMind conversation not found.",
        )


@echomind_router.get(
    "/conversations",
    response_model=list[EchoMindConversationSummary],
    response_model_by_alias=True,
)
async def list_echomind_conversations(
    http_request: Request,
    limit: Annotated[
        int,
        Query(title="The max number of conversations", gt=0, lte=100),
    ] = 100,
    sort_order: Annotated[
        ConversationSortOrder,
        Query(title="Sort order for conversations"),
    ] = ConversationSortOrder.UPDATED_AT_DESC,
    conversation_service: ConversationService = Depends(get_conversation_service),
) -> list[EchoMindConversationSummary]:
    """List only EchoMind conversations for the current user."""
    page = await conversation_service.search_conversations(
        limit=limit,
        sort_order=sort_order,
        user_id=get_current_user_id(http_request),
    )
    return [
        EchoMindConversationSummary(
            conversationId=conversation.id,
            agentProfile="echomind",
            title=conversation.title,
            executionStatus=conversation.execution_status,
            createdAt=conversation.created_at,
            updatedAt=conversation.updated_at,
        )
        for conversation in page.items
        if conversation.tags.get(PYROMIND_AGENT_PROFILE_TAG_KEY)
        == ECHOMIND_PROFILE.tag_value
    ]


@echomind_router.get(
    "/conversations/{conversation_id}",
    response_model=EchoMindConversationInfo,
    response_model_by_alias=True,
    responses={404: {"description": "EchoMind conversation not found"}},
)
async def get_echomind_conversation(
    http_request: Request,
    conversation_id: UUID,
    conversation_service: ConversationService = Depends(get_conversation_service),
) -> EchoMindConversationInfo:
    """Read an EchoMind conversation through its independent Chat contract."""
    user_id = get_current_user_id(http_request)
    event_service = await conversation_service.get_event_service(
        conversation_id,
        user_id=user_id,
    )
    if event_service is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="EchoMind conversation not found.",
        )
    _require_echomind_conversation(event_service)
    conversation = await conversation_service.get_conversation(
        conversation_id,
        user_id=user_id,
    )
    if conversation is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="EchoMind conversation not found.",
        )
    return EchoMindConversationInfo(
        conversationId=conversation.id,
        agentProfile="echomind",
        conversation=conversation,
    )


@echomind_router.post(
    "/conversations",
    response_model=EchoMindConversationInfo,
    response_model_by_alias=True,
)
async def create_echomind_conversation(
    http_request: Request,
    request: EchoMindCreateConversationRequest,
    response: Response,
    conversation_service: ConversationService = Depends(get_conversation_service),
) -> EchoMindConversationInfo:
    """Create an independent EchoMind Chat with no Project or Run lifecycle."""
    import openhands.tools.data_preparation.definition  # noqa: F401
    import openhands.tools.data_preparation.platform_submit  # noqa: F401

    register_default_tools(enable_browser=False)

    knowledge_base_path = request.extra.get(
        "knowledge_base_path", _DEFAULT_KNOWLEDGE_BASE_PATH
    )
    knowledge_base_path = os.path.abspath(knowledge_base_path)
    if not os.path.isdir(knowledge_base_path):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Knowledge base path does not exist: {knowledge_base_path}",
        )

    user_id = get_current_user_id(http_request)
    conversation_id = uuid.uuid4()
    conversation_dir = conversation_service.conversations_dir / conversation_id.hex
    conversation_dir.mkdir(parents=True, exist_ok=True)
    conversation_dir.chmod(0o700)
    (conversation_dir / "public_data").mkdir(mode=0o700, exist_ok=True)

    llm = _build_agent_llm(request.llm, usage_id=ECHOMIND_PROFILE.usage_id)
    agent, agent_secrets = _build_echomind_agent(
        http_request,
        request=request,
        llm=llm,
        conversation_dir=conversation_dir,
        knowledge_base_path=knowledge_base_path,
    )
    workspace = LocalWorkspace(working_dir=str(conversation_dir))

    workflow_dsl = _workflow_dsl_from_xyflow(request.workflow_xyflow)
    if workflow_dsl:
        workflow_file = conversation_dir / WORKFLOW_RELATIVE_PATH
        workflow_file.parent.mkdir(parents=True, exist_ok=True)
        workflow_file.write_text(workflow_dsl, encoding="utf-8")

    start_request = _build_agent_start_request(
        agent=agent,
        workspace=workspace,
        conversation_id=conversation_id,
        secrets={
            **agent_secrets,
            **(
                {
                    PYROMIND_PRODUCT_USER_ID_SECRET: StaticSecret(
                        value=SecretStr(user_id)
                    )
                }
                if user_id is not None
                else {}
            ),
        },
        tags={
            PYROMIND_APP_TAG_KEY: PYROMIND_APP_TAG_VALUE,
            PYROMIND_AGENT_PROFILE_TAG_KEY: ECHOMIND_PROFILE.tag_value,
        },
        user_id=user_id,
    )

    info: ConversationInfo | None = None
    is_new = False
    try:
        info, is_new = await conversation_service.start_conversation(start_request)
        current_user = getattr(http_request.state, "current_user", None)
        if is_dev() and isinstance(current_user, CurrentLoginUser):
            bind_debug_current_login_user_to_conversation(info.id, current_user)

        if is_new and request.message:
            event_service = await conversation_service.get_event_service(
                info.id,
                user_id=user_id,
            )
            if event_service is None:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Conversation event service not found: {info.id}",
                )
            initial_reminder = _workflow_state_reminder(conversation_dir)
            await event_service.send_message(
                Message(role="user", content=[TextContent(text=request.message)]),
                run=True,
                extended_content=[initial_reminder],
                workflow_dsl_snapshot=workflow_dsl,
                workflow_xyflow_snapshot=request.workflow_xyflow,
            )
            refreshed = await conversation_service.get_conversation(
                info.id,
                user_id=user_id,
            )
            if refreshed is not None:
                info = refreshed
    except Exception:
        if is_new and info is not None:
            try:
                await conversation_service.delete_conversation(
                    info.id,
                    user_id=user_id,
                )
            except Exception:
                logger.warning(
                    "Failed to roll back EchoMind conversation %s",
                    info.id,
                    exc_info=True,
                )
        raise

    assert info is not None
    response.status_code = status.HTTP_201_CREATED if is_new else status.HTTP_200_OK
    return EchoMindConversationInfo(
        conversationId=info.id,
        agentProfile="echomind",
        conversation=info,
    )


@echomind_router.get(
    "/conversations/{conversation_id}/messages",
    response_model=EventPage,
    responses={404: {"description": "EchoMind conversation not found"}},
)
async def list_echomind_messages(
    page_id: Annotated[
        str | None,
        Query(title="Optional next_page_id from the previously returned page"),
    ] = None,
    limit: Annotated[
        int,
        Query(title="The max number of results in the page", gt=0, lte=100),
    ] = 100,
    sort_order: Annotated[
        EventSortOrder,
        Query(title="Sort order for events"),
    ] = EventSortOrder.TIMESTAMP,
    event_service: EventService = Depends(get_event_service),
) -> EventPage:
    """Read persisted messages and tool events for one EchoMind Chat."""
    _require_echomind_conversation(event_service)
    return await event_service.search_events(
        page_id=page_id,
        limit=limit,
        sort_order=sort_order,
    )


@echomind_router.post(
    "/conversations/{conversation_id}/messages",
    response_model=Success,
    responses={404: {"description": "EchoMind conversation not found"}},
)
async def send_echomind_message(
    http_request: Request,
    request: EchoMindSendMessageRequest,
    event_service: EventService = Depends(get_event_service),
) -> Success:
    """Send a message only through the EchoMind Chat contract."""
    _require_echomind_conversation(event_service)
    current_user = getattr(http_request.state, "current_user", None)
    if isinstance(current_user, CurrentLoginUser):
        await apply_pyromind_validation_context(event_service, current_user)

    conversation = event_service.get_conversation()
    working_dir = Path(conversation.workspace.working_dir)
    workflow_dsl = _workflow_dsl_from_xyflow(request.workflow_xyflow)
    reminder = _sync_workflow_with_canvas(working_dir, workflow_dsl)

    await event_service.send_message(
        Message(role="user", content=[TextContent(text=request.text)]),
        run=request.run,
        extended_content=[reminder],
        workflow_dsl_snapshot=workflow_dsl,
        workflow_xyflow_snapshot=request.workflow_xyflow,
    )
    return Success()


@echomind_router.get(
    "/conversations/{conversation_id}/project-request",
    response_model=ProjectRequest,
    responses={404: {"description": "EchoMind project request not found"}},
)
async def get_echomind_project_request(
    event_service: EventService = Depends(get_event_service),
) -> ProjectRequest:
    """Return the server-validated request written by the EchoMind Agent."""
    _require_echomind_conversation(event_service)
    conversation = event_service.get_conversation()
    try:
        return read_project_request(Path(conversation.workspace.working_dir))
    except ProjectRequestNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="EchoMind project request does not exist yet.",
        ) from exc
    except ProjectRequestInvalidError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"EchoMind project request is invalid: {exc}",
        ) from exc
