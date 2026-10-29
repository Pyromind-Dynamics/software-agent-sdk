import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast
from uuid import UUID

import pytest
from fastapi import HTTPException, Response, status
from pydantic import ValidationError
from starlette.requests import Request

from openhands.agent_server.agent_profiles import ECHOMIND_PROFILE, PYROMIND_PROFILE
from openhands.agent_server.conversation_service import ConversationService
from openhands.agent_server.echomind_projects import (
    PROJECT_REQUEST_RELATIVE_PATH,
    ProjectRequest,
    ProjectStore,
    read_project_request,
)
from openhands.agent_server.echomind_projects_router import (
    CreateEchoMindProjectRequest,
    create_echomind_project,
    run_echomind_project_node,
)
from openhands.agent_server.echomind_router import (
    EchoMindCreateConversationRequest,
    EchoMindSendMessageRequest,
    create_echomind_conversation,
    get_echomind_conversation,
    get_echomind_project_request,
    list_echomind_conversations,
    list_echomind_messages,
    send_echomind_message,
)
from openhands.agent_server.event_service import EventService
from openhands.agent_server.models import ConversationInfo, ConversationPage, EventPage
from openhands.agent_server.pyromind_auth import CurrentLoginUser
from openhands.agent_server.pyromind_constants import (
    PYROMIND_AGENT_PROFILE_TAG_KEY,
    PYROMIND_APP_TAG_KEY,
    PYROMIND_APP_TAG_VALUE,
)
from openhands.agent_server.pyromind_router import (
    PYROMIND_AUTH_TOKEN_SECRET,
    PYROMIND_PRODUCT_USER_ID_SECRET,
    PYROMIND_X_CLUSTER_SECRET,
    AgentLLMConfig,
    PyromindSendMessageRequest,
    _require_pyromind_tags,
    send_pyromind_message,
)
from openhands.sdk.conversation.request import StartConversationRequest
from openhands.sdk.conversation.state import ConversationExecutionStatus
from openhands.sdk.llm.message import Message
from openhands.tools.pyromind_dataset.definition import (
    PYROMIND_STORAGE_AUTH_COOKIE_SECRET,
    PYROMIND_STORAGE_HEADERS_STATE_KEY,
)
from openhands.tools.workflow.validate_workflow_dsl import (
    PYROMIND_VALIDATE_AUTH_COOKIE_SECRET,
    PYROMIND_VALIDATE_HEADERS_STATE_KEY,
)


def _valid_request_payload() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "name": "AVI SFT",
        "description": "Build a single-turn AVI SFT workflow.",
        "skill": "data-processing",
        "requirements": [
            {
                "key": "dataset",
                "label": "Dataset",
                "value": "avi-golden",
                "status": "confirmed",
            }
        ],
    }


def _avi_request_payload(*, train_mode: str = "sft") -> dict[str, Any]:
    values = {
        "dataset_uri": "datasets/avi/train.jsonl",
        "dataset_format": "jsonl with defect/diff/gt images",
        "image_roles": "defect,diff,gt",
        "label_schema": "pass,fail,defect",
        "split_policy": "batch-isolated train/val/test",
        "inference_model": "Qwen/Qwen3-VL-4B-Instruct",
        "train_mode": train_mode,
        "evaluation_rule": "result match; primary metric fail_recall",
        "artifact_goal": "LoRA checkpoint and ATIF evaluation report",
    }
    return {
        "schema_version": 1,
        "name": "PCB AVI single-turn",
        "description": (
            "Build the AVI Dataset -> Inference -> Evaluator -> Train graph."
        ),
        "skill": "avi-single-turn",
        "requirements": [
            {
                "key": key,
                "label": key.replace("_", " ").title(),
                "value": value,
                "status": "confirmed",
            }
            for key, value in values.items()
        ],
    }


def _write_request(working_dir: Path, payload: dict[str, Any]) -> None:
    path = working_dir / PROJECT_REQUEST_RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


class _FakeEchoMindEventService:
    def __init__(
        self,
        working_dir: Path,
        *,
        profile: str = ECHOMIND_PROFILE.tag_value,
    ) -> None:
        self.stored = _StoredConversation(
            id=UUID("00000000-0000-0000-0000-000000000001"),
            tags={
                PYROMIND_APP_TAG_KEY: PYROMIND_APP_TAG_VALUE,
                PYROMIND_AGENT_PROFILE_TAG_KEY: profile,
            },
        )
        self._conversation = _ConversationWithWorkspace(
            workspace=_Workspace(working_dir=str(working_dir))
        )
        self.sent_message: Message | None = None
        self.run: bool | None = None
        self.extended_content: list[Any] | None = None
        self.workflow_dsl_snapshot: str | None = None
        self.workflow_xyflow_snapshot: dict[str, Any] | None = None
        self.secrets: dict[str, Any] = {}
        self.agent_state: dict[str, Any] = {}
        self.event_page = EventPage(items=[])

    def get_conversation(self):
        return self._conversation

    async def send_message(
        self,
        message: Message,
        run: bool = False,
        _from_goal_loop: bool = False,
        extended_content: list[Any] | None = None,
        workflow_dsl_snapshot: str | None = None,
        workflow_xyflow_snapshot: dict[str, Any] | None = None,
    ) -> None:
        self.sent_message = message
        self.run = run
        self.extended_content = extended_content
        self.workflow_dsl_snapshot = workflow_dsl_snapshot
        self.workflow_xyflow_snapshot = workflow_xyflow_snapshot

    async def update_secrets(self, secrets: dict[str, Any]) -> None:
        self.secrets.update(secrets)

    async def update_agent_state(self, state: dict[str, Any]) -> None:
        self.agent_state.update(state)

    async def search_events(
        self,
        page_id: str | None = None,
        limit: int = 100,
        sort_order=None,
    ) -> EventPage:
        return self.event_page


@dataclass(frozen=True)
class _StoredConversation:
    id: UUID
    tags: dict[str, str]


@dataclass(frozen=True)
class _Workspace:
    working_dir: str


@dataclass(frozen=True)
class _ConversationWithWorkspace:
    workspace: _Workspace


class _FakeEchoMindConversationService:
    def __init__(self, conversations_dir: Path, working_dir: Path) -> None:
        self.conversations_dir = conversations_dir
        self.start_request: StartConversationRequest | None = None
        self.info: ConversationInfo | None = None
        self.event_service = _FakeEchoMindEventService(working_dir)

    async def start_conversation(
        self,
        request: StartConversationRequest,
    ) -> tuple[ConversationInfo, bool]:
        self.start_request = request
        assert request.conversation_id is not None
        self.info = ConversationInfo(
            id=request.conversation_id,
            agent=request.agent,
            workspace=request.workspace,
            execution_status=ConversationExecutionStatus.IDLE,
            tags=request.tags,
            title="EchoMind test",
        )
        return self.info, True

    async def get_event_service(
        self,
        conversation_id: UUID,
        user_id: str | None = None,
    ) -> EventService | None:
        assert self.info is not None
        assert conversation_id == self.info.id
        return cast(EventService, self.event_service)

    async def get_conversation(
        self,
        conversation_id: UUID,
        user_id: str | None = None,
    ) -> ConversationInfo | None:
        assert self.info is not None
        assert conversation_id == self.info.id
        return self.info

    async def search_conversations(
        self,
        page_id: str | None = None,
        limit: int = 100,
        execution_status=None,
        sort_order=None,
        user_id=None,
    ) -> ConversationPage:
        return ConversationPage(
            items=[self.info] if self.info is not None else [],
            next_page_id=None,
        )

    async def delete_conversation(
        self,
        conversation_id: UUID,
        user_id: str | None = None,
    ) -> bool:
        return True


def _request() -> Request:
    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/api/echomind/conversations",
            "headers": [],
        }
    )


@pytest.mark.asyncio
async def test_echomind_chat_has_independent_contract_and_tools(tmp_path):
    service = _FakeEchoMindConversationService(
        tmp_path / "conversations",
        tmp_path / "echomind-workspace",
    )
    response = Response()

    result = await create_echomind_conversation(
        _request(),
        EchoMindCreateConversationRequest(
            llm=AgentLLMConfig(model="gpt-4o", api_key="test-key"),
            message="我要做一个 AVI 图片质检训练流程",
        ),
        response,
        conversation_service=cast(ConversationService, service),
    )

    assert response.status_code == status.HTTP_201_CREATED
    assert result.agent_profile == "echomind"
    assert service.start_request is not None
    assert service.start_request.tags == {
        PYROMIND_APP_TAG_KEY: PYROMIND_APP_TAG_VALUE,
        PYROMIND_AGENT_PROFILE_TAG_KEY: ECHOMIND_PROFILE.tag_value,
    }
    assert "EchoMind Agent" in service.start_request.agent.static_system_message
    tool_names = {tool.name for tool in service.start_request.agent.tools}
    assert all(name in ECHOMIND_PROFILE.tool_allowlist for name in tool_names)
    assert "terminal" in tool_names
    assert "apply_patch" in tool_names
    assert "sandbox" in ECHOMIND_PROFILE.skill_allowlist
    assert {
        "sandbox_create",
        "sandbox_delete",
        "sandbox_read_file",
        "sandbox_write_file",
        "sandbox_delete_file",
        "sandbox_terminal",
    } <= tool_names
    assert {"skills_list", "skills_read"} <= tool_names
    assert service.start_request.agent.agent_context is not None
    assert {
        skill.name for skill in service.start_request.agent.agent_context.skills
    } == set(ECHOMIND_PROFILE.skill_allowlist)
    assert service.event_service.sent_message is not None
    assert service.event_service.workflow_xyflow_snapshot is None
    assert service.event_service.extended_content is not None

    with pytest.raises(ValidationError):
        EchoMindCreateConversationRequest.model_validate(
            {
                "llm": {"model": "gpt-4o", "api_key": "test-key"},
                "scenario_id": "avi-single-turn",
            }
        )


@pytest.mark.asyncio
async def test_echomind_and_pyromind_message_contracts_are_isolated(
    tmp_path,
):
    echomind_event = _FakeEchoMindEventService(tmp_path)
    await send_echomind_message(
        _request(),
        EchoMindSendMessageRequest(text="继续"),
        event_service=cast(EventService, echomind_event),
    )
    assert echomind_event.sent_message is not None

    with pytest.raises(HTTPException) as echomind_rejects_pyromind:
        await send_echomind_message(
            _request(),
            EchoMindSendMessageRequest(text="继续"),
            event_service=cast(
                EventService,
                _FakeEchoMindEventService(tmp_path, profile="pyromind"),
            ),
        )
    assert echomind_rejects_pyromind.value.status_code == status.HTTP_404_NOT_FOUND

    with pytest.raises(HTTPException) as pyromind_rejects_echomind:
        await send_pyromind_message(
            _request(),
            PyromindSendMessageRequest(text="继续"),
            event_service=cast(EventService, echomind_event),
        )
    assert pyromind_rejects_echomind.value.status_code == status.HTTP_404_NOT_FOUND

    with pytest.raises(HTTPException) as pyromind_rejects_unknown:
        await send_pyromind_message(
            _request(),
            PyromindSendMessageRequest(text="继续"),
            event_service=cast(
                EventService,
                _FakeEchoMindEventService(tmp_path, profile="unknown"),
            ),
        )
    assert pyromind_rejects_unknown.value.status_code == status.HTTP_404_NOT_FOUND


@pytest.mark.asyncio
async def test_echomind_read_contracts_restore_only_echomind_sessions(tmp_path):
    service = _FakeEchoMindConversationService(
        tmp_path / "conversations",
        tmp_path / "echomind-workspace",
    )
    created = await create_echomind_conversation(
        _request(),
        EchoMindCreateConversationRequest(
            llm=AgentLLMConfig(model="gpt-4o", api_key="test-key"),
        ),
        Response(),
        conversation_service=cast(ConversationService, service),
    )

    restored = await get_echomind_conversation(
        _request(),
        created.conversation_id,
        conversation_service=cast(ConversationService, service),
    )
    assert restored.conversation_id == created.conversation_id
    assert restored.agent_profile == "echomind"
    conversations = await list_echomind_conversations(
        _request(),
        conversation_service=cast(ConversationService, service),
    )
    assert [item.conversation_id for item in conversations] == [created.conversation_id]

    page = await list_echomind_messages(
        event_service=cast(EventService, service.event_service),
    )
    assert page == EventPage(items=[])

    pyromind_event_service = _FakeEchoMindEventService(
        tmp_path / "pyromind-workspace",
        profile=PYROMIND_PROFILE.tag_value,
    )
    with pytest.raises(HTTPException) as wrong_profile:
        await list_echomind_messages(
            event_service=cast(EventService, pyromind_event_service),
        )
    assert wrong_profile.value.status_code == status.HTTP_404_NOT_FOUND


def test_echomind_starts_with_all_pyromind_agent_capabilities():
    assert set(PYROMIND_PROFILE.skill_allowlist) < set(ECHOMIND_PROFILE.skill_allowlist)
    assert set(ECHOMIND_PROFILE.skill_allowlist) - set(
        PYROMIND_PROFILE.skill_allowlist
    ) == {"avi-single-turn"}
    assert ECHOMIND_PROFILE.tool_allowlist == PYROMIND_PROFILE.tool_allowlist


def test_echomind_request_accepts_shared_agent_configuration():
    request = EchoMindCreateConversationRequest.model_validate(
        {
            "llm": {"model": "gpt-4o", "api_key": "test-key"},
            "workflowXyflow": {"name": "Empty", "nodes": [], "edges": []},
            "extra": {"skills_path": "/tmp/skills"},
        }
    )

    assert request.workflow_xyflow == {"name": "Empty", "nodes": [], "edges": []}
    assert request.extra == {"skills_path": "/tmp/skills"}

    for forbidden_id in ("scenario_id", "scenario_run_id", "project_id", "run_id"):
        with pytest.raises(ValidationError):
            EchoMindCreateConversationRequest.model_validate(
                {
                    "llm": {"model": "gpt-4o", "api_key": "test-key"},
                    forbidden_id: "client-supplied",
                }
            )


@pytest.mark.asyncio
async def test_echomind_message_refreshes_platform_auth_context(tmp_path):
    event_service = _FakeEchoMindEventService(tmp_path)
    http_request = _request()
    http_request.state.current_user = CurrentLoginUser(
        username="debug-user-42",
        email="debug-user-42@example.test",
        user_id=42,
        cookie="auth_token=context-token; other=value",
        x_cluster="context-cluster",
    )

    await send_echomind_message(
        http_request,
        EchoMindSendMessageRequest(text="继续搭建 AVI 工作流"),
        event_service=cast(EventService, event_service),
    )

    assert event_service.secrets == {
        PYROMIND_VALIDATE_AUTH_COOKIE_SECRET: "auth_token=context-token; other=value",
        PYROMIND_AUTH_TOKEN_SECRET: "context-token",
        PYROMIND_STORAGE_AUTH_COOKIE_SECRET: "auth_token=context-token; other=value",
        PYROMIND_X_CLUSTER_SECRET: "context-cluster",
        PYROMIND_PRODUCT_USER_ID_SECRET: "42",
    }
    assert event_service.agent_state == {
        PYROMIND_VALIDATE_HEADERS_STATE_KEY: {"x-cluster": "context-cluster"},
        PYROMIND_STORAGE_HEADERS_STATE_KEY: {"x-cluster": "context-cluster"},
    }


def test_pyromind_profile_keeps_legacy_conversation_compatibility():
    _require_pyromind_tags({})
    _require_pyromind_tags({PYROMIND_AGENT_PROFILE_TAG_KEY: PYROMIND_PROFILE.tag_value})
    _require_pyromind_tags({PYROMIND_AGENT_PROFILE_TAG_KEY: "workbench"})

    with pytest.raises(HTTPException):
        _require_pyromind_tags(
            {PYROMIND_AGENT_PROFILE_TAG_KEY: ECHOMIND_PROFILE.tag_value}
        )


def test_project_request_rejects_graph_and_unresolved_requirements():
    with_graph = _valid_request_payload()
    with_graph["graph"] = {"nodes": [], "edges": []}
    with pytest.raises(ValidationError, match="graph"):
        ProjectRequest.model_validate(with_graph)

    incomplete = _valid_request_payload()
    incomplete["requirements"][0]["status"] = "missing"
    with pytest.raises(ValidationError, match="unresolved requirements"):
        ProjectRequest.model_validate(incomplete)


def test_avi_project_request_requires_all_scenario_requirements():
    incomplete = _avi_request_payload()
    incomplete["requirements"] = incomplete["requirements"][:-1]

    with pytest.raises(ValidationError, match="artifact_goal"):
        ProjectRequest.model_validate(incomplete)


@pytest.mark.asyncio
async def test_project_creation_requires_complete_echomind_request(tmp_path):
    conversations_dir = tmp_path / "conversations"
    working_dir = conversations_dir / "echomind"
    working_dir.mkdir(parents=True)
    event_service = _FakeEchoMindEventService(working_dir)
    conversation_id = event_service.stored.id
    service = type(
        "Service",
        (),
        {
            "conversations_dir": conversations_dir,
            "get_event_service": lambda _self, _conversation_id, user_id=None: (
                _async_value(event_service)
            ),
        },
    )()

    with pytest.raises(HTTPException) as missing:
        await create_echomind_project(
            _request(),
            CreateEchoMindProjectRequest(
                conversation_id=conversation_id,
                confirmed=True,
            ),
            conversation_service=cast(ConversationService, service),
        )
    assert missing.value.status_code == status.HTTP_409_CONFLICT

    invalid = _valid_request_payload()
    invalid["requirements"][0]["status"] = "missing"
    _write_request(working_dir, invalid)

    with pytest.raises(HTTPException) as invalid_response:
        await create_echomind_project(
            _request(),
            CreateEchoMindProjectRequest(
                conversation_id=conversation_id,
                confirmed=True,
            ),
            conversation_service=cast(ConversationService, service),
        )
    assert invalid_response.value.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT
    with pytest.raises(ValueError):
        read_project_request(working_dir)
    assert not (conversations_dir.parent / "projects").exists()


@pytest.mark.asyncio
async def test_project_creation_keeps_empty_graph_for_unregistered_skill(tmp_path):
    conversations_dir = tmp_path / "conversations"
    working_dir = conversations_dir / "echomind"
    working_dir.mkdir(parents=True)
    _write_request(working_dir, _valid_request_payload())
    project_request = await get_echomind_project_request(
        event_service=cast(
            EventService,
            _FakeEchoMindEventService(working_dir),
        )
    )
    assert project_request.skill == "data-processing"
    assert not hasattr(project_request, "graph")

    event_service = _FakeEchoMindEventService(working_dir)

    class _Service:
        def __init__(self) -> None:
            self.conversations_dir = conversations_dir

        async def get_event_service(
            self,
            conversation_id: UUID,
            user_id: str | None = None,
        ):
            assert conversation_id == event_service.stored.id
            return event_service

    service = _Service()

    project = await create_echomind_project(
        _request(),
        CreateEchoMindProjectRequest(
            conversation_id=event_service.stored.id,
            confirmed=True,
        ),
        conversation_service=cast(ConversationService, service),
    )

    UUID(project.project.project_id)
    UUID(project.graph.graph_id)
    assert project.graph.project_id == project.project.project_id
    assert project.graph.nodes == []
    assert project.graph.edges == []
    assert project.project.echomind_conversation_id == str(event_service.stored.id)

    store = ProjectStore.for_conversations_dir(conversations_dir)
    persisted = store.load(project.project.project_id, owner_user_id=None)
    assert persisted.graph.nodes == project.graph.nodes
    assert persisted.graph.edges == project.graph.edges


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("train_mode", "expected_title"), [("sft", "Train (SFT)"), ("dpo", "Train (DPO)")]
)
async def test_avi_project_creation_builds_stage_graph_and_executor_shell(
    tmp_path,
    train_mode,
    expected_title,
):
    conversations_dir = tmp_path / "conversations"
    working_dir = conversations_dir / "echomind"
    working_dir.mkdir(parents=True)
    _write_request(working_dir, _avi_request_payload(train_mode=train_mode))
    event_service = _FakeEchoMindEventService(working_dir)

    class _Service:
        def __init__(self) -> None:
            self.conversations_dir = conversations_dir

        async def get_event_service(
            self,
            conversation_id: UUID,
            user_id: str | None = None,
        ):
            assert conversation_id == event_service.stored.id
            return event_service

    service = _Service()
    result = await create_echomind_project(
        _request(),
        CreateEchoMindProjectRequest(
            conversation_id=event_service.stored.id,
            confirmed=True,
        ),
        conversation_service=cast(ConversationService, service),
    )

    assert result.graph.template == "avi-single-turn/v1"
    assert [node.id for node in result.graph.nodes] == [
        "dataset",
        "inference",
        "evaluator",
        "train",
    ]
    assert [node.type for node in result.graph.nodes] == [
        "dataset",
        "inference",
        "evaluator",
        "train",
    ]
    assert [(edge.source, edge.target) for edge in result.graph.edges] == [
        ("dataset", "inference"),
        ("inference", "evaluator"),
        ("evaluator", "train"),
    ]
    assert all(node.status == "idle" for node in result.graph.nodes)
    assert all(node.executor == "shell" for node in result.graph.nodes)
    train_node = next(node for node in result.graph.nodes if node.id == "train")
    assert train_node.title == expected_title
    assert train_node.config["method"] == train_mode

    execution = await run_echomind_project_node(
        result.project.project_id,
        "train",
        _request(),
        conversation_service=cast(ConversationService, service),
    )
    assert execution.status == "not_implemented"
    assert execution.executor == "shell"
    assert execution.node_id == "train"


async def _async_value(value):
    return value
