import json
import logging
import re
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock

from pydantic import SecretStr

from openhands.agent_server.pyromind_subagent import (
    GENERAL_PURPOSE_AGENT_NAME,
    GENERAL_PURPOSE_AGENT_PROMPT,
    SEARCH_AGENT_NAME,
    SEARCH_AGENT_PROMPT,
    SUBAGENT_TOOL_DESCRIPTION,
    KnowledgeBaseGrepAction,
    KnowledgeBaseGrepObservation,
    KnowledgeBaseGrepTool,
    KnowledgeBaseReadAction,
    KnowledgeBaseReadObservation,
    KnowledgeBaseReadTool,
    PyromindSubAgentTool,
    SubAgentAction,
    SubAgentExecutor,
    SubAgentObservation,
    SubAgentType,
    _general_purpose_agent_factory,
    _search_agent_factory,
    _subagent_factories,
    configure_subagents,
)
from openhands.sdk import Agent, Conversation, Tool
from openhands.sdk.conversation.impl.local_conversation import LocalConversation
from openhands.sdk.conversation.state import (
    ConversationExecutionStatus,
    ConversationState,
)
from openhands.sdk.event.llm_convertible.observation import ObservationEvent
from openhands.sdk.llm import LLM, Message, MessageToolCall, TextContent
from openhands.sdk.testing import TestLLM
from openhands.tools.file_editor import FileEditorObservation, FileEditorTool
from openhands.tools.task.manager import TaskStatus


def _conversation_state(workspace: Path) -> ConversationState:
    return cast(
        ConversationState,
        SimpleNamespace(workspace=SimpleNamespace(working_dir=str(workspace))),
    )


def _tool_call(call_id: str, name: str, arguments: dict[str, object]) -> Message:
    return Message(
        role="assistant",
        content=[TextContent(text="")],
        tool_calls=[
            MessageToolCall(
                id=call_id,
                name=name,
                arguments=json.dumps(arguments),
                origin="completion",
            )
        ],
    )


def _text_message(text: str) -> Message:
    return Message(role="assistant", content=[TextContent(text=text)])


def test_subagent_description_routes_intermediate_knowledge_lookups_to_search():
    assert "including an intermediate lookup inside" in SUBAGENT_TOOL_DESCRIPTION
    assert "search, grep, or read" in SUBAGENT_TOOL_DESCRIPTION


def test_knowledge_index_covers_every_searchable_document():
    knowledge = Path(__file__).parents[2] / "knowledge"
    index = (knowledge / "index.md").read_text(encoding="utf-8")
    linked_paths = set(re.findall(r"\]\(([^)]+)\)", index))
    searchable_paths = {
        path.relative_to(knowledge).as_posix()
        for path in knowledge.rglob("*")
        if path.is_file()
        and path.suffix in {".md", ".mdx", ".py"}
        and path.name not in {"README.md", "index.md"}
    }

    assert linked_paths == searchable_paths


def test_knowledge_grep_returns_logical_paths_and_rejects_traversal(tmp_path, caplog):
    caplog.set_level(logging.INFO)
    workspace = tmp_path / "conversation"
    workspace.mkdir()
    knowledge = tmp_path / "knowledge"
    (knowledge / "studio").mkdir(parents=True)
    (knowledge / "studio" / "dpo-training.mdx").write_text(
        "---\ntitle: DPO 训练\n---\n## 关键参数\nbeta 控制偏好强度\n",
        encoding="utf-8",
    )
    outside = tmp_path / "outside.mdx"
    outside.write_text("secret outside content", encoding="utf-8")
    (knowledge / "studio" / "outside-link.mdx").symlink_to(outside)
    configure_subagents(workspace, knowledge)

    tool = KnowledgeBaseGrepTool.create(_conversation_state(workspace))[0]
    result = tool(
        KnowledgeBaseGrepAction(
            pattern="DPO|beta",
            path="knowledge/studio",
            include="*.mdx",
        )
    )

    assert isinstance(result, KnowledgeBaseGrepObservation)
    assert not result.is_error
    assert [match.path for match in result.matches] == [
        "knowledge/studio/dpo-training.mdx",
        "knowledge/studio/dpo-training.mdx",
    ]
    assert str(knowledge) not in result.text

    escaped = tool(
        KnowledgeBaseGrepAction(pattern="secret", path="knowledge/../outside")
    )
    assert escaped.is_error
    assert "escapes the knowledge base" in escaped.text

    symlinked = tool(KnowledgeBaseGrepAction(pattern="secret", path="knowledge"))
    assert isinstance(symlinked, KnowledgeBaseGrepObservation)
    assert not symlinked.is_error
    assert symlinked.matches == []
    assert "event=query_started" in caplog.text
    assert "path='knowledge/studio'" in caplog.text
    assert "pattern='DPO|beta'" in caplog.text
    assert "event=query_completed" in caplog.text
    assert "matches=2" in caplog.text
    assert "event=query_failed" in caplog.text
    assert str(knowledge) not in caplog.text


def test_knowledge_read_opens_only_logical_knowledge_documents(tmp_path, caplog):
    caplog.set_level(logging.INFO)
    workspace = tmp_path / "conversation"
    workspace.mkdir()
    knowledge = tmp_path / "knowledge"
    knowledge.mkdir()
    document = knowledge / "guide.mdx"
    document.write_text("# Guide\nfirst\nsecond\nthird\n", encoding="utf-8")
    configure_subagents(workspace, knowledge)

    tool = KnowledgeBaseReadTool.create(_conversation_state(workspace))[0]
    result = tool(
        KnowledgeBaseReadAction(
            path="knowledge/guide.mdx",
            start_line=2,
            end_line=3,
        )
    )

    assert isinstance(result, KnowledgeBaseReadObservation)
    assert not result.is_error
    assert result.path == "knowledge/guide.mdx"
    assert result.start_line == 2
    assert result.end_line == 3
    assert "first" in result.text
    assert "second" in result.text
    assert "third" not in result.text
    assert str(document) not in result.text

    absolute = tool(KnowledgeBaseReadAction(path=str(document)))
    assert absolute.is_error
    assert "logical `knowledge/` root" in absolute.text
    assert "event=read_started" in caplog.text
    assert "path='knowledge/guide.mdx'" in caplog.text
    assert "start_line=2 end_line=3" in caplog.text
    assert "event=read_completed" in caplog.text
    assert "lines=2-3 total_lines=4" in caplog.text
    assert "event=read_failed" in caplog.text
    assert str(document) not in caplog.text


def test_search_agent_has_only_read_only_knowledge_tools():
    factory = _search_agent_factory()
    llm = LLM(model="gpt-4o", api_key=SecretStr("test-key"))
    agent = factory.factory_func(llm)

    assert factory.definition.name == SEARCH_AGENT_NAME
    assert factory.definition.permission_mode == "never_confirm"
    assert factory.definition.max_iteration_per_run == 12
    assert [tool.name for tool in agent.tools] == [
        KnowledgeBaseGrepTool.name,
        KnowledgeBaseReadTool.name,
    ]
    assert agent.agent_context is not None
    assert agent.agent_context.system_message_suffix == SEARCH_AGENT_PROMPT


def test_general_purpose_agent_has_workspace_read_write_tools():
    factory = _general_purpose_agent_factory()
    llm = LLM(model="gpt-4o", api_key=SecretStr("test-key"))
    agent = factory.factory_func(llm)

    assert factory.definition.name == GENERAL_PURPOSE_AGENT_NAME
    assert factory.definition.permission_mode is None
    assert factory.definition.max_iteration_per_run == 30
    assert [tool.name for tool in agent.tools] == [
        "grep",
        "file_editor",
        "terminal",
        "task_tracker",
    ]
    assert agent.agent_context is not None
    assert agent.agent_context.system_message_suffix == GENERAL_PURPOSE_AGENT_PROMPT


def test_subagent_executor_selects_conversation_scoped_profile(caplog):
    caplog.set_level(logging.INFO)
    manager = MagicMock()
    manager.start_task_with_factory.return_value = SimpleNamespace(
        id="task_00000001",
        conversation_id="child-conversation",
        status=TaskStatus.COMPLETED,
        result="DPO uses beta.\nSource: knowledge/studio/dpo-training.mdx",
        error=None,
    )
    executor = SubAgentExecutor(manager, _subagent_factories())
    parent = MagicMock(spec=LocalConversation)
    parent.state.id = "parent-conversation"

    result = executor(
        SubAgentAction(
            type=SubAgentType.SEARCH,
            task="DPO 有哪些关键参数？",
        ),
        conversation=parent,
    )

    assert not result.is_error
    assert result.task_id == "task_00000001"
    assert result.type == SubAgentType.SEARCH
    assert result.child_conversation_id == "child-conversation"
    assert "knowledge/studio/dpo-training.mdx" in result.text
    call = manager.start_task_with_factory.call_args.kwargs
    assert call["prompt"] == "DPO 有哪些关键参数？"
    assert call["factory"].definition.name == SEARCH_AGENT_NAME
    assert call["conversation"] is parent
    assert "event=delegation_started" in caplog.text
    assert "parent_conversation_id=parent-conversation" in caplog.text
    assert "type=search" in caplog.text
    assert "task='DPO 有哪些关键参数？'" in caplog.text
    assert "event=delegation_completed" in caplog.text
    assert "task_id=task_00000001" in caplog.text


def test_search_subagent_reads_index_then_source_end_to_end(tmp_path, caplog):
    caplog.set_level(logging.INFO)
    workspace = tmp_path / "conversation"
    workspace.mkdir()
    knowledge = tmp_path / "knowledge"
    (knowledge / "studio").mkdir(parents=True)
    (knowledge / "studio" / "dpo-training.mdx").write_text(
        "---\ntitle: DPO 训练\n---\n## 关键参数\nbeta 控制偏好强度\n",
        encoding="utf-8",
    )
    (knowledge / "index.md").write_text(
        "# Knowledge Index\n\n"
        "| Page | Summary | Tags |\n"
        "|---|---|---|\n"
        "| [DPO 训练](studio/dpo-training.mdx) | DPO 参数说明 | dpo, beta |\n",
        encoding="utf-8",
    )
    configure_subagents(workspace, knowledge)

    llm = TestLLM.from_messages(
        [
            _tool_call(
                "parent-subagent",
                PyromindSubAgentTool.name,
                {
                    "type": "search",
                    "task": "DPO 的 beta 参数有什么作用？",
                },
            ),
            _tool_call(
                "child-index",
                KnowledgeBaseReadTool.name,
                {"path": "knowledge/index.md"},
            ),
            _tool_call(
                "child-read",
                KnowledgeBaseReadTool.name,
                {"path": "knowledge/studio/dpo-training.mdx"},
            ),
            _text_message(
                "beta 控制偏好强度。\n"
                "来源：knowledge/studio/dpo-training.mdx（关键参数）"
            ),
            _text_message("beta 用于控制偏好优化强度。"),
        ]
    )
    conversation = Conversation(
        agent=Agent(llm=llm, tools=[Tool(name=PyromindSubAgentTool.name)]),
        workspace=str(workspace),
        visualizer=None,
    )

    conversation.send_message("DPO 的 beta 参数有什么作用？")
    conversation.run()

    assert conversation.state.execution_status == ConversationExecutionStatus.FINISHED
    observations = [
        event.observation
        for event in conversation.state.events
        if isinstance(event, ObservationEvent)
        and isinstance(event.observation, SubAgentObservation)
    ]
    assert len(observations) == 1
    assert not any(
        isinstance(event, ObservationEvent)
        and isinstance(
            event.observation,
            (KnowledgeBaseGrepObservation, KnowledgeBaseReadObservation),
        )
        for event in conversation.state.events
    )
    assert observations[0].status == TaskStatus.COMPLETED
    assert "knowledge/studio/dpo-training.mdx" in observations[0].text
    assert llm.remaining_responses == 0
    assert "event=delegation_started" in caplog.text
    assert "[subagent] event=started" in caplog.text
    assert "type=search" in caplog.text
    assert "subagent=search" in caplog.text
    assert "child_conversation_id=" in caplog.text
    assert "path='knowledge/index.md'" in caplog.text
    assert caplog.text.count("event=read_started") == 2
    assert caplog.text.count("event=read_completed") == 2
    assert "event=delegation_completed" in caplog.text
    assert str(knowledge) not in caplog.text

    subagent_events = [
        record.getMessage()
        for record in caplog.records
        if record.getMessage().startswith("[subagent-event]")
    ]
    assert len(subagent_events) == 7
    assert [
        int(index)
        for message in subagent_events
        for index in re.findall(r"event_index=(\d+)", message)
    ] == list(range(7))
    assert [
        kind
        for message in subagent_events
        for kind in re.findall(r"kind=(\w+)", message)
    ] == [
        "SystemPromptEvent",
        "MessageEvent",
        "ActionEvent",
        "ObservationEvent",
        "ActionEvent",
        "ObservationEvent",
        "MessageEvent",
    ]
    event_ids = [
        event_id
        for message in subagent_events
        for event_id in re.findall(r"event_id=([0-9a-f-]+)", message)
    ]
    assert len(event_ids) == len(set(event_ids)) == 7
    assert all(
        f"parent_conversation_id={conversation.state.id}" in message
        for message in subagent_events
    )
    child_conversation_ids = {
        child_id
        for message in subagent_events
        for child_id in re.findall(r"child_conversation_id=([0-9a-f-]+)", message)
    }
    assert len(child_conversation_ids) == 1
    assert all("task_id=task_00000001" in message for message in subagent_events)


def test_general_purpose_subagent_writes_without_leaking_child_events(tmp_path, caplog):
    caplog.set_level(logging.INFO)
    workspace = tmp_path / "conversation"
    workspace.mkdir()
    knowledge = tmp_path / "knowledge"
    knowledge.mkdir()
    configure_subagents(workspace, knowledge)

    llm = TestLLM.from_messages(
        [
            _tool_call(
                "parent-subagent",
                PyromindSubAgentTool.name,
                {
                    "type": "general_purpose",
                    "task": "Create result.txt containing done and report the change.",
                },
            ),
            _tool_call(
                "child-write",
                FileEditorTool.name,
                {
                    "command": "create",
                    "path": "result.txt",
                    "file_text": "done\n",
                },
            ),
            _text_message(
                "Created result.txt. Files changed: result.txt. Tests: not needed."
            ),
            _text_message("The delegated workspace task completed."),
        ]
    )
    conversation = Conversation(
        agent=Agent(llm=llm, tools=[Tool(name=PyromindSubAgentTool.name)]),
        workspace=str(workspace),
        visualizer=None,
    )

    conversation.send_message("Create the requested file through a subagent.")
    conversation.run()

    assert (workspace / "result.txt").read_text(encoding="utf-8") == "done\n"
    assert conversation.state.execution_status == ConversationExecutionStatus.FINISHED
    observations = [
        event.observation
        for event in conversation.state.events
        if isinstance(event, ObservationEvent)
    ]
    assert len(observations) == 1
    assert isinstance(observations[0], SubAgentObservation)
    assert not any(
        isinstance(observation, FileEditorObservation) for observation in observations
    )
    assert observations[0].type == SubAgentType.GENERAL_PURPOSE
    assert observations[0].status == TaskStatus.COMPLETED
    assert "Created result.txt" in observations[0].text
    assert "type=general_purpose" in caplog.text
    assert "subagent=general-purpose" in caplog.text
