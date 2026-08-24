from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal, cast
from uuid import UUID, uuid4

import pytest

from openhands.agent_server.conversation_service import ConversationService
from openhands.agent_server.run_workflow_callback import (
    _extract_node_names_from_dsl,
    _failed_node_names_from_error_log,
    build_run_workflow_terminal_reminder,
    deliver_run_workflow_status,
)
from openhands.sdk.llm import Message, TextContent
from openhands.tools.node_signature import (
    NodeSignatureObservation,
)
from openhands.tools.workflow.definition import WORKFLOW_RELATIVE_PATH


class _FakeEventService:
    def __init__(self, conversation: Any = None) -> None:
        self.run: bool | None = None
        self.internal_context: list[TextContent] | None = None
        self.conversation = conversation
        self.visible: bool = False
        self.calls: list[dict] = []
        self.removed_tasks: dict[str, Any] = {}

    async def remove_active_long_task(self, task_id: str):
        return self.removed_tasks.pop(task_id, None)

    async def send_internal_context(
        self,
        content: list[TextContent],
        run: bool = False,
        visible: bool = False,
        extended_content: list[TextContent] | None = None,
    ) -> str:
        self.run = run
        self.internal_context = content
        self.visible = visible
        self.extended = extended_content
        self.calls.append(
            {
                "content": content,
                "run": run,
                "visible": visible,
                "extended_content": extended_content,
            }
        )
        return "internal-event"

    def get_conversation(self):
        if self.conversation is None:
            raise ValueError("inactive_service")
        return self.conversation


class _FakeConversationService:
    def __init__(self, conversations_dir: Path, conversation: Any = None) -> None:
        self.conversations_dir = conversations_dir
        self.event_service = _FakeEventService(conversation=conversation)
        self.requested_conversation_id: UUID | None = None

    async def get_event_service(self, conversation_id: UUID):
        self.requested_conversation_id = conversation_id
        return self.event_service


class _FakeSecretRegistry:
    def __init__(self, token: str = "auth-token") -> None:
        self.token = token

    def get_secret_value(self, name: str) -> str:
        return self.token


class _FakeConversation:
    def __init__(
        self,
        working_dir: Path,
        *,
        llm: Any = None,
        agent_state: dict | None = None,
    ) -> None:
        self.workspace = SimpleNamespace(working_dir=str(working_dir))
        self.state = SimpleNamespace(
            agent=SimpleNamespace(llm=llm),
            agent_state=agent_state or {},
            secret_registry=_FakeSecretRegistry(),
        )


class _FakeNodeSignatureExecutor:
    def __init__(self, observation: NodeSignatureObservation) -> None:
        self.observation = observation
        self.node_names: list[str] | None = None

    def __call__(self, action, conversation=None):
        self.node_names = list(action.node_names)
        return self.observation


class _FakeLLM:
    def __init__(self, text: str = "", error: Exception | None = None) -> None:
        self.text = text
        self.error = error

    async def acompletion(self, messages, **kwargs):
        if self.error is not None:
            raise self.error
        return SimpleNamespace(
            message=Message(role="assistant", content=[TextContent(text=self.text)])
        )


def _signature_observation(
    *,
    success: bool = True,
    status: Literal["success", "error"] = "success",
) -> NodeSignatureObservation:
    if not success:
        return NodeSignatureObservation(
            status="error",
            node_name="CloneAndCacheDataset",
            error_message="HTTP error fetching node signatures",
        )
    return NodeSignatureObservation(
        status=status,
        results=[
            {
                "node_name": "CloneAndCacheDataset",
                "success": True,
                "function_signature": "def CloneAndCacheDataset(dataset, target_path)",
                "docstring": "Clone a dataset into the workspace.",
                "parameters": [
                    {
                        "name": "dataset",
                        "type": "STRING",
                        "required": True,
                        "default": None,
                    }
                ],
                "source_code": (
                    "def CloneAndCacheDataset(dataset, target_path):\n    pass\n"
                ),
            }
        ],
    )


@pytest.mark.asyncio
async def test_generic_callback_silently_resumes_conversation(tmp_path):
    conversation_id = uuid4()
    task_id = f"workflow-task-{uuid4()}"
    service = _FakeConversationService(tmp_path / "conversations")

    result = await deliver_run_workflow_status(
        task_id=task_id,
        status="Succeeded",
        conversation_id=str(conversation_id),
        auto_run=False,
        conversation_service=cast(ConversationService, service),
    )

    assert result.outcome == "delivered_async"
    assert result.normalized_status == "Succeeded"
    assert result.conversation_id == str(conversation_id)
    assert service.requested_conversation_id == conversation_id
    assert service.event_service.run is False
    assert service.event_service.internal_context is not None
    visible_text = service.event_service.internal_context[0].text
    assert "工作流已完成" in visible_text
    assert task_id in visible_text
    assert "Succeeded" in visible_text
    # The <system_reminder> is in extended_content, not in the visible text
    assert service.event_service.extended is not None
    assert len(service.event_service.extended) == 1
    reminder = service.event_service.extended[0].text
    assert task_id in reminder
    assert "Resume the tool invocation associated with this task" in reminder
    assert "most recent non-empty visible message" in reminder
    assert "workflow_debug" not in reminder
    assert "Review the outcome and continue helping the user" not in reminder
    assert "stats.json" not in reminder

    # Verify single call with both visible and extended content
    assert len(service.event_service.calls) == 1
    call = service.event_service.calls[0]
    assert call["visible"] is True
    assert call["run"] is False
    assert call["extended_content"] is not None

    duplicate = await deliver_run_workflow_status(
        task_id=task_id,
        status="Succeeded",
        conversation_id=str(conversation_id),
        auto_run=False,
        conversation_service=cast(ConversationService, service),
    )
    assert duplicate.outcome == "duplicate_terminal"


@pytest.mark.asyncio
async def test_terminal_callback_removes_active_long_task(tmp_path):
    conversation_id = uuid4()
    task_id = f"workflow-task-{uuid4()}"
    service = _FakeConversationService(tmp_path / "conversations")
    service.event_service.removed_tasks[task_id] = SimpleNamespace(status="Running")

    result = await deliver_run_workflow_status(
        task_id=task_id,
        status="Succeeded",
        conversation_id=str(conversation_id),
        auto_run=True,
        conversation_service=cast(ConversationService, service),
    )

    assert result.outcome == "delivered_async"
    assert task_id not in service.event_service.removed_tasks
    assert service.event_service.run is True


@pytest.mark.asyncio
async def test_terminal_callback_stopped_task_does_not_resume(tmp_path):
    conversation_id = uuid4()
    task_id = f"workflow-task-{uuid4()}"
    service = _FakeConversationService(tmp_path / "conversations")
    service.event_service.removed_tasks[task_id] = SimpleNamespace(status="Stopped")

    result = await deliver_run_workflow_status(
        task_id=task_id,
        status="Stopped",
        conversation_id=str(conversation_id),
        auto_run=True,
        conversation_service=cast(ConversationService, service),
    )

    assert result.outcome == "delivered_async"
    assert result.normalized_status == "Terminated"
    assert task_id not in service.event_service.removed_tasks
    assert service.event_service.run is False


@pytest.mark.asyncio
async def test_terminated_callback_wakes_conversation_for_in_flight_task(tmp_path):
    """A platform-cancelled (Terminated) task still wakes the conversation.

    Mirrors the Kafka payload for a task cancelled on the platform while it
    was running: the task is not marked ``Stopped`` (that only happens when
    the user interrupted the conversation), so the terminal callback must
    deliver the visible notification and re-run the agent.
    """
    conversation_id = uuid4()
    task_id = "7796"
    service = _FakeConversationService(tmp_path / "conversations")
    service.event_service.removed_tasks[task_id] = SimpleNamespace(status="Running")

    result = await deliver_run_workflow_status(
        task_id=task_id,
        status="Terminated",
        conversation_id=str(conversation_id),
        auto_run=True,
        conversation_service=cast(ConversationService, service),
    )

    assert result.outcome == "delivered_async"
    assert result.normalized_status == "Terminated"
    assert task_id not in service.event_service.removed_tasks
    assert service.event_service.run is True


@pytest.mark.asyncio
async def test_workflow_debug_callback_success_uses_debug_guidance(tmp_path):
    conversation_id = uuid4()
    task_id = f"workflow-task-{uuid4()}"
    service = _FakeConversationService(tmp_path / "conversations")

    result = await deliver_run_workflow_status(
        task_id=task_id,
        status="Succeeded",
        conversation_id=str(conversation_id),
        auto_run=False,
        from_workflow_debug=True,
        conversation_service=cast(ConversationService, service),
    )

    assert result.outcome == "delivered_async"
    assert service.event_service.internal_context is not None
    visible_text = service.event_service.internal_context[0].text
    assert "工作流调试运行成功" in visible_text
    assert task_id in visible_text

    # <system_reminder> is in extended_content
    assert service.event_service.extended is not None
    reminder = service.event_service.extended[0].text
    assert "workflow_debug (test) run that passed" in reminder
    assert "wait for their next message" in reminder
    assert "Resume the tool invocation associated with this task" not in reminder

    # Verify single call
    assert len(service.event_service.calls) == 1


@pytest.mark.asyncio
async def test_workflow_debug_callback_failure_uses_debug_guidance(tmp_path):
    conversation_id = uuid4()
    task_id = f"workflow-task-{uuid4()}"
    service = _FakeConversationService(tmp_path / "conversations")

    result = await deliver_run_workflow_status(
        task_id=task_id,
        status="Failed",
        error_log="node X failed",
        conversation_id=str(conversation_id),
        auto_run=False,
        from_workflow_debug=True,
        conversation_service=cast(ConversationService, service),
    )

    assert result.outcome == "delivered_async"
    assert service.event_service.internal_context is not None
    visible_text = service.event_service.internal_context[0].text
    assert "工作流调试运行失败" in visible_text
    assert task_id in visible_text
    assert "node X failed" in visible_text

    # <system_reminder> is in extended_content
    assert service.event_service.extended is not None
    reminder = service.event_service.extended[0].text
    assert "workflow_debug (test) run that failed" in reminder
    assert "call workflow_debug again" in reminder
    assert "node X failed" in reminder
    assert "Resume the tool invocation associated with this task" not in reminder

    # Verify single call
    assert len(service.event_service.calls) == 1


@pytest.mark.asyncio
async def test_generic_callback_without_conversation_is_unknown_task(tmp_path):
    task_id = f"workflow-task-{uuid4()}"
    service = _FakeConversationService(tmp_path / "conversations")

    result = await deliver_run_workflow_status(
        task_id=task_id,
        status="Succeeded",
        conversation_service=cast(ConversationService, service),
    )

    assert result.outcome == "unknown_task"
    assert service.requested_conversation_id is None
    assert service.event_service.internal_context is None


def test_build_reminder_production_ignores_debug_status_semantics():
    reminder = build_run_workflow_terminal_reminder(
        task_id="t1",
        status="Succeeded",
        from_workflow_debug=False,
    )
    assert "Resume the tool invocation associated with this task" in reminder
    assert "workflow_debug" not in reminder


def test_build_reminder_debug_terminated_guidance():
    reminder = build_run_workflow_terminal_reminder(
        task_id="t1",
        status="Terminated",
        error_log="cancelled by user",
        from_workflow_debug=True,
    )
    assert "was terminated" in reminder
    assert "cancelled by user" in reminder
    assert "Resume the tool invocation associated with this task" not in reminder


def test_build_reminder_debug_error_matches_failure_guidance():
    reminder = build_run_workflow_terminal_reminder(
        task_id="t1",
        status="Error",
        error_log="boom",
        from_workflow_debug=True,
    )
    assert "workflow_debug (test) run that failed" in reminder
    assert "call workflow_debug again" in reminder
    assert "boom" in reminder


@pytest.mark.asyncio
async def test_workflow_debug_callback_terminated_uses_debug_guidance(tmp_path):
    conversation_id = uuid4()
    task_id = f"workflow-task-{uuid4()}"
    service = _FakeConversationService(tmp_path / "conversations")

    result = await deliver_run_workflow_status(
        task_id=task_id,
        status="Terminated",
        conversation_id=str(conversation_id),
        auto_run=False,
        from_workflow_debug=True,
        conversation_service=cast(ConversationService, service),
    )

    assert result.outcome == "delivered_async"
    assert service.event_service.internal_context is not None
    visible_text = service.event_service.internal_context[0].text
    assert "工作流调试运行已终止" in visible_text
    assert task_id in visible_text

    # <system_reminder> is in extended_content
    assert service.event_service.extended is not None
    reminder = service.event_service.extended[0].text
    assert "was terminated" in reminder
    assert "Resume the tool invocation associated with this task" not in reminder
    # Verify single call
    assert len(service.event_service.calls) == 1


def test_extract_node_names_from_dsl():
    dsl = """# workflow: workflow
na54dbd4 = CloneAndCacheDataset(id=15, dataset="x", target_path=base.dataset_path)
n9bdbba6 = DatasetConfigBuilderVisionNode(id=24, image_field="image_path")
na702f68 = RewardItemBuilderNode(id=26, entry="reward:func")
"""
    assert _extract_node_names_from_dsl(dsl) == [
        "CloneAndCacheDataset",
        "DatasetConfigBuilderVisionNode",
        "RewardItemBuilderNode",
    ]


def test_extract_node_names_from_dsl_deduplicates_and_skips_comments():
    dsl = """# comment = NotANode(id=1)
a = Foo(id=1)
b = Foo(id=2)
c = Bar(id=3)
"""
    assert _extract_node_names_from_dsl(dsl) == ["Foo", "Bar"]


def test_failed_node_names_from_error_log_matches_group_headers():
    dsl = """na54dbd4 = CloneAndCacheDataset(id=15, dataset="x")
n9bdbba6 = DatasetConfigBuilderVisionNode(id=24, image_field="image_path")
na702f68 = RewardItemBuilderNode(id=26, entry="reward:func")
"""
    error_log = "--- 15 ---\nboom\n--- 26 ---\nbad\n"
    assert _failed_node_names_from_error_log(error_log, dsl) == [
        "CloneAndCacheDataset",
        "RewardItemBuilderNode",
    ]


@pytest.mark.parametrize(
    "error_log",
    [None, "", "node X failed", "--- 999 ---\nboom\n"],
)
def test_failed_node_names_from_error_log_returns_none_without_match(error_log):
    dsl = 'na54dbd4 = CloneAndCacheDataset(id=15, dataset="x")\n'
    assert _failed_node_names_from_error_log(error_log, dsl) is None


def test_failed_node_names_from_error_log_keeps_only_matching_ids():
    dsl = 'na54dbd4 = CloneAndCacheDataset(id=15, dataset="x")\n'
    error_log = "--- 15 ---\nboom\n--- 999 ---\nunknown\n"
    assert _failed_node_names_from_error_log(error_log, dsl) == ["CloneAndCacheDataset"]


def test_failed_node_names_from_error_log_falls_back_to_line_number_without_id():
    dsl = 'a = Foo(dataset="x")\n'
    error_log = "--- 1 ---\nboom\n"
    assert _failed_node_names_from_error_log(error_log, dsl) == ["Foo"]


def test_build_reminder_debug_failure_includes_signature_guidance():
    reminder = build_run_workflow_terminal_reminder(
        task_id="t1",
        status="Failed",
        error_log="boom",
        from_workflow_debug=True,
        node_signature_guidance="Node CloneAndCacheDataset requires dataset (STRING).",
    )
    assert "Node signature guidance:" in reminder
    assert "requires dataset" in reminder
    assert reminder.index("Runtime error log:") < reminder.index(
        "Node signature guidance:"
    )


@pytest.mark.asyncio
async def test_workflow_debug_callback_failure_fetches_only_failed_nodes(
    tmp_path, monkeypatch
):
    import openhands.tools.node_signature.impl as node_signature_impl

    conversation_id = uuid4()
    conversation_dir = tmp_path / "conversations" / conversation_id.hex
    workflow = conversation_dir / WORKFLOW_RELATIVE_PATH
    workflow.parent.mkdir(parents=True)
    workflow.write_text(
        'na54dbd4 = CloneAndCacheDataset(id=15, dataset="x")\n'
        'n9bdbba6 = DatasetConfigBuilderVisionNode(id=24, image_field="x")\n'
    )

    captured: list[_FakeNodeSignatureExecutor] = []

    def executor_factory(**kw):
        executor = _FakeNodeSignatureExecutor(_signature_observation())
        captured.append(executor)
        return executor

    monkeypatch.setattr(node_signature_impl, "NodeSignatureExecutor", executor_factory)
    service = _FakeConversationService(
        tmp_path / "conversations",
        conversation=_FakeConversation(conversation_dir, llm=_FakeLLM(text="x")),
    )

    result = await deliver_run_workflow_status(
        task_id=f"workflow-task-{uuid4()}",
        status="Failed",
        error_log="--- 15 ---\nboom\n",
        conversation_id=str(conversation_id),
        auto_run=False,
        from_workflow_debug=True,
        conversation_service=cast(ConversationService, service),
    )

    assert result.outcome == "delivered_async"
    assert captured[0].node_names == ["CloneAndCacheDataset"]
    assert service.event_service.extended is not None
    reminder = service.event_service.extended[0].text
    assert "Node signature guidance:" in reminder


@pytest.mark.asyncio
async def test_workflow_debug_callback_failure_injects_summarized_node_signature(
    tmp_path, monkeypatch
):
    import openhands.tools.node_signature.impl as node_signature_impl

    conversation_id = uuid4()
    conversation_dir = tmp_path / "conversations" / conversation_id.hex
    workflow = conversation_dir / WORKFLOW_RELATIVE_PATH
    workflow.parent.mkdir(parents=True)
    workflow.write_text('na54dbd4 = CloneAndCacheDataset(id=15, dataset="x")\n')

    monkeypatch.setattr(
        node_signature_impl,
        "NodeSignatureExecutor",
        lambda **kw: _FakeNodeSignatureExecutor(_signature_observation()),
    )
    conversation = _FakeConversation(
        conversation_dir,
        llm=_FakeLLM(text="简洁摘要：dataset 为必填 STRING 参数。"),
        agent_state={
            "pyromind_validate_workflow_dsl_headers": {"x-cluster": "us-west-1#pre"}
        },
    )
    service = _FakeConversationService(
        tmp_path / "conversations", conversation=conversation
    )

    result = await deliver_run_workflow_status(
        task_id=f"workflow-task-{uuid4()}",
        status="Failed",
        error_log="node X failed",
        conversation_id=str(conversation_id),
        auto_run=False,
        from_workflow_debug=True,
        conversation_service=cast(ConversationService, service),
    )

    assert result.outcome == "delivered_async"
    assert service.event_service.extended is not None
    reminder = service.event_service.extended[0].text
    assert "Node signature guidance:" in reminder
    assert "简洁摘要" in reminder
    assert "def CloneAndCacheDataset" not in reminder


@pytest.mark.asyncio
async def test_workflow_debug_callback_failure_falls_back_to_source_free_signatures(
    tmp_path, monkeypatch
):
    import openhands.tools.node_signature.impl as node_signature_impl

    conversation_id = uuid4()
    conversation_dir = tmp_path / "conversations" / conversation_id.hex
    workflow = conversation_dir / WORKFLOW_RELATIVE_PATH
    workflow.parent.mkdir(parents=True)
    workflow.write_text('na54dbd4 = CloneAndCacheDataset(id=15, dataset="x")\n')

    monkeypatch.setattr(
        node_signature_impl,
        "NodeSignatureExecutor",
        lambda **kw: _FakeNodeSignatureExecutor(_signature_observation()),
    )
    conversation = _FakeConversation(
        conversation_dir,
        llm=_FakeLLM(error=RuntimeError("llm down")),
    )
    service = _FakeConversationService(
        tmp_path / "conversations", conversation=conversation
    )

    result = await deliver_run_workflow_status(
        task_id=f"workflow-task-{uuid4()}",
        status="Error",
        error_log="node X failed",
        conversation_id=str(conversation_id),
        auto_run=False,
        from_workflow_debug=True,
        conversation_service=cast(ConversationService, service),
    )

    assert result.outcome == "delivered_async"
    assert service.event_service.extended is not None
    reminder = service.event_service.extended[0].text
    assert "Node signature guidance:" in reminder
    # Fallback keeps the source-free signature/parameter contract...
    assert "Signature: def CloneAndCacheDataset(dataset, target_path)" in reminder
    # ...and must not leak the node source body.
    assert "    pass" not in reminder


@pytest.mark.asyncio
async def test_workflow_debug_callback_failure_skips_guidance_when_fetch_fails(
    tmp_path, monkeypatch
):
    import openhands.tools.node_signature.impl as node_signature_impl

    conversation_id = uuid4()
    conversation_dir = tmp_path / "conversations" / conversation_id.hex
    workflow = conversation_dir / WORKFLOW_RELATIVE_PATH
    workflow.parent.mkdir(parents=True)
    workflow.write_text('na54dbd4 = CloneAndCacheDataset(id=15, dataset="x")\n')

    monkeypatch.setattr(
        node_signature_impl,
        "NodeSignatureExecutor",
        lambda **kw: _FakeNodeSignatureExecutor(_signature_observation(success=False)),
    )
    service = _FakeConversationService(
        tmp_path / "conversations",
        conversation=_FakeConversation(conversation_dir, llm=_FakeLLM(text="x")),
    )

    result = await deliver_run_workflow_status(
        task_id=f"workflow-task-{uuid4()}",
        status="Failed",
        error_log="node X failed",
        conversation_id=str(conversation_id),
        auto_run=False,
        from_workflow_debug=True,
        conversation_service=cast(ConversationService, service),
    )

    assert result.outcome == "delivered_async"
    assert service.event_service.internal_context is not None
    reminder = service.event_service.internal_context[0].text
    assert "Node signature guidance:" not in reminder
    assert "node X failed" in reminder


@pytest.mark.asyncio
async def test_workflow_debug_callback_failure_skips_guidance_without_workflow(
    tmp_path, monkeypatch
):
    import openhands.tools.node_signature.impl as node_signature_impl

    conversation_id = uuid4()
    conversation_dir = tmp_path / "conversations" / conversation_id.hex
    conversation_dir.mkdir(parents=True)

    monkeypatch.setattr(
        node_signature_impl,
        "NodeSignatureExecutor",
        lambda **kw: _FakeNodeSignatureExecutor(_signature_observation()),
    )
    service = _FakeConversationService(
        tmp_path / "conversations",
        conversation=_FakeConversation(conversation_dir, llm=_FakeLLM(text="x")),
    )

    result = await deliver_run_workflow_status(
        task_id=f"workflow-task-{uuid4()}",
        status="Failed",
        error_log="node X failed",
        conversation_id=str(conversation_id),
        auto_run=False,
        from_workflow_debug=True,
        conversation_service=cast(ConversationService, service),
    )

    assert result.outcome == "delivered_async"
    assert service.event_service.internal_context is not None
    reminder = service.event_service.internal_context[0].text
    assert "Node signature guidance:" not in reminder
    # Verify single call
    assert len(service.event_service.calls) == 1
