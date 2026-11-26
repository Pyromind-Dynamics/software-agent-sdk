import pytest

from harness_adapter.pi_adapter.adapter import PiAdapter, _PiSession
from harness_adapter.pi_adapter.persistence import PiSessionFiles
from pyromind_runtime.application.conversation_runtime import ConversationRuntime
from pyromind_runtime.domain.context import RequestContext
from pyromind_runtime.infrastructure.file_product_store import FileProductStore
from pyromind_runtime.ports.harness import SessionSpec

from .fake_adapter import FakeAdapter


@pytest.fixture
async def waiting_runtime(tmp_path):
    adapter = FakeAdapter()
    runtime = ConversationRuntime(tmp_path, adapter)
    context = RequestContext(user_id="42")
    await runtime.create_conversation(
        SessionSpec(
            conversation_id="waiting",
            user_id="42",
            workspace_root=str(tmp_path / "waiting"),
        ),
        context,
    )
    try:
        yield runtime, adapter, context
    finally:
        await runtime.close()


async def test_wait_uses_registered_work_for_the_current_main_run(waiting_runtime):
    runtime, _, context = waiting_runtime
    for task_id, overrides in (
        ("pending", {"status": "pending"}),
        ("running", {}),
        ("older-run", {"run_id": "previous"}),
        ("done", {"status": "succeeded"}),
        ("human-gate", {"status": "waiting"}),
        ("history", {"kind": "historical_experience"}),
        ("graph", {"graph_run_id": "graph-1"}),
        (
            "child",
            {
                "correlation": {
                    "conversation_id": "waiting",
                    "task_kind": "workflow_run",
                    "type": "subagent",
                    "subagent_id": "child-1",
                }
            },
        ),
    ):
        runtime.register_external_task(
            "waiting",
            {
                "task_id": task_id,
                "kind": "workflow_run",
                "run_id": "origin",
                "status": "running",
                "submitted_at": "2026-11-25T06:00:00Z",
                "updated_at": "2026-11-25T06:00:00Z",
                **overrides,
            },
        )
    assert await runtime.pending_external_tasks("waiting", "origin", context) == (
        "pending",
        "running",
    )
    assert await runtime.pending_external_tasks("waiting", "new-user", context) == ()
    with pytest.raises(PermissionError):
        await runtime.pending_external_tasks(
            "waiting", "origin", RequestContext(user_id="other")
        )


async def test_terminal_callback_releases_wait_without_suspending_resume(
    waiting_runtime, tmp_path
):
    runtime, adapter, context = waiting_runtime
    runtime.register_external_task(
        "waiting",
        {
            "task_id": "platform-1",
            "kind": "workflow_run",
            "run_id": "origin",
            "status": "running",
            "submitted_at": "2026-11-25T06:00:00Z",
            "updated_at": "2026-11-25T06:00:00Z",
        },
    )
    assert await runtime.pending_external_tasks("waiting", "origin", context) == (
        "platform-1",
    )
    await runtime.deliver_external_task_status(
        "waiting", task_id="platform-1", status="Succeeded"
    )
    assert await runtime.pending_external_tasks("waiting", "origin", context) == ()
    assert len(adapter.external_task_notifications) == 1
    assert adapter.external_task_notifications[0][1].trigger_turn is True
    snapshot = FileProductStore(tmp_path / "waiting").load_snapshot()
    assert snapshot.automatic_turns_suspended is False


async def test_adapter_wait_query_uses_its_own_session_context(tmp_path):
    adapter = PiAdapter(tmp_path, terminal_backend="os-sandbox")
    context = RequestContext(user_id="42")
    session = _PiSession("waiting", tmp_path, PiSessionFiles(tmp_path), {}, context)
    calls = []

    async def waiting(conversation_id, run_id, request_context):
        calls.append((conversation_id, run_id, request_context))
        return ("platform-1",)

    adapter.set_external_wait_provider(waiting)
    assert await adapter._runner_request(
        session, "external_task.waiting", {"run_id": "origin"}
    ) == {"task_ids": ["platform-1"]}
    assert calls == [("waiting", "origin", context)]
    with pytest.raises(ValueError, match="run_id"):
        await adapter._runner_request(session, "external_task.waiting", {})
