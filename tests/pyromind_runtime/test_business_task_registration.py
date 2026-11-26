"""Enrich immediate SDK receipts without regressing a fast terminal callback."""

from pyromind_agent_server.external_task_registry import WorkflowExternalTaskRegistry
from pyromind_runtime.application.conversation_runtime import ConversationRuntime
from pyromind_runtime.domain.context import RequestContext
from pyromind_runtime.ports.harness import SessionSpec

from .fake_adapter import FakeAdapter


async def test_sdk_receipt_normalization_and_late_metadata_preserve_terminal_state(
    tmp_path,
):
    registry = WorkflowExternalTaskRegistry(tmp_path)
    runtime = ConversationRuntime(tmp_path, FakeAdapter(), external_tasks=registry)
    context = RequestContext(user_id="42")
    try:
        await runtime.create_conversation(
            SessionSpec(
                conversation_id="c", user_id="42", workspace_root=str(tmp_path / "c")
            ),
            context,
        )
        initial = {
            "task_id": "sdk-task",
            "kind": "environment_processing",
            "status": "Pending",
        }
        runtime.register_external_task("c", initial)
        first = (await runtime.get_snapshot("c", context)).external_tasks[0]
        assert first.status == "pending"
        assert first.submitted_at and first.updated_at
        await runtime.deliver_external_task_status(
            "c", task_id="sdk-task", status="succeeded", auto_run=False
        )
        complete = {
            **initial,
            "run_id": "actual-edp-run",
            "output_dir": "/workspace/edp/run",
            "attempt": 1,
            "max_attempts": 10,
            "keep_ui_lock": True,
        }
        runtime.register_external_task("c", complete)
        snapshot = await runtime.get_snapshot("c", context)
        task = snapshot.external_tasks[0]
        assert task.status == "succeeded"
        assert task.run_id == "actual-edp-run"
        assert task.output_dir == "/workspace/edp/run"
        assert task.attempt == 1 and task.max_attempts == 10
        assert not task.keep_ui_lock
        assert task.submitted_at == first.submitted_at
        runtime.register_external_task("c", complete)
        assert (
            await runtime.get_snapshot("c", context)
        ).through_seq == snapshot.through_seq
        saved = registry.resolve("c", "sdk-task")
        assert saved is not None and saved["run_id"] == "actual-edp-run"
        assert saved["status"] == "succeeded"
    finally:
        await runtime.close()
