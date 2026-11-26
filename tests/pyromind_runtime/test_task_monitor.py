import json
from unittest.mock import AsyncMock, Mock

import pytest

from pyromind_runtime.application.task_monitor import TaskMonitor
from pyromind_runtime.domain.task_correlation import TaskCorrelation
from pyromind_runtime.domain.workflow_execution import WorkflowExecutionRequest
from pyromind_runtime.infrastructure.file_product_store import FileProductStore
from pyromind_runtime.infrastructure.sqlite_task_monitor_store import (
    SQLiteTaskMonitorStore,
)
from pyromind_runtime.ports.task_monitor import TaskStatus

from .test_workflow_execution import _runtime


@pytest.mark.parametrize("initial_status", ["Pending", "Succeeded"])
@pytest.mark.parametrize("kind", ["data_preparation", "workflow_run"])
async def test_monitor_accepts_owned_sdk_tasks_and_releases_the_check(
    tmp_path, initial_status, kind
):
    runtime, adapter, _, context = await _runtime(tmp_path)
    scope = "workflow-conversation"
    environment = await runtime.platform_sdk_context(scope, context)
    correlation = TaskCorrelation.model_validate_json(
        json.loads(environment["PYROMIND_TASK_CONTEXTS"])[kind]
    )
    checks = SQLiteTaskMonitorStore(tmp_path / "task-checks.sqlite3")
    query = Mock()
    query.query = AsyncMock(return_value=TaskStatus("succeeded"))
    monitor = TaskMonitor(checks, query, runtime, clock=lambda: 2)
    try:
        await runtime.deliver_external_task_status(
            scope,
            task_id="sdk-monitored",
            status=initial_status,
            correlation=correlation,
            trusted_platform=True,
        )
        store = FileProductStore(tmp_path / scope)
        if kind == "workflow_run":
            store.save_workflow_execution_request(
                WorkflowExecutionRequest(
                    request_id="sdk:sdk-monitored",
                    conversation_id=scope,
                    workflow_path="version://sdk-monitored",
                    fingerprint="metadata-fingerprint",
                    workflow={},
                    summary={},
                    correlation=correlation,
                    task_id="sdk-monitored",
                    status="submitted",
                ),
                expected_revision=None,
            )
        monitored = runtime.task_monitor_state(scope, "sdk-monitored")
        assert monitored is not None and monitored.resume_auto_run
        checks.register(scope, "sdk-monitored", due_at=1)
        await monitor.check_due()
        task = FileProductStore(tmp_path / scope).load_snapshot().external_tasks[0]
        assert task.status == "succeeded"
        assert len(adapter.external_task_notifications) == 1
        assert adapter.external_task_notifications[0][1].trigger_turn
        assert checks.claim(1000, 60) is None
        if initial_status == "Pending":
            query.query.assert_awaited_once()
            actual_task, actual_context = query.query.call_args.args
            assert actual_task == "sdk-monitored"
            assert actual_context.user_id == context.user_id
            assert actual_context.cookie == context.cookie
        else:
            query.query.assert_not_awaited()
    finally:
        await runtime.close()
