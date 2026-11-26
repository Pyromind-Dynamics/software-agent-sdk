"""Share accepted debug submissions across SDK processes and legacy Tools."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest

from harness_adapter.pi_adapter import PiAdapter
from harness_adapter.pi_adapter.adapter import _PiSession
from harness_adapter.pi_adapter.business_tool_host import (
    DEBUG_STATE_RELATIVE_PATH,
    PyromindBusinessToolHost,
    ToolExecutionContext,
)
from harness_adapter.pi_adapter.persistence import PiSessionFiles
from harness_adapter.pi_adapter.runner import PiRunnerProcess
from pyromind_common.platform.workflows import DebugBudget
from pyromind_runtime.domain.context import RequestContext
from pyromind_runtime.ports.harness import ExternalTaskNotification


async def test_sdk_budget_migrates_legacy_count_and_survives_fallback_and_reset(
    tmp_path: Path,
) -> None:
    files = PiSessionFiles(tmp_path)
    files.save_business_state(
        {"pyromind_workflow_attempts": 7, "unrelated_state": "must-not-copy"}
    )

    path = PyromindBusinessToolHost.prepare_debug_budget(tmp_path)

    assert path == tmp_path / DEBUG_STATE_RELATIVE_PATH
    assert json.loads(path.read_text()) == {"pyromind_workflow_attempts": 7}
    budget = DebugBudget.load(path)
    budget.accepted = 10
    budget.save()
    workflow = tmp_path / "public_data/workflow_canvas/workflow.py"
    workflow.parent.mkdir(parents=True, exist_ok=True)
    workflow.write_text("workflow = InputNode()\n")
    restarted = PiAdapter(tmp_path, terminal_backend="os-sandbox")
    result = await restarted._business_tools.execute(
        "workflow_debug",
        {},
        ToolExecutionContext(
            conversation_id="budget-fallback",
            workspace_root=tmp_path,
            request_context=RequestContext(user_id="42"),
            model_configuration={},
        ),
    )

    assert result["is_error"] is True
    assert result["details"]["attempt"] == 10
    assert "maximum" in json.dumps(result["content"])
    assert files.load_business_state() == {"pyromind_workflow_attempts": 10}
    budget = DebugBudget.load(path)
    budget.accepted = 0
    budget.save()
    PyromindBusinessToolHost.prepare_debug_budget(tmp_path)
    assert files.load_business_state() == {"pyromind_workflow_attempts": 0}
    budget.accepted = 4
    budget.save()
    PyromindBusinessToolHost.reset_attempt_budget(tmp_path)
    assert DebugBudget.load(path).accepted == 0
    assert files.load_business_state() == {"pyromind_workflow_attempts": 0}


def test_shared_debug_budget_rejects_a_path_outside_public_data(tmp_path: Path) -> None:
    public_data = tmp_path / "public_data"
    public_data.mkdir()
    outside = tmp_path / "private"
    outside.mkdir()
    (public_data / ".pyromind").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="inside public_data"):
        PyromindBusinessToolHost.prepare_debug_budget(tmp_path)

    assert not list(outside.iterdir())


async def test_remote_legacy_debug_pulls_sdk_count_and_pushes_authorized_reset(
    tmp_path: Path, monkeypatch
) -> None:
    adapter = PiAdapter(
        tmp_path, terminal_backend="sandbox", platform_sdk_enabled=False
    )
    files = PiSessionFiles(tmp_path)
    files.initialize({})
    files.save_sandbox({"sandbox_id": "existing-sandbox"})
    context = RequestContext(user_id="42")
    session = _PiSession("remote-budget", tmp_path, files, {}, context)
    adapter._sessions[session.session_id] = session
    tool_context = adapter._control_tool_context(session)
    path = adapter._business_tools.prepare_debug_budget(tmp_path)
    workflow = tmp_path / "public_data/workflow_canvas/workflow.py"
    workflow.parent.mkdir(parents=True, exist_ok=True)
    workflow.write_text("workflow = InputNode()\n")
    synchronized = []

    async def pull(_context, _files):
        budget = DebugBudget.load(path)
        budget.accepted = 10
        budget.save()
        synchronized.append("pull:10")
        return "/workspace/remote/public_data/.pyromind/workflow-debug-state.json"

    async def push(_context, _files):
        synchronized.append(f"push:{DebugBudget.load(path).accepted}")

    monkeypatch.setattr(adapter._sandbox, "debug_state_path", pull)
    monkeypatch.setattr(adapter._sandbox, "push_debug_budget", push)

    result = await adapter._execute_business_tool(
        session, "workflow_debug", {}, tool_context
    )

    assert result["is_error"] is True
    details = result["details"]
    assert isinstance(details, dict)
    assert details["attempt"] == 10
    assert synchronized == ["pull:10", "push:10"]
    runner = Mock(spec=PiRunnerProcess)
    runner.request = AsyncMock(return_value={"accepted": True})
    session.runner = runner
    monkeypatch.setattr(adapter, "_ensure_runner", AsyncMock())
    await adapter.notify_external_task(
        adapter._handle(session),
        ExternalTaskNotification(
            task_id="debug-task",
            kind="workflow_debug",
            status="succeeded",
            hidden_text="The authorized round has finished.",
            reset_attempt_budget=True,
        ),
        context,
    )

    assert synchronized == ["pull:10", "push:10", "push:0"]
    assert files.load_business_state() == {"pyromind_workflow_attempts": 0}
