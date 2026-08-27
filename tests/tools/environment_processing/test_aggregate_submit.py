"""Tests for the edp_aggregate platform submission tool."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from openhands.tools.environment_processing.aggregate_submit import (
    AGGREGATE_TASK_KIND,
    EdpAggregateAction,
    EdpAggregateTool,
    build_aggregate_command,
)


# ---------------------------------------------------------------------------
# build_aggregate_command
# ---------------------------------------------------------------------------


def test_build_aggregate_command_pod_paths_and_args() -> None:
    command = build_aggregate_command(
        out_dir="edp/out",
        run_dirs=["edp/batch-001/run-1", "edp/batch-002/run-2"],
        protocol="tmax",
        min_reward=1.0,
        system_prompt="You are a coding agent.",
        limit=5,
    )
    # pure standard library: no venv, no pip — node python3 directly
    assert "venv" not in command
    assert "pip" not in command
    steps = command.split(" && ")
    assert steps[0] == "true"  # leading no-op absorbs the dropped first segment
    assert steps[1] == "test -f /target-workspace/edp/out/aggregate_results.py"
    assert steps[2] == "test -f /target-workspace/edp/out/convert_to_slime.py"
    assert steps[3] == "test -f /target-workspace/edp/out/convert_to_sft.py"
    run_step = steps[4]
    assert run_step.startswith("python3 /target-workspace/edp/out/aggregate_results.py")
    assert (
        "--run-dirs /target-workspace/edp/batch-001/run-1 "
        "/target-workspace/edp/batch-002/run-2" in run_step
    )
    assert "--out-dir /target-workspace/edp/out" in run_step
    assert "--protocol tmax" in run_step
    assert "--min-reward 1.0" in run_step
    assert "--system-prompt" in run_step
    assert "--limit 5" in run_step


def test_build_aggregate_command_omits_optional_args() -> None:
    command = build_aggregate_command(
        out_dir="edp/out",
        run_dirs=["edp/batch-001/run-1"],
        protocol="tmax",
        min_reward=0.5,
        system_prompt=None,
        limit=None,
    )
    assert "--system-prompt" not in command
    assert "--limit" not in command
    assert "--min-reward 0.5" in command


# ---------------------------------------------------------------------------
# Executor: staging + submission
# ---------------------------------------------------------------------------


def _conversation_with_secrets(secrets: dict[str, str]) -> MagicMock:
    registry = MagicMock()
    registry.get_secret_value.side_effect = lambda name: secrets.get(name)
    state = SimpleNamespace(secret_registry=registry, agent_state={})
    conversation = MagicMock()
    conversation.state = state
    conversation.id = "conv-1"
    return conversation


def _runtime_tmp(tmp_path: Path) -> Path:
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "aggregate_results.py").write_text("print('aggregate')\n")
    (scripts / "convert_to_slime.py").write_text("print('slime')\n")
    (scripts / "convert_to_sft.py").write_text("print('sft')\n")
    return scripts


def _executor(runtime_dir: str | None = None) -> Any:
    from openhands.tools.environment_processing.aggregate_submit import (
        EdpAggregateExecutor,
    )

    return EdpAggregateExecutor(
        env="pre",
        cluster="us-west-1",
        headers={"x-cluster": "us-west-1#pre"},
        runtime_dir=runtime_dir,
        storage_base_url=None,
        storage_headers={},
        storage_secret_headers={},
        timeout=10,
    )


def _patch_submission(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    module = sys.modules["openhands.tools.environment_processing.aggregate_submit"]
    upload = MagicMock()
    monkeypatch.setattr(module, "upload_local_file_to_pyromind", upload)
    response = SimpleNamespace(task_id="task-77", status="Pending")
    submit = MagicMock(return_value=response)
    monkeypatch.setattr(module, "submit_workflow_task", submit)
    client = MagicMock()
    monkeypatch.setattr(
        module, "create_workflow_api_client", MagicMock(return_value=client)
    )
    return SimpleNamespace(upload=upload, submit=submit, client=client)


def test_executor_submits_aggregate_task(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mocks = _patch_submission(monkeypatch)
    conversation = _conversation_with_secrets({"auth_token": "platform-tok"})
    executor = _executor(runtime_dir=str(_runtime_tmp(tmp_path)))
    observation = executor(
        EdpAggregateAction(
            run_dirs=["/edp/batch-001/run-1", "/edp/batch-002/run-2"],
            out_dir="edp/out",
            min_reward=0.8,
        ),
        conversation,
    )

    assert observation.status == "Pending"
    assert observation.task_id == "task-77"
    assert observation.out_dir == "/edp/out"
    # all three aggregation scripts are staged into the out dir
    assert mocks.upload.call_count == 3
    staged = {Path(c.kwargs["local_path"]).name for c in mocks.upload.call_args_list}
    assert staged == {
        "aggregate_results.py",
        "convert_to_slime.py",
        "convert_to_sft.py",
    }
    assert all(
        c.kwargs["target_dir"] == "/edp/out" for c in mocks.upload.call_args_list
    )
    # one CustomCommandCPUNode workflow; the command carries no credentials
    workflow = mocks.submit.call_args.kwargs["workflow"]
    node = workflow["nodes"][0]
    assert node["data"]["nodeType"] == "CustomCommandCPUNode"
    command = node["data"]["config"]["command"]
    assert "python3 /target-workspace/edp/out/aggregate_results.py" in command
    assert "--min-reward 0.8" in command
    assert "platform-tok" not in command
    assert workflow["name"].startswith("agent-edp-aggregate-")
    conversation.register_active_long_task.assert_called_once()
    task = conversation.register_active_long_task.call_args.args[0]
    assert task.kind == AGGREGATE_TASK_KIND


def test_executor_missing_auth_token_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mocks = _patch_submission(monkeypatch)
    conversation = _conversation_with_secrets({})
    executor = _executor(runtime_dir=str(_runtime_tmp(tmp_path)))
    observation = executor(
        EdpAggregateAction(run_dirs=["/edp/batch-001/run-1"], out_dir="edp/out"),
        conversation,
    )
    assert observation.status == "Failed"
    assert "auth_token" in observation.text
    mocks.upload.assert_not_called()


def test_executor_missing_scripts_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_submission(monkeypatch)
    conversation = _conversation_with_secrets({"auth_token": "tok"})
    empty = tmp_path / "scripts"
    empty.mkdir()
    executor = _executor(runtime_dir=str(empty))
    observation = executor(
        EdpAggregateAction(run_dirs=["/edp/batch-001/run-1"], out_dir="edp/out"),
        conversation,
    )
    assert observation.status == "Failed"
    assert "aggregate_results.py" in observation.text


def test_edp_aggregate_action_requires_run_dirs() -> None:
    with pytest.raises(ValueError):
        EdpAggregateAction(run_dirs=[], out_dir="edp/out")


def test_edp_aggregate_tool_rejects_unknown_params() -> None:
    with pytest.raises(ValueError, match="unknown params"):
        EdpAggregateTool.create(task_store_dir="/tmp/x")
