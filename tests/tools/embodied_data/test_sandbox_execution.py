import ast
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock
from uuid import UUID

import pytest
from openhands_embodied_runtime.sandbox_runner import run_full, run_plan
from pydantic import ValidationError
from pyromind_sdk.client.models import TrainingTaskCreateResponse

from openhands.sdk.conversation.secret_registry import SecretRegistry
from openhands.sdk.conversation.state import ActiveLongTask
from openhands.tools.embodied_data import validate_lerobot_v21_dataset
from openhands.tools.embodied_data.platform_submit import (
    EmbodiedTaskAssociation,
    EmbodiedTaskStore,
    RunEmbodiedSandboxAction,
    RunEmbodiedSandboxExecutor,
)


_CONVERSATION_ID = UUID("00000000-0000-0000-0000-000000000123")


def _fake_conversation(tmp_path: Path):
    registry = SecretRegistry()
    registry.update_secrets({"auth_token": "session-token"})
    state = type(
        "FakeState",
        (),
        {
            "secret_registry": registry,
            "active_long_tasks": [],
            "__enter__": lambda self: self,
            "__exit__": lambda self, *args: None,
        },
    )()
    workspace = type(
        "FakeWorkspace",
        (),
        {"working_dir": str(tmp_path / "conversations" / _CONVERSATION_ID.hex)},
    )()

    def register_active_long_task(self, task: ActiveLongTask) -> None:
        cast(Any, state).active_long_tasks = [
            *cast(Any, state).active_long_tasks,
            task,
        ]

    return type(
        "FakeConversation",
        (),
        {
            "id": _CONVERSATION_ID,
            "workspace": workspace,
            "state": state,
            "register_active_long_task": register_active_long_task,
        },
    )()


def test_sandbox_runner_plans_batches_validates_and_publishes(
    self_collected_path: Path,
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "sandbox-run"
    target = tmp_path / "published_lerobot_v21"

    planned = run_plan(
        self_collected_path,
        run_dir,
        idle_min_duration_s=10,
        runtime_revision="openhands-embodied-runtime==1.29.3",
    )

    assert planned["complete"] is True
    assert planned["episode_count"] == 1
    assert planned["runtime_revision"] == "openhands-embodied-runtime==1.29.3"
    assert (run_dir / "inspection.json").is_file()
    assert (run_dir / "representative_plan.json").is_file()

    completed = run_full(
        self_collected_path,
        run_dir,
        target,
        task_text="Pick and place the item on the table",
        confirm_subtasks=True,
        confirm_derived_action=True,
        idle_min_duration_s=10,
        runtime_revision="openhands-embodied-runtime==1.29.3",
    )

    assert completed["complete"] is True
    assert completed["published"] is True
    assert completed["accepted_episode_count"] == 1
    assert completed["validation"]["valid"] is True
    assert validate_lerobot_v21_dataset(target).valid
    assert not any("plan" in path.name for path in target.rglob("*"))
    assert (run_dir / "full_report.json").is_file()

    resumed = run_full(
        self_collected_path,
        run_dir,
        target,
        task_text="Pick and place the item on the table",
        confirm_subtasks=True,
        confirm_derived_action=True,
        idle_min_duration_s=10,
        resume=True,
        runtime_revision="openhands-embodied-runtime==1.29.3",
    )

    assert resumed["complete"] is True
    assert resumed["published"] is True
    assert resumed["phase"] == "resume"
    assert resumed["runtime_revision"] == "openhands-embodied-runtime==1.29.3"


def test_sandbox_runtime_sources_parse_as_python_310() -> None:
    runtime_root = (
        Path(__file__).parents[3]
        / "openhands-embodied-runtime"
        / "openhands_embodied_runtime"
    )

    for source_path in runtime_root.glob("*.py"):
        ast.parse(
            source_path.read_text(encoding="utf-8"),
            filename=str(source_path),
            feature_version=(3, 10),
        )


def test_full_sandbox_run_requires_plan_artifacts(
    self_collected_path: Path,
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="requires a completed plan phase"):
        run_full(
            self_collected_path,
            tmp_path / "missing-plan",
            tmp_path / "target",
            task_text="Pick and place the item on the table",
            confirm_subtasks=True,
            confirm_derived_action=True,
        )


def test_sandbox_runner_rejects_audit_directory_inside_source(
    self_collected_path: Path,
) -> None:
    with pytest.raises(ValueError, match="source and run directory paths"):
        run_plan(self_collected_path, self_collected_path / "sandbox-run")


def test_sandbox_action_requires_confirmation_for_full() -> None:
    with pytest.raises(ValidationError, match="subtask confirmation"):
        RunEmbodiedSandboxAction(
            source_path="/robot/source",
            mode="full",
            run_id=UUID("10000000-0000-0000-0000-000000000001"),
            target_path="/robot/target",
            task_text="Pick and place",
            confirm_derived_action=True,
        )


def test_sandbox_submit_uses_storage_mount_without_local_materialization(
    monkeypatch,
    tmp_path: Path,
) -> None:
    client = MagicMock()
    create_client = MagicMock(return_value=client)
    submit = MagicMock(
        return_value=TrainingTaskCreateResponse(
            task_id="task-plan",
            name="embodied-plan",
            status="Pending",
        )
    )
    monkeypatch.setattr(
        "openhands.tools.embodied_data.platform_submit.create_workflow_api_client",
        create_client,
    )
    monkeypatch.setattr(
        "openhands.tools.embodied_data.platform_submit.submit_workflow_task",
        submit,
    )
    task_store_dir = tmp_path / "tasks"
    conversation = _fake_conversation(tmp_path)

    observation = RunEmbodiedSandboxExecutor(
        env="pre",
        cluster="test-cluster",
        runtime_package="openhands-embodied-runtime==1.29.3",
        task_store_dir=str(task_store_dir),
    )(
        RunEmbodiedSandboxAction(
            source_path="/workspace/robot/source",
            mode="plan",
            idle_min_duration_s=10,
        ),
        cast(Any, conversation),
    )

    assert not observation.is_error
    assert observation.run_id is not None
    assert observation.output_dir is not None
    workflow = submit.call_args.kwargs["workflow"]
    node = workflow["nodes"][0]
    assert node["data"]["nodeType"] == "CustomCommandCPUNode"
    command = node["data"]["config"]["command"]
    assert (
        "pip install --disable-pip-version-check "
        "openhands-embodied-runtime==1.29.3" in command
    )
    assert "-m openhands_embodied_runtime.sandbox_runner" in command
    assert "--source /target-workspace/robot/source" in command
    assert f"--run-dir /target-workspace{observation.output_dir}" in command
    assert "--runtime-revision openhands-embodied-runtime==1.29.3" in command
    assert "materialize_storage_files" not in command
    assert "--target" not in command
    create_client.assert_called_once_with(
        env="pre",
        cluster="test-cluster",
        auth_token="session-token",
        headers={},
        timeout=30,
    )
    association = EmbodiedTaskStore(task_store_dir).get_by_run_id(observation.run_id)
    assert association is not None
    assert association.phase == "plan"
    assert association.source_path == "/robot/source"
    assert cast(Any, conversation).state.active_long_tasks == [
        ActiveLongTask(
            task_id="task-plan",
            kind="embodied_data_cleaning",
            status="Pending",
        )
    ]


def test_sandbox_full_continues_plan_with_same_run_and_target(
    monkeypatch,
    tmp_path: Path,
) -> None:
    run_id = UUID("10000000-0000-0000-0000-000000000001")
    task_store_dir = tmp_path / "tasks"
    output_dir = f"/.pyromind-agent/{_CONVERSATION_ID}/embodied_cleaning/{run_id}"
    EmbodiedTaskStore(task_store_dir).save(
        EmbodiedTaskAssociation(
            task_id="task-plan",
            conversation_id=str(_CONVERSATION_ID),
            run_id=str(run_id),
            phase="plan",
            output_dir=output_dir,
            source_path="/robot/source",
            idle_min_duration_s=10,
        )
    )
    monkeypatch.setattr(
        "openhands.tools.embodied_data.platform_submit.create_workflow_api_client",
        MagicMock(return_value=MagicMock()),
    )
    submit = MagicMock(
        return_value=TrainingTaskCreateResponse(
            task_id="task-full",
            name="embodied-full",
            status="Pending",
        )
    )
    monkeypatch.setattr(
        "openhands.tools.embodied_data.platform_submit.submit_workflow_task",
        submit,
    )

    observation = RunEmbodiedSandboxExecutor(
        env="prod",
        cluster="test-cluster",
        runtime_wheel_storage_path=(
            "/runtime/openhands_embodied_runtime-1.29.3-py3-none-any.whl"
        ),
        task_store_dir=str(task_store_dir),
    )(
        RunEmbodiedSandboxAction(
            source_path="/robot/source",
            mode="full",
            run_id=run_id,
            target_path="/workspace/robot/cleaned",
            task_text="Pick and place the item on the table",
            confirm_subtasks=True,
            confirm_derived_action=True,
            idle_min_duration_s=10,
        ),
        cast(Any, _fake_conversation(tmp_path)),
    )

    assert not observation.is_error
    assert observation.run_id == str(run_id)
    assert observation.output_dir == output_dir
    assert observation.target_path == "/robot/cleaned"
    command = submit.call_args.kwargs["workflow"]["nodes"][0]["data"]["config"][
        "command"
    ]
    assert (
        "test -f "
        "/target-workspace/runtime/"
        "openhands_embodied_runtime-1.29.3-py3-none-any.whl"
    ) in command
    assert "--mode full" in command
    assert "--target /target-workspace/robot/cleaned" in command
    assert (
        "--runtime-revision /runtime/openhands_embodied_runtime-1.29.3-py3-none-any.whl"
    ) in command
    assert "--confirm-subtasks --confirm-derived-action" in command


def test_sandbox_submit_fails_closed_without_fixed_runtime(tmp_path: Path) -> None:
    observation = RunEmbodiedSandboxExecutor()(
        RunEmbodiedSandboxAction(source_path="/robot/source"),
        cast(Any, _fake_conversation(tmp_path)),
    )

    assert observation.is_error
    assert "runtime is not configured" in observation.text


def test_sandbox_submit_rejects_unpinned_runtime_package() -> None:
    with pytest.raises(ValueError, match="exact package pin"):
        RunEmbodiedSandboxExecutor(runtime_package="openhands-embodied-runtime")


def test_sandbox_submit_rejects_main_tools_wheel() -> None:
    with pytest.raises(ValueError, match="openhands_embodied_runtime"):
        RunEmbodiedSandboxExecutor(
            runtime_wheel_storage_path=(
                "/runtime/embodied-test/openhands_tools-1.29.3-py3-none-any.whl"
            )
        )
