from pathlib import Path
from typing import Any, cast
from uuid import UUID

import httpx
import pytest
from pydantic import SecretStr

from openhands.sdk.conversation.secret_registry import SecretRegistry
from openhands.sdk.secret import StaticSecret
from openhands.tools.pyromind_cleaning import (
    DatasetCleaningTaskAssociation,
    DatasetCleaningTaskStore,
    RunDatasetCleaningAction,
    RunDatasetCleaningExecutor,
)
from openhands.tools.pyromind_dataset.definition import (
    PYROMIND_STORAGE_AUTH_COOKIE_SECRET,
    PYROMIND_STORAGE_HEADERS_STATE_KEY,
)


_CONVERSATION_ID = UUID("00000000-0000-0000-0000-000000000123")


class _Response:
    def __init__(self, status_code: int, payload: Any, text: str = "") -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self) -> Any:
        return self._payload


def _secret_registry() -> SecretRegistry:
    registry = SecretRegistry()
    registry.update_secrets(
        {
            PYROMIND_STORAGE_AUTH_COOKIE_SECRET: StaticSecret(
                value=SecretStr("auth_token=session-token")
            )
        }
    )
    return registry


def _fake_conversation(tmp_path: Path):
    workspace = type(
        "FakeWorkspace",
        (),
        {"working_dir": str(tmp_path / "conversations" / _CONVERSATION_ID.hex)},
    )()
    state = type(
        "FakeState",
        (),
        {
            "secret_registry": _secret_registry(),
            "agent_state": {
                PYROMIND_STORAGE_HEADERS_STATE_KEY: {"x-cluster": "us-west-1#pre"}
            },
        },
    )()
    return type(
        "FakeConversation",
        (),
        {"id": _CONVERSATION_ID, "workspace": workspace, "state": state},
    )()


def test_run_dataset_cleaning_submits_fixed_workflow_and_persists_task(
    monkeypatch,
    tmp_path,
):
    calls: list[dict[str, Any]] = []

    def fake_post(url, *, headers, json, timeout):
        calls.append({"url": url, "headers": headers, "json": json, "timeout": timeout})
        return _Response(200, {"success": True, "data": {"task_id": 9876}})

    monkeypatch.setattr(httpx, "post", fake_post)
    task_store_dir = tmp_path / "tasks"
    conversation = _fake_conversation(tmp_path)

    observation = RunDatasetCleaningExecutor(
        endpoint_url="https://portal.test/std2/studio_api/api/prompt",
        task_store_dir=str(task_store_dir),
        timeout=5,
    )(
        RunDatasetCleaningAction(
            input_path="datasets/source.jsonl",
            script_path="/agentTest/clean.py",
            limit=25,
        ),
        cast(Any, conversation),
    )

    assert not observation.is_error
    assert observation.task_id == "9876"
    assert observation.run_id is not None
    assert observation.output_dir == (f"/agentTest/data_cleaning/{observation.run_id}")
    assert observation.resumed is False
    assert calls[0]["url"] == "https://portal.test/std2/studio_api/api/prompt"
    assert calls[0]["timeout"] == 5
    assert calls[0]["headers"]["cookie"] == "auth_token=session-token"
    assert calls[0]["headers"]["x-cluster"] == "us-west-1#pre"

    workflow = calls[0]["json"]
    assert workflow["id"] == observation.run_id
    assert workflow["edges"] == []
    assert len(workflow["nodes"]) == 1
    node = workflow["nodes"][0]
    assert node["data"]["nodeType"] == "CustomCommandNode"
    assert node["data"]["config"]["cpu"] == 4
    command = node["data"]["config"]["command"]
    pod_output_dir = f"/target-workspace{observation.output_dir}"
    assert "mkdir -p /target-workspace/agentTest/data_cleaning" in command
    assert f"mkdir {pod_output_dir}" in command
    assert (
        f"cp /target-workspace/agentTest/clean.py {pod_output_dir}/clean_script.py"
    ) in command
    assert "--input /target-workspace/datasets/source.jsonl" in command
    assert f"--output {pod_output_dir}/output.jsonl" in command
    assert f"--state-dir {pod_output_dir}" in command
    assert "--limit 25" in command
    assert "--resume" not in command

    association = DatasetCleaningTaskStore(task_store_dir).get("9876")
    assert association is not None
    assert association.conversation_id == str(_CONVERSATION_ID)
    assert association.run_id == observation.run_id
    assert association.output_dir == observation.output_dir
    assert association.input_path == "/datasets/source.jsonl"
    assert association.script_path == "/agentTest/clean.py"
    assert association.limit == 25
    assert association.resumed is False


def test_run_dataset_cleaning_resume_uses_frozen_script(monkeypatch, tmp_path):
    calls: list[dict[str, Any]] = []

    def fake_post(url, *, headers, json, timeout):
        calls.append({"json": json})
        return _Response(200, {"success": True, "data": {"task_id": "task-2"}})

    monkeypatch.setattr(httpx, "post", fake_post)
    run_id = UUID("10000000-0000-0000-0000-000000000001")
    task_store_dir = tmp_path / "tasks"
    task_store = DatasetCleaningTaskStore(task_store_dir)
    task_store.save(
        DatasetCleaningTaskAssociation(
            task_id="task-1",
            conversation_id=str(_CONVERSATION_ID),
            run_id=str(run_id),
            output_dir=f"/agentTest/data_cleaning/{run_id}",
            input_path="/datasets/source.jsonl",
            script_path="/agentTest/original.py",
        )
    )

    observation = RunDatasetCleaningExecutor(
        endpoint_url="https://portal.test/std2/studio_api/api/prompt",
        task_store_dir=str(task_store_dir),
    )(
        RunDatasetCleaningAction(
            input_path="/datasets/source.jsonl",
            resume_run_id=run_id,
        ),
        cast(Any, _fake_conversation(tmp_path)),
    )

    assert not observation.is_error
    assert observation.run_id == str(run_id)
    assert observation.resumed is True
    command = calls[0]["json"]["nodes"][0]["data"]["config"]["command"]
    frozen_script = (
        f"/target-workspace/agentTest/data_cleaning/{run_id}/clean_script.py"
    )
    assert command.startswith(f"test -f {frozen_script} && python3 {frozen_script}")
    assert "--resume" in command
    assert "cp " not in command
    association = task_store.get("task-2")
    assert association is not None
    assert association.script_path == "/agentTest/original.py"


def test_run_dataset_cleaning_rejects_unknown_resume(tmp_path):
    run_id = UUID("20000000-0000-0000-0000-000000000001")

    observation = RunDatasetCleaningExecutor(
        endpoint_url="https://portal.test/std2/studio_api/api/prompt",
        task_store_dir=str(tmp_path / "tasks"),
    )(
        RunDatasetCleaningAction(
            input_path="/datasets/source.jsonl",
            resume_run_id=run_id,
        ),
        cast(Any, _fake_conversation(tmp_path)),
    )

    assert observation.is_error
    assert f"unknown dataset cleaning run {run_id}" in observation.text


@pytest.mark.parametrize(
    ("input_path", "script_path", "expected_error"),
    [
        ("/datasets/../secret.jsonl", "/agentTest/clean.py", "contain '..'"),
        ("/datasets/source.jsonl", "/agentTest/clean.sh", "Python .py file"),
    ],
)
def test_run_dataset_cleaning_rejects_invalid_paths(
    input_path,
    script_path,
    expected_error,
    tmp_path,
):
    observation = RunDatasetCleaningExecutor(
        endpoint_url="https://portal.test/std2/studio_api/api/prompt",
        task_store_dir=str(tmp_path / "tasks"),
    )(
        RunDatasetCleaningAction(
            input_path=input_path,
            script_path=script_path,
        ),
        cast(Any, _fake_conversation(tmp_path)),
    )

    assert observation.is_error
    assert expected_error in observation.text
