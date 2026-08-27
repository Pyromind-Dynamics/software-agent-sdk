"""Tests for the env-validation platform submission tool (edp_submit)."""

from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from openhands.tools.environment_processing.platform_submit import (
    EdpSubmitAction,
    EdpSubmitTool,
    build_edp_command,
    build_edp_workflow,
    resolve_llm_env,
)


# ---------------------------------------------------------------------------
# resolve_llm_env
# ---------------------------------------------------------------------------


def _conversation_with_secrets(secrets: dict[str, str]) -> MagicMock:
    registry = MagicMock()
    registry.get_secret_value.side_effect = lambda name: secrets.get(name)
    state = SimpleNamespace(secret_registry=registry, agent_state={})
    conversation = MagicMock()
    conversation.state = state
    conversation.id = "conv-1"
    return conversation


_LLM_ENV_KEYS = (
    "LLM_BASE_URL",
    "LLM_AUTH_TOKEN",
    "LLM_MODEL",
    "DF_API_BASE_URL",
    "DF_API_KEY",
    "DF_MODEL_NAME",
    "ANTHROPIC_BASE_URL",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_MODEL",
)


def _clear_llm_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in _LLM_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


def test_resolve_llm_env_uses_llm_secrets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_llm_env(monkeypatch)
    conversation = _conversation_with_secrets(
        {
            "LLM_BASE_URL": "https://gw.example",
            "LLM_AUTH_TOKEN": "sk-abc",
            "LLM_MODEL": "openai/deepseek-v4-flash-0731",
        }
    )
    resolved = resolve_llm_env(conversation)
    assert resolved == {
        "LLM_BASE_URL": "https://gw.example",
        "LLM_AUTH_TOKEN": "sk-abc",
        "LLM_MODEL": "openai/deepseek-v4-flash-0731",
    }


def test_resolve_llm_env_falls_back_to_df_env_stripping_v1(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_llm_env(monkeypatch)
    conversation = _conversation_with_secrets(
        {
            "DF_API_BASE_URL": "https://gw.example/v1",
            "DF_API_KEY": "df-key",
            "DF_MODEL_NAME": "openai/gpt-5.6-luna",
        }
    )
    resolved = resolve_llm_env(conversation)
    assert resolved == {
        "LLM_BASE_URL": "https://gw.example",
        "LLM_AUTH_TOKEN": "df-key",
        "LLM_MODEL": "openai/gpt-5.6-luna",
    }


def test_resolve_llm_env_falls_back_to_anthropic_secrets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_llm_env(monkeypatch)
    conversation = _conversation_with_secrets(
        {
            "ANTHROPIC_BASE_URL": "https://gw.example/anthropic/v1",
            "ANTHROPIC_AUTH_TOKEN": "sk-legacy",
            "ANTHROPIC_MODEL": "anthropic/claude-sonnet-4",
        }
    )
    resolved = resolve_llm_env(conversation)
    assert resolved == {
        "LLM_BASE_URL": "https://gw.example/anthropic",
        "LLM_AUTH_TOKEN": "sk-legacy",
        "LLM_MODEL": "anthropic/claude-sonnet-4",
    }


def test_resolve_llm_env_process_env_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_llm_env(monkeypatch)
    conversation = _conversation_with_secrets({})
    monkeypatch.setenv("LLM_BASE_URL", "https://gw.example")
    monkeypatch.setenv("LLM_AUTH_TOKEN", "env-key")
    monkeypatch.setenv("LLM_MODEL", "model-x")
    resolved = resolve_llm_env(conversation)
    assert resolved["LLM_AUTH_TOKEN"] == "env-key"


def test_resolve_llm_env_missing_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_llm_env(monkeypatch)
    conversation = _conversation_with_secrets({"LLM_MODEL": "m"})
    with pytest.raises(ValueError, match="LLM_BASE_URL"):
        resolve_llm_env(conversation)


# ---------------------------------------------------------------------------
# build_edp_command / build_edp_workflow
# ---------------------------------------------------------------------------


def _command(limit: int | None = None, dedup_by_image: bool = False) -> str:
    return build_edp_command(
        output_dir="/agent/conv-1/environment_processing/run-uuid",
        profile_name="tmax-validation",
        platform_env="pre",
        cluster="us-west-1",
        auth_token="platform-tok",
        llm_env={
            "LLM_BASE_URL": "https://gw.example",
            "LLM_AUTH_TOKEN": "sk-secret",
            "LLM_MODEL": "model-x",
        },
        limit=limit,
        dedup_by_image=dedup_by_image,
    )


def test_build_edp_command_pod_paths_and_credentials() -> None:
    command = _command(limit=3, dedup_by_image=True)
    assert "python3 -m venv /tmp/edp-venv" in command
    assert "pip install --quiet" in command
    assert "> /tmp/edp-install.log 2>&1" in command
    assert (
        "-r /target-workspace/agent/conv-1/environment_processing/"
        "run-uuid/pod_requirements.txt" in command
    )
    assert "openhands-tools==" not in command
    assert (
        "PYTHONPATH=/target-workspace/agent/conv-1/environment_processing/run-uuid"
        in command
    )
    pipeline = command.split(" && ")[-1]
    # The pod executor drops the very first `export ...;` segment, so the
    # pipeline starts with a no-op `true;` and every real export follows.
    assert pipeline.startswith("true; export PYROMIND_ENV=pre;")
    # the auth token additionally travels as an explicit --auth-token
    # argument, expanded from the export, so it can never be lost again.
    assert '--auth-token "$PYROMIND_AUTH_TOKEN"' in command
    assert (
        "/target-workspace/agent/conv-1/environment_processing/run-uuid/sandbox_runner.py"
        in command
    )
    assert (
        "--profile /target-workspace/agent/conv-1/environment_processing/"
        "run-uuid/tmax-validation.json" in command
    )
    assert (
        "--manifest /target-workspace/agent/conv-1/environment_processing/"
        "run-uuid/manifest.jsonl" in command
    )
    assert (
        "--output-dir /target-workspace/agent/conv-1/environment_processing/"
        "run-uuid/run" in command
    )
    assert "--env pre --cluster us-west-1" in command
    assert "--limit 3" in command
    assert "--dedup-by-image" in command
    # Credentials are exported as real `export` statements BEFORE the $VAR
    # references in --set, so same-shell argument expansion sees the values
    # (a prefix assignment K=v python ... is invisible to "$K" expansion).
    assert "sk-secret" in command  # lives in the export prefix
    assert "export LLM_AUTH_TOKEN=" in command
    assert "export LLM_BASE_URL=" in command
    assert "export LLM_MODEL=" in command
    assert "export PYROMIND_AUTH_TOKEN=" in command
    # the export statements must precede the --set $VAR references in order
    export_pos = command.index("export LLM_AUTH_TOKEN=")
    set_pos = command.index('--set LLM_AUTH_TOKEN="$LLM_AUTH_TOKEN"')
    assert export_pos < set_pos


def test_build_edp_command_without_limit() -> None:
    command = _command()
    assert "--limit" not in command
    assert "--dedup-by-image" not in command


def test_build_edp_workflow_uses_custom_command_cpu_node() -> None:
    run_id = uuid.uuid4()
    workflow = build_edp_workflow(run_id=run_id, command="echo hi", cpu=8, memory=32)
    node = workflow["nodes"][0]
    assert node["data"]["nodeType"] == "CustomCommandCPUNode"
    assert node["data"]["config"]["command"] == "echo hi"
    assert node["data"]["config"]["cpu"] == 8
    assert node["data"]["config"]["memory"] == 32
    assert workflow["id"] == str(run_id)
    assert workflow["name"].startswith("agent-edp-")
    assert workflow["edges"] == []


# ---------------------------------------------------------------------------
# Executor: staging + submission
# ---------------------------------------------------------------------------


def _executor(runtime_dir: str | None = None) -> Any:
    from openhands.tools.environment_processing.platform_submit import (
        EdpSubmitExecutor,
    )

    return EdpSubmitExecutor(
        env="pre",
        cluster="us-west-1",
        headers={"x-cluster": "us-west-1#pre"},
        runtime_dir=runtime_dir,
        storage_base_url=None,
        storage_headers={},
        storage_secret_headers={},
        timeout=10,
    )


def _runtime_tmp(tmp_path: Path, with_pod_runtime: bool = True) -> Path:
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "sandbox_runner.py").write_text("print('runner')\n")
    if with_pod_runtime:
        pod = scripts / "pod_runtime"
        (pod / "openhands" / "sdk" / "profiles").mkdir(parents=True)
        (pod / "openhands" / "tools" / "utils").mkdir(parents=True)
        (pod / "openhands" / "tools" / "sandbox").mkdir(parents=True)
        (pod / "pod_requirements.txt").write_text("pyromind-sdk>=0.1.9\n")
        (pod / "openhands" / "__init__.py").write_text("")
        (pod / "openhands" / "sdk" / "__init__.py").write_text("")
        (pod / "openhands" / "sdk" / "profiles" / "__init__.py").write_text("")
        (pod / "openhands" / "sdk" / "profiles" / "processing_profile.py").write_text(
            ""
        )
        (pod / "openhands" / "tools" / "__init__.py").write_text("")
        (pod / "openhands" / "tools" / "utils" / "__init__.py").write_text("")
        (pod / "openhands" / "tools" / "utils" / "pyromind_api_client.py").write_text(
            ""
        )
        (pod / "openhands" / "tools" / "sandbox" / "__init__.py").write_text("")
    profiles = tmp_path / "profiles"
    profiles.mkdir()
    (profiles / "tmax-validation.json").write_text('{"name": "tmax-validation"}\n')
    return scripts


def _pod_runtime_file_count() -> int:
    return 9  # requirements + 6 __init__ + processing_profile + api_client + shim


def _patch_submission(
    monkeypatch: pytest.MonkeyPatch, task_ids: tuple[str, ...] = ("task-99",)
) -> SimpleNamespace:
    module = sys.modules["openhands.tools.environment_processing.platform_submit"]
    upload = MagicMock()
    monkeypatch.setattr(module, "upload_local_file_to_pyromind", upload)
    responses = [SimpleNamespace(task_id=tid, status="Pending") for tid in task_ids]
    submit = MagicMock(side_effect=responses)
    monkeypatch.setattr(module, "submit_workflow_task", submit)
    client = MagicMock()
    monkeypatch.setattr(
        module, "create_workflow_api_client", MagicMock(return_value=client)
    )
    return SimpleNamespace(upload=upload, submit=submit, client=client)


def test_executor_submits_single_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mocks = _patch_submission(monkeypatch)
    conversation = _conversation_with_secrets(
        {
            "auth_token": "platform-tok",
            "LLM_BASE_URL": "https://gw.example",
            "LLM_AUTH_TOKEN": "sk-secret",
            "LLM_MODEL": "model-x",
        }
    )
    executor = _executor(runtime_dir=str(_runtime_tmp(tmp_path)))
    observation = executor(
        EdpSubmitAction(manifest="edp/batch-001/manifest.jsonl"), conversation
    )

    assert observation.status == "Pending"
    assert observation.task_ids == ["task-99"]
    assert observation.resumed is False
    assert len(observation.output_dirs) == 1
    assert observation.output_dirs[0].startswith("/edp/batch-001/")
    # runner + profile + pod_runtime tree staged; the manifest already lives
    # on Storage and is never re-uploaded
    assert mocks.upload.call_count == 2 + _pod_runtime_file_count()
    uploaded_names = {
        Path(c.kwargs["local_path"]).name for c in mocks.upload.call_args_list
    }
    assert "manifest.jsonl" not in uploaded_names
    # one task created with the CustomCommandCPUNode workflow
    workflow = mocks.submit.call_args.kwargs["workflow"]
    assert workflow["nodes"][0]["data"]["nodeType"] == "CustomCommandCPUNode"
    # the runner reads the manifest straight from the Storage mount
    command = workflow["nodes"][0]["data"]["config"]["command"]
    assert "--manifest /target-workspace/edp/batch-001/manifest.jsonl" in command
    conversation.register_active_long_task.assert_called_once()
    kind = conversation.register_active_long_task.call_args.args[0].kind
    assert kind == "environment_processing"


def test_executor_resume_reuses_run_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_submission(monkeypatch)
    conversation = _conversation_with_secrets(
        {
            "auth_token": "platform-tok",
            "LLM_BASE_URL": "https://gw.example",
            "LLM_AUTH_TOKEN": "sk-secret",
            "LLM_MODEL": "model-x",
        }
    )
    run_id = "3f9a6c1d-4f5e-4a6b-8c7d-9e0f1a2b3c4d"
    executor = _executor(runtime_dir=str(_runtime_tmp(tmp_path)))
    observation = executor(
        EdpSubmitAction(manifest="edp/batch-001/manifest.jsonl", run_id=run_id),
        conversation,
    )

    assert observation.resumed is True
    assert observation.run_ids == [run_id]
    assert observation.output_dirs == [f"/edp/batch-001/{run_id}"]


def test_executor_shards_index_download_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mocks = _patch_submission(monkeypatch)
    module = sys.modules["openhands.tools.environment_processing.platform_submit"]
    download = MagicMock(
        side_effect=ValueError("Pyromind storage get_url API: file not found")
    )
    monkeypatch.setattr(module, "download_file_from_pyromind", download)
    conversation = _conversation_with_secrets({"auth_token": "tok"})
    executor = _executor()
    observation = executor(
        EdpSubmitAction(shards="/edp/nope/shards.json"), conversation
    )
    assert observation.status == "Failed"
    assert "not found" in observation.text
    mocks.upload.assert_not_called()
    mocks.submit.assert_not_called()


def test_executor_missing_llm_env_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mocks = _patch_submission(monkeypatch)
    _clear_llm_env(monkeypatch)
    conversation = _conversation_with_secrets({"auth_token": "tok"})
    executor = _executor(runtime_dir=str(_runtime_tmp(tmp_path)))
    observation = executor(
        EdpSubmitAction(manifest="edp/batch-001/manifest.jsonl"), conversation
    )
    assert observation.status == "Failed"
    assert "LLM_BASE_URL" in observation.text
    # credential resolution fails before staging/submission
    mocks.upload.assert_not_called()
    mocks.submit.assert_not_called()


def test_executor_missing_profile_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_submission(monkeypatch)
    conversation = _conversation_with_secrets(
        {
            "auth_token": "tok",
            "LLM_BASE_URL": "https://gw.example",
            "LLM_AUTH_TOKEN": "sk",
            "LLM_MODEL": "m",
        }
    )
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "sandbox_runner.py").write_text("print('runner')\n")
    executor = _executor(runtime_dir=str(scripts))
    observation = executor(
        EdpSubmitAction(manifest="edp/batch-001/manifest.jsonl"), conversation
    )
    assert observation.status == "Failed"
    assert "tmax-validation" in observation.text


def test_edp_submit_tool_rejects_unknown_params() -> None:
    with pytest.raises(ValueError, match="unknown params"):
        EdpSubmitTool.create(task_store_dir="/tmp/x")


def test_edp_submit_action_requires_exactly_one_manifest_source() -> None:
    with pytest.raises(ValueError, match="exactly one of manifest"):
        EdpSubmitAction()
    with pytest.raises(ValueError, match="exactly one of manifest"):
        EdpSubmitAction(manifest="edp/b/manifest.jsonl", shards="/edp/shards.json")


def _patch_shards_index(
    monkeypatch: pytest.MonkeyPatch, shards: list[str]
) -> MagicMock:
    module = sys.modules["openhands.tools.environment_processing.platform_submit"]
    download = MagicMock(return_value=json.dumps({"shards": shards}).encode("utf-8"))
    monkeypatch.setattr(module, "download_file_from_pyromind", download)
    return download


_SHARDS = [
    "/edp/batch-001/manifest.jsonl",
    "/edp/batch-002/manifest.jsonl",
    "/edp/batch-003/manifest.jsonl",
]


def _full_conversation() -> MagicMock:
    return _conversation_with_secrets(
        {
            "auth_token": "platform-tok",
            "LLM_BASE_URL": "https://gw.example",
            "LLM_AUTH_TOKEN": "sk-secret",
            "LLM_MODEL": "model-x",
        }
    )


def test_executor_submits_shard_batch_with_offset_and_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mocks = _patch_submission(monkeypatch, task_ids=("task-1", "task-2"))
    download = _patch_shards_index(monkeypatch, _SHARDS)
    conversation = _full_conversation()
    executor = _executor(runtime_dir=str(_runtime_tmp(tmp_path)))
    observation = executor(
        EdpSubmitAction(shards="/edp/shards.json", shard_offset=1, shard_count=2),
        conversation,
    )

    assert observation.status == "Pending"
    assert observation.task_ids == ["task-1", "task-2"]
    # the index download is bounded
    assert download.call_args.kwargs["max_bytes"] == 4 * 1024 * 1024
    # each shard is its own workflow nested under its manifest's directory
    commands = [
        c.kwargs["workflow"]["nodes"][0]["data"]["config"]["command"]
        for c in mocks.submit.call_args_list
    ]
    assert "--manifest /target-workspace/edp/batch-002/manifest.jsonl" in commands[0]
    assert "--manifest /target-workspace/edp/batch-003/manifest.jsonl" in commands[1]
    assert observation.output_dirs[0].startswith("/edp/batch-002/")
    assert observation.output_dirs[1].startswith("/edp/batch-003/")
    # a fresh batch gives each shard an independent run dir
    assert observation.run_ids[0] != observation.run_ids[1]
    assert conversation.register_active_long_task.call_count == 2
    kinds = [
        c.args[0].kind for c in conversation.register_active_long_task.call_args_list
    ]
    assert kinds == ["environment_processing", "environment_processing"]


def test_executor_shard_batch_resume_shares_run_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_submission(monkeypatch, task_ids=("task-1", "task-2", "task-3"))
    _patch_shards_index(monkeypatch, _SHARDS)
    conversation = _full_conversation()
    run_id = "3f9a6c1d-4f5e-4a6b-8c7d-9e0f1a2b3c4d"
    executor = _executor(runtime_dir=str(_runtime_tmp(tmp_path)))
    observation = executor(
        EdpSubmitAction(shards="/edp/shards.json", run_id=run_id),
        conversation,
    )

    assert observation.resumed is True
    # all shards share the run id so each resume skips its own checkpoint
    assert observation.run_ids == [run_id, run_id, run_id]
    assert observation.output_dirs == [
        f"/edp/batch-001/{run_id}",
        f"/edp/batch-002/{run_id}",
        f"/edp/batch-003/{run_id}",
    ]


def test_executor_shard_batch_stops_on_first_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mocks = _patch_submission(monkeypatch)
    mocks.submit.side_effect = [
        SimpleNamespace(task_id="task-1", status="Pending"),
        RuntimeError("boom"),
    ]
    _patch_shards_index(monkeypatch, _SHARDS)
    conversation = _full_conversation()
    executor = _executor(runtime_dir=str(_runtime_tmp(tmp_path)))
    observation = executor(EdpSubmitAction(shards="/edp/shards.json"), conversation)

    # the first shard keeps running; the batch stops before the third
    assert observation.task_ids == ["task-1"]
    assert observation.is_error is True
    assert "Failed to submit shard" in observation.text
    assert "boom" in observation.text
    assert conversation.register_active_long_task.call_count == 1
