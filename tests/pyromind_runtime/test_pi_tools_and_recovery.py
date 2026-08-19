from __future__ import annotations

import json
from types import SimpleNamespace

import httpx
from harness_adapter.pi_adapter import PiAdapter
from harness_adapter.pi_adapter.adapter import _resolve_model
from harness_adapter.pi_adapter.business_tools import (
    execute_validation_tool,
    validation_tool_spec,
)
from harness_adapter.pi_adapter.permissions import TerminalPermissionPolicy
from harness_adapter.pi_adapter.persistence import PiSessionFiles
from pyromind_runtime.domain.context import RequestContext
from pyromind_runtime.ports.harness import SessionSpec


def test_validation_schema_is_generated_and_only_exposes_dsl_path() -> None:
    schema = validation_tool_spec()["input_schema"]
    assert "dsl_path" in schema["properties"]
    assert "dsl" not in schema["properties"]


async def test_validation_reads_conversation_file_and_forwards_request_context(
    tmp_path, monkeypatch
) -> None:
    workflow = tmp_path / "public_data" / "workflow_canvas" / "workflow.py"
    workflow.parent.mkdir(parents=True)
    workflow.write_text("workflow = InputNode()")
    captured = {}

    def fake_post(url, *, headers, json, timeout):
        captured.update(url=url, headers=headers, json=json, timeout=timeout)
        return SimpleNamespace(
            status_code=200,
            text="ok",
            json=lambda: {
                "success": True,
                "data": {
                    "valid": True,
                    "workflow_id": "w1",
                    "errors": [],
                    "warnings": [],
                },
            },
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    result = await execute_validation_tool(
        tmp_path,
        {"dsl_path": "public_data/workflow_canvas/workflow.py"},
        RequestContext(
            user_id="42",
            cookie="auth=secret",
            authorization="Bearer secret",
            x_cluster="cluster-a",
            accept_language="zh-CN",
        ),
    )

    assert result["is_error"] is False
    assert captured["json"]["dsl"] == "workflow = InputNode()"
    assert captured["headers"]["cookie"] == "auth=secret"
    assert captured["headers"]["authorization"] == "Bearer secret"
    assert captured["headers"]["x-cluster"] == "cluster-a"
    assert captured["headers"]["accept-language"] == "zh-CN"


async def test_validation_projects_401_without_retry(tmp_path, monkeypatch) -> None:
    workflow = tmp_path / "workflow.py"
    workflow.write_text("workflow = InputNode()")
    calls = 0

    def fake_post(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return SimpleNamespace(status_code=401, text="unauthorized")

    monkeypatch.setattr(httpx, "post", fake_post)
    result = await execute_validation_tool(
        tmp_path,
        {"dsl_path": "workflow.py"},
        RequestContext(user_id="42", cookie="auth=secret"),
    )
    assert result["is_error"] is True
    assert calls == 1
    assert "401" in json.dumps(result)


def test_terminal_only_confirms_high_risk_commands() -> None:
    policy = TerminalPermissionPolicy()
    assert not policy.requires_confirmation("safe", {"command": "pwd"})
    assert policy.requires_confirmation("danger", {"command": "rm -rf /tmp/example"})


def test_checkpoint_is_atomic_and_session_is_non_secret(tmp_path) -> None:
    files = PiSessionFiles(tmp_path)
    files.initialize({"model": {"provider": "openai", "id": "gpt-5"}})
    files.save_checkpoint([{"role": "user", "content": "hello"}])
    files.save_inflight({"run_id": "run-1", "operation_id": "op-1"})
    assert files.load_checkpoint()[0]["role"] == "user"
    assert files.load_inflight() == {"run_id": "run-1", "operation_id": "op-1"}
    assert "api_key" not in files.session_path.read_text()
    files.clear_inflight()
    assert files.load_inflight() is None


def test_model_resolution_rules() -> None:
    assert _resolve_model("deepseek/deepseek-chat", "https://openrouter.ai/api/v1") == (
        "openrouter",
        "deepseek/deepseek-chat",
    )
    assert _resolve_model("anthropic/claude-sonnet", None) == (
        "anthropic",
        "claude-sonnet",
    )
    assert _resolve_model("gpt-5", "http://localhost:8000/v1") == (
        "openai",
        "gpt-5",
    )
    assert _resolve_model("deepseek/deepseek-chat", "http://localhost:8000/v1") == (
        "openai",
        "deepseek/deepseek-chat",
    )


async def test_real_runner_starts_without_persisting_request_api_key(tmp_path) -> None:
    conversations = tmp_path / "conversations"
    conversations.mkdir()
    adapter = PiAdapter(conversations)
    handle = await adapter.create_session(
        SessionSpec(
            conversation_id="conversation-1",
            user_id="42",
            workspace_root=str(conversations / "conversation-1"),
            model_configuration={"model": "gpt-4o", "api_key": "request-secret"},
        ),
        RequestContext(user_id="42", cookie="cookie-secret"),
    )
    try:
        persisted = (
            conversations / "conversation-1" / "pi" / "session.json"
        ).read_text()
        assert "request-secret" not in persisted
        assert "cookie-secret" not in persisted
        event = await anext(adapter.subscribe(handle))
        assert event.type == "history.synced"
    finally:
        await adapter.close(handle)
