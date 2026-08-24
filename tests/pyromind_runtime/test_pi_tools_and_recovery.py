from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import harness_adapter.pi_adapter.adapter as pi_adapter_module
import httpx
import pytest
from harness_adapter.pi_adapter import PiAdapter
from harness_adapter.pi_adapter.adapter import (
    _is_workflow_mutation,
    _resolve_model,
    _session_config,
)
from harness_adapter.pi_adapter.business_tool_host import PyromindBusinessToolHost
from harness_adapter.pi_adapter.business_tools import (
    execute_validation_tool,
    validation_tool_spec,
)
from harness_adapter.pi_adapter.permissions import TerminalPermissionPolicy
from harness_adapter.pi_adapter.persistence import PiSessionFiles
from pydantic import ValidationError
from pyromind_runtime.domain.context import RequestContext
from pyromind_runtime.ports.harness import SessionSpec

from openhands.agent_server.pyromind_router import PyromindLLMConfig


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


def test_session_metadata_is_non_secret_and_uses_pi_jsonl(tmp_path) -> None:
    files = PiSessionFiles(tmp_path)
    files.initialize({"model": {"provider": "openai", "id": "gpt-5"}})
    files.save_inflight({"run_id": "run-1", "operation_id": "op-1"})
    assert files.session_log_path == tmp_path / "pi" / "session.jsonl"
    assert files.load_inflight() == {"run_id": "run-1", "operation_id": "op-1"}
    assert "api_key" not in files.session_path.read_text()
    files.clear_inflight()
    assert files.load_inflight() is None


def test_adapter_resolves_available_knowledge_root(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("PYROMIND_KNOWLEDGE_BASE_PATH", raising=False)
    explicit = tmp_path / "knowledge"
    explicit.mkdir()
    assert PiAdapter(
        tmp_path / "explicit", knowledge_root=explicit
    )._knowledge_root == (explicit.resolve())
    assert (
        PiAdapter(
            tmp_path / "missing-adapter",
            knowledge_root=tmp_path / "missing-knowledge",
        )._knowledge_root
        is None
    )

    repository = Path(pi_adapter_module.__file__).parents[3]
    assert (
        PiAdapter(tmp_path / "default")._knowledge_root
        == (repository / "knowledge").resolve()
    )


async def test_runner_start_receives_configured_knowledge_root(
    tmp_path, monkeypatch
) -> None:
    captured = {}

    class FakeRunner:
        def __init__(self, **_kwargs) -> None:
            self.running = True

        async def start(self, config) -> None:
            captured.update(config)

        async def close(self) -> None:
            self.running = False

    monkeypatch.setattr(pi_adapter_module, "PiRunnerProcess", FakeRunner)
    conversations = tmp_path / "conversations"
    conversations.mkdir()
    knowledge = tmp_path / "knowledge"
    knowledge.mkdir()
    adapter = PiAdapter(conversations, knowledge_root=knowledge)
    handle = await adapter.create_session(
        SessionSpec(
            conversation_id="conversation-1",
            user_id="42",
            workspace_root=str(conversations / "conversation-1"),
            model_configuration={"model": "gpt-5", "api_key": "test-key"},
        ),
        RequestContext(user_id="42"),
    )
    try:
        assert captured["knowledge_root"] == str(knowledge.resolve())
    finally:
        await adapter.close(handle)


async def test_runner_loads_three_named_skills_and_eight_business_tools(
    tmp_path, monkeypatch
) -> None:
    captured = {}

    class FakeRunner:
        def __init__(self, **_kwargs) -> None:
            self.running = True

        async def start(self, config) -> None:
            captured.update(config)

        async def close(self) -> None:
            self.running = False

    monkeypatch.setattr(pi_adapter_module, "PiRunnerProcess", FakeRunner)
    conversations = tmp_path / "conversations"
    conversations.mkdir()
    adapter = PiAdapter(conversations)
    handle = await adapter.create_session(
        SessionSpec(
            conversation_id="conversation-skills",
            user_id="42",
            workspace_root=str(conversations / "conversation-skills"),
            model_configuration={"model": "gpt-5", "api_key": "test-key"},
        ),
        RequestContext(user_id="42"),
    )
    try:
        assert [item["name"] for item in captured["skill_roots"]] == [
            "generate-workflow-dsl",
            "data-cleaning",
            "data-preparation",
        ]
        assert {item["name"] for item in captured["tools"]} == {
            "validate_workflow_dsl",
            "preview_dataset",
            "upload_file_to_pyromind",
            "run_dataset_cleaning",
            "df_run_pipeline",
            "df_submit_pipeline",
            "df_check_progress",
            "df_stop_task",
        }
    finally:
        await adapter.close(handle)


def test_business_tool_specs_are_generated_from_openhands_definitions() -> None:
    repository = Path(pi_adapter_module.__file__).parents[3]
    roots = [
        repository / ".agents" / "skills" / "data-cleaning",
        repository / ".agents" / "skills" / "data-preparation",
    ]
    specs = PyromindBusinessToolHost(roots).specs()
    assert len(specs) == 8
    assert all(spec["input_schema"].get("type") == "object" for spec in specs)


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


def test_session_config_persists_model_api_resolution() -> None:
    def model_config(configuration):
        return _session_config(
            SessionSpec(
                conversation_id="conversation-1",
                user_id="42",
                workspace_root="/tmp/conversation-1",
                model_configuration=configuration,
            )
        )["model"]

    assert (
        model_config({"model": "custom-model", "base_url": "http://localhost:8000/v1"})[
            "api"
        ]
        == "openai-completions"
    )
    assert (
        model_config(
            {
                "model": "deepseek-v4-flash-0731",
                "base_url": "http://localhost:8000/v1",
            }
        )["context_window"]
        == 200_000
    )
    assert (
        model_config(
            {
                "model": "deepseek-v4-flash-0731",
                "base_url": "http://localhost:8000/v1",
                "context_window": 196_608,
            }
        )["context_window"]
        == 196_608
    )
    assert (
        model_config({"model": "custom-model", "base_url": "http://localhost:8000/v1"})[
            "context_window"
        ]
        == 128_000
    )
    assert (
        model_config(
            {
                "model": "deepseek/deepseek-chat",
                "base_url": "https://openrouter.ai/api/v1",
            }
        )["api"]
        == "openai-completions"
    )
    assert (
        model_config(
            {
                "model": "custom-model",
                "base_url": "http://localhost:8000/v1",
                "api": "openai-responses",
            }
        )["api"]
        == "openai-responses"
    )
    assert "api" not in model_config({"model": "gpt-5"})
    assert "context_window" not in model_config({"model": "gpt-5"})
    assert "api" not in model_config(
        {"model": "gpt-5", "base_url": "https://api.openai.com/v1"}
    )


def test_pyromind_llm_config_rejects_unknown_model_api() -> None:
    with pytest.raises(ValidationError):
        PyromindLLMConfig(model="gpt-5", api="other")


def test_pyromind_llm_config_rejects_invalid_context_window() -> None:
    with pytest.raises(ValidationError):
        PyromindLLMConfig(model="gpt-5", context_window=0)


async def test_adapter_ignores_duplicate_run_finished(tmp_path) -> None:
    adapter = PiAdapter(tmp_path)
    files = PiSessionFiles(tmp_path / "conversation-1")
    files.initialize({"model": {"provider": "openai", "id": "gpt-5"}})
    session = SimpleNamespace(
        session_id="conversation-1",
        finished_runs=set(),
        running=True,
        files=files,
        queue=asyncio.Queue(),
    )
    frame = {
        "protocolVersion": 2,
        "type": "pi.event",
        "eventId": "e1",
        "sessionId": "conversation-1",
        "runId": "run-1",
        "occurredAt": "2026-08-20T00:00:00Z",
        "kind": "run.finished",
        "payload": {"outcome": {"status": "completed", "stop_reason": "stop"}},
    }

    await adapter._runner_event(session, frame)
    await adapter._runner_event(session, {**frame, "eventId": "e2"})

    assert session.queue.qsize() == 1
    assert (await session.queue.get()).payload["status"] == "idle"


async def test_adapter_projects_generic_write_completion_to_workflow(
    tmp_path, monkeypatch
) -> None:
    adapter = PiAdapter(tmp_path)
    workspace = tmp_path / "conversation-1"
    files = PiSessionFiles(workspace)
    files.initialize({"model": {"provider": "openai", "id": "gpt-5"}})
    session = SimpleNamespace(
        session_id="conversation-1",
        workspace_root=workspace,
        finished_runs=set(),
        running=True,
        files=files,
        queue=asyncio.Queue(),
    )
    emitted = []

    async def capture_workflow(_session, event_id, **kwargs) -> None:
        emitted.append((event_id, kwargs.get("source_event_id")))

    monkeypatch.setattr(adapter, "_emit_workflow", capture_workflow)
    frame = {
        "protocolVersion": 2,
        "type": "pi.event",
        "eventId": "write-event",
        "sessionId": "conversation-1",
        "runId": "run-1",
        "occurredAt": "2026-08-20T00:00:00Z",
        "kind": "tool.completed",
        "payload": {
            "tool_call_id": "call-write",
            "tool_name": "write",
            "arguments": {
                "path": "public_data/workflow_canvas/workflow.py",
                "content": "workflow = SFTWorkflow()",
            },
            "content": [],
            "details": None,
        },
    }

    await adapter._runner_event(session, frame)

    assert emitted == [("write-event:workflow", "write-event")]
    assert (await session.queue.get()).type == "operation.completed"
    assert not _is_workflow_mutation(
        workspace,
        {
            "tool_name": "write",
            "arguments": {"path": "other.py"},
        },
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
