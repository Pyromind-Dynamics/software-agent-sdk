from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import harness_adapter.pi_adapter.adapter as pi_adapter_module
import httpx
import pytest
from harness_adapter.pi_adapter import PiAdapter
from harness_adapter.pi_adapter.adapter import (
    _is_workflow_mutation,
    _resolve_model,
    _session_config,
)
from harness_adapter.pi_adapter.business_tool_host import (
    PyromindBusinessToolHost,
    ToolExecutionContext,
    _cap_details_size,
    _cap_response_text,
)
from harness_adapter.pi_adapter.business_tools import (
    execute_validation_tool,
    validation_tool_spec,
)
from harness_adapter.pi_adapter.permissions import TerminalPermissionPolicy
from harness_adapter.pi_adapter.persistence import PiSessionFiles
from pydantic import BaseModel, ValidationError
from pyromind_runtime.domain.context import RequestContext
from pyromind_runtime.domain.snapshot import WorkflowState
from pyromind_runtime.ports.harness import (
    ExternalTaskNotification,
    ForkSpec,
    ProductCheckpoint,
    RestoreWorkflowSpec,
    SessionSpec,
)

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


async def test_runner_loads_five_named_skills_and_fourteen_business_tools(
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
            "data-processing",
            "debug-workflow",
            "training-analysis",
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
            "edp_render",
            "edp_submit",
            "edp_aggregate",
            "workflow_debug",
            "analyze_task_failure",
            "training_analysis",
        }
    finally:
        await adapter.close(handle)


def test_business_tool_specs_are_generated_from_openhands_definitions() -> None:
    repository = Path(pi_adapter_module.__file__).parents[3]
    roots = [
        repository / ".agents" / "skills" / "data-processing",
        repository / ".agents" / "skills" / "training-analysis",
    ]
    specs = PyromindBusinessToolHost(roots).specs()
    assert len(specs) == 14
    assert {"edp_render", "edp_submit", "edp_aggregate"} <= {
        spec["name"] for spec in specs
    }
    assert all(spec["input_schema"].get("type") == "object" for spec in specs)


def test_tool_response_text_is_capped_to_runner_frame_budget() -> None:
    image_block = {"type": "image_url", "image_url": {"url": "data:image/png;base64,x"}}
    capped = _cap_response_text(
        [
            {"type": "text", "text": "a" * 300_000},
            {"type": "text", "text": "tail"},
            image_block,
        ]
    )
    assert capped[0]["text"].endswith(
        "[output truncated: exceeded the runner frame budget]"
    )
    assert len(capped[0]["text"]) < 250_000 + 100
    assert capped[1] == {
        "type": "text",
        "text": "\n\n[output truncated: exceeded the runner frame budget]",
    }
    assert capped[2] == image_block
    assert _cap_response_text([{"type": "text", "text": "ok"}]) == [
        {"type": "text", "text": "ok"}
    ]


def test_details_are_capped_to_runner_frame_budget() -> None:
    oversized_entries = [
        {"path": f"datasets/allenai/tmax/task-data/task_{index:06d}", "size": 1}
        for index in range(30_000)
    ]
    capped = _cap_details_size(
        {"entries": oversized_entries, "dataset_path": "datasets/allenai/tmax/"}
    )
    assert len(capped["entries"]) == 50
    assert capped["entries_truncated"] == {"shown": 50, "total": 30_000}
    assert capped["dataset_path"] == "datasets/allenai/tmax/"

    pasted = _cap_details_size({"blob": "x" * 300_000})
    assert pasted["blob"].endswith(
        "[output truncated: exceeded the runner frame budget]"
    )

    def _deep_value(size: int) -> list[dict[str, list[str]]]:
        return [{"nested": ["y" * size]}]

    irreducible = _cap_details_size(
        {
            "entries": _deep_value(150_000),
            "directory_summary": {"sampled_child_folders": _deep_value(150_000)},
        }
    )
    assert irreducible == {
        "truncated": True,
        "keys": ["directory_summary", "entries"],
    }

    untouched = {"entries": [{"path": "a"}], "n": 1}
    assert _cap_details_size(untouched) is untouched


async def test_pi_host_synthesizes_debug_task_and_persists_only_attempt_budget(
    tmp_path,
) -> None:
    repository = Path(pi_adapter_module.__file__).parents[3]
    host = PyromindBusinessToolHost(
        [
            repository / ".agents" / "skills" / "data-processing",
            repository / ".agents" / "skills" / "training-analysis",
        ]
    )

    class FakeAction(BaseModel):
        pass

    class FakeObservation:
        is_error = False
        to_llm_content = ()

        @staticmethod
        def model_dump(*_args, **_kwargs):
            return {
                "task_id": "debug-task",
                "status": "Pending",
                "attempt": 4,
                "max_attempts": 10,
                "keep_ui_lock": True,
            }

    class FakeTool:
        action_type = FakeAction
        executor = None

        def __call__(self, _action, facade):
            facade.state.agent_state["pyromind_workflow_attempts"] = 4
            facade.state.agent_state["must_not_persist"] = "secret"
            return FakeObservation()

    host._factories["workflow_debug"] = cast(Any, lambda _context: FakeTool())
    result = await host.execute(
        "workflow_debug",
        {},
        ToolExecutionContext(
            conversation_id="conversation-1",
            workspace_root=tmp_path,
            request_context=RequestContext(
                user_id="42",
                cookie="auth=request-secret",
                authorization="Bearer request-secret",
            ),
            model_configuration={"model": "gpt-5", "api_key": "request-secret"},
        ),
    )

    assert result["signals"] == [
        {
            "type": "external_task.submitted",
            "task": {
                "task_id": "debug-task",
                "kind": "workflow_debug",
                "status": "Pending",
            },
        }
    ]
    persisted = PiSessionFiles(tmp_path).business_state_path.read_text()
    assert json.loads(persisted) == {"pyromind_workflow_attempts": 4}
    assert "request-secret" not in persisted
    assert "must_not_persist" not in persisted


async def test_facade_registers_session_llm_credentials(tmp_path) -> None:
    """resolve_llm_env must see the session's own working key, not DF_API_KEY."""
    repository = Path(pi_adapter_module.__file__).parents[3]
    host = PyromindBusinessToolHost(
        [
            repository / ".agents" / "skills" / "data-processing",
            repository / ".agents" / "skills" / "training-analysis",
        ]
    )
    seen: dict[str, str | None] = {}

    class FakeAction(BaseModel):
        pass

    class FakeTool:
        action_type = FakeAction
        executor = None

        def __call__(self, _action, facade):
            registry = facade.state.secret_registry
            for name in ("LLM_AUTH_TOKEN", "LLM_BASE_URL", "LLM_MODEL"):
                seen[name] = registry.get_secret_value(name)
            return SimpleNamespace(
                is_error=False,
                to_llm_content=(),
                details={},
                model_dump=lambda mode=None, exclude=None: {},
            )

    host._factories["edp_render"] = cast(Any, lambda _context: FakeTool())
    monkeypatch_invalid_df_key = "sk-or-v1-should-never-be-used"
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("DF_API_KEY", monkeypatch_invalid_df_key)
        await host.execute(
            "edp_render",
            {},
            ToolExecutionContext(
                conversation_id="conversation-1",
                workspace_root=tmp_path,
                request_context=RequestContext(user_id="42"),
                model_configuration={
                    "model": "openai/deepseek-v4-flash-0731",
                    "api_key": "sk-session-key",
                    "base_url": "http://208.64.254.189:8000/v1",
                },
            ),
        )

    assert seen == {
        "LLM_AUTH_TOKEN": "sk-session-key",
        "LLM_BASE_URL": "http://208.64.254.189:8000/v1",
        "LLM_MODEL": "openai/deepseek-v4-flash-0731",
    }


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
        PyromindLLMConfig(model="gpt-5", api=cast(Any, "other"))


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

    await adapter._runner_event(cast(Any, session), frame)
    await adapter._runner_event(cast(Any, session), {**frame, "eventId": "e2"})

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

    await adapter._runner_event(cast(Any, session), frame)

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


async def test_pi_adapter_forks_native_session_and_sanitizes_workspace(
    tmp_path, monkeypatch
) -> None:
    class FakeRunner:
        def __init__(self, **_kwargs) -> None:
            self.running = True

        async def start(self, _config) -> None:
            return None

        async def request(self, method, params):
            assert method == "fork"
            target = Path(params["target_session_dir"]) / "branched.jsonl"
            target.write_text('{"type":"session"}\n', encoding="utf-8")
            return {"session_path": str(target)}

        async def close(self) -> None:
            self.running = False

    monkeypatch.setattr(pi_adapter_module, "PiRunnerProcess", FakeRunner)
    conversations = tmp_path / "conversations"
    conversations.mkdir()
    adapter = PiAdapter(conversations)
    context = RequestContext(user_id="42")
    source_handle = await adapter.create_session(
        SessionSpec(
            conversation_id="source",
            user_id="42",
            workspace_root=str(conversations / "source"),
            model_configuration={"model": "gpt-5", "api_key": "request-secret"},
        ),
        context,
    )
    source = conversations / "source"
    (source / "public_data").mkdir()
    (source / "public_data" / "artifact.txt").write_text("safe", encoding="utf-8")
    current_workflow = source / "public_data" / "workflow_canvas" / "workflow.py"
    current_workflow.parent.mkdir()
    current_workflow.write_text("workflow = OutputNode()", encoding="utf-8")
    (source / "product").mkdir()
    (source / "product" / "snapshot.json").write_text("private", encoding="utf-8")
    (source / ".env").write_text("TOKEN=secret", encoding="utf-8")
    PiSessionFiles(source).save_checkpoint_index({"workflow-v1": "leaf-1"})
    checkpoint = ProductCheckpoint(
        event_id="workflow-v1",
        through_seq=2,
        workflow=WorkflowState(
            resource_id="pyromind_workflow",
            version="v1",
            dsl="workflow = InputNode()",
            canvas=None,
        ),
    )

    target_handle = await adapter.fork(
        source_handle,
        ForkSpec(
            source_conversation_id="source",
            target_conversation_id="target",
            event_id="workflow-v1",
        ),
        checkpoint,
        context,
    )

    target = conversations / "target"
    assert target_handle.session_id == "target"
    assert (target / "public_data" / "artifact.txt").read_text() == "safe"
    assert (target / "public_data" / "workflow_canvas" / "workflow.py").read_text() == (
        "workflow = InputNode()"
    )
    assert not (target / "product").exists()
    assert not (target / ".env").exists()
    assert (target / "pi" / "session.jsonl").is_file()
    assert "request-secret" not in (target / "pi" / "session.json").read_text()
    await adapter.close(target_handle)
    await adapter.close(source_handle)


async def test_pi_adapter_restores_workflow_and_resets_debug_budget(
    tmp_path, monkeypatch
) -> None:
    requests = []

    class FakeRunner:
        def __init__(self, **_kwargs) -> None:
            self.running = True

        async def start(self, _config) -> None:
            return None

        async def request(self, method, params):
            requests.append((method, params))
            return {"accepted": True, "checkpoint_entry_id": "leaf-restored"}

        async def close(self) -> None:
            self.running = False

    monkeypatch.setattr(pi_adapter_module, "PiRunnerProcess", FakeRunner)
    conversations = tmp_path / "conversations"
    conversations.mkdir()
    adapter = PiAdapter(conversations)
    context = RequestContext(user_id="42")
    handle = await adapter.create_session(
        SessionSpec(
            conversation_id="restore-source",
            user_id="42",
            workspace_root=str(conversations / "restore-source"),
            model_configuration={"model": "gpt-5", "api_key": "request-secret"},
        ),
        context,
    )
    root = conversations / "restore-source"
    files = PiSessionFiles(root)
    files.save_business_state({"pyromind_workflow_attempts": 7})
    checkpoint = ProductCheckpoint(
        event_id="workflow-v1",
        through_seq=2,
        workflow=WorkflowState(
            resource_id="pyromind_workflow",
            version="v1",
            dsl="workflow = InputNode()",
            canvas=None,
        ),
    )

    restored = await adapter.restore_workflow(
        handle,
        RestoreWorkflowSpec(command_id="rollback-1", checkpoint=checkpoint),
        context,
    )
    await adapter.notify_external_task(
        handle,
        ExternalTaskNotification(
            task_id="debug-task",
            kind="workflow_debug",
            run_id="debug-task",
            status="succeeded",
            hidden_text="<system_reminder>done</system_reminder>",
            reset_attempt_budget=True,
        ),
        context,
    )

    workflow = root / "public_data" / "workflow_canvas" / "workflow.py"
    assert restored.workflow_file_action == "updated"
    assert workflow.read_text() == "workflow = InputNode()"
    assert files.load_checkpoint_index()["rollback:rollback-1:workflow"] == (
        "leaf-restored"
    )
    assert files.load_business_state() == {"pyromind_workflow_attempts": 0}
    assert [method for method, _params in requests] == ["context.append", "notify"]
    await adapter.close(handle)
