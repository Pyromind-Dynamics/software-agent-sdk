from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar, cast

import harness_adapter.pi_adapter.adapter as pi_adapter_module
import httpx
import pytest
from harness_adapter.pi_adapter import PiAdapter, resolve_pi_terminal_backend
from harness_adapter.pi_adapter.adapter import (
    _is_workflow_mutation,
    _resolve_model,
    _session_config,
)
from harness_adapter.pi_adapter.business_tool_host import (
    PyromindBusinessToolHost,
    ToolExecutionContext,
)
from harness_adapter.pi_adapter.business_tools import (
    execute_validation_tool,
    validation_tool_spec,
)
from harness_adapter.pi_adapter.permissions import TerminalPermissionPolicy
from harness_adapter.pi_adapter.persistence import PiSessionFiles
from harness_adapter.pi_adapter.runner import PiRunnerExit, PlannedPiRunnerExitReason
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


class _LifecycleFakeRunner:
    instances: ClassVar[list[_LifecycleFakeRunner]] = []

    def __init__(self, **kwargs) -> None:
        self.exit_handler = kwargs["exit_handler"]
        self.running = True
        self.pid = 7000 + len(self.instances)
        self.instances.append(self)

    async def start(self, _config) -> None:
        return None

    async def request(self, _method, _params):
        return {"accepted": True}

    async def close(self, *, reason: PlannedPiRunnerExitReason = "shutdown") -> None:
        if not self.running:
            return
        self.running = False
        await self.exit_handler(PiRunnerExit(returncode=0, reason=reason))


def test_validation_schema_is_generated_and_only_exposes_dsl_path() -> None:
    schema = validation_tool_spec()["input_schema"]
    assert "dsl_path" in schema["properties"]
    assert "dsl" not in schema["properties"]
    assert "dsl_path" not in schema.get("required", [])


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
        {},
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


async def test_validation_path_errors_are_returned_as_tool_results(tmp_path) -> None:
    result = await execute_validation_tool(
        tmp_path,
        {"dsl_path": "../workflow.py"},
        RequestContext(user_id="42"),
    )

    assert result["is_error"] is True
    assert "must stay inside the workspace" in json.dumps(result)


def test_terminal_only_confirms_high_risk_commands() -> None:
    policy = TerminalPermissionPolicy()
    assert not policy.requires_confirmation("safe", {"command": "pwd"})
    assert policy.requires_confirmation("danger", {"command": "rm -rf /tmp/example"})


def test_pi_terminal_backend_requires_os_sandbox(monkeypatch) -> None:
    monkeypatch.setenv("APP_ENV", "dev")
    monkeypatch.setenv("PYROMIND_PI_TERMINAL_BACKEND", "os-sandbox")
    assert resolve_pi_terminal_backend() == "os-sandbox"
    with pytest.raises(RuntimeError, match="expected one of: os-sandbox"):
        resolve_pi_terminal_backend(terminal_backend="local")


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
        tmp_path / "explicit", terminal_backend="os-sandbox", knowledge_root=explicit
    )._knowledge_root == (explicit.resolve())
    assert (
        PiAdapter(
            tmp_path / "missing-adapter",
            terminal_backend="os-sandbox",
            knowledge_root=tmp_path / "missing-knowledge",
        )._knowledge_root
        is None
    )

    repository = Path(pi_adapter_module.__file__).parents[3]
    assert (
        PiAdapter(tmp_path / "default", terminal_backend="os-sandbox")._knowledge_root
        == (repository / "knowledge").resolve()
    )


def test_adapter_resolves_skill_roots_from_env(tmp_path, monkeypatch) -> None:
    skills = tmp_path / "skills"
    names = (
        "generate-workflow-dsl",
        "data-cleaning",
        "data-preparation",
        "debug-workflow",
        "training-analysis",
    )
    for name in names:
        (skills / name).mkdir(parents=True)
    monkeypatch.setenv("PYROMIND_SKILLS_PATH", str(skills))
    adapter = PiAdapter(tmp_path / "conversations", terminal_backend="os-sandbox")
    assert adapter._skill_roots == [skills.resolve() / name for name in names]


async def test_runner_start_receives_configured_knowledge_root(
    tmp_path, monkeypatch
) -> None:
    captured = {}

    class FakeRunner:
        def __init__(self, **_kwargs) -> None:
            self.running = True

        async def start(self, config) -> None:
            workspace = Path(config["workspace_root"])
            assert (workspace / "public_data").is_dir()
            assert (workspace / "pi" / "terminal-output").is_dir()
            captured.update(config)

        async def close(self) -> None:
            self.running = False

    monkeypatch.setattr(pi_adapter_module, "PiRunnerProcess", FakeRunner)
    conversations = tmp_path / "conversations"
    conversations.mkdir()
    knowledge = tmp_path / "knowledge"
    knowledge.mkdir()
    adapter = PiAdapter(
        conversations, terminal_backend="os-sandbox", knowledge_root=knowledge
    )
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


async def test_idle_unexpected_runner_exit_is_silent_and_restarts_lazily(
    tmp_path, monkeypatch
) -> None:
    _LifecycleFakeRunner.instances.clear()
    monkeypatch.setattr(pi_adapter_module, "PiRunnerProcess", _LifecycleFakeRunner)
    conversations = tmp_path / "conversations"
    conversations.mkdir()
    adapter = PiAdapter(conversations, terminal_backend="os-sandbox")
    handle = await adapter.create_session(
        SessionSpec(
            conversation_id="idle-exit",
            user_id="42",
            workspace_root=str(conversations / "idle-exit"),
            model_configuration={"model": "gpt-5", "api_key": "test-key"},
        ),
        RequestContext(user_id="42"),
    )
    session = adapter._session(handle.session_id)
    original = _LifecycleFakeRunner.instances[0]
    history = await session.queue.get()
    assert history is not None
    assert history.type == "history.synced"

    original.running = False
    await original.exit_handler(PiRunnerExit(returncode=-2, reason="unexpected"))

    assert session.runner is None
    assert session.queue.empty()
    assert session.files.load_inflight() is None

    await adapter._ensure_runner(session)
    assert len(_LifecycleFakeRunner.instances) == 2
    assert session.runner is _LifecycleFakeRunner.instances[1]
    await adapter.close(handle)


async def test_busy_unexpected_runner_exit_fails_original_run_once(
    tmp_path, monkeypatch
) -> None:
    _LifecycleFakeRunner.instances.clear()
    monkeypatch.setattr(pi_adapter_module, "PiRunnerProcess", _LifecycleFakeRunner)
    conversations = tmp_path / "conversations"
    conversations.mkdir()
    adapter = PiAdapter(conversations, terminal_backend="os-sandbox")
    handle = await adapter.create_session(
        SessionSpec(
            conversation_id="busy-exit",
            user_id="42",
            workspace_root=str(conversations / "busy-exit"),
            model_configuration={"model": "gpt-5", "api_key": "test-key"},
        ),
        RequestContext(user_id="42"),
    )
    session = adapter._session(handle.session_id)
    runner = _LifecycleFakeRunner.instances[0]
    history = await session.queue.get()
    assert history is not None
    assert history.type == "history.synced"
    session.files.save_inflight({"run_id": "run-active"})

    runner.running = False
    exit_result = PiRunnerExit(returncode=-9, reason="unexpected")
    await runner.exit_handler(exit_result)
    await runner.exit_handler(exit_result)

    events = []
    while not session.queue.empty():
        events.append(session.queue.get_nowait())
    notices = [event for event in events if event.type == "notice.raised"]
    assert len(notices) == 1
    assert notices[0].run_id == "run-active"
    assert notices[0].payload["code"] == "pi_runner_exited"
    assert session.files.load_inflight() is None
    await adapter.close(handle)


async def test_stale_runner_exit_does_not_clear_replacement_or_emit_error(
    tmp_path, monkeypatch
) -> None:
    _LifecycleFakeRunner.instances.clear()
    monkeypatch.setattr(pi_adapter_module, "PiRunnerProcess", _LifecycleFakeRunner)
    conversations = tmp_path / "conversations"
    conversations.mkdir()
    adapter = PiAdapter(conversations, terminal_backend="os-sandbox")
    handle = await adapter.create_session(
        SessionSpec(
            conversation_id="stale-exit",
            user_id="42",
            workspace_root=str(conversations / "stale-exit"),
            model_configuration={"model": "gpt-5", "api_key": "test-key"},
        ),
        RequestContext(user_id="42"),
    )
    session = adapter._session(handle.session_id)
    original = _LifecycleFakeRunner.instances[0]
    history = await session.queue.get()
    assert history is not None
    assert history.type == "history.synced"

    await adapter._start_runner(session, "test-key")
    replacement = _LifecycleFakeRunner.instances[1]
    original.running = False
    await original.exit_handler(PiRunnerExit(returncode=-2, reason="unexpected"))

    assert session.runner is replacement
    assert session.queue.empty()
    await adapter.close(handle)


async def test_planned_shutdown_preserves_inflight_for_attach_recovery(
    tmp_path, monkeypatch
) -> None:
    _LifecycleFakeRunner.instances.clear()
    monkeypatch.setattr(pi_adapter_module, "PiRunnerProcess", _LifecycleFakeRunner)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    conversations = tmp_path / "conversations"
    conversations.mkdir()
    adapter = PiAdapter(conversations, terminal_backend="os-sandbox")
    handle = await adapter.create_session(
        SessionSpec(
            conversation_id="shutdown-recovery",
            user_id="42",
            workspace_root=str(conversations / "shutdown-recovery"),
            model_configuration={"model": "gpt-5", "api_key": "test-key"},
        ),
        RequestContext(user_id="42"),
    )
    session = adapter._session(handle.session_id)
    runner = _LifecycleFakeRunner.instances[0]
    history = await session.queue.get()
    assert history is not None
    assert history.type == "history.synced"
    session.files.save_inflight(
        {"run_id": "run-interrupted", "operation_id": "tool-interrupted"}
    )

    runner.running = False
    await runner.exit_handler(PiRunnerExit(returncode=0, reason="shutdown"))

    assert session.runner is None
    assert session.queue.empty()
    assert session.files.load_inflight() == {
        "run_id": "run-interrupted",
        "operation_id": "tool-interrupted",
    }
    await adapter.close(handle)

    restarted = PiAdapter(conversations, terminal_backend="os-sandbox")
    restarted_handle = await restarted.attach_session(
        "shutdown-recovery", RequestContext(user_id="42")
    )
    restarted_session = restarted._session(restarted_handle.session_id)
    recovered = []
    while not restarted_session.queue.empty():
        recovered.append(restarted_session.queue.get_nowait())

    failures = [event for event in recovered if event.type == "operation.failed"]
    assert len(failures) == 1
    assert failures[0].run_id == "run-interrupted"
    assert failures[0].payload["error_code"] == "runner_restarted"
    assert restarted_session.files.load_inflight() is None
    await restarted.close(restarted_handle)


async def test_create_stages_public_data_and_workflow_before_runner(
    tmp_path, monkeypatch
) -> None:
    expected_dsl = "workflow = InputNode()"

    class FakeRunner:
        def __init__(self, **_kwargs) -> None:
            self.running = True

        async def start(self, config) -> None:
            workflow = (
                Path(config["workspace_root"])
                / "public_data"
                / "workflow_canvas"
                / "workflow.py"
            )
            assert workflow.read_text(encoding="utf-8") == expected_dsl

        async def request(self, method, _params):
            assert method == "context.append"
            return {"checkpoint_entry_id": "initial-workflow"}

        async def close(self) -> None:
            self.running = False

    monkeypatch.setattr(pi_adapter_module, "PiRunnerProcess", FakeRunner)
    monkeypatch.setattr(
        pi_adapter_module, "convert_xyflow_to_dsl", lambda _canvas: expected_dsl
    )
    conversations = tmp_path / "conversations"
    conversations.mkdir()
    adapter = PiAdapter(conversations, terminal_backend="os-sandbox")
    handle = await adapter.create_session(
        SessionSpec(
            conversation_id="workflow-first",
            user_id="42",
            workspace_root=str(conversations / "workflow-first"),
            workflow_xyflow={"nodes": [], "edges": []},
            model_configuration={"model": "gpt-5", "api_key": "test-key"},
        ),
        RequestContext(user_id="42"),
    )
    try:
        root = conversations / "workflow-first"
        assert (root / "public_data").is_dir()
        assert (root / "pi" / "terminal-output").is_dir()
    finally:
        await adapter.close(handle)


async def test_create_failure_removes_only_new_workspace(tmp_path, monkeypatch) -> None:
    class FailingRunner:
        def __init__(self, **_kwargs) -> None:
            self.running = True

        async def start(self, _config) -> None:
            raise RuntimeError("sandbox unavailable")

        async def close(self) -> None:
            self.running = False

    monkeypatch.setattr(pi_adapter_module, "PiRunnerProcess", FailingRunner)
    conversations = tmp_path / "conversations"
    conversations.mkdir()
    adapter = PiAdapter(conversations, terminal_backend="os-sandbox")

    with pytest.raises(RuntimeError, match="sandbox unavailable"):
        await adapter.create_session(
            SessionSpec(
                conversation_id="failed-create",
                user_id="42",
                workspace_root=str(conversations / "failed-create"),
                model_configuration={"model": "gpt-5", "api_key": "test-key"},
            ),
            RequestContext(user_id="42"),
        )

    assert not (conversations / "failed-create").exists()


async def test_attach_repairs_missing_public_data_before_runner(
    tmp_path, monkeypatch
) -> None:
    class FakeRunner:
        def __init__(self, **_kwargs) -> None:
            self.running = True

        async def start(self, config) -> None:
            assert (Path(config["workspace_root"]) / "public_data").is_dir()

        async def close(self) -> None:
            self.running = False

    monkeypatch.setattr(pi_adapter_module, "PiRunnerProcess", FakeRunner)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    conversations = tmp_path / "conversations"
    root = conversations / "attach-source"
    root.mkdir(parents=True)
    PiSessionFiles(root).initialize(
        {
            "session_id": "attach-source",
            "model": {"provider": "openai", "id": "gpt-5"},
        }
    )
    adapter = PiAdapter(conversations, terminal_backend="os-sandbox")
    handle = await adapter.attach_session("attach-source", RequestContext(user_id="42"))
    try:
        assert (root / "public_data").is_dir()
    finally:
        await adapter.close(handle)


async def test_attach_rejects_symlinked_public_data(tmp_path) -> None:
    conversations = tmp_path / "conversations"
    root = conversations / "unsafe-attach"
    outside = tmp_path / "outside"
    root.mkdir(parents=True)
    outside.mkdir()
    (root / "public_data").symlink_to(outside, target_is_directory=True)
    PiSessionFiles(root).initialize(
        {
            "session_id": "unsafe-attach",
            "model": {"provider": "openai", "id": "gpt-5"},
        }
    )
    adapter = PiAdapter(conversations, terminal_backend="os-sandbox")

    with pytest.raises(RuntimeError, match="public_data must not be a symbolic link"):
        await adapter.attach_session("unsafe-attach", RequestContext(user_id="42"))


async def test_attach_rejects_symlinked_conversation_root(tmp_path) -> None:
    conversations = tmp_path / "conversations"
    outside = tmp_path / "outside-conversation"
    conversations.mkdir()
    outside.mkdir()
    (conversations / "unsafe-root").symlink_to(outside, target_is_directory=True)
    adapter = PiAdapter(conversations, terminal_backend="os-sandbox")

    with pytest.raises(RuntimeError, match="conversation root must not be"):
        await adapter.attach_session("unsafe-root", RequestContext(user_id="42"))


async def test_attach_rejects_symlinked_pi_before_loading_state(tmp_path) -> None:
    conversations = tmp_path / "conversations"
    root = conversations / "unsafe-pi"
    outside = tmp_path / "outside-pi"
    root.mkdir(parents=True)
    (root / "public_data").mkdir()
    outside.mkdir()
    (outside / "session.json").write_text(
        '{"session_id":"unsafe-pi","model":{"provider":"openai","id":"gpt-5"}}',
        encoding="utf-8",
    )
    (outside / "session.jsonl").write_text("", encoding="utf-8")
    (root / "pi").symlink_to(outside, target_is_directory=True)
    adapter = PiAdapter(conversations, terminal_backend="os-sandbox")

    with pytest.raises(RuntimeError, match="pi must not be a symbolic link"):
        await adapter.attach_session("unsafe-pi", RequestContext(user_id="42"))


async def test_runner_loads_five_named_skills_and_eleven_business_tools(
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
    adapter = PiAdapter(conversations, terminal_backend="os-sandbox")
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
            "workflow_debug",
            "analyze_task_failure",
            "training_analysis",
        }
    finally:
        await adapter.close(handle)


def test_business_tool_specs_are_generated_from_openhands_definitions() -> None:
    repository = Path(pi_adapter_module.__file__).parents[3]
    roots = [
        repository / ".agents" / "skills" / "data-cleaning",
        repository / ".agents" / "skills" / "data-preparation",
        repository / ".agents" / "skills" / "training-analysis",
    ]
    specs = PyromindBusinessToolHost(roots).specs()
    assert len(specs) == 11
    assert all(spec["input_schema"].get("type") == "object" for spec in specs)


async def test_preview_dataset_timeout_returns_a_tool_error(tmp_path) -> None:
    repository = Path(pi_adapter_module.__file__).parents[3]
    host = PyromindBusinessToolHost(
        [
            repository / ".agents" / "skills" / "data-cleaning",
            repository / ".agents" / "skills" / "data-preparation",
            repository / ".agents" / "skills" / "training-analysis",
        ]
    )

    class FakeAction(BaseModel):
        dataset_path: str

    class SlowPreviewTool:
        action_type = FakeAction
        executor = None

        def __call__(self, _action, _facade):
            time.sleep(0.05)
            return object()

    host._factories["preview_dataset"] = cast(Any, lambda _context: SlowPreviewTool())
    result = await host.execute(
        "preview_dataset",
        {"dataset_path": "datasets/large"},
        ToolExecutionContext(
            conversation_id="conversation-timeout",
            workspace_root=tmp_path,
            request_context=RequestContext(user_id="42"),
            model_configuration={"model": "gpt-5"},
            extra={"preview_dataset_timeout_seconds": 0.01},
        ),
    )

    assert result["is_error"] is True
    assert result["details"] == {
        "error_code": "tool_timeout",
        "timeout_seconds": 0.01,
    }
    assert "Narrow dataset_path" in result["content"][0]["text"]


async def test_pi_host_synthesizes_debug_task_and_persists_only_attempt_budget(
    tmp_path,
) -> None:
    repository = Path(pi_adapter_module.__file__).parents[3]
    host = PyromindBusinessToolHost(
        [
            repository / ".agents" / "skills" / "data-cleaning",
            repository / ".agents" / "skills" / "data-preparation",
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

    config = _session_config(
        SessionSpec(
            conversation_id="conversation-timeout",
            user_id="42",
            workspace_root="/tmp/conversation-timeout",
            model_configuration={"model": "gpt-5"},
            extra={
                "preview_dataset_timeout_seconds": 45,
                "untrusted_extra": "ignored",
            },
        )
    )
    assert config["extra"] == {"preview_dataset_timeout_seconds": 45}


def test_pyromind_llm_config_rejects_unknown_model_api() -> None:
    with pytest.raises(ValidationError):
        PyromindLLMConfig(model="gpt-5", api=cast(Any, "other"))


def test_pyromind_llm_config_rejects_invalid_context_window() -> None:
    with pytest.raises(ValidationError):
        PyromindLLMConfig(model="gpt-5", context_window=0)


async def test_adapter_ignores_duplicate_run_finished(tmp_path) -> None:
    adapter = PiAdapter(tmp_path, terminal_backend="os-sandbox")
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
    adapter = PiAdapter(tmp_path, terminal_backend="os-sandbox")
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


async def test_runner_start_does_not_persist_request_api_key(
    tmp_path, monkeypatch
) -> None:
    class FakeRunner:
        def __init__(self, **_kwargs) -> None:
            self.running = True

        async def start(self, _config) -> None:
            return None

        async def close(self) -> None:
            self.running = False

    monkeypatch.setattr(pi_adapter_module, "PiRunnerProcess", FakeRunner)
    conversations = tmp_path / "conversations"
    conversations.mkdir()
    adapter = PiAdapter(conversations, terminal_backend="os-sandbox")
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
    adapter = PiAdapter(conversations, terminal_backend="os-sandbox")
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
    (source / "public_data" / "artifact.txt").write_text("safe", encoding="utf-8")
    current_workflow = source / "public_data" / "workflow_canvas" / "workflow.py"
    current_workflow.parent.mkdir()
    current_workflow.write_text("workflow = OutputNode()", encoding="utf-8")
    (source / "product").mkdir()
    (source / "product" / "snapshot.json").write_text("private", encoding="utf-8")
    (source / ".env").write_text("TOKEN=secret", encoding="utf-8")
    (source / "public_data" / "secret-link").symlink_to(source / ".env")
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
    assert not (target / "public_data" / "secret-link").exists()
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
    adapter = PiAdapter(conversations, terminal_backend="os-sandbox")
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
