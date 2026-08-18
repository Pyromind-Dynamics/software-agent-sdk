from __future__ import annotations

import os
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path

import pytest
from pydantic import JsonValue
from pyromind_runtime.adapters.pi import (
    BoundedSandboxGateway,
    LocalPiRunnerLauncher,
    PiRunnerCallbackRouter,
    PiWorkflowMirror,
    PyromindSandboxLeaseProvider,
    SandboxedPiRunnerLauncher,
    StaticPiModelConfigResolver,
)
from pyromind_runtime.contracts import (
    ModelProfile,
    SandboxRef,
    SessionSpec,
    TextContentBlock,
    ToolResult,
    WorkspaceRef,
)
from pyromind_runtime.contracts.sandbox import (
    SandboxExecResult,
    SandboxFileInfo,
)
from pyromind_runtime.tool_host import (
    PREVIEW_DATASET_TOOL_SPEC,
    PythonToolHost,
    ToolExecutionContext,
)


class LocalTestSandboxBackend:
    def __init__(self) -> None:
        self.exec_calls: list[tuple[str, str, Mapping[str, str], float]] = []
        self.create_requests: list[dict[str, JsonValue]] = []
        self.deleted_sandboxes: list[str] = []
        self.closed = False

    async def create_sandbox(self, request: dict[str, JsonValue]) -> str:
        self.create_requests.append(request)
        return "sandbox-1"

    async def wait_until_running(
        self,
        sandbox_id: str,
        *,
        timeout_seconds: float = 300.0,
    ) -> None:
        assert sandbox_id == "sandbox-1"
        assert timeout_seconds > 0

    async def delete_sandbox(self, sandbox_id: str) -> None:
        self.deleted_sandboxes.append(sandbox_id)

    async def exists(self, sandbox_id: str, path: str) -> bool:
        return Path(path).exists() or Path(path).is_symlink()

    async def canonical_path(self, sandbox_id: str, path: str) -> str:
        return str(Path(path).resolve(strict=True))

    async def read_file(self, sandbox_id: str, path: str) -> bytes:
        return Path(path).read_bytes()

    async def write_file(
        self,
        sandbox_id: str,
        path: str,
        content: bytes,
    ) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)

    async def append_file(
        self,
        sandbox_id: str,
        path: str,
        content: bytes,
    ) -> None:
        with Path(path).open("ab") as stream:
            stream.write(content)

    async def rename_file(
        self,
        sandbox_id: str,
        source_path: str,
        destination_path: str,
    ) -> None:
        Path(source_path).replace(destination_path)

    async def file_info(
        self,
        sandbox_id: str,
        path: str,
    ) -> SandboxFileInfo:
        target = Path(path)
        stat = target.lstat()
        if target.is_symlink():
            kind = "symlink"
        elif target.is_dir():
            kind = "directory"
        else:
            kind = "file"
        return SandboxFileInfo(
            name=target.name,
            path=str(target),
            kind=kind,
            size=stat.st_size,
            mtime_ms=stat.st_mtime * 1000,
        )

    async def list_dir(
        self,
        sandbox_id: str,
        path: str,
    ) -> tuple[SandboxFileInfo, ...]:
        return tuple(
            [
                await self.file_info(sandbox_id, str(child))
                for child in Path(path).iterdir()
            ]
        )

    async def create_dir(
        self,
        sandbox_id: str,
        path: str,
        *,
        recursive: bool,
    ) -> None:
        Path(path).mkdir(parents=recursive)

    async def remove(
        self,
        sandbox_id: str,
        path: str,
        *,
        recursive: bool,
        force: bool,
    ) -> None:
        target = Path(path)
        if not target.exists() and force:
            return
        if target.is_dir() and recursive:
            shutil.rmtree(target)
        elif target.is_dir():
            target.rmdir()
        else:
            target.unlink()

    async def create_temp_dir(
        self,
        sandbox_id: str,
        parent: str,
        prefix: str,
    ) -> str:
        return tempfile.mkdtemp(prefix=prefix, dir=parent)

    async def create_temp_file(
        self,
        sandbox_id: str,
        parent: str,
        prefix: str,
        suffix: str,
    ) -> str:
        descriptor, path = tempfile.mkstemp(prefix=prefix, suffix=suffix, dir=parent)
        os.close(descriptor)
        return path

    async def exec(
        self,
        sandbox_id: str,
        command: str,
        *,
        cwd: str,
        environment: Mapping[str, str],
        timeout_seconds: float,
    ) -> SandboxExecResult:
        self.exec_calls.append((command, cwd, environment, timeout_seconds))
        return SandboxExecResult(stdout="x" * 100, stderr="", exit_code=0)

    async def close(self) -> None:
        self.closed = True


def _object(value: JsonValue) -> dict[str, JsonValue]:
    assert isinstance(value, dict)
    return value


async def test_sandbox_gateway_rejects_lexical_and_symlink_escape(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (workspace / "escape").symlink_to(outside, target_is_directory=True)
    gateway = BoundedSandboxGateway(
        LocalTestSandboxBackend(),
        sandbox_id="sandbox-1",
        workspace_root=str(workspace),
    )

    lexical = await gateway.handle("env.readTextFile", {"path": "../outside/data"})
    symlink = await gateway.handle("env.readTextFile", {"path": "escape/data"})

    assert lexical == {
        "ok": False,
        "error": {
            "code": "permission_denied",
            "message": "Path escapes workspace",
            "path": str(outside / "data"),
        },
    }
    symlink_object = _object(symlink)
    assert symlink_object["ok"] is False
    assert _object(symlink_object["error"])["code"] == "permission_denied"


async def test_sandbox_gateway_bounds_files_output_and_environment(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    backend = LocalTestSandboxBackend()
    gateway = BoundedSandboxGateway(
        backend,
        sandbox_id="sandbox-1",
        workspace_root=str(workspace),
        environment={"LANG": "C.UTF-8"},
        max_output_bytes=10,
    )

    written = await gateway.handle(
        "env.writeFile",
        {"path": "data.txt", "content": "hello", "encoding": "utf8"},
    )
    read = await gateway.handle("env.readTextFile", {"path": "data.txt"})
    denied = await gateway.handle(
        "env.exec",
        {"command": "pwd", "env": {"OPENAI_API_KEY": "secret"}},
    )
    executed = await gateway.handle(
        "env.exec",
        {"command": "pwd", "env": {"LANG": "C"}, "timeout": 2},
    )

    assert written == {"ok": True, "value": None}
    assert read == {"ok": True, "value": "hello"}
    denied_object = _object(denied)
    assert denied_object["ok"] is False
    assert _object(denied_object["error"])["code"] == "permission_denied"
    assert executed == {
        "ok": True,
        "value": {"stdout": "x" * 10, "stderr": "", "exitCode": 0},
    }
    assert backend.exec_calls[0][2] == {"LANG": "C"}


async def test_pi_callback_router_executes_only_session_tools(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    spec = SessionSpec(
        product_session_id="session-1",
        user_id="user-1",
        workspace=WorkspaceRef(workspace_id="workspace-1", root=str(workspace)),
        sandbox=SandboxRef(sandbox_id="sandbox-1", backend="test"),
        model_profile=ModelProfile(profile_id="test"),
        tools=(PREVIEW_DATASET_TOOL_SPEC,),
    )
    seen_context: ToolExecutionContext | None = None
    host = PythonToolHost()

    async def preview_handler(arguments, context):
        nonlocal seen_context
        seen_context = context
        return ToolResult(
            content=(TextContentBlock(text=str(arguments["dataset_path"])),),
            details={"source": "test"},
        )

    host.register(PREVIEW_DATASET_TOOL_SPEC, preview_handler)
    router = PiRunnerCallbackRouter(
        spec,
        BoundedSandboxGateway(
            LocalTestSandboxBackend(),
            sandbox_id="sandbox-1",
            workspace_root=str(workspace),
        ),
        host,
    )

    result = await router.handle(
        "tool.execute",
        {"tool_name": "preview_dataset", "arguments": {"dataset_path": "data/a"}},
    )
    result_object = _object(result)
    assert result_object["is_error"] is False
    assert result_object["details"] == {"source": "test"}
    assert seen_context is not None
    assert seen_context.session_id == "session-1"


async def test_sandboxed_pi_launcher_acquires_and_releases_one_lease(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    skill_directory = tmp_path / "host-skills" / "generate-workflow-dsl"
    (skill_directory / "references").mkdir(parents=True)
    (skill_directory / "SKILL.md").write_text(
        "---\nname: generate-workflow-dsl\n"
        "description: Generate workflow DSL.\n---\nInstructions.\n",
        encoding="utf-8",
    )
    (skill_directory / "references" / "contracts.md").write_text(
        "contracts",
        encoding="utf-8",
    )
    (skill_directory / "ignored.pyc").write_bytes(b"ignored")
    backend = LocalTestSandboxBackend()
    provider = PyromindSandboxLeaseProvider(
        lambda spec: backend,
        create_request={"sandbox_type": "swebench", "image": "python:3.13"},
        workspace_root=str(workspace),
        environment={"PATH": "/usr/bin:/bin"},
    )
    launcher = SandboxedPiRunnerLauncher(
        command=("node", "src/index.ts"),
        lease_provider=provider,
        model_resolver=StaticPiModelConfigResolver(
            profile_id="test",
            provider="openai",
            model_id="gpt-5.5",
            api_key="model-token",
        ),
        tool_host=PythonToolHost(),
        system_prompt="Test system prompt",
        skill_directories=(skill_directory,),
    )
    spec = SessionSpec(
        product_session_id="session-1",
        user_id="user-1",
        workspace=WorkspaceRef(workspace_id="workspace-1", root=str(workspace)),
        sandbox=SandboxRef(sandbox_id="requested", backend="pyromind"),
        model_profile=ModelProfile(profile_id="test"),
    )

    launch = await launcher.prepare(spec)

    assert workspace.is_dir()
    assert backend.create_requests == [
        {
            "sandbox_type": "swebench",
            "image": "python:3.13",
            "name": "pi-session-1",
        }
    ]
    assert launch.runtime_config["workspace_root"] == str(workspace)
    assert launch.runtime_config["native_tools"] == ["read", "write", "edit", "bash"]
    assert launch.runtime_config["skill_dirs"] == [str(workspace / ".pyromind/skills")]
    published_skill = workspace / ".pyromind/skills/generate-workflow-dsl"
    assert (published_skill / "SKILL.md").is_file()
    assert (published_skill / "references/contracts.md").read_text() == "contracts"
    assert not (published_skill / "ignored.pyc").exists()
    assert launch.request_handler is not None
    assert await launch.request_handler(
        "env.absolutePath", {"path": "created.txt"}
    ) == {"ok": True, "value": str(workspace / "created.txt")}
    assert launch.cleanup is not None
    await launch.cleanup()
    await launch.cleanup()

    assert backend.deleted_sandboxes == ["sandbox-1"]
    assert backend.closed


async def test_local_pi_launcher_uses_node_execution_env(tmp_path) -> None:
    workspace_root = tmp_path / "workspaces"
    session_workspace = workspace_root / "session-1"
    skill_directory = tmp_path / "skills" / "generate-workflow-dsl"
    skill_directory.mkdir(parents=True)
    (skill_directory / "SKILL.md").write_text("Instructions.\n", encoding="utf-8")
    launcher = LocalPiRunnerLauncher(
        command=("node", "src/index.ts"),
        workspace_root=workspace_root,
        model_resolver=StaticPiModelConfigResolver(
            profile_id="test",
            provider="openai",
            model_id="gpt-5.5",
            api_key="model-token",
        ),
        tool_host=PythonToolHost(),
        system_prompt="Test system prompt",
        skill_directories=(skill_directory,),
    )
    spec = SessionSpec(
        product_session_id="session-1",
        user_id="user-1",
        workspace=WorkspaceRef(
            workspace_id="workspace-1",
            root=str(session_workspace),
        ),
        sandbox=SandboxRef(sandbox_id="unused", backend="local"),
        model_profile=ModelProfile(profile_id="test"),
    )

    launch = await launcher.prepare(spec)

    assert session_workspace.is_dir()
    assert launch.runtime_config["execution_env"] == "node"
    assert launch.runtime_config["workspace_root"] == str(session_workspace)
    assert launch.runtime_config["skill_dirs"] == [str(skill_directory)]
    assert launch.request_handler is not None
    with pytest.raises(ValueError, match="unavailable for a local Pi session"):
        await launch.request_handler("env.absolutePath", {"path": "file.txt"})


async def test_local_pi_launcher_rejects_workspace_outside_configured_root(
    tmp_path,
) -> None:
    launcher = LocalPiRunnerLauncher(
        command=("node", "src/index.ts"),
        workspace_root=tmp_path / "workspaces",
        model_resolver=StaticPiModelConfigResolver(
            profile_id="test",
            provider="openai",
            model_id="gpt-5.5",
            api_key="model-token",
        ),
        tool_host=PythonToolHost(),
        system_prompt="Test system prompt",
    )
    spec = SessionSpec(
        product_session_id="session-1",
        user_id="user-1",
        workspace=WorkspaceRef(
            workspace_id="workspace-1",
            root=str(tmp_path / "outside"),
        ),
        sandbox=SandboxRef(sandbox_id="unused", backend="local"),
        model_profile=ModelProfile(profile_id="test"),
    )

    with pytest.raises(ValueError, match="must be a session directory"):
        await launcher.prepare(spec)


async def test_pi_callback_mirrors_only_the_fixed_workflow_resource(tmp_path) -> None:
    remote_workspace = tmp_path / "remote"
    remote_workspace.mkdir()
    product_workspace = tmp_path / "product"
    backend = LocalTestSandboxBackend()
    spec = SessionSpec(
        product_session_id="session-1",
        user_id="user-1",
        workspace=WorkspaceRef(
            workspace_id="workspace-1",
            root=str(product_workspace),
        ),
        sandbox=SandboxRef(sandbox_id="sandbox-1", backend="test"),
        model_profile=ModelProfile(profile_id="test"),
    )
    router = PiRunnerCallbackRouter(
        spec,
        BoundedSandboxGateway(
            backend,
            sandbox_id="sandbox-1",
            workspace_root=str(remote_workspace),
        ),
        PythonToolHost(),
        workflow_mirror=PiWorkflowMirror(
            remote_workspace_root=str(remote_workspace),
            local_workspace_root=str(product_workspace),
        ),
    )

    await router.handle(
        "env.writeFile",
        {
            "path": str(remote_workspace / "public_data/workflow_canvas/workflow.py"),
            "content": "workflow = Workflow()",
            "encoding": "utf8",
        },
    )
    await router.handle(
        "env.writeFile",
        {
            "path": str(remote_workspace / "notes.txt"),
            "content": "not mirrored",
            "encoding": "utf8",
        },
    )

    mirrored = product_workspace / "public_data/workflow_canvas/workflow.py"
    assert mirrored.read_text(encoding="utf-8") == "workflow = Workflow()"
    assert not (product_workspace / "notes.txt").exists()


async def test_workflow_mirror_rejects_state_symlink_before_replacing_workflow(
    tmp_path,
) -> None:
    product_workspace = tmp_path / "product-workspace"
    workflow_directory = product_workspace / "public_data" / "workflow_canvas"
    workflow_directory.mkdir(parents=True)
    workflow_path = workflow_directory / "workflow.py"
    workflow_path.write_text("original", encoding="utf-8")
    outside = tmp_path / "outside-state.json"
    outside.write_text("outside", encoding="utf-8")
    (workflow_directory / "state.json").symlink_to(outside)
    mirror = PiWorkflowMirror(
        remote_workspace_root="/workspace",
        local_workspace_root=str(product_workspace),
    )

    with pytest.raises(ValueError, match="state target must not be a symlink"):
        await mirror.write(b"replacement")

    assert workflow_path.read_text(encoding="utf-8") == "original"
    assert outside.read_text(encoding="utf-8") == "outside"
