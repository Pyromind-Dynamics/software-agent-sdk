from __future__ import annotations

import shutil
from pathlib import Path

import httpx
import pytest
from pyromind_runtime.adapters.pi import (
    LocalPiRunnerLauncher,
    PiAdapter,
    StaticPiModelConfigResolver,
    safe_runner_environment,
)
from pyromind_runtime.contracts import (
    ModelProfile,
    SandboxRef,
    SessionSpec,
    WorkspaceRef,
)
from pyromind_runtime.tool_host import PythonToolHost, first_version_tool_specs

from openhands.agent_server.api import create_app
from openhands.agent_server.config import Config


_REPO_ROOT = Path(__file__).parents[3]
_PI_RUNTIME = _REPO_ROOT / "openhands-agent-server" / "pi-runtime"
_NODE = shutil.which("node")


@pytest.mark.skipif(
    _NODE is None or not (_PI_RUNTIME / "node_modules").is_dir(),
    reason="Node or the locked Pi runtime dependencies are unavailable",
)
async def test_python_adapter_handshakes_with_the_real_pi_runner(tmp_path) -> None:
    workspace_root = tmp_path / "workspaces"
    launcher = LocalPiRunnerLauncher(
        command=(_NODE or "node", str(_PI_RUNTIME / "src" / "index.ts")),
        workspace_root=workspace_root,
        cwd=str(_PI_RUNTIME),
        environment=safe_runner_environment(),
        model_resolver=StaticPiModelConfigResolver(
            profile_id="default",
            provider="openai",
            model_id="gpt-5.5",
            api_key="model-token",
        ),
        tool_host=PythonToolHost(),
        system_prompt="Test system prompt",
    )
    adapter = PiAdapter(launcher)
    spec = SessionSpec(
        product_session_id="session-1",
        user_id="user-1",
        workspace=WorkspaceRef(
            workspace_id="session-1",
            root=str(workspace_root / "session-1"),
        ),
        sandbox=SandboxRef(sandbox_id="session-1", backend="local"),
        model_profile=ModelProfile(profile_id="default"),
        tools=first_version_tool_specs(),
    )

    handle = await adapter.create_session(spec)
    await adapter.close(handle.session_id)

    assert handle.harness_id == "pi"
    assert (workspace_root / "session-1").is_dir()


@pytest.mark.skipif(
    _NODE is None or not (_PI_RUNTIME / "node_modules").is_dir(),
    reason="Node or the locked Pi runtime dependencies are unavailable",
)
async def test_create_conversation_starts_local_pi_without_sandbox_config(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("PYROMIND_DEFAULT_HARNESS", "pi")
    monkeypatch.setenv("PYROMIND_PI_MODEL_API_KEY", "model-token")
    monkeypatch.setenv("PYROMIND_PI_SKILLS", "")
    for name in (
        "PYROMIND_API_KEY",
        "PYROMIND_SANDBOX_IMAGE",
        "PYROMIND_SANDBOX_TYPE",
        "PYROMIND_CLUSTER",
    ):
        monkeypatch.delenv(name, raising=False)
    app = create_app(
        Config(
            conversations_path=tmp_path / "legacy-conversations",
            workspace_path=tmp_path / "workspaces",
            enable_session_api_key_auth=False,
            enable_pyromind_jwt_auth=False,
        )
    )
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        response = await client.post("/api/v2/pyromind/conversations", json={})

    assert response.status_code == 201
    conversation_id = response.json()["conversation_id"]
    assert (tmp_path / "workspaces" / conversation_id).is_dir()
    await app.state.product_runtime.close()
