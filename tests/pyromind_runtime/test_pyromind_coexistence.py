import os
import subprocess
import sys
from pathlib import Path

import httpx

from pyromind_agent_server import bootstrap
from pyromind_agent_server.app import create_app
from pyromind_agent_server.config import Config
from pyromind_runtime.application.conversation_runtime import ConversationRuntime


ROOT = Path(__file__).resolve().parents[2]


def _python(script: str) -> None:
    subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        env={**os.environ, "OPENHANDS_SUPPRESS_BANNER": "1"},
        check=True,
        capture_output=True,
        text=True,
    )


def test_product_and_pi_import_without_the_legacy_namespace():
    _python("""
import importlib.abc
import sys
class RejectLegacy(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "openhands" or fullname.startswith("openhands."):
            raise AssertionError("legacy dependency: " + fullname)
sys.meta_path.insert(0, RejectLegacy())
from pyromind_agent_server.app import create_app
from harness_adapter.pi_adapter.business_tool_host import PyromindBusinessToolHost
from pyromind_common.tools.workflow.task_submission import workflow_submission_scope
paths = create_app().openapi()["paths"]
assert "/api/v2/pyromind/conversations" in paths
assert "/api/v2/echomind/conversations" in paths
assert "/api/conversations" not in paths
""")


def test_canonical_sandbox_bundle_can_start_without_legacy_packages():
    bundle = ROOT / ".agents/skills/data-processing/scripts/edp"
    subprocess.run(
        [sys.executable, str(bundle / "sandbox_runner.py"), "--help"],
        env={**os.environ, "PYTHONPATH": str(bundle / "pod_runtime")},
        check=True,
        capture_output=True,
        text=True,
    )


async def test_deferred_product_service_initializes_pi_and_closes(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("APP_ENV", "dev")
    monkeypatch.setenv("PYROMIND_PI_TERMINAL_BACKEND", "os-sandbox")
    monkeypatch.setenv("PYROMIND_AGENTGENOME_ENABLED", "0")
    monkeypatch.setattr(bootstrap, "build_experiment_plan_store", lambda: None)
    app = create_app(
        Config(conversations_path=tmp_path / "dormant", deferred_init=True)
    )
    async with app.router.lifespan_context(app):
        assert app.state.product_runtime is None
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get("/api/v2/pyromind/conversations")
            assert response.status_code == 503
            response = await client.post(
                "/api/init", json={"conversations_path": str(tmp_path / "mounted")}
            )
            assert response.status_code == 200
            runtime = app.state.product_runtime
            assert isinstance(runtime, ConversationRuntime)
            assert set(runtime.adapters) == {"pi"}
            assert runtime.default_harness_id == "pi"
    assert app.state.product_runtime is None
