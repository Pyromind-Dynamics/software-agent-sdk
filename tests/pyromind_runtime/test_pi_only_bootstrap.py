from importlib.util import find_spec
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from harness_adapter.pi_adapter import PiAdapter
from pyromind_agent_server import bootstrap
from pyromind_agent_server.app import create_app
from pyromind_agent_server.config import Config
from pyromind_runtime.domain.context import RequestContext

from openhands.agent_server.api import create_app as create_sdk_app
from openhands.agent_server.config import Config as LegacyConfig


def test_removed_openhands_adapter_cannot_be_imported():
    assert find_spec("harness_adapter.openhands_adapter") is None


@pytest.mark.parametrize("configured", [None, "pi"])
async def test_product_runtime_registers_only_pi(tmp_path, monkeypatch, configured):
    monkeypatch.setenv("APP_ENV", "dev")
    monkeypatch.setenv("PYROMIND_PI_TERMINAL_BACKEND", "os-sandbox")
    if configured is None:
        monkeypatch.delenv("PYROMIND_HARNESS_BACKEND", raising=False)
    else:
        monkeypatch.setenv("PYROMIND_HARNESS_BACKEND", configured)
    monkeypatch.setenv("PYROMIND_VERSION_DB_PATH", str(tmp_path / "versions.sqlite3"))
    monkeypatch.setenv("PYROMIND_AGENTGENOME_ENABLED", "0")
    monkeypatch.setattr(bootstrap, "build_experiment_plan_store", lambda: None)
    monkeypatch.setattr(bootstrap, "set_workflow_status_dispatcher", lambda _: None)
    app = FastAPI()
    app.state.conversation_service = SimpleNamespace(
        conversations_dir=tmp_path / "conversations", idle_eviction_timeout=0
    )
    runtime = bootstrap.ensure_product_runtime(app)
    assert runtime is not None
    try:
        assert runtime.default_harness_id == "pi"
        assert set(runtime.adapters) == {"pi"}
        adapter = runtime.adapters["pi"]
        assert isinstance(adapter, PiAdapter)
        hook = adapter._business_tools.training_artifact_hook
        assert hook is not None
        record = AsyncMock(return_value=1)
        monkeypatch.setattr(runtime, "record_training_artifacts", record)
        context = RequestContext(user_id="42")
        assert await hook("c", "task", [], context) == 1
        record.assert_awaited_once_with("c", "task", [], context)
    finally:
        await runtime.close()


def test_retired_backend_is_rejected_before_creating_a_session(monkeypatch, tmp_path):
    monkeypatch.setenv("PYROMIND_HARNESS_BACKEND", "openhands")
    app = FastAPI()
    app.state.conversation_service = SimpleNamespace(conversations_dir=tmp_path)
    with pytest.raises(RuntimeError, match="Unsupported PYROMIND_HARNESS_BACKEND"):
        bootstrap.ensure_product_runtime(app)


def test_pi_service_exposes_product_routes_without_legacy_execution_routes():
    app = create_app(Config(static_files_path=None))
    paths = set(app.openapi()["paths"])
    assert "/api/v2/pyromind/conversations" in paths
    assert "/api/v2/echomind/conversations" in paths
    assert "/api/pyromind/workflow/callback" in paths
    assert "/api/pyromind/debug/callback" in paths
    assert "/api/pyromind/data-preparation/progress" in paths
    assert "/api/pyromind/conversations" not in paths
    assert "/api/conversations" not in paths
    assert not any(
        p.startswith("/api/conversations/") and "/workspace" not in p for p in paths
    )
    assert "/api/conversations/{conversation_id}/workspace/{file_path}" in paths
    assert not any(p.endswith("/chat/completions") for p in paths)
    assert not any(getattr(r, "path", "").startswith("/sockets") for r in app.routes)


def test_retained_sdk_service_keeps_its_own_routes():
    app = create_sdk_app(LegacyConfig(static_files_path=None))
    paths = set(app.openapi()["paths"])
    assert "/api/conversations" in paths
