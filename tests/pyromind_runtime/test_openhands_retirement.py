import json
import subprocess
import sys
from datetime import UTC, datetime
from unittest.mock import AsyncMock

import httpx
import pytest

from pyromind_agent_server import bootstrap
from pyromind_agent_server.app import create_app
from pyromind_agent_server.config import Config
from pyromind_runtime.application.conversation_runtime import ConversationRuntime
from pyromind_runtime.domain.capabilities import HarnessCapabilities
from pyromind_runtime.domain.commands import UserMessageCommand
from pyromind_runtime.domain.content import TextContent
from pyromind_runtime.domain.context import RequestContext
from pyromind_runtime.domain.errors import ProductRuntimeError
from pyromind_runtime.domain.snapshot import ConversationSnapshot, TimelineNotice
from pyromind_runtime.infrastructure.file_product_store import FileProductStore

from .fake_adapter import FakeAdapter


def test_product_imports_without_openhands_packages():
    script = """
import importlib.abc
import sys
class RejectLegacy(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "openhands" or fullname.startswith("openhands."):
            raise AssertionError("retired import: " + fullname)
sys.meta_path.insert(0, RejectLegacy())
from pyromind_agent_server.app import create_app
paths = create_app().openapi()["paths"]
assert "/api/v2/pyromind/conversations" in paths
assert "/api/v2/echomind/conversations" in paths
assert "/api/pyromind/workflow/callback" in paths
assert "/api/conversations" not in paths
"""
    subprocess.run([sys.executable, "-c", script], check=True, capture_output=True)


@pytest.mark.parametrize("missing_harness_id", [False, True])
async def test_legacy_snapshot_is_read_only_without_attach_or_recovery(
    tmp_path, missing_harness_id
):
    directory = tmp_path / "legacy"
    directory.mkdir()
    store = FileProductStore(directory)
    store.create(
        ConversationSnapshot(
            conversation_id="legacy",
            status="running",
            capabilities=HarnessCapabilities(cancel=True, resume=True),
        ),
        user_id="owner",
        harness_id="openhands",
    )
    if missing_harness_id:
        data = json.loads(store.metadata_path.read_text())
        del data["harness_id"]
        store.metadata_path.write_text(json.dumps(data))
    original = store.metadata_path.read_bytes()
    adapter = FakeAdapter()
    runtime = ConversationRuntime(tmp_path, adapter)
    runtime._user_input.coordinate_recovery = AsyncMock()
    try:
        context = RequestContext(user_id="owner", x_cluster="test-cluster#pre")
        snapshot = await runtime.get_snapshot("legacy", context)
        assert snapshot.capabilities.read_only and not snapshot.capabilities.resume
        assert snapshot.status == "closed"
        assert isinstance(snapshot.timeline[-1], TimelineNotice)
        assert snapshot.timeline[-1].code == "legacy_harness_retired"
        assert not runtime._read_recoveries
        runtime._user_input.coordinate_recovery.assert_not_awaited()
        with pytest.raises(ProductRuntimeError, match="archived") as error:
            await runtime.submit_command(
                "legacy",
                UserMessageCommand(
                    command_id="retry", content=(TextContent(text="continue"),)
                ),
                context,
            )
        assert error.value.code == "legacy_harness_retired"
        assert store.harness_id() == "openhands"
        assert store.metadata_path.read_bytes() == original
        with pytest.raises(PermissionError):
            await runtime.get_snapshot("legacy", RequestContext(user_id="foreign"))
    finally:
        await runtime.close()


async def test_deferred_product_initialization_and_shutdown(tmp_path, monkeypatch):
    monkeypatch.setenv("APP_ENV", "dev")
    monkeypatch.setenv("PYROMIND_PI_TERMINAL_BACKEND", "os-sandbox")
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
            assert response.status_code == 200, response.text
            assert app.state.product_runtime is not None
            assert app.state.product_runtime.conversation_root == tmp_path / "mounted"
            response = await client.post("/api/init", json={})
            assert response.status_code == 400
    assert app.state.product_runtime is None


async def test_legacy_task_callback_is_idempotent_without_engine_resume(tmp_path):
    directory = tmp_path / "legacy"
    directory.mkdir()
    store = FileProductStore(directory)
    store.create(
        ConversationSnapshot(
            conversation_id="legacy", capabilities=HarnessCapabilities()
        ),
        user_id="owner",
        harness_id="openhands",
    )
    runtime = ConversationRuntime(tmp_path, FakeAdapter())
    try:
        runtime.register_external_task(
            "legacy",
            {
                "task_id": "task",
                "kind": "workflow_run",
                "status": "running",
                "submitted_at": datetime.now(UTC).isoformat(),
                "updated_at": datetime.now(UTC).isoformat(),
            },
        )
        first = await runtime.deliver_external_task_status(
            "legacy", task_id="task", status="Succeeded", auto_run=True
        )
        repeated = await runtime.deliver_external_task_status(
            "legacy", task_id="task", status="Succeeded", auto_run=True
        )
        assert first.seq == repeated.seq
        assert first.payload["resume_pending"] is False
        assert first.payload["resume_auto_run"] is False
        assert not runtime._active
        assert store.load_snapshot().external_tasks[0].status == "succeeded"
    finally:
        await runtime.close()


async def test_profile_workspace_artifact_owner_can_read_and_foreign_user_cannot(
    tmp_path, monkeypatch
):
    from uuid import uuid4

    from pyromind_agent_server.service_context import ProductServiceContext

    monkeypatch.setenv("APP_ENV", "dev")
    identity = uuid4().hex
    directory = tmp_path / identity
    (directory / "public_data").mkdir(parents=True)
    (directory / "public_data/report.html").write_text("saved report")
    FileProductStore(directory).create(
        ConversationSnapshot(
            conversation_id=identity, capabilities=HarnessCapabilities()
        ),
        user_id="42",
        profile_id="echomind-manager",
    )
    app = create_app(Config())
    app.state.conversation_service = ProductServiceContext(tmp_path)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        url = f"/api/conversations/{identity}/workspace/public_data/report.html"
        response = await client.get(url, headers={"x-pyromind-debug-user-id": "42"})
        assert response.status_code == 200
        assert response.text == "saved report"
        response = await client.get(url, headers={"x-pyromind-debug-user-id": "7"})
        assert response.status_code == 404


async def test_archived_workflow_http_request_cannot_reach_submission(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace
    from typing import cast

    from pyromind_agent_server.service_context import ProductServiceContext
    from pyromind_runtime.ports.workflow_submission import WorkflowSubmissionPort

    monkeypatch.setenv("APP_ENV", "dev")
    directory = tmp_path / "legacy"
    directory.mkdir()
    store = FileProductStore(directory)
    store.create(
        ConversationSnapshot(
            conversation_id="legacy", capabilities=HarnessCapabilities()
        ),
        user_id="42",
        harness_id="openhands",
    )
    submission = SimpleNamespace(prepare=AsyncMock(return_value={}))
    runtime = ConversationRuntime(
        tmp_path,
        FakeAdapter(),
        workflow_submission=cast(WorkflowSubmissionPort, submission),
    )
    app = create_app(Config())
    app.state.conversation_service = ProductServiceContext(tmp_path)
    app.state.product_runtime = runtime
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/api/v2/pyromind/conversations/legacy/workflow-executions",
                headers={"x-pyromind-debug-user-id": "42"},
                json={"workflow_path": "public_data/workflow.py"},
            )
        assert response.status_code == 409, response.text
        assert response.json()["detail"]["code"] == "legacy_harness_retired"
        submission.prepare.assert_not_awaited()
        assert not store.load_workflow_execution_requests()
    finally:
        await runtime.close()
