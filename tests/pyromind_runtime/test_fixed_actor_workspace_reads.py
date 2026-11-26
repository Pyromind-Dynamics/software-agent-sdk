from unittest.mock import Mock

import httpx
import pytest

from harness_adapter.pi_adapter.actor_workspace import ActorTransport
from harness_adapter.pi_adapter.adapter import PiAdapter
from harness_adapter.pi_adapter.business_tool_host import ToolExecutionContext
from harness_adapter.pi_adapter.fixed_actor import FixedActorExecutionManager
from harness_adapter.pi_adapter.persistence import PiSessionFiles
from pyromind_runtime.domain.capabilities import HarnessCapabilities
from pyromind_runtime.domain.context import RequestContext
from pyromind_runtime.domain.errors import ProductRuntimeError
from pyromind_runtime.ports.harness import SessionHandle


async def test_persisted_plan_reads_use_the_recorded_actor_disk(tmp_path, monkeypatch):
    monkeypatch.setenv("PYROMIND_SANDBOX_ACTOR_URL", "http://router.test")
    monkeypatch.setenv("PYROMIND_SANDBOX_ACTOR_TARGET", "dev/sandbox")
    adapter = PiAdapter(tmp_path, terminal_backend="sandbox")
    assert isinstance(adapter._sandbox, FixedActorExecutionManager)
    cid = "dev-conversation"
    root = tmp_path / cid
    context = RequestContext(user_id="developer", cookie="auth_token=fixture")
    files = PiSessionFiles(root)
    files.initialize({})
    files.save_sandbox(
        {
            "sandbox_id": "platform",
            "storage_host_path": "/workspace",
            "mount_path": "/workspace",
        }
    )
    handle = SessionHandle(
        session_id=cid, adapter_session_ref=cid, capabilities=HarnessCapabilities()
    )
    read = Mock(return_value=b'{"source":"actor"}')
    monkeypatch.setattr(ActorTransport, "read_file", read)
    plan = "public_data/workflowPLAN.json"
    assert await adapter.read_persisted_workspace_text(handle, plan, context) is None
    assert await adapter.development_storage_proxy(cid) is None
    read.assert_not_called()

    monkeypatch.setattr(FixedActorExecutionManager, "_prepare", lambda *_: None)
    tool = ToolExecutionContext(
        conversation_id=cid,
        workspace_root=root,
        request_context=context,
        model_configuration={},
        extra={},
    )
    await adapter._sandbox.ensure(tool, files)
    assert (
        await adapter.read_persisted_workspace_text(
            handle, plan, context, max_bytes=128
        )
        == '{"source":"actor"}'
    )
    read.assert_called_once_with(
        f"/workspace/.pyromind-agent/{cid}/{plan}", max_bytes=128, timeout=3
    )
    req = httpx.Request("GET", "http://router.test/files")
    read.side_effect = httpx.HTTPStatusError(
        "missing", request=req, response=httpx.Response(404)
    )
    assert await adapter.read_persisted_workspace_text(handle, plan, context) is None
    read.side_effect = httpx.ConnectError("gateway unavailable", request=req)
    with pytest.raises(ProductRuntimeError, match="unreachable"):
        await adapter.read_persisted_workspace_text(handle, plan, context)
