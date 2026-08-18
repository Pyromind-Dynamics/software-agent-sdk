from __future__ import annotations

import asyncio

import pytest
from pyromind_runtime.contracts import (
    SandboxRef,
    TextContentBlock,
    ToolResult,
    ToolSpec,
    WorkspaceRef,
)
from pyromind_runtime.tool_host import (
    PREVIEW_DATASET_TOOL_SPEC,
    PythonToolHost,
    SessionToolContextStore,
    ToolExecutionContext,
    ToolRequestContext,
    ToolRequestContextNotAvailableError,
    ToolRiskPolicy,
    first_version_tool_specs,
)


def _context() -> ToolExecutionContext:
    return ToolExecutionContext(
        session_id="session-1",
        user_id="user-1",
        workspace=WorkspaceRef(workspace_id="workspace-1", root="/workspace"),
        sandbox=SandboxRef(
            sandbox_id="sandbox-1",
            backend="managed",
            lease_id="lease-1",
        ),
    )


async def test_tool_host_validates_arguments_before_calling_handler() -> None:
    called = False

    async def handler(arguments, context) -> ToolResult:
        nonlocal called
        called = True
        return ToolResult(content=(TextContentBlock(text="unexpected"),))

    host = PythonToolHost()
    host.register(PREVIEW_DATASET_TOOL_SPEC, handler)

    result = await host.execute("preview_dataset", {"n": 10}, _context())

    assert result.is_error is True
    assert result.error_code == "invalid_tool_arguments"
    assert called is False


async def test_tool_host_executes_registered_low_risk_tool() -> None:
    async def handler(arguments, context) -> ToolResult:
        return ToolResult(
            content=(TextContentBlock(text=str(arguments["dataset_path"])),),
            details={"session_id": context.session_id},
        )

    host = PythonToolHost()
    host.register(PREVIEW_DATASET_TOOL_SPEC, handler)

    result = await host.execute(
        "preview_dataset",
        {"dataset_path": "datasets/train.jsonl", "n": 3},
        _context(),
    )

    assert result.is_error is False
    assert result.content[0] == TextContentBlock(text="datasets/train.jsonl")
    assert result.details == {"session_id": "session-1"}


async def test_tool_host_enforces_risk_timeout_and_output_limits() -> None:
    async def handler(arguments, context) -> ToolResult:
        await asyncio.sleep(0.02)
        return ToolResult(content=(TextContentBlock(text="x" * 1000),))

    denied_spec = ToolSpec(
        name="dangerous_tool",
        description="Test risk policy.",
        input_schema={"type": "object"},
        timeout_seconds=1,
        risk_level="high",
    )
    denied_host = PythonToolHost()
    denied_host.register(denied_spec, handler)
    denied = await denied_host.execute("dangerous_tool", {}, _context())
    assert denied.error_code == "tool_risk_denied"

    timeout_spec = denied_spec.model_copy(
        update={"name": "slow_tool", "timeout_seconds": 0.001, "risk_level": "low"}
    )
    timeout_host = PythonToolHost()
    timeout_host.register(timeout_spec, handler)
    timed_out = await timeout_host.execute("slow_tool", {}, _context())
    assert timed_out.error_code == "tool_timeout"

    output_spec = timeout_spec.model_copy(
        update={"name": "large_tool", "timeout_seconds": 1}
    )
    output_host = PythonToolHost(max_result_bytes=100)
    output_host.register(output_spec, handler)
    too_large = await output_host.execute("large_tool", {}, _context())
    assert too_large.error_code == "tool_result_too_large"


async def test_tool_host_does_not_leak_handler_exception_text() -> None:
    async def handler(arguments, context) -> ToolResult:
        raise RuntimeError("secret-token-value")

    host = PythonToolHost(ToolRiskPolicy(allowed=frozenset({"low", "medium"})))
    host.register(PREVIEW_DATASET_TOOL_SPEC, handler)

    result = await host.execute(
        "preview_dataset",
        {"dataset_path": "dataset.jsonl"},
        _context(),
    )

    serialized = result.model_dump_json()
    assert result.error_code == "tool_handler_failed"
    assert "secret-token-value" not in serialized


def test_first_version_specs_are_harness_neutral() -> None:
    specs = first_version_tool_specs()

    assert [spec.name for spec in specs] == [
        "preview_dataset",
        "validate_workflow_dsl",
    ]
    assert all(spec.risk_level == "low" for spec in specs)


def test_session_tool_context_is_memory_only_allowlisted_and_redacted_from_repr() -> (
    None
):
    store = SessionToolContextStore(
        allowed_header_names={"cookie", "authorization", "x-cluster"}
    )
    context = ToolRequestContext(
        headers={"Cookie": "auth_token=secret", "X-Cluster": "cluster-a"}
    )

    store.bind("session-1", context)

    stored = store.get("session-1")
    assert stored.headers == {
        "cookie": "auth_token=secret",
        "x-cluster": "cluster-a",
    }
    assert "secret" not in repr(stored)
    with pytest.raises(ValueError, match="host"):
        store.bind("session-2", ToolRequestContext(headers={"host": "evil"}))

    store.remove("session-1")
    with pytest.raises(ToolRequestContextNotAvailableError):
        store.get("session-1")
