from __future__ import annotations

from pyromind_runtime.contracts import SandboxRef, TextContentBlock, WorkspaceRef
from pyromind_runtime.tool_host import (
    SessionToolContextStore,
    ToolExecutionContext,
    ToolRequestContext,
)
from starlette.requests import Request

from openhands.agent_server import pyromind_router
from openhands.agent_server.pyromind_auth import CurrentLoginUser
from openhands.tools.pyromind_dataset import PreviewDatasetObservation
from openhands.tools.workflow import ValidateWorkflowDslObservation


def _execution_context() -> ToolExecutionContext:
    return ToolExecutionContext(
        session_id="session-1",
        user_id="user-1",
        workspace=WorkspaceRef(workspace_id="workspace-1", root="/workspace"),
        sandbox=SandboxRef(sandbox_id="sandbox-1", backend="pyromind"),
    )


def test_product_tool_request_context_uses_only_allowed_request_credentials() -> None:
    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/api/v2/pyromind/conversations",
            "headers": [
                (b"authorization", b"Bearer model-user"),
                (b"host", b"untrusted.example"),
            ],
        }
    )
    request.state.current_user = CurrentLoginUser(
        username="user",
        email="user@example.test",
        user_id=1,
        cookie="auth_token=session-secret",
        x_cluster="cluster-a",
    )

    context = pyromind_router.get_product_tool_request_context(request)

    assert context.headers == {
        "authorization": "Bearer model-user",
        "cookie": "auth_token=session-secret",
        "x-cluster": "cluster-a",
    }


async def test_product_tool_host_reuses_python_executors_and_redacts_details(
    monkeypatch,
) -> None:
    seen_headers: list[dict[str, str]] = []

    class FakePreviewExecutor:
        def __init__(self, *, headers) -> None:
            seen_headers.append(headers)

        def __call__(self, action, conversation):
            assert conversation is None
            return PreviewDatasetObservation.from_text(
                text=f"preview:{action.dataset_path}",
                dataset_path=action.dataset_path,
                metadata={"download_url": "https://secret.example/signed"},
            )

    class FakeValidateExecutor:
        def __init__(self, *, headers) -> None:
            seen_headers.append(headers)

        def __call__(self, action, conversation):
            assert conversation is None
            return ValidateWorkflowDslObservation.from_text(
                text=f"validate:{action.name}",
                valid=True,
                raw_response={"authorization": "secret", "valid": True},
            )

    monkeypatch.setattr(
        pyromind_router,
        "PreviewDatasetExecutor",
        FakePreviewExecutor,
    )
    monkeypatch.setattr(
        pyromind_router,
        "ValidateWorkflowDslExecutor",
        FakeValidateExecutor,
    )
    context_store = SessionToolContextStore(
        allowed_header_names={"authorization", "cookie", "x-cluster"}
    )
    context_store.bind(
        "session-1",
        ToolRequestContext(
            headers={"cookie": "auth_token=session-secret", "x-cluster": "cluster-a"}
        ),
    )
    host = pyromind_router.build_product_tool_host(context_store)

    preview = await host.execute(
        "preview_dataset",
        {"dataset_path": "datasets/train.jsonl", "n": 3},
        _execution_context(),
    )
    validation = await host.execute(
        "validate_workflow_dsl",
        {"dsl": "workflow = Workflow()", "name": "demo"},
        _execution_context(),
    )

    assert isinstance(preview.content[0], TextContentBlock)
    assert preview.content[0].text == "preview:datasets/train.jsonl"
    assert preview.details is not None
    assert preview.details["metadata"] == {"download_url": "[REDACTED]"}
    assert isinstance(validation.content[0], TextContentBlock)
    assert validation.content[0].text == "validate:demo"
    assert validation.details is not None
    assert validation.details["raw_response"] == {
        "authorization": "[REDACTED]",
        "valid": True,
    }
    assert seen_headers == [
        {"cookie": "auth_token=session-secret", "x-cluster": "cluster-a"},
        {"cookie": "auth_token=session-secret", "x-cluster": "cluster-a"},
    ]
