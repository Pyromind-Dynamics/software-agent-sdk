"""Publish the exact validated graph without persisting execution credentials."""

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from pydantic import SecretStr

from harness_adapter.pi_adapter import PiAdapter
from harness_adapter.pi_adapter.business_tool_host import ToolExecutionContext
from pyromind_common.sdk.conversation.secret_registry import SecretRegistry
from pyromind_common.sdk.secret import StaticSecret
from pyromind_runtime.domain.context import RequestContext

from .test_sdk_execution_metadata import sdk_recipe_factory as sdk_recipe_factory


@pytest.fixture
def validation_context(tmp_path: Path) -> ToolExecutionContext:
    secrets = SecretRegistry()
    secrets.update_secrets(
        {"SDK_KEY": StaticSecret(value=SecretStr("derived-sdk-secret"))}
    )
    secrets.get_secret_value("SDK_KEY")
    return ToolExecutionContext(
        conversation_id="validation-artifact",
        workspace_root=tmp_path,
        request_context=RequestContext(
            user_id="42", cookie="auth_token=request-secret"
        ),
        model_configuration={},
        sdk_submissions_enabled=True,
        sdk_secrets=secrets,
    )


def validated_graph(config: dict) -> dict:
    return {
        "name": "validated",
        "nodes": [
            {
                "id": "1",
                "data": {
                    "config": config,
                    "nodeDefinition": {"properties": {"api_key": {"type": "string"}}},
                },
            }
        ],
        "edges": [],
    }


async def test_validation_publishes_the_same_graph_idempotently(validation_context):
    host = PiAdapter(
        validation_context.workspace_root, terminal_backend="os-sandbox"
    )._business_tools
    graph = validated_graph({"cpu": 12, "command": "python3 evaluation.py"})
    details = {"raw_response": {"data": {"workflow": graph}}}

    first = await host._publish_validated_workflow(details, validation_context)
    second = await host._publish_validated_workflow(details, validation_context)

    assert first == second
    path = validation_context.workspace_root / first["validated_workflow_path"]
    assert json.loads(path.read_text()) == graph
    assert hashlib.sha256(path.read_bytes()).hexdigest() == first["sha256"]
    assert list(path.parent.glob("*.json")) == [path]


async def test_dataset_replay_persists_shared_model_routing(
    validation_context, sdk_recipe_factory
):
    context = replace(
        validation_context,
        model_configuration={
            "base_url": "https://model.test/v1",
            "model": "model",
            "api_key": "sdk-test-model-secret",
        },
    )
    host = PiAdapter(
        context.workspace_root, terminal_backend="os-sandbox"
    )._business_tools
    graph = sdk_recipe_factory("124", parent="123")["workflow"]
    details = {"raw_response": {"data": {"workflow": graph}}}

    result = await host._publish_validated_workflow(details, context)

    path = context.workspace_root / result["validated_workflow_path"]
    assert json.loads(path.read_text()) == graph
    assert "sdk-test-model-secret" not in path.read_text()


@pytest.mark.parametrize(
    ("model_config", "protected_value"),
    [
        (
            {"base_url": "https://gateway.test/v1?token=private"},
            "https://gateway.test/v1?token=private",
        ),
        (
            {"base_url": "https://user:secret@gateway.test"},
            "https://user:secret@gateway.test",
        ),
        (
            {"base_url": "https://gateway.test/private-token"},
            "https://gateway.test/private-token",
        ),
        ({"model": "shared-value", "api_key": "shared-value"}, "shared-value"),
    ],
)
async def test_routing_metadata_cannot_exempt_credentials(
    validation_context, model_config, protected_value
):
    context = replace(validation_context, model_configuration=model_config)
    host = PiAdapter(
        context.workspace_root, terminal_backend="os-sandbox"
    )._business_tools
    graph = validated_graph({"command": f"printenv; echo {protected_value}"})
    result = await host._publish_validated_workflow(
        {"raw_response": {"data": {"workflow": graph}}}, context
    )
    assert "artifact_unavailable" in result
    assert not list(context.workspace_root.rglob("*.json"))


@pytest.mark.parametrize(
    "config",
    [
        {"api_key": "execution-secret"},
        {"command": "curl --cookie auth_token=request-secret https://example.com"},
        {"command": "curl -H X-Api-Key:derived-sdk-secret https://example.com"},
        {"input_url": "https://example.com/data?signature=execution-secret"},
    ],
)
async def test_validation_refuses_credential_graphs_without_redacting_and_saving_them(
    validation_context, config
):
    host = PiAdapter(
        validation_context.workspace_root, terminal_backend="os-sandbox"
    )._business_tools
    details = {"raw_response": {"data": {"workflow": validated_graph(config)}}}

    result = await host._publish_validated_workflow(details, validation_context)

    assert "artifact_unavailable" in result
    assert "SDK submission must stop" in result["artifact_unavailable"]
    assert "raw_response" not in details
    assert not list(validation_context.workspace_root.rglob("*.json"))
    assert "execution-secret" not in json.dumps(result)
    assert "request-secret" not in json.dumps(result)
    assert "derived-sdk-secret" not in json.dumps(result)


async def test_artifact_failure_keeps_valid_result_and_legacy_summary(
    tmp_path: Path, monkeypatch
):
    host = PiAdapter(tmp_path, terminal_backend="os-sandbox")._business_tools

    async def validate(*args, **kwargs):
        return {
            "is_error": False,
            "content": [{"type": "text", "text": "Validation passed."}],
            "details": {
                "valid": True,
                "raw_response": {"data": {"workflow": validated_graph({"cpu": 12})}},
            },
        }

    monkeypatch.setattr(
        "harness_adapter.pi_adapter.business_tool_host.execute_validation_tool",
        validate,
    )
    publish = AsyncMock(
        return_value={
            "artifact_unavailable": "Publishing failed; SDK submission must stop."
        }
    )
    monkeypatch.setattr(host, "_publish_validated_workflow", publish)
    for sdk_enabled in (False, True):
        result = await host.execute(
            "validate_workflow_dsl",
            {},
            ToolExecutionContext(
                conversation_id="validation-artifact",
                workspace_root=tmp_path,
                request_context=RequestContext(user_id="42"),
                model_configuration={},
                sdk_submissions_enabled=sdk_enabled,
            ),
        )
        assert result["is_error"] is False
        assert result["details"]["valid"] is True
        if sdk_enabled:
            assert "artifact_unavailable" in result["details"]
            assert "SDK submission must stop" in json.dumps(result["content"])
        else:
            assert result["content"] == [{"type": "text", "text": "Validation passed."}]
    publish.assert_awaited_once()


@pytest.mark.parametrize(
    ("field", "secret"),
    [
        ("workflow", "request-secret"),
        ("workflow", "derived-sdk-secret"),
        ("workflow", "session-model-secret"),
        ("code", "managed-dataflow-secret"),
        ("code", "process-model-secret"),
    ],
)
async def test_metadata_binding_refuses_actual_credentials_before_runtime_call(
    validation_context, monkeypatch, field, secret
):
    monkeypatch.setenv("DF_API_KEY", "managed-dataflow-secret")
    monkeypatch.setenv("LLM_AUTH_TOKEN", "process-model-secret")
    context = replace(
        validation_context,
        model_configuration={"api_key": "session-model-secret"},
    )
    host = PiAdapter(
        context.workspace_root, terminal_backend="os-sandbox"
    )._business_tools
    host.version_call = AsyncMock()
    payload = {
        "task_id": "task-1",
        "workflow": validated_graph({"command": "python3 pipeline.py"}),
    }
    if field == "workflow":
        payload["workflow"] = validated_graph({"command": f"echo {secret}"})
    else:
        payload[field] = f"print({secret!r})"

    result = await host.execute(
        "product_versions",
        {
            "action": "bind_execution_metadata",
            "command_id": "bind",
            "arguments": payload,
        },
        context,
    )

    assert result["is_error"] is True
    assert "cannot contain credentials" in json.dumps(result)
    assert secret not in json.dumps(result)
    host.version_call.assert_not_awaited()
    assert not list(context.workspace_root.rglob("*.json"))


async def test_metadata_binding_preserves_runtime_credential_markers(
    validation_context,
):
    host = PiAdapter(
        validation_context.workspace_root, terminal_backend="os-sandbox"
    )._business_tools
    host.version_call = AsyncMock(return_value={"task_id": "task-1", "bound": True})
    payload = {
        "task_id": "task-1",
        "workflow": validated_graph(
            {
                "command": (
                    'export DF_API_KEY="$param"; {{PYROMIND_SECRET:LLM_AUTH_TOKEN}}'
                )
            }
        ),
        "code": 'client = PyroMindAPIClient(api_key=os.environ["PYROMIND_API_KEY"])',
    }

    result = await host.execute(
        "product_versions",
        {
            "action": "bind_execution_metadata",
            "command_id": "bind",
            "arguments": payload,
        },
        validation_context,
    )

    assert result["is_error"] is False
    assert host.version_call.await_args is not None
    assert host.version_call.await_args.args[1].arguments == payload
