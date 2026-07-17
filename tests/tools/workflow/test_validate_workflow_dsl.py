import json as jsonlib
from pathlib import Path
from typing import Any, cast

import httpx
from pydantic import SecretStr

from openhands.agent_server.pyromind_router import (
    PYROMIND_VALIDATE_AUTH_COOKIE_SECRET,
    PYROMIND_VALIDATE_HEADERS_STATE_KEY,
)
from openhands.sdk.conversation.secret_registry import SecretRegistry
from openhands.sdk.secret import StaticSecret
from openhands.sdk.tool import Tool
from openhands.sdk.tool.registry import resolve_tool
from openhands.tools.workflow import (
    ValidateWorkflowDslAction,
    ValidateWorkflowDslExecutor,
    ValidateWorkflowDslObservation,
    ValidateWorkflowDslTool,
    WorkflowValidationIssue,
)
from openhands.tools.workflow.validate_workflow_dsl import (
    PRE_VALIDATE_URL,
    PROD_VALIDATE_URL,
)


class _FakeWorkspace:
    def __init__(self, working_dir: Path) -> None:
        self.working_dir = str(working_dir)


class _Response:
    def __init__(self, status_code: int, payload: Any, text: str = "") -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self) -> Any:
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


def _valid_response() -> _Response:
    return _Response(
        200,
        {
            "success": True,
            "data": {
                "valid": True,
                "workflow_id": "workflow-1",
                "errors": [],
                "warnings": [],
            },
            "message": None,
            "error_code": None,
        },
    )


def _fake_conversation(
    tmp_path: Path,
    *,
    secret_registry: SecretRegistry | None = None,
    agent_state: dict[str, Any] | None = None,
):
    return type(
        "FakeConversation",
        (),
        {
            "workspace": _FakeWorkspace(tmp_path),
            "state": type(
                "FakeState",
                (),
                {
                    "secret_registry": secret_registry or SecretRegistry(),
                    "agent_state": agent_state or {},
                },
            )(),
        },
    )()


def test_validate_workflow_dsl_defaults_to_pre_endpoint_for_local_and_pre(
    monkeypatch,
):
    urls: list[str] = []

    def fake_post(url, *, headers, json, timeout):
        urls.append(url)
        return _valid_response()

    monkeypatch.setattr(httpx, "post", fake_post)
    monkeypatch.delenv("APP_ENV", raising=False)
    observation = ValidateWorkflowDslExecutor()(
        ValidateWorkflowDslAction(dsl="# workflow: demo\n")
    )
    assert not observation.is_error
    assert urls[-1] == PRE_VALIDATE_URL

    for app_env in ("dev", "pre", "local"):
        monkeypatch.setenv("APP_ENV", app_env)
        observation = ValidateWorkflowDslExecutor()(
            ValidateWorkflowDslAction(dsl="# workflow: demo\n")
        )
        assert not observation.is_error
        assert urls[-1] == PRE_VALIDATE_URL


def test_validate_workflow_dsl_defaults_to_prod_endpoint_for_online_envs(
    monkeypatch,
):
    urls: list[str] = []

    def fake_post(url, *, headers, json, timeout):
        urls.append(url)
        return _valid_response()

    monkeypatch.setattr(httpx, "post", fake_post)
    for app_env in ("prod", "production", "online"):
        monkeypatch.setenv("APP_ENV", app_env)
        observation = ValidateWorkflowDslExecutor()(
            ValidateWorkflowDslAction(dsl="# workflow: demo\n")
        )
        assert not observation.is_error
        assert urls[-1] == PROD_VALIDATE_URL


def test_validate_workflow_dsl_posts_payload_and_preserves_issue_fields(monkeypatch):
    calls: dict[str, Any] = {}

    def fake_post(url, *, headers, json, timeout):
        calls.update(
            {
                "url": url,
                "headers": headers,
                "json": json,
                "timeout": timeout,
            }
        )
        return _Response(
            200,
            {
                "success": True,
                "data": {
                    "valid": False,
                    "workflow_id": "8ca009ae-d7c9-4a5a-ae51-6faee07eded4",
                    "errors": [
                        {
                            "code": "NODE_NOT_FOUND",
                            "level": "error",
                            "workflow_id": ("8ca009ae-d7c9-4a5a-ae51-6faee07eded4"),
                            "node_id": "1",
                            "node_type": "CloneAndCacheModel1",
                            "node_name": "Clone model",
                            "field": "nodes[1]",
                            "message": "Node not found",
                            "source": "k8s",
                            "detail": {
                                "location": "nodes[1]",
                                "target_node_line": 4,
                                "node_code": (
                                    "n6a71806 = CloneAndCacheModel1"
                                    "(id=1, model=nfbc9d56.value)"
                                ),
                                "target_node_code": (
                                    "n6a71806 = CloneAndCacheModel1"
                                    "(id=1, model=nfbc9d56.value)"
                                ),
                            },
                        }
                    ],
                    "warnings": [],
                },
                "message": None,
                "error_code": None,
            },
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    executor = ValidateWorkflowDslExecutor(
        endpoint_url="https://validator.test/validate",
        headers={"x-cluster": "us-west-1#pre"},
        timeout=5.0,
    )

    observation = executor(
        ValidateWorkflowDslAction(
            name="workflow",
            dsl="# workflow: clone-xyflow\nnode = CloneAndCacheModel1(id=1)\n",
        )
    )

    assert isinstance(observation, ValidateWorkflowDslObservation)
    assert not observation.is_error
    assert observation.success is True
    assert observation.valid is False
    assert observation.workflow_id == "8ca009ae-d7c9-4a5a-ae51-6faee07eded4"
    assert observation.errors[0].code == "NODE_NOT_FOUND"
    assert observation.errors[0].node_id == "1"
    assert observation.errors[0].node_name == "Clone model"
    assert observation.errors[0].detail["target_node_line"] == 4
    assert observation.errors[0].detail["node_code"].startswith("n6a71806")
    assert observation.errors[0].detail["target_node_code"].startswith("n6a71806")
    assert observation.retryable is False
    assert observation.failure_stage == "platform_schema"
    assert "line=4" in observation.text
    assert "dsl_code: n6a71806 = CloneAndCacheModel1" in observation.text
    assert calls == {
        "url": "https://validator.test/validate",
        "headers": {
            "accept": "*/*",
            "content-type": "application/json",
            "x-cluster": "us-west-1#pre",
        },
        "json": {
            "name": "workflow",
            "dsl": "# workflow: clone-xyflow\nnode = CloneAndCacheModel1(id=1)\n",
        },
        "timeout": 5.0,
    }


def test_validate_workflow_dsl_reads_workflow_file_when_dsl_omitted(
    monkeypatch, tmp_path
):
    calls: dict[str, Any] = {}

    def fake_post(url, *, headers, json, timeout):
        calls["json"] = json
        return _valid_response()

    (tmp_path / "public_data" / "workflow_canvas").mkdir(parents=True)
    (tmp_path / "public_data" / "workflow_canvas" / "workflow.py").write_text(
        "# workflow: from-file\n", encoding="utf-8"
    )
    monkeypatch.setattr(httpx, "post", fake_post)

    observation = ValidateWorkflowDslExecutor()(
        ValidateWorkflowDslAction(),
        cast(Any, _fake_conversation(tmp_path)),
    )

    assert not observation.is_error
    assert calls["json"] == {
        "name": "workflow",
        "dsl": "# workflow: from-file\n",
    }


def test_validate_workflow_dsl_reports_missing_workflow_file(monkeypatch, tmp_path):
    def fake_post(url, *, headers, json, timeout):
        raise AssertionError("validation API should not be called")

    monkeypatch.setattr(httpx, "post", fake_post)

    observation = ValidateWorkflowDslExecutor()(
        ValidateWorkflowDslAction(),
        cast(Any, _fake_conversation(tmp_path)),
    )

    assert observation.is_error
    assert "workflow.py" in observation.text
    assert "does not exist" in observation.text


def test_validate_workflow_dsl_resolves_headers_from_conversation_secrets(
    monkeypatch,
):
    calls: dict[str, Any] = {}

    def fake_post(url, *, headers, json, timeout):
        calls["headers"] = headers
        return _Response(
            200,
            {
                "success": True,
                "data": {
                    "valid": True,
                    "workflow_id": "workflow-1",
                    "errors": [],
                    "warnings": [],
                },
                "message": None,
                "error_code": None,
            },
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    secret_registry = SecretRegistry()
    secret_registry.update_secrets(
        {
            "PYROMIND_VALIDATE_AUTH_COOKIE": StaticSecret(
                value=SecretStr("auth_token=session-token")
            )
        }
    )
    conversation = _fake_conversation(Path("/tmp"), secret_registry=secret_registry)
    executor = ValidateWorkflowDslExecutor(
        secret_headers={"cookie": "PYROMIND_VALIDATE_AUTH_COOKIE"}
    )

    observation = executor(
        ValidateWorkflowDslAction(dsl="# workflow: demo\n"),
        cast(Any, conversation),
    )

    assert not observation.is_error
    assert calls["headers"]["cookie"] == "auth_token=session-token"


def test_validate_workflow_dsl_uses_websocket_validation_context(monkeypatch):
    calls: dict[str, Any] = {}

    def fake_post(url, *, headers, json, timeout):
        calls["headers"] = headers
        return _valid_response()

    monkeypatch.setattr(httpx, "post", fake_post)
    secret_registry = SecretRegistry()
    secret_registry.update_secrets(
        {
            PYROMIND_VALIDATE_AUTH_COOKIE_SECRET: StaticSecret(
                value=SecretStr("auth_token=websocket-token")
            )
        }
    )
    conversation = _fake_conversation(
        Path("/tmp"),
        secret_registry=secret_registry,
        agent_state={
            PYROMIND_VALIDATE_HEADERS_STATE_KEY: {"x-cluster": "websocket-cluster"}
        },
    )
    executor = ValidateWorkflowDslExecutor(headers={"x-cluster": "request-cluster"})

    observation = executor(
        ValidateWorkflowDslAction(dsl="# workflow: demo\n"),
        cast(Any, conversation),
    )

    assert not observation.is_error
    assert calls["headers"]["cookie"] == "auth_token=websocket-token"
    assert calls["headers"]["x-cluster"] == "websocket-cluster"


def test_validate_workflow_dsl_marks_api_failure_as_tool_error(monkeypatch):
    def fake_post(url, *, headers, json, timeout):
        return _Response(
            200,
            {
                "success": False,
                "data": None,
                "message": "login required",
                "error_code": "AUTH_REQUIRED",
            },
        )

    monkeypatch.setattr(httpx, "post", fake_post)

    observation = ValidateWorkflowDslExecutor()(
        ValidateWorkflowDslAction(dsl="# workflow: demo\n")
    )

    assert observation.is_error
    assert observation.success is False
    assert observation.message == "login required"
    assert observation.error_code == "AUTH_REQUIRED"
    assert observation.retryable is False
    assert observation.failure_stage == "transport"
    assert "AUTH_REQUIRED" in observation.text


def test_validate_workflow_dsl_reports_transport_and_json_errors(monkeypatch):
    request = httpx.Request("POST", "https://validator.test/validate")

    def raise_connect_error(url, *, headers, json, timeout):
        raise httpx.ConnectError("network down", request=request)

    monkeypatch.setattr(httpx, "post", raise_connect_error)
    observation = ValidateWorkflowDslExecutor()(
        ValidateWorkflowDslAction(dsl="# workflow: demo\n")
    )
    assert observation.is_error
    assert "ConnectError" in observation.text
    assert observation.retryable is True
    assert observation.failure_stage == "transport"

    def return_invalid_json(url, *, headers, json, timeout):
        return _Response(
            200,
            jsonlib.JSONDecodeError("bad json", doc="{", pos=0),
        )

    monkeypatch.setattr(httpx, "post", return_invalid_json)
    observation = ValidateWorkflowDslExecutor()(
        ValidateWorkflowDslAction(dsl="# workflow: demo\n")
    )
    assert observation.is_error
    assert "invalid JSON" in observation.text
    assert observation.retryable is False
    assert observation.failure_stage == "transport"

    def return_service_unavailable(url, *, headers, json, timeout):
        return _Response(503, {}, text="temporarily unavailable")

    monkeypatch.setattr(httpx, "post", return_service_unavailable)
    observation = ValidateWorkflowDslExecutor()(
        ValidateWorkflowDslAction(dsl="# workflow: demo\n")
    )
    assert observation.is_error
    assert observation.retryable is True
    assert observation.failure_stage == "transport"


def test_validate_workflow_dsl_401_exposes_non_retryable_stop_guidance(monkeypatch):
    def return_unauthorized(url, *, headers, json, timeout):
        return _Response(401, {}, text="login required")

    monkeypatch.setattr(httpx, "post", return_unauthorized)

    observation = ValidateWorkflowDslExecutor()(
        ValidateWorkflowDslAction(dsl="# workflow: demo\n")
    )

    assert observation.is_error
    assert observation.retryable is False
    assert observation.failure_stage == "transport"
    assert "retryable=false" in observation.text
    assert "do not call validate_workflow_dsl again" in observation.text
    assert "do not use terminal commands" in observation.text


def test_validate_workflow_dsl_classifies_deterministic_failure_stages(monkeypatch):
    issue: dict[str, Any] = {}

    def fake_post(url, *, headers, json, timeout):
        return _Response(
            200,
            {
                "success": True,
                "data": {
                    "valid": False,
                    "workflow_id": "workflow-1",
                    "errors": [issue],
                    "warnings": [],
                },
            },
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    executor = ValidateWorkflowDslExecutor()
    action = ValidateWorkflowDslAction(dsl="# workflow: demo\n")

    issue.update(
        code="DSL_PARSE_FAILED",
        level="error",
        source="dsl",
        message="invalid syntax",
    )
    observation = executor(action)
    assert observation.failure_stage == "dsl_parse"
    assert observation.retryable is False

    issue.clear()
    issue.update(
        code="WORKFLOW_SCHEMA_INVALID",
        level="error",
        source="xyflow",
        message="invalid workflow schema",
    )
    observation = executor(action)
    assert observation.failure_stage == "sdk_schema"
    assert observation.retryable is False


def test_validate_workflow_dsl_tool_is_explicitly_available() -> None:
    tool = ValidateWorkflowDslTool.create(
        endpoint_url="https://validator.test/validate",
        headers={"x-cluster": "us-west-1#pre"},
        timeout=1,
    )[0]
    assert tool.name == "validate_workflow_dsl"
    assert tool.annotations is not None
    assert tool.annotations.readOnlyHint is True
    assert tool.annotations.openWorldHint is True
    assert "when retryable=false" in tool.description
    assert "do not use terminal commands" in tool.description
    assert "headers" not in tool.action_type.model_fields
    assert "secret_headers" not in tool.action_type.model_fields

    resolved = resolve_tool(
        Tool(
            name="validate_workflow_dsl",
            params={"endpoint_url": "https://validator.test/validate"},
        ),
        cast(Any, None),
    )
    assert isinstance(resolved[0], ValidateWorkflowDslTool)


def test_validate_workflow_dsl_output_schema_describes_api_fields() -> None:
    success_description = ValidateWorkflowDslObservation.model_fields[
        "success"
    ].description
    valid_description = ValidateWorkflowDslObservation.model_fields["valid"].description
    detail_description = WorkflowValidationIssue.model_fields["detail"].description
    source_description = WorkflowValidationIssue.model_fields["source"].description
    retryable_description = ValidateWorkflowDslObservation.model_fields[
        "retryable"
    ].description
    failure_stage_description = ValidateWorkflowDslObservation.model_fields[
        "failure_stage"
    ].description

    assert success_description is not None
    assert "BizResponse.success" in success_description
    assert valid_description is not None
    assert "WorkflowValidationResult" in valid_description
    assert detail_description is not None
    assert "1-based line numbers" in detail_description
    assert "node_code is the DSL statement for issue.node_id" in detail_description
    assert "target_node_line" in detail_description
    assert "target_node_code" in detail_description
    assert "source_node_code" in detail_description
    assert "source_node_id" in detail_description
    assert "available_inputs" in detail_description
    assert source_description is not None
    assert "dsl" in source_description
    assert "xyflow" in source_description
    assert "k8s" in source_description
    assert retryable_description is not None
    assert "408/429/5xx" in retryable_description
    assert failure_stage_description is not None
    assert "SDK workflow schema" in failure_stage_description
