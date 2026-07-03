import json as jsonlib
from typing import Any, cast

import httpx
from pydantic import SecretStr

from openhands.sdk.conversation.secret_registry import SecretRegistry
from openhands.sdk.secret import StaticSecret
from openhands.sdk.tool import Tool
from openhands.sdk.tool.registry import resolve_tool
from openhands.tools.workflow import (
    ValidateWorkflowDslAction,
    ValidateWorkflowDslExecutor,
    ValidateWorkflowDslObservation,
    ValidateWorkflowDslTool,
)


class _Response:
    def __init__(self, status_code: int, payload: Any, text: str = "") -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self) -> Any:
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


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
                            "field": "nodes[1]",
                            "message": "Node not found",
                            "source": "k8s",
                            "detail": {
                                "location": "nodes[1]",
                                "target_node_line": 4,
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
    assert observation.errors[0].detail["target_node_line"] == 4
    assert "line=4" in observation.text
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
    conversation = type(
        "FakeConversation",
        (),
        {"state": type("FakeState", (), {"secret_registry": secret_registry})()},
    )()
    executor = ValidateWorkflowDslExecutor(
        secret_headers={"cookie": "PYROMIND_VALIDATE_AUTH_COOKIE"}
    )

    observation = executor(
        ValidateWorkflowDslAction(dsl="# workflow: demo\n"),
        cast(Any, conversation),
    )

    assert not observation.is_error
    assert calls["headers"]["cookie"] == "auth_token=session-token"


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
