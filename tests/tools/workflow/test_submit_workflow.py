from __future__ import annotations

from types import SimpleNamespace
from uuid import uuid4

from openhands.tools.workflow import submit_workflow as module


def test_submit_workflow_tool_stages_confirmation_with_current_identity(monkeypatch):
    captured: dict = {}

    class Secrets:
        def get_secret_value(self, name):
            return {
                module.PYROMIND_VALIDATE_AUTH_COOKIE_SECRET: "auth_token=ephemeral",
                module.PYROMIND_PRODUCT_USER_ID_SECRET: "42",
            }.get(name)

    def post(url, *, headers, json, timeout):
        captured.update({"url": url, "headers": headers, "body": json})
        return SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {
                "request_id": json["request_id"],
                "status": "awaiting_confirmation",
                "summary": {"name": "sft"},
            },
        )

    monkeypatch.setattr(module.httpx, "post", post)
    monkeypatch.setenv("OH_INTERNAL_SERVER_URL", "http://127.0.0.1:9999")
    conversation_id = uuid4()
    conversation = SimpleNamespace(
        id=conversation_id,
        state=SimpleNamespace(secret_registry=Secrets(), agent_state={}),
    )
    observation = module.SubmitWorkflowExecutor()(
        module.SubmitWorkflowAction(workflow_path="storage/project/workflow.py"),
        conversation,
    )
    assert observation.status == "awaiting_confirmation"
    assert captured["url"].endswith(
        f"/conversations/{conversation_id.hex}/workflow-executions"
    )
    assert captured["headers"]["x-pyromind-debug-user-id"] == "42"
    assert captured["headers"]["cookie"] == "auth_token=ephemeral"
    assert captured["body"]["workflow_path"] == "storage/project/workflow.py"
