from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock

import httpx
import pytest
from pydantic import BaseModel

from harness_adapter.pi_adapter.business_tool_host import (
    PyromindBusinessToolHost,
    ToolExecutionContext,
)
from pyromind_common.tools.data_preparation.submission import SubmissionObservation
from pyromind_common.tools.workflow.task_submission import submit_workflow_task
from pyromind_runtime.domain.context import RequestContext
from tests.pyromind_runtime.test_workflow_resource_validation import _resource_response


@pytest.mark.parametrize(
    "tool_name",
    [
        "df_submit_evaluation",
        "edp_render",
        "edp_submit",
        "edp_aggregate",
        "workflow_debug",
    ],
)
@pytest.mark.parametrize("resource_state", ["shortage", "unavailable", "valid"])
async def test_pi_direct_submissions_check_actual_resources(
    tmp_path, monkeypatch, tool_name, resource_state
):
    root = Path(__file__).parents[2]
    host = PyromindBusinessToolHost(
        [
            root / ".agents/skills/data-processing",
            root / ".agents/skills/training-analysis",
        ]
    )
    workflow = {
        "id": "frozen-workflow",
        "nodes": [{"data": {"nodeType": "CustomCommandCPUNode", "config": {"cpu": 2}}}],
        "edges": [],
    }
    client = MagicMock()
    client.studio.create.return_value.task_id = "platform-task"
    client.studio.create.return_value.status = "Pending"
    calls = []

    def post(url, **kwargs):
        calls.append((url, kwargs))
        if resource_state == "unavailable":
            raise httpx.ReadTimeout("resource service unavailable")
        return _resource_response(valid=resource_state == "valid")

    monkeypatch.setattr(httpx, "post", post)

    class Action(BaseModel):
        pass

    class Tool:
        action_type = Action
        executor = None
        meta = None

        def __call__(self, action, conversation):
            try:
                response = submit_workflow_task(
                    client=client,
                    workflow=workflow,
                    name="test-resource-admission",
                    conversation_id=str(conversation.id),
                    test_mode=tool_name == "workflow_debug",
                )
                return SubmissionObservation.from_text(
                    text="submitted", task_id=response.task_id, status="Pending"
                )
            except ValueError as exc:
                return SubmissionObservation.from_text(text=str(exc), is_error=True)

    host._factories[tool_name] = cast(Any, lambda context: Tool())
    result = await host.execute(
        tool_name,
        {},
        ToolExecutionContext(
            conversation_id="resource-admission",
            workspace_root=tmp_path,
            request_context=RequestContext(
                user_id="42",
                authorization="Bearer user-auth",
                cookie="auth_token=user-cookie",
                x_cluster="test#pre",
            ),
            model_configuration={},
        ),
        tool_call_id="call-1",
    )
    assert len(calls) == 1 and calls[0][0].endswith("/resources/validate")
    assert calls[0][1]["json"] == {"workflow": workflow}
    assert calls[0][1]["headers"]["authorization"] == "Bearer user-auth"
    assert calls[0][1]["headers"]["cookie"] == "auth_token=user-cookie"
    assert calls[0][1]["headers"]["x-cluster"] == "test#pre"
    assert result["is_error"] is (resource_state != "valid")
    assert client.studio.create.call_count == (1 if resource_state == "valid" else 0)
    if resource_state != "valid":
        assert result["signals"] == []
        assert "must-not-leak" not in str(result)
