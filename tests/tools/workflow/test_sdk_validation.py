from copy import deepcopy

import httpx
import pytest

from pyromind_common.tools.workflow import validate_workflow_dsl as validation


@pytest.mark.parametrize("test_mode", [False, True])
def test_validation_prepares_sdk_graph_without_creating_a_task(monkeypatch, test_mode):
    graph = {
        "name": "workflow",
        "nodes": [],
        "edges": [],
        "execution_argos": [{"execution_mode": "prod"}, {"other": "preserved"}],
    }
    payload = {"success": True, "data": {"valid": True, "workflow": graph}}
    original = deepcopy(payload)
    requests = []

    def post(url, **kwargs):
        requests.append(kwargs["json"])
        return httpx.Response(200, json=payload)

    monkeypatch.setattr(validation.httpx, "post", post)
    result = validation.ValidateWorkflowDslExecutor()(
        validation.ValidateWorkflowDslAction(
            dsl="source = Node(id=1)", test_mode=test_mode
        )
    )
    assert result.valid is True
    assert result.raw_response is not None
    returned = result.raw_response["data"]["workflow"]
    expected = (
        [{"other": "preserved"}, {"execution_mode": "test"}]
        if test_mode
        else graph["execution_argos"]
    )
    assert returned["execution_argos"] == expected
    assert payload == original
    assert requests == [{"name": "workflow", "dsl": "source = Node(id=1)"}]


def test_test_mode_validation_rejects_missing_actual_graph(monkeypatch):
    monkeypatch.setattr(
        validation.httpx,
        "post",
        lambda *args, **kwargs: httpx.Response(
            200, json={"success": True, "data": {"valid": True}}
        ),
    )
    result = validation.ValidateWorkflowDslExecutor()(
        validation.ValidateWorkflowDslAction(dsl="source = Node(id=1)", test_mode=True)
    )
    assert result.is_error
    assert result.valid is False
    assert "no graph" in result.text
