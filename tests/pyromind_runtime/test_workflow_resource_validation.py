import httpx
import pytest

from harness_adapter.workflow_validation import (
    ResourceValidatedWorkflowDslExecutor,
    WorkflowResourceValidationObservation,
)
from pyromind_common.tools.workflow.validate_workflow_dsl import (
    ValidateWorkflowDslAction,
)
from tests.tools.workflow.test_validate_workflow_dsl import (
    _Response,
    _valid_response as _structure_valid_response,
)


VALIDATED_WORKFLOW = {
    "id": "workflow-1",
    "name": "normalized-by-server",
    "nodes": [
        {"id": "train", "data": {"nodeType": "Train", "config": {"gpu_count": 4}}}
    ],
    "edges": [],
}


def _valid_response():
    response = _structure_valid_response()
    response.json()["data"]["workflow"] = VALIDATED_WORKFLOW
    return response


def _resource_response(*, valid=False, code="WORKFLOW_RESOURCES_INSUFFICIENT"):
    return _Response(
        200,
        {
            "success": True,
            "data": {
                "valid": valid,
                "errors": []
                if valid
                else [
                    {
                        "code": code,
                        "node_id": "train",
                        "message": "资源不足",
                        "source": "k8s",
                    }
                ],
                "warnings": [],
                "available_resources": []
                if valid
                else [
                    {
                        "gpu_type": "NVIDIA-H100-NVL",
                        "max_gpu_count": 4,
                        "max_cpu_cores": 16,
                        "max_mem_gib": 64,
                        "node": "must-not-leak",
                    }
                ],
            },
        },
    )


@pytest.mark.parametrize("cpu_only", [False, True])
def test_resource_shortage_is_tool_error_and_exposes_only_selected_node_resources(
    monkeypatch,
    cpu_only,
):
    calls = []
    response = _resource_response()
    if cpu_only:
        response.json()["data"]["available_resources"][0].update(
            gpu_type=None, max_gpu_count=0
        )

    def post(url, **kwargs):
        calls.append((url, kwargs))
        return _valid_response() if url.endswith("/dsl/validate") else response

    monkeypatch.setattr(httpx, "post", post)
    result = ResourceValidatedWorkflowDslExecutor(
        headers={"authorization": "Bearer test"}
    )(ValidateWorkflowDslAction(dsl="train = Train()", name="demo"))
    assert len(calls) == 2
    assert calls[0][1]["json"] == {"dsl": "train = Train()", "name": "demo"}
    assert calls[1][1]["json"] == {"workflow": VALIDATED_WORKFLOW}
    assert isinstance(result, WorkflowResourceValidationObservation)
    assert result.validated_workflow == VALIDATED_WORKFLOW
    assert "validated_workflow" not in result.model_dump()
    assert "dsl" not in calls[1][1]["json"]
    assert calls[0][1]["headers"] == calls[1][1]["headers"]
    assert not result.valid and result.is_error
    assert result.failure_stage == "resources" and not result.retryable
    assert result.errors[0].node_id == "train"
    assert result.available_resources[0].max_gpu_count == (0 if cpu_only else 4)
    assert result.model_dump()["available_resources"][0]["gpu_type"] == (
        None if cpu_only else "NVIDIA-H100-NVL"
    )
    assert ("仅申请 CPU 和内存" in result.text) is cpu_only
    assert result.available_resources[0].max_cpu_cores == 16
    assert "max" in result.text and "sum" in result.text
    assert "must-not-leak" not in result.model_dump_json()
    assert "single_node_options" not in result.model_dump_json()
    assert "single_node_options" not in result.text


def test_legacy_node_options_are_not_exposed_to_agent(monkeypatch):
    response = _resource_response()
    response.json()["data"]["available_resources"][0]["single_node_options"] = [
        {"gpu_count": 4, "cpu_cores": 16, "mem_gib": 64, "node": "private-host"}
    ]
    monkeypatch.setattr(
        httpx,
        "post",
        lambda url, **kwargs: (
            _valid_response() if url.endswith("/dsl/validate") else response
        ),
    )
    result = ResourceValidatedWorkflowDslExecutor()(
        ValidateWorkflowDslAction(dsl="test")
    )
    assert (
        not result.valid and result.errors[0].code == "WORKFLOW_RESOURCES_INSUFFICIENT"
    )
    assert "single_node_options" not in result.model_dump_json()
    assert "private-host" not in result.model_dump_json()
    assert "single_node_options" not in result.text


def test_structural_failure_does_not_query_resources(monkeypatch):
    calls = []

    def post(url, **kwargs):
        calls.append(url)
        return _Response(
            200,
            {
                "success": True,
                "data": {"valid": False, "errors": [{"code": "DSL_PARSE_FAILED"}]},
            },
        )

    monkeypatch.setattr(httpx, "post", post)
    result = ResourceValidatedWorkflowDslExecutor()(
        ValidateWorkflowDslAction(dsl="bad")
    )
    assert result.valid is False and len(calls) == 1


@pytest.mark.parametrize(
    "response",
    [
        _Response(503, {}, text="private-host secret-token"),
        _Response(200, {"success": False, "message": "private-host secret-token"}),
        _Response(200, {"success": True, "data": {"valid": "true"}}),
        _Response(200, {"success": True, "data": {"valid": False}}),
    ],
)
def test_resource_transport_and_malformed_results_fail_closed(monkeypatch, response):
    monkeypatch.setattr(httpx, "post", lambda *a, **kw: response)
    result = ResourceValidatedWorkflowDslExecutor().validate_resources(
        VALIDATED_WORKFLOW
    )
    assert result.valid is False and result.is_error
    assert result.errors[0].code == "RESOURCE_CHECK_FAILED"
    assert "private-host" not in result.text and "secret-token" not in result.text


@pytest.mark.parametrize("issues", ["bad", {}, [None], ["bad"]])
def test_resource_malformed_issues_cannot_silently_pass(monkeypatch, issues):
    response = _Response(
        200, {"success": True, "data": {"valid": True, "errors": issues}}
    )
    monkeypatch.setattr(httpx, "post", lambda *a, **kw: response)
    result = ResourceValidatedWorkflowDslExecutor().validate_resources(
        VALIDATED_WORKFLOW
    )
    assert result.is_error and result.valid is False


def test_resource_endpoint_can_be_explicitly_configured(monkeypatch):
    calls = []
    monkeypatch.setattr(
        httpx, "post", lambda url, **kw: calls.append(url) or _valid_response()
    )
    executor = ResourceValidatedWorkflowDslExecutor(
        endpoint_url="https://custom/schema",
        resource_endpoint_url="https://custom/resources",
    )
    result = executor(ValidateWorkflowDslAction(dsl="test"))
    assert result.valid and calls == [
        "https://custom/schema",
        "https://custom/resources",
    ]


def test_legacy_openhands_validator_remains_structure_only(monkeypatch):
    from pyromind_common.tools.workflow.validate_workflow_dsl import (
        ValidateWorkflowDslExecutor,
        ValidateWorkflowDslObservation,
        ValidateWorkflowDslTool,
    )

    calls = []
    monkeypatch.setattr(
        httpx, "post", lambda url, **kw: calls.append(url) or _valid_response()
    )
    tool = ValidateWorkflowDslTool.create()[0]
    assert type(tool.executor) is ValidateWorkflowDslExecutor
    assert tool.executor(ValidateWorkflowDslAction(dsl="test")).valid
    assert len(calls) == 1 and calls[0].endswith("/dsl/validate")
    assert "available_resources" not in ValidateWorkflowDslObservation.model_fields
    assert "single_node_options" not in tool.description


@pytest.mark.asyncio
async def test_pi_validation_uses_new_resource_checker(tmp_path, monkeypatch):
    from harness_adapter.pi_adapter.business_tools import (
        execute_validation_tool,
        validation_tool_spec,
    )
    from pyromind_common.sdk.workspace.local import LocalWorkspace
    from pyromind_runtime.domain.context import RequestContext

    (tmp_path / "workflow.py").write_text("test = Train()")
    calls = []

    def post(url, **kwargs):
        calls.append(url)
        return (
            _valid_response() if url.endswith("/dsl/validate") else _resource_response()
        )

    monkeypatch.setattr(httpx, "post", post)
    result = await execute_validation_tool(
        LocalWorkspace(working_dir=tmp_path),
        {"dsl_path": "workflow.py"},
        RequestContext(user_id="42"),
    )
    assert result["is_error"] and result["details"]["valid"] is False
    assert result["details"]["available_resources"][0]["max_gpu_count"] == 4
    assert len(calls) == 2 and calls[-1].endswith("/resources/validate")
    assert "single_node_options" not in validation_tool_spec()["description"]
    assert "same machine" in validation_tool_spec()["description"]
    assert "gpu_type=null" in validation_tool_spec()["description"]
    assert "CPU/memory-only" in validation_tool_spec()["description"]
    assert "single_node_options" not in result["details"]["available_resources"][0]


@pytest.mark.parametrize("workflow", [None, "bad", {}, {"nodes": None, "edges": []}])
def test_missing_validated_workflow_fails_without_resource_request(
    monkeypatch, workflow
):
    calls = []
    response = _structure_valid_response()
    response.json()["data"]["workflow"] = workflow
    monkeypatch.setattr(httpx, "post", lambda url, **kw: calls.append(url) or response)
    result = ResourceValidatedWorkflowDslExecutor()(
        ValidateWorkflowDslAction(dsl="test")
    )
    assert result.valid is False and result.is_error
    assert result.retryable is False
    assert len(calls) == 1 and calls[0].endswith("/dsl/validate")
