from __future__ import annotations

from types import SimpleNamespace

import pytest

from harness_adapter import workflow_submission as module
from pyromind_runtime.domain.context import RequestContext


async def test_query_preserves_platform_failure_details(monkeypatch):
    task = SimpleNamespace(status="Failed", error_message="Model preparation failed")
    client = SimpleNamespace(studio=SimpleNamespace(get_task=lambda _: task))
    monkeypatch.setattr(module, "create_workflow_api_client", lambda **kwargs: client)

    result = await module.PlatformWorkflowSubmission().query(
        "42", RequestContext(user_id="42", cookie="auth_token=ephemeral")
    )

    assert result == {
        "task_id": "42",
        "status": "Failed",
        "error_summary": "Model preparation failed",
    }


def test_prepare_uses_storage_dsl_validation_and_marks_training(monkeypatch):
    downloaded: list[str] = []
    monkeypatch.setattr(module, "_default_storage_base_url", lambda: "https://storage")

    def download(**kwargs):
        downloaded.append(kwargs["storage_path"])
        return b"workflow = 'sft'"

    monkeypatch.setattr(module, "download_file_from_pyromind", download)
    monkeypatch.setattr(
        module,
        "ResourceValidatedWorkflowDslExecutor",
        lambda **kwargs: (
            lambda action: SimpleNamespace(
                is_error=False,
                valid=True,
                validated_workflow={
                    "name": "server-normalized",
                    "nodes": [{"data": {"nodeType": "ModelTrainSFTNode"}}],
                    "edges": [],
                },
            )
        ),
    )

    result = module.PlatformWorkflowSubmission._prepare(
        "/workspace/.pyromind-agent/conversation/workflow.py",
        RequestContext(user_id="42", cookie="auth_token=ephemeral"),
    )
    assert downloaded == ["/.pyromind-agent/conversation/workflow.py"]
    summary = result["summary"]
    assert isinstance(summary, dict)
    assert summary["has_training_node"] is True
    workflow = result["workflow"]
    assert isinstance(workflow, dict)
    assert workflow["name"] == "server-normalized"
    assert result["fingerprint"]


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("/workspace/project/workflow.py", "/project/workflow.py"),
        ("/project/workflow.py", "/project/workflow.py"),
    ],
)
def test_prepare_accepts_public_and_storage_api_workflow_paths(
    monkeypatch, path, expected
):
    downloaded: list[str] = []
    monkeypatch.setattr(module, "_default_storage_base_url", lambda: "https://storage")
    monkeypatch.setattr(
        module,
        "download_file_from_pyromind",
        lambda **kwargs: (
            downloaded.append(kwargs["storage_path"]) or b"workflow = 'sft'"
        ),
    )
    monkeypatch.setattr(
        module,
        "ResourceValidatedWorkflowDslExecutor",
        lambda **kwargs: (
            lambda action: SimpleNamespace(
                is_error=False,
                valid=True,
                validated_workflow={
                    "name": "server-normalized",
                    "nodes": [{"data": {"nodeType": "ModelTrainSFTNode"}}],
                    "edges": [],
                },
            )
        ),
    )

    module.PlatformWorkflowSubmission._prepare(path, RequestContext(user_id="42"))

    assert downloaded == [expected]


def test_prepare_rejects_the_former_pi_mount():
    with pytest.raises(ValueError, match="execution-only path"):
        module.PlatformWorkflowSubmission._prepare(
            "/target-workspace/project/workflow.py",
            RequestContext(user_id="42"),
        )


@pytest.mark.parametrize(
    "path",
    [
        "storage/../workflow.py",
        "storage/project/file.txt",
        "/workflow.py",
        # 已删除的别名不能再用于指定工作流。
        "storage/project/workflow.py",
    ],
)
def test_prepare_rejects_non_storage_workflow(path):
    with pytest.raises(ValueError, match="workflow_path"):
        module.PlatformWorkflowSubmission._prepare(path, RequestContext(user_id="42"))


@pytest.mark.parametrize(
    ("route", "env", "cluster", "header"),
    [("c#pre", "pre", "c", "c#pre"), (None, "prod", "us-west-1", "us-west-1")],
)
def test_submit_never_retries_platform_create(monkeypatch, route, env, cluster, header):
    monkeypatch.setenv("APP_ENV", "prod")
    seen: dict = {}

    def create_client(**kwargs):
        seen["routing"] = kwargs
        return object()

    monkeypatch.setattr(module, "create_workflow_api_client", create_client)

    def submit(**kwargs):
        seen.update(kwargs)
        return SimpleNamespace(task_id="task-123")

    monkeypatch.setattr(module, "submit_workflow_task", submit)
    task_id = module.PlatformWorkflowSubmission._submit(
        {"name": "sft"},
        "conversation-1",
        RequestContext(user_id="42", cookie="auth_token=ephemeral", x_cluster=route),
    )
    assert task_id == "task-123"
    assert seen["routing"]["env"] == env
    assert seen["routing"]["cluster"] == cluster
    assert seen["routing"]["headers"] == {"x-cluster": header}
    assert seen["retry_create"] is False
    assert seen["conversation_id"] == "conversation-1"


def test_final_resource_check_uses_effective_workflow_and_request_identity(monkeypatch):
    seen = {}
    effective = {
        "name": "frozen",
        "nodes": [{"id": "gpu", "data": {"config": {"gpu_count": 8}}}],
    }

    class Validator:
        def __init__(self, headers):
            seen["headers"] = headers

        def validate_resources(self, workflow):
            seen["workflow"] = workflow
            return SimpleNamespace(
                valid=False, is_error=True, text="资源不足，max_gpu_count=4"
            )

    monkeypatch.setattr(module, "ResourceValidatedWorkflowDslExecutor", Validator)
    with pytest.raises(ValueError, match="资源不足"):
        module.PlatformWorkflowSubmission._validate_resources(
            effective,
            RequestContext(
                user_id="42",
                cookie="auth_token=test",
                authorization="Bearer test",
                x_cluster="cluster#pre",
            ),
        )
    assert seen["workflow"] == effective
    assert seen["headers"]["x-cluster"] == "cluster#pre"
    assert seen["headers"]["authorization"] == "Bearer test"


def test_final_resource_check_does_not_omit_values_equal_to_client_defaults(
    monkeypatch,
):
    seen = []
    workflow = {
        "name": "w",
        "nodes": [
            {
                "id": "1",
                "data": {
                    "nodeType": "CustomCommandNode",
                    "config": {"gpu_count": 4, "cpu": 8, "memory": 64},
                    "nodeDefinition": {
                        "input": {
                            "required": {
                                "gpu_count": ["INT", {"default": 4}],
                                "cpu": ["INT", {"default": 8}],
                                "memory": ["INT", {"default": 64}],
                            }
                        }
                    },
                },
            }
        ],
        "edges": [],
    }

    class Validator:
        def __init__(self, **kwargs):
            pass

        def validate_resources(self, workflow):
            seen.append(workflow)
            return SimpleNamespace(is_error=False, valid=True)

    monkeypatch.setattr(module, "ResourceValidatedWorkflowDslExecutor", Validator)
    module.PlatformWorkflowSubmission._validate_resources(
        workflow, RequestContext(user_id="42")
    )
    assert seen[0]["nodes"][0]["data"]["config"] == {
        "gpu_count": 4,
        "cpu": 8,
        "memory": 64,
    }
    assert "nodeDefinition" in workflow["nodes"][0]["data"]


def test_prepare_requires_the_validated_workflow_instead_of_reparsing(monkeypatch):
    monkeypatch.setattr(module, "download_file_from_pyromind", lambda **kw: b"workflow")
    monkeypatch.setattr(
        module,
        "ResourceValidatedWorkflowDslExecutor",
        lambda **kw: lambda action: SimpleNamespace(is_error=False, valid=True),
    )
    with pytest.raises(ValueError, match="no validated definition"):
        module.PlatformWorkflowSubmission._prepare(
            "/project/workflow.py", RequestContext(user_id="42")
        )
