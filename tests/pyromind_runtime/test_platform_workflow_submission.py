from __future__ import annotations

from types import SimpleNamespace

import pytest
from harness_adapter import workflow_submission as module
from pyromind_runtime.domain.context import RequestContext


def test_prepare_uses_storage_dsl_validation_and_marks_training(monkeypatch):
    downloaded: list[str] = []
    monkeypatch.setattr(module, "_default_storage_base_url", lambda: "https://storage")

    def download(**kwargs):
        downloaded.append(kwargs["storage_path"])
        return b"workflow = 'sft'"

    monkeypatch.setattr(module, "download_file_from_pyromind", download)
    monkeypatch.setattr(
        module,
        "ValidateWorkflowDslExecutor",
        lambda **kwargs: lambda action: SimpleNamespace(is_error=False, valid=True),
    )
    monkeypatch.setattr(
        module,
        "convert_dsl_to_xyflow",
        lambda dsl, name: {
            "name": name,
            "nodes": [{"data": {"nodeType": "ModelTrainSFTNode"}}],
        },
    )
    result = module.PlatformWorkflowSubmission._prepare(
        "storage/.pyromind-agent/conversation/workflow.py",
        RequestContext(user_id="42", cookie="auth_token=ephemeral"),
    )
    assert downloaded == ["/.pyromind-agent/conversation/workflow.py"]
    summary = result["summary"]
    assert isinstance(summary, dict)
    assert summary["has_training_node"] is True
    assert result["fingerprint"]


@pytest.mark.parametrize(
    "path",
    ["/tmp/workflow.py", "storage/../workflow.py", "storage/project/file.txt"],
)
def test_prepare_rejects_non_storage_workflow(path):
    with pytest.raises(ValueError, match="workflow_path"):
        module.PlatformWorkflowSubmission._prepare(path, RequestContext(user_id="42"))


def test_submit_never_retries_platform_create(monkeypatch):
    seen: dict = {}
    monkeypatch.setattr(module, "create_workflow_api_client", lambda **kwargs: object())

    def submit(**kwargs):
        seen.update(kwargs)
        return SimpleNamespace(task_id="task-123")

    monkeypatch.setattr(module, "submit_workflow_task", submit)
    task_id = module.PlatformWorkflowSubmission._submit(
        {"name": "sft"},
        "conversation-1",
        RequestContext(user_id="42", cookie="auth_token=ephemeral", x_cluster="c#pre"),
    )
    assert task_id == "task-123"
    assert seen["retry_create"] is False
    assert seen["conversation_id"] == "conversation-1"
