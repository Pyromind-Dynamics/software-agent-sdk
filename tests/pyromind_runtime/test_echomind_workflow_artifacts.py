import logging
from pathlib import Path
from traceback import format_exception
from unittest.mock import Mock
from uuid import uuid4

import pytest

from pyromind_agent_server import echomind_train_backend
from pyromind_agent_server.echomind_evaluator_backend import (
    PlatformEvaluatorWorkflowBackend,
)
from pyromind_agent_server.echomind_runtime_contracts import (
    RuntimeSubmissionNotAttemptedError,
)
from pyromind_agent_server.echomind_train_backend import PlatformTrainWorkflowBackend
from pyromind_agent_server.echomind_train_runtime import (
    TrainRuntimeError,
    TrainWorkflowSpec,
)
from pyromind_runtime.domain.capabilities import HarnessCapabilities
from pyromind_runtime.domain.content import JsonObject
from pyromind_runtime.domain.snapshot import ConversationSnapshot
from pyromind_runtime.domain.workflow_execution import WorkflowExecutionRequest
from pyromind_runtime.infrastructure.file_product_store import FileProductStore


@pytest.fixture
def archived_wandb_workflow(tmp_path: Path) -> tuple[Path, TrainWorkflowSpec, str]:
    conversation_id = uuid4().hex
    conversation = tmp_path / conversation_id
    conversation.mkdir()
    store = FileProductStore(conversation)
    store.create(
        ConversationSnapshot(
            conversation_id=conversation_id,
            capabilities=HarnessCapabilities(cancel=True),
        ),
        user_id="owner",
        profile_id="echomind",
    )
    store.save_x_cluster("us-west-1")
    key = "0123456789abcdef" * 2 + "01234567"

    def workflow(value: str) -> JsonObject:
        return {
            "nodes": [
                {
                    "id": "wandb",
                    "data": {
                        "nodeType": "WandbConfigBuilderNode",
                        "config": {"wandb_api_key": value},
                    },
                }
            ]
        }

    store.save_workflow_execution_request(
        WorkflowExecutionRequest(
            request_id="archived-request",
            conversation_id=conversation_id,
            workflow_path="workflow.py",
            fingerprint="fingerprint",
            workflow=workflow(key),
            summary={},
            status="submitted",
            task_id="10237",
        ),
        expected_revision=None,
    )
    spec = TrainWorkflowSpec(
        task_name="train",
        conversation_id=conversation_id,
        cluster="us-west-1",
        owner_user_id="owner",
        workflow=workflow("WANDB_API_KEY"),
        wandb_credential_task_id="10237",
    )
    return tmp_path, spec, key


def test_train_resolves_archived_wandb_key_only_in_submission(
    monkeypatch, caplog, archived_wandb_workflow
):
    conversations_dir, spec, key = archived_wandb_workflow
    client = Mock()
    client.studio.create.return_value.task_id = 42
    monkeypatch.setattr(echomind_train_backend, "_client", lambda **_: client)

    backend = PlatformTrainWorkflowBackend(conversations_dir=conversations_dir)
    assert backend.submit_workflow(spec=spec, request_context=None) == "42"
    submitted = client.studio.create.call_args.args[0].workflow
    assert submitted["nodes"][0]["data"]["nodeType"] == "PyroMindWandbConfigBuilderNode"
    assert submitted["nodes"][0]["data"]["config"]["wandb_api_key"] == key
    assert spec.workflow is not None
    assert spec.workflow["nodes"][0]["data"]["nodeType"] == "WandbConfigBuilderNode"
    assert (
        spec.workflow["nodes"][0]["data"]["config"]["wandb_api_key"] == "WANDB_API_KEY"
    )

    sdk_logger = logging.getLogger("pyromind_sdk")

    def fail_submission(_request):
        sdk_logger.error("Invalid workflow value: %s", key)
        raise ValueError(f"Invalid workflow value: {key}")

    client.studio.create.side_effect = fail_submission
    with pytest.raises(TrainRuntimeError, match=r"\[REDACTED\]") as failure:
        backend.submit_workflow(spec=spec, request_context=None)
    assert key not in "".join(format_exception(failure.value))
    assert key not in caplog.text
    sdk_logger.error("SDK logging restored")
    assert "SDK logging restored" in caplog.text


def test_train_redacts_archived_wandb_key_echoed_in_task_status(
    monkeypatch, caplog, archived_wandb_workflow
):
    _, spec, key = archived_wandb_workflow
    client = Mock()
    task = Mock(status="Failed", error_message=f"W&B rejected {key}", task_id=10304)

    def failed_task(_task_id):
        logging.getLogger("pyromind_sdk").error("Returned task status: %s", key)
        return task

    client.studio.get_task.side_effect = failed_task
    monkeypatch.setattr(echomind_train_backend, "_client", lambda **_: client)

    backend = PlatformTrainWorkflowBackend()
    state = backend.query_task(task_id="10304", spec=spec, request_context=None)
    assert state.message == "Train task failed: W&B rejected [REDACTED]"

    def running_task():
        logging.getLogger("pyromind_sdk").error("Listed task workflow: %s", key)
        task.status = "Running"
        task.name = "train"
        return [task]

    def stop_task(_task_id):
        logging.getLogger("pyromind_sdk").error("Stopped task workflow: %s", key)
        task.status = "Canceled"
        return task

    client.studio.list.side_effect = running_task
    client.studio.stop.side_effect = stop_task
    assert (
        backend.find_task_by_name(name="train", spec=spec, request_context=None)
        == "10304"
    )
    backend.cancel_task(task_id="10304", spec=spec, request_context=None)
    client.studio.stop.assert_called_once_with("10304")
    assert key not in caplog.text


@pytest.mark.parametrize(
    "update", [{"owner_user_id": "someone-else"}, {"cluster": "other"}]
)
def test_train_denies_out_of_scope_wandb_credential_source(
    monkeypatch, archived_wandb_workflow, update
):
    conversations_dir, spec, _ = archived_wandb_workflow
    client = Mock()
    monkeypatch.setattr(echomind_train_backend, "_client", lambda **_: client)

    backend = PlatformTrainWorkflowBackend(conversations_dir=conversations_dir)
    with pytest.raises(RuntimeSubmissionNotAttemptedError, match="credential source"):
        backend.submit_workflow(
            spec=spec.model_copy(update=update), request_context=None
        )
    client.studio.create.assert_not_called()


def test_train_resolves_latest_complete_checkpoint(monkeypatch):
    client = Mock()
    client.studio.get_task.return_value.status = "Succeeded"
    storage = Mock()
    storage.list.side_effect = {
        "/models/run/candidate": {"checkpoint-9", "checkpoint-10", "checkpoint-25"},
        "/models/run/candidate/checkpoint-25": {"trainer_state.json"},
        "/models/run/candidate/checkpoint-10": {
            "adapter_config.json",
            "adapter_model.safetensors",
        },
    }.__getitem__
    monkeypatch.setattr(echomind_train_backend, "_client", lambda **_: client)
    monkeypatch.setattr(
        echomind_train_backend, "build_runtime_storage", lambda **_: storage
    )
    spec = TrainWorkflowSpec(
        task_name="train",
        conversation_id="conversation",
        cluster="us-west-1",
        workflow={},
        output_uri="/workspace/models/run/candidate",
        artifact_selection="latest_lora_checkpoint",
    )

    state = PlatformTrainWorkflowBackend().query_task(
        task_id="42", spec=spec, request_context=None
    )

    assert state.status == "succeeded"
    assert state.output_uri == "/workspace/models/run/candidate/checkpoint-10"
    client.studio.get_node_output.assert_not_called()


@pytest.mark.parametrize("missing_shard", [False, True])
def test_train_verifies_merged_model_weights(monkeypatch, missing_shard):
    client = Mock()
    client.studio.get_task.return_value.status = "Succeeded"
    storage = Mock()
    directory = "/models/run/merged"
    storage.list.return_value = {
        "config.json",
        "model.safetensors.index.json",
        "model-00001.safetensors",
    }
    if not missing_shard:
        storage.list.return_value.add("model-00002.safetensors")
    storage.read.side_effect = {
        f"{directory}/config.json": b'{"model_type":"qwen3_vl"}',
        f"{directory}/model.safetensors.index.json": (
            b'{"weight_map":{"first":"model-00001.safetensors",'
            b'"second":"model-00002.safetensors"}}'
        ),
    }.__getitem__
    monkeypatch.setattr(echomind_train_backend, "_client", lambda **_: client)
    monkeypatch.setattr(
        echomind_train_backend, "build_runtime_storage", lambda **_: storage
    )
    spec = TrainWorkflowSpec(
        task_name="train",
        conversation_id="conversation",
        cluster="us-west-1",
        workflow={},
        output_uri=f"/workspace{directory}",
        artifact_selection="merged_model",
    )

    state = PlatformTrainWorkflowBackend().query_task(
        task_id="42", spec=spec, request_context=None
    )

    if missing_shard:
        assert state.status == "failed"
        assert state.failure_category == "validation"
        assert state.output_uri is None
    else:
        assert state.status == "succeeded"
        assert state.output_uri == spec.output_uri
    client.studio.get_node_output.assert_not_called()


def test_evaluation_requires_successful_report_with_metrics(monkeypatch):
    client = Mock()
    client.studio.get_task.return_value.status = "Succeeded"
    storage = Mock()
    storage.read.side_effect = [
        b'{"status":"running"}',
        b'{"status":"succeeded","metrics":{"accuracy":0.75}}',
    ]
    monkeypatch.setattr(echomind_train_backend, "_client", lambda **_: client)
    monkeypatch.setattr(
        echomind_train_backend, "build_runtime_storage", lambda **_: storage
    )
    spec = TrainWorkflowSpec(
        task_name="evaluate",
        conversation_id="conversation",
        cluster="us-west-1",
        workflow={},
        output_uri="/workspace/models/run/report.json",
        artifact_selection="evaluation_report",
    )
    backend = PlatformEvaluatorWorkflowBackend()

    incomplete = backend.query_task(task_id="43", spec=spec, request_context=None)
    completed = backend.query_task(task_id="43", spec=spec, request_context=None)

    assert incomplete.status == "failed"
    assert incomplete.failure_category == "validation"
    assert completed.status == "succeeded"
    assert completed.output_uri == spec.output_uri
