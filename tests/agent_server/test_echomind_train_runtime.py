from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import pytest
from pyromind_agent_server import echomind_train_backend as backend_module
from pyromind_agent_server.echomind_train_backend import PlatformTrainWorkflowBackend
from pyromind_runtime.domain.context import RequestContext

from openhands.agent_server.echomind_graph_runs import TaskRun
from openhands.agent_server.echomind_projects import (
    ProjectGraph,
    ProjectInfo,
    ProjectNode,
    ProjectNodeOutput,
    ProjectRecord,
    ProjectStore,
)
from openhands.agent_server.echomind_runtime_contracts import (
    RuntimeCredentialUnavailableError,
    RuntimeStageContext,
)
from openhands.agent_server.echomind_storage import canonical_sha256
from openhands.agent_server.echomind_train_runtime import (
    TrainRuntimeExecutor,
    TrainTaskState,
    TrainWorkflowSpec,
    bind_workflow_inputs,
    bind_workflow_output,
    prepare_workflow_for_submission,
)


def test_bind_workflow_inputs_resolves_artifact_uris():
    workflow = {
        "nodes": [
            {
                "data": {
                    "config": {
                        "train_data_path": "",
                        "val_data_path": "",
                    }
                }
            }
        ]
    }

    result = bind_workflow_inputs(
        workflow,
        bindings={
            "nodes[0].data.config.train_data_path": "train_messages",
            "nodes[0].data.config.val_data_path": "val_messages",
        },
        inputs={
            "train_messages": {
                "uri": "artifact://train/1",
                "metadata": {"source_uri": "storage://train.jsonl"},
            },
            "val_messages": "storage://val.jsonl",
        },
    )

    config = result["nodes"][0]["data"]["config"]
    assert config == {
        "train_data_path": "storage://train.jsonl",
        "val_data_path": "storage://val.jsonl",
    }


def test_bind_workflow_output_rewrites_only_the_selected_node():
    workflow = {
        "nodes": [
            {"id": "train", "data": {"config": {"output_path": "old"}}},
            {"id": "other", "data": {"config": {"output_path": "untouched"}}},
        ]
    }

    result = bind_workflow_output(
        workflow,
        node_id="train",
        parameter="output_path",
        value="/workspace/models/run-1/candidate",
    )

    assert result["nodes"][0]["data"]["config"]["output_path"] == (
        "/workspace/models/run-1/candidate"
    )
    assert result["nodes"][1]["data"]["config"]["output_path"] == "untouched"
    assert workflow["nodes"][0]["data"]["config"]["output_path"] == "old"


def test_prepare_workflow_renders_command_and_run_scoped_output():
    workflow = {
        "nodes": [
            {
                "id": "evaluate",
                "data": {
                    "config": {
                        "command": "old",
                        "merged_model_path": "old",
                        "output_path": "old",
                    }
                },
            }
        ]
    }
    spec = TrainWorkflowSpec(
        task_name="eval-task",
        conversation_id="conversation-1",
        cluster="us-west-1",
        workflow=workflow,
        inputs={
            "candidate_model": "/workspace/models/run-1/candidate",
            "validation_dataset": "/workspace/datasets/val.jsonl",
        },
        output_uri="/workspace/models/run-1/merged",
        output_node_id="evaluate",
        workflow_templates={
            "nodes[0].data.config.merged_model_path": (
                "/workspace/models/{graph_run_id}/merged"
            )
        },
        command_template=(
            "python evaluate.py --model {candidate_model} "
            "--dataset {validation_dataset} --run {graph_run_id} "
            "--report {output_uri}"
        ),
        command_binding_path="nodes[0].data.config.command",
        template_values={
            "graph_run_id": "graph-run-1",
            "output_uri": "/workspace/models/run-1/report.json",
        },
    )

    result = prepare_workflow_for_submission(workflow, spec=spec)

    config = result["nodes"][0]["data"]["config"]
    assert config["command"] == (
        "python evaluate.py --model /workspace/models/run-1/candidate "
        "--dataset /workspace/datasets/val.jsonl --run graph-run-1 "
        "--report /workspace/models/run-1/report.json"
    )
    assert config["output_path"] == "/workspace/models/run-1/merged"
    assert config["merged_model_path"] == "/workspace/models/graph-run-1/merged"


def test_train_executor_submits_polls_and_emits_model(tmp_path):
    projects_root = tmp_path / "projects"
    project, context, task_run = _train_fixture(projects_root)
    backend = _FakeBackend()
    executor = TrainRuntimeExecutor(
        projects_root=str(projects_root),
        backend=backend,
        request_context=RequestContext(
            user_id="42",
            cookie="auth_token=request-token",
            x_cluster=project.project.cluster,
        ),
    )

    prepared = executor.prepare(context)
    assert prepared.status == "waiting_external"
    assert prepared.external_task is not None
    assert prepared.external_task.provider == "pyromind.studio.training"

    submitted = executor.submit(context, task_run)
    assert submitted.status == "submitted"
    assert submitted.provider_run_id == "task-1"
    assert backend.submitted_spec is not None
    assert backend.submitted_spec.bindings == {
        "nodes[0].data.config.train_data_path": "train_messages"
    }

    running = executor.poll(
        context,
        task_run.model_copy(update={"provider_run_id": "task-1"}),
    )
    completed = executor.poll(
        context,
        task_run.model_copy(update={"provider_run_id": "task-1"}),
    )

    assert running.status == "running"
    assert completed.status == "completed"
    assert completed.artifacts[0].port == "trained_model"
    assert completed.artifacts[0].payload.kind == "model.checkpoint"
    assert completed.artifacts[0].payload.content["uri"] == (
        f"storage://models/{context.graph_run_id}/candidate/checkpoint-25"
    )


def test_train_executor_recovers_existing_task_without_resubmitting(tmp_path):
    projects_root = tmp_path / "projects"
    _, context, task_run = _train_fixture(projects_root)
    backend = _FakeBackend(existing_task_id="existing-task")
    executor = TrainRuntimeExecutor(
        projects_root=str(projects_root),
        backend=backend,
    )

    submitted = executor.submit(context, task_run)

    assert submitted.provider_run_id == "existing-task"
    assert backend.submitted_spec is None


def test_platform_train_backend_binds_workflow_before_submit(monkeypatch):
    submitted: dict[str, Any] = {}
    client = _FakeClient()

    monkeypatch.setattr(backend_module, "_client", lambda **kwargs: client)

    def submit_workflow_task(**kwargs):
        submitted.update(kwargs)
        return SimpleNamespace(task_id="platform-task-1")

    monkeypatch.setattr(
        backend_module,
        "submit_workflow_task",
        submit_workflow_task,
    )
    spec = TrainWorkflowSpec(
        task_name="echomind-train-test",
        conversation_id="conversation-1",
        cluster="us-west-1#pre",
        workflow={
            "name": "old",
            "nodes": [
                {
                    "id": "train",
                    "data": {
                        "config": {
                            "train_data_path": "",
                            "output_path": "old-output",
                        }
                    },
                }
            ],
        },
        bindings={"nodes[0].data.config.train_data_path": "train_messages"},
        inputs={"train_messages": "storage://train.jsonl"},
        output_uri="/workspace/models/run-1/candidate",
        output_node_id="train",
    )

    task_id = PlatformTrainWorkflowBackend().submit_workflow(
        spec=spec,
        request_context=None,
    )

    assert task_id == "platform-task-1"
    workflow = submitted["workflow"]
    assert workflow["name"] == "echomind-train-test"
    assert workflow["nodes"][0]["data"]["config"]["train_data_path"] == (
        "storage://train.jsonl"
    )
    assert workflow["nodes"][0]["data"]["config"]["output_path"] == (
        "/workspace/models/run-1/candidate"
    )
    assert submitted["retry_create"] is False
    assert submitted["task_kind"] == "echomind_train"
    assert client.closed is True


def test_platform_train_backend_requires_owner_credentials(monkeypatch):
    monkeypatch.setenv("PYROMIND_RUNTIME_API_KEY", "unrelated-global-key")
    spec = TrainWorkflowSpec(
        task_name="echomind-train-test",
        conversation_id="conversation-1",
        cluster="us-west-1#pre",
        workflow={"nodes": []},
        output_uri="/workspace/models/run-1/candidate",
    )

    with pytest.raises(
        RuntimeCredentialUnavailableError,
        match="owner's Pyromind access is unavailable",
    ):
        PlatformTrainWorkflowBackend().submit_workflow(
            spec=spec,
            request_context=None,
        )


def test_platform_train_backend_waits_for_workflow_path_credentials():
    spec = TrainWorkflowSpec(
        task_name="echomind-train-test",
        conversation_id="conversation-1",
        cluster="us-west-1#pre",
        workflow_path="/workspace/workflows/train.json",
        output_uri="/workspace/models/run-1/candidate",
    )

    with pytest.raises(
        RuntimeCredentialUnavailableError,
        match="workflow_path requires",
    ):
        PlatformTrainWorkflowBackend().submit_workflow(
            spec=spec,
            request_context=None,
        )


def _train_fixture(
    projects_root,
) -> tuple[ProjectInfo, RuntimeStageContext, TaskRun]:
    project_id = str(uuid4())
    graph_run_id = str(uuid4())
    stage_run_id = str(uuid4())
    node = ProjectNode(
        id="train",
        type="train",
        title="Train",
        executor="train_runtime",
        implementation_ref="builtin://echomind/train_runtime/v1",
        implementation_digest=canonical_sha256({"executor": "train_runtime"}),
        outputs={
            "trained_model": ProjectNodeOutput(
                artifact_type="model.checkpoint",
                schema_ref="echomind://schemas/model/v1",
            )
        },
        config={
            "workflow": {
                "name": "train",
                "nodes": [{"data": {"config": {"train_data_path": ""}}}],
            },
            "workflow_bindings": {
                "nodes[0].data.config.train_data_path": "train_messages"
            },
            "output_uri": "storage://models/{graph_run_id}/candidate",
            "artifact_uri": ("storage://models/{graph_run_id}/candidate/checkpoint-25"),
        },
    )
    project = ProjectInfo(
        project=ProjectRecord(
            project_id=project_id,
            name="Train runtime test",
            description="Train runtime test",
            skill="experiment-planning",
            owner_user_id="42",
            conversation_id="conversation-1",
            cluster="us-west-1#pre",
        ),
        graph=ProjectGraph(
            graph_id=str(uuid4()),
            project_id=project_id,
            nodes=[node],
            edges=[],
        ),
    )
    ProjectStore(projects_root).save(project)
    context = RuntimeStageContext(
        project_id=project_id,
        release_id="release-1",
        graph_run_id=graph_run_id,
        stage_run_id=stage_run_id,
        node=node,
        inputs={
            "train_messages": "storage://train.jsonl",
            "base_model": "storage://models/base",
        },
    )
    task_run = TaskRun(
        task_run_id=str(uuid4()),
        stage_run_id=stage_run_id,
        provider="pyromind.studio.training",
        idempotency_key="train-once",
        request_digest=canonical_sha256({"train": 1}),
    )
    return project, context, task_run


class _FakeBackend:
    def __init__(self, *, existing_task_id: str | None = None) -> None:
        self.existing_task_id = existing_task_id
        self.submitted_spec: TrainWorkflowSpec | None = None
        self.poll_count = 0

    def find_task_by_name(self, *, name, spec, request_context):
        del name, spec, request_context
        return self.existing_task_id

    def submit_workflow(self, *, spec, request_context):
        del request_context
        self.submitted_spec = spec
        return "task-1"

    def query_task(self, *, task_id, spec, request_context):
        del task_id, spec, request_context
        self.poll_count += 1
        if self.poll_count == 1:
            return TrainTaskState(status="running", message="Train is running.")
        return TrainTaskState(
            status="succeeded",
            message="Train completed.",
            output_uri="storage://models/candidate-1",
        )

    def cancel_task(self, *, task_id, spec, request_context):
        del task_id, spec, request_context


class _FakeClient:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True
