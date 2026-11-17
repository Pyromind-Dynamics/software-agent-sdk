from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from uuid import uuid4

from pyromind_agent_server import echomind_train_backend as backend_module
from pyromind_agent_server.echomind_evaluator_backend import (
    PlatformEvaluatorWorkflowBackend,
)
from pyromind_runtime.domain.context import RequestContext

from openhands.agent_server.echomind_evaluator_runtime import (
    EvaluatorRuntimeExecutor,
    EvaluatorTaskState,
    EvaluatorWorkflowSpec,
)
from openhands.agent_server.echomind_graph_runs import TaskRun
from openhands.agent_server.echomind_projects import (
    ProjectGraph,
    ProjectInfo,
    ProjectNode,
    ProjectNodeOutput,
    ProjectRecord,
    ProjectStore,
)
from openhands.agent_server.echomind_runtime_contracts import RuntimeStageContext
from openhands.agent_server.echomind_storage import canonical_sha256


def test_evaluator_executor_submits_polls_and_emits_report(tmp_path):
    projects_root = tmp_path / "projects"
    project, context, task_run = _evaluator_fixture(projects_root)
    backend = _FakeEvaluatorBackend()
    executor = EvaluatorRuntimeExecutor(
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
    assert prepared.external_task.provider == "pyromind.studio.evaluation"

    submitted = executor.submit(context, task_run)
    assert submitted.status == "submitted"
    assert submitted.provider_run_id == "eval-task-1"
    assert backend.submitted_spec is not None
    assert backend.submitted_spec.bindings == {
        "nodes[0].data.config.model_path": "candidate_model",
        "nodes[0].data.config.dataset_path": "validation_dataset",
    }

    running = executor.poll(
        context,
        task_run.model_copy(update={"provider_run_id": "eval-task-1"}),
    )
    completed = executor.poll(
        context,
        task_run.model_copy(update={"provider_run_id": "eval-task-1"}),
    )

    assert running.status == "running"
    assert completed.status == "completed"
    assert completed.artifacts[0].port == "eval_report"
    assert completed.artifacts[0].payload.kind == "evaluation.report"
    assert completed.artifacts[0].payload.content["uri"] == (
        "storage://evaluations/candidate-1/report.json"
    )


def test_evaluator_executor_recovers_existing_task_without_resubmitting(tmp_path):
    projects_root = tmp_path / "projects"
    _, context, task_run = _evaluator_fixture(projects_root)
    backend = _FakeEvaluatorBackend(existing_task_id="existing-eval-task")
    executor = EvaluatorRuntimeExecutor(
        projects_root=str(projects_root),
        backend=backend,
    )

    submitted = executor.submit(context, task_run)

    assert submitted.provider_run_id == "existing-eval-task"
    assert backend.submitted_spec is None


def test_platform_evaluator_backend_uses_evaluation_task_kind(monkeypatch):
    submitted: dict[str, Any] = {}
    client = _FakeClient()

    monkeypatch.setattr(backend_module, "_client", lambda **kwargs: client)

    def submit_workflow_task(**kwargs):
        submitted.update(kwargs)
        return SimpleNamespace(task_id="platform-eval-task-1")

    monkeypatch.setattr(
        backend_module,
        "submit_workflow_task",
        submit_workflow_task,
    )
    spec = EvaluatorWorkflowSpec(
        task_name="echomind-eval-test",
        conversation_id="conversation-1",
        cluster="us-west-1#pre",
        workflow={
            "name": "old",
            "nodes": [
                {
                    "id": "evaluate",
                    "data": {
                        "config": {
                            "model_path": "",
                            "command": "old",
                        }
                    },
                }
            ],
        },
        bindings={"nodes[0].data.config.model_path": "candidate_model"},
        inputs={
            "candidate_model": "storage://models/checkpoint-1",
            "validation_dataset": "storage://datasets/val.jsonl",
        },
        command_template=(
            "python evaluate.py --model {candidate_model} "
            "--dataset {validation_dataset} --run {graph_run_id}"
        ),
        command_binding_path="nodes[0].data.config.command",
        template_values={"graph_run_id": "run-1"},
    )

    task_id = PlatformEvaluatorWorkflowBackend().submit_workflow(
        spec=spec,
        request_context=None,
    )

    assert task_id == "platform-eval-task-1"
    assert submitted["task_kind"] == "echomind_evaluator"
    assert submitted["retry_create"] is False
    assert submitted["workflow"]["nodes"][0]["data"]["config"]["model_path"] == (
        "storage://models/checkpoint-1"
    )
    assert submitted["workflow"]["nodes"][0]["data"]["config"]["command"] == (
        "python evaluate.py --model storage://models/checkpoint-1 "
        "--dataset storage://datasets/val.jsonl --run run-1"
    )
    assert client.closed is True


def _evaluator_fixture(
    projects_root,
) -> tuple[ProjectInfo, RuntimeStageContext, TaskRun]:
    project_id = str(uuid4())
    graph_run_id = str(uuid4())
    stage_run_id = str(uuid4())
    node = ProjectNode(
        id="evaluator",
        type="evaluator",
        title="Evaluator",
        executor="evaluator_runtime",
        implementation_ref="builtin://echomind/evaluator_runtime/v1",
        implementation_digest=canonical_sha256({"executor": "evaluator_runtime"}),
        outputs={
            "eval_report": ProjectNodeOutput(
                artifact_type="evaluation.report",
                schema_ref="echomind://schemas/evaluation-report/v1",
            )
        },
        config={
            "evaluator_runtime": {
                "workflow": {
                    "name": "evaluation",
                    "nodes": [
                        {
                            "data": {
                                "config": {
                                    "model_path": "",
                                    "dataset_path": "",
                                }
                            }
                        }
                    ],
                },
                "workflow_bindings": {
                    "nodes[0].data.config.model_path": "candidate_model",
                    "nodes[0].data.config.dataset_path": "validation_dataset",
                },
                "output_uri": ("storage://evaluations/{graph_run_id}/report.json"),
            }
        },
    )
    project = ProjectInfo(
        project=ProjectRecord(
            project_id=project_id,
            name="Evaluator runtime test",
            description="Evaluator runtime test",
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
            "candidate_model": "storage://models/checkpoint-1",
            "validation_dataset": "storage://datasets/val.jsonl",
        },
    )
    task_run = TaskRun(
        task_run_id=str(uuid4()),
        stage_run_id=stage_run_id,
        provider="pyromind.studio.evaluation",
        idempotency_key="evaluate-once",
        request_digest=canonical_sha256({"evaluate": 1}),
    )
    return project, context, task_run


class _FakeEvaluatorBackend:
    def __init__(self, *, existing_task_id: str | None = None) -> None:
        self.existing_task_id = existing_task_id
        self.submitted_spec: EvaluatorWorkflowSpec | None = None
        self.poll_count = 0

    def find_task_by_name(self, *, name, spec, request_context):
        del name, spec, request_context
        return self.existing_task_id

    def submit_workflow(self, *, spec, request_context):
        del request_context
        self.submitted_spec = spec
        return "eval-task-1"

    def query_task(self, *, task_id, spec, request_context):
        del task_id, spec, request_context
        self.poll_count += 1
        if self.poll_count == 1:
            return EvaluatorTaskState(
                status="running",
                message="Evaluation is running.",
            )
        return EvaluatorTaskState(
            status="succeeded",
            message="Evaluation completed.",
            output_uri="storage://evaluations/candidate-1/report.json",
        )

    def cancel_task(self, *, task_id, spec, request_context):
        del task_id, spec, request_context


class _FakeClient:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True
