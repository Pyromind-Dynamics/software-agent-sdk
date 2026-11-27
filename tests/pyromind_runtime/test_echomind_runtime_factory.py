from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import Mock
from uuid import uuid4

import pytest

from pyromind_agent_server import (
    echomind_dataset_backend,
    echomind_runtime_factory as runtime_factory,
    echomind_train_backend,
    echomind_workflow_storage,
)
from pyromind_agent_server.echomind_dataset_runtime import (
    DatasetRuntimeExecutor,
    DatasetWorkflowSpec,
)
from pyromind_agent_server.echomind_projects import (
    ProjectGraph,
    ProjectInfo,
    ProjectRecord,
    ProjectStore,
)
from pyromind_agent_server.echomind_runtime_credentials import RuntimeCredentialStore
from pyromind_agent_server.echomind_runtime_factory import (
    GraphRuntimeCallbackService,
)
from pyromind_agent_server.echomind_train_runtime import (
    TrainRuntimeExecutor,
    TrainWorkflowSpec,
)


@pytest.mark.parametrize("kind", ["dataset", "train", "evaluator"])
@pytest.mark.parametrize("use_binding", [True, False])
def test_background_runtime_verifies_each_stage_via_bound_or_recorded_storage(
    tmp_path, monkeypatch, kind, use_binding
):
    credentials = Mock(spec=RuntimeCredentialStore)
    credentials.load.return_value = "owner-key"
    client = Mock()
    client.studio.get_task.return_value = SimpleNamespace(
        task_id="42", status="Succeeded"
    )
    monkeypatch.setattr(echomind_train_backend, "_client", lambda **_: client)
    monkeypatch.setattr(echomind_dataset_backend, "_client", lambda **_: client)
    storage = Mock()
    storage.list.return_value = {
        "train_messages.jsonl",
        "val_messages.jsonl",
        "adapter_config.json",
        "adapter_model.safetensors",
    }
    storage.read.return_value = b'{"status":"succeeded","metrics":{"pass_rate":1}}'
    project_id = str(uuid4())
    recorded = Mock(return_value=storage)
    load_project = Mock(
        return_value=SimpleNamespace(
            project=SimpleNamespace(conversation_id="conversation", cluster="cn-east-1")
        )
    )
    monkeypatch.setattr(ProjectStore, "load", load_project)
    bound = Mock(return_value=storage if use_binding else None)
    monkeypatch.setattr(
        echomind_workflow_storage, "build_bound_evaluation_storage", bound
    )
    scheduler = runtime_factory.build_echomind_runtime_scheduler(
        projects_root=tmp_path, credential_store=credentials, recorded_storage=recorded
    )
    executor = scheduler.executor_registry.get(f"{kind}_runtime")
    base = TrainWorkflowSpec(
        task_name="stage-task",
        conversation_id="conversation",
        owner_user_id="owner",
        cluster="cn-east-1",
        workflow={},
        output_uri="/workspace/.pyromind-agent/conversation/echomind-runs/round/output",
        template_values={"project_id": project_id},
    )
    if isinstance(executor, DatasetRuntimeExecutor):
        assert executor.workflow_executor is not None
        backend = executor.workflow_executor.backend
        spec = DatasetWorkflowSpec(
            **base.model_dump(),
            train_filename="train_messages.jsonl",
            val_filename="val_messages.jsonl",
        )
    else:
        assert isinstance(executor, TrainRuntimeExecutor)
        backend = executor.backend
        spec = base.model_copy(
            update={
                "artifact_selection": "latest_lora_checkpoint"
                if kind == "train"
                else "evaluation_report"
            }
        )

    result = backend.query_task(task_id="42", spec=spec, request_context=None)

    assert result.status == "succeeded"
    assert bound.call_args.kwargs["projects_root"] == tmp_path
    assert bound.call_args.kwargs["credential_store"] is credentials
    assert bound.call_args.kwargs["project_id"] == base.template_values["project_id"]
    context = bound.call_args.kwargs["context"]
    assert context.user_id == "owner"
    assert context.x_cluster == "cn-east-1"
    assert context.cookie is None and context.authorization is None
    if use_binding:
        recorded.assert_not_called()
    else:
        recorded.assert_called_once()
        cid, actor_context = recorded.call_args.args
        assert cid == "conversation" and actor_context.profile_id == "echomind"
        assert (
            actor_context.user_id == "owner" and actor_context.x_cluster == "cn-east-1"
        )
        load_project.assert_called_once_with(project_id, owner_user_id="owner")
    client.close.assert_called_once()


@pytest.mark.parametrize(
    ("cid", "cluster"),
    [("another-conversation", "cn-east-1"), ("conversation", "us-west-1")],
)
def test_recorded_storage_rejects_an_unrelated_project(
    tmp_path, monkeypatch, cid, cluster
):
    project = SimpleNamespace(
        project=SimpleNamespace(conversation_id=cid, cluster=cluster)
    )
    monkeypatch.setattr(ProjectStore, "load", Mock(return_value=project))
    monkeypatch.setattr(
        echomind_workflow_storage,
        "build_bound_evaluation_storage",
        Mock(return_value=None),
    )
    reader = Mock()
    spec = TrainWorkflowSpec(
        task_name="task",
        conversation_id="conversation",
        cluster="cn-east-1",
        owner_user_id="owner",
        workflow={},
        template_values={"project_id": str(uuid4())},
        output_uri="/workspace/.pyromind-agent/conversation/echomind-runs/g/s/out",
    )
    with pytest.raises(PermissionError, match="Project context"):
        echomind_workflow_storage.build_runtime_storage(
            spec=spec,
            request_context=None,
            projects_root=tmp_path,
            recorded_storage=reader,
        )
    reader.assert_not_called()


async def test_graph_callback_service_returns_owning_conversation(
    tmp_path,
    monkeypatch,
) -> None:
    projects_root = tmp_path / "projects"
    project_id = str(uuid4())
    project = ProjectInfo(
        project=ProjectRecord(
            project_id=project_id,
            name="Runtime callback test",
            description="Runtime callback test",
            skill="experiment-planning",
            conversation_id="conversation-1",
            cluster="us-west-1#pre",
            lifecycle="frozen",
            active_release_id="release-1",
            frozen_at=datetime.now(UTC),
        ),
        graph=ProjectGraph(
            graph_id=str(uuid4()),
            project_id=project_id,
            lifecycle="frozen",
            nodes=[],
            edges=[],
        ),
    )
    ProjectStore(projects_root).save(project)

    class _Scheduler:
        def callback_by_provider_task(
            self,
            *,
            provider_run_id: str,
            status: str,
            conversation_id: str,
            task_kind: str | None,
        ):
            assert provider_run_id == "platform-task-1"
            assert status == "Succeeded"
            assert conversation_id == "conversation-1"
            assert task_kind == "echomind_train"
            return SimpleNamespace(project_id=project_id)

    monkeypatch.setattr(
        runtime_factory,
        "build_echomind_runtime_scheduler",
        lambda **kwargs: _Scheduler(),
    )

    conversation_id = await GraphRuntimeCallbackService(projects_root).handle(
        task_id="platform-task-1",
        status="Succeeded",
        conversation_id="conversation-1",
        task_kind="echomind_train",
    )

    assert conversation_id == "conversation-1"


async def test_graph_callback_service_falls_back_on_runtime_error(
    tmp_path,
    monkeypatch,
) -> None:
    class _Scheduler:
        def callback_by_provider_task(self, **kwargs):
            del kwargs
            raise RuntimeError("platform is temporarily unavailable")

    monkeypatch.setattr(
        runtime_factory,
        "build_echomind_runtime_scheduler",
        lambda **kwargs: _Scheduler(),
    )

    conversation_id = await GraphRuntimeCallbackService(tmp_path / "projects").handle(
        task_id="platform-task-1",
        status="Succeeded",
        conversation_id="conversation-1",
        task_kind="echomind_train",
    )

    assert conversation_id is None
