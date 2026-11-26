from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import httpx
import pytest
from fastapi import FastAPI

from pyromind_agent_server.api import (
    demo_collection as demo,
    echomind_projects as projects_api,
)
from pyromind_agent_server.config import Config
from pyromind_agent_server.echomind_acquisition import (
    AcquisitionSimulationStore,
    AcquisitionSource,
)
from pyromind_agent_server.echomind_acquisition_runtime import AcquisitionPaused
from pyromind_agent_server.echomind_acquisition_tasks import (
    EchoMindAcquisitionTaskManager,
)
from pyromind_agent_server.echomind_projects import (
    ProjectGraph,
    ProjectInfo,
    ProjectRecord,
    ProjectStore,
)
from pyromind_agent_server.echomind_publication import PublicationStore
from pyromind_agent_server.echomind_runtime_slots import RuntimeSlotStore
from pyromind_agent_server.echomind_sandbox_storage import SandboxEvaluationStorage
from pyromind_agent_server.service_context import ProductServiceContext


@pytest.fixture
async def project_client(tmp_path, monkeypatch):
    monkeypatch.setenv("APP_ENV", "dev")
    conversations = tmp_path / "conversations"
    app = FastAPI()
    app.state.config = Config(
        conversations_path=conversations,
        enable_session_api_key_auth=False,
        enable_pyromind_jwt_auth=False,
    )
    app.state.conversation_service = ProductServiceContext(
        conversations_dir=conversations
    )
    app.include_router(demo.router)
    projects = ProjectStore.for_conversations_dir(conversations)
    pid = str(uuid4())
    project = ProjectInfo(
        project=ProjectRecord(
            project_id=pid,
            conversation_id=uuid4().hex,
            name="Demo",
            description="",
            skill="test",
            owner_user_id="42",
            cluster="cn-east-1",
            lifecycle="frozen",
            active_release_id="release",
        ),
        graph=ProjectGraph(project_id=pid, graph_id=str(uuid4()), lifecycle="frozen"),
    )
    projects.save(project)
    async with httpx.AsyncClient(
        base_url="http://test",
        transport=httpx.ASGITransport(app=app),
        headers={"x-pyromind-debug-user-id": "42", "x-cluster": "cn-east-1"},
    ) as client:
        yield client, project, AcquisitionSimulationStore(projects.root)


def seed(simulations, pid, *, manual):
    return simulations.create(
        project_id=pid,
        request_id="fixture",
        requested_samples=5,
        requested_rounds=1,
        source=AcquisitionSource(
            train_uri="/workspace/train.jsonl",
            val_uri="/workspace/val.jsonl",
            image_root="/workspace/images",
        ),
        splits=["train", "val"],
        auto_start_graph_run=True,
        manual_start=manual,
    )


def selection():
    return {
        "request_id": str(uuid4()),
        "offset": 17,
        "limit": 5,
        "image_directory": "/workspace/images/",
    }


async def test_collection_requires_an_explicit_manual_seed(project_client):
    client, project, simulations = project_client
    pid = project.project.project_id
    url = f"/api/echomind/projects/{pid}/demo-collection"
    seed(simulations, str(uuid4()), manual=True)
    assert (await client.post(url, json=selection())).status_code == 404
    seed(simulations, pid, manual=False)
    assert (await client.post(url, json=selection())).status_code == 404
    assert len(simulations.list(pid)) == 1


@pytest.mark.parametrize("dataset_labels", [True, False])
async def test_selected_page_starts_one_real_tracer_runner(
    project_client, monkeypatch, dataset_labels
):
    client, project, simulations = project_client
    pid = project.project.project_id
    original = seed(simulations, pid, manual=True)
    slot = Mock(
        status="ready", active_release_id="release", batch_execution_available=True
    )
    slot.serving_inference.resource_id = "model-1"
    slot.tracer.resource_id = "tracer-1"
    slot.tracer.endpoint = "https://tracer.test"
    slot.tracer.custom_runtime_script_path = None
    slot.tracer.buffer_config.batch_size = 5
    monkeypatch.setattr(RuntimeSlotStore, "load", Mock(return_value=slot))
    monkeypatch.setattr(
        PublicationStore,
        "load_active",
        Mock(
            return_value=Mock(
                publication_mode="production",
                release_id="release",
                nodes=[
                    Mock(
                        type="dataset",
                        config={
                            "batch_label_source": "dataset",
                            "batch_validation_source": "fixed",
                        }
                        if dataset_labels
                        else {},
                    )
                ],
            )
        ),
    )
    service = Mock()
    service.provider.serving_binding.return_value = (
        Mock(endpoint="https://model.test"),
        "test",
    )
    service.provider.secret_key.return_value = "test"
    for module in (demo, projects_api):
        monkeypatch.setattr(
            module, "_runtime_slot_service", AsyncMock(return_value=service)
        )
    storage = Mock(spec=SandboxEvaluationStorage)
    monkeypatch.setattr(
        demo, "build_bound_evaluation_storage", Mock(return_value=storage)
    )
    monkeypatch.setattr(
        projects_api, "build_acquisition_storage", Mock(return_value=storage)
    )
    monkeypatch.setattr(
        projects_api, "load_tracer_admin_key", Mock(return_value="test")
    )
    monkeypatch.setattr(projects_api, "_runtime_scheduler", Mock())
    monkeypatch.setattr(
        demo.HttpTracerAcquisitionTransport, "models", Mock(return_value=["model"])
    )
    status = Mock(side_effect=AcquisitionPaused("test tracer is paused"))
    monkeypatch.setattr(demo.HttpTracerAcquisitionTransport, "status", status)

    def prepare(*, project, raw_directory, request_id, **kwargs):
        return demo.demo_expected_source(project, request_id, raw_directory)

    prepared = Mock(side_effect=prepare)
    monkeypatch.setattr(demo, "prepare_demo_source", prepared)
    monkeypatch.setattr(
        EchoMindAcquisitionTaskManager,
        "start",
        lambda self, *, runner, simulation: runner.run(simulation),
    )
    body = selection()
    url = f"/api/echomind/projects/{pid}/demo-collection"
    response = await client.post(url, json=body)
    assert response.status_code == 202
    simulation_id = response.json()["simulation_id"]
    simulation = simulations.load(project_id=pid, simulation_id=simulation_id)
    assert simulation.status == "paused" and simulation.error == "test tracer is paused"
    assert simulation.simulation_id != original.simulation_id
    assert (
        simulations.load(project_id=pid, simulation_id=original.simulation_id).status
        == "pending"
    )
    prepared.assert_called_once()
    args = prepared.call_args.kwargs
    assert (args["offset"], args["limit"], args["raw_directory"]) == (
        17,
        5,
        "/workspace/images",
    )
    assert args["seed_source"] == original.source
    assert simulation.splits == (["train"] if dataset_labels else ["train", "val"])
    assert args["fixed_validation"] == dataset_labels
    assert (args["infer"] is None) == dataset_labels
    assert service.provider.serving_binding.call_count == (0 if dataset_labels else 1)
    again = await client.post(url, json=body)
    assert again.status_code == 202 and again.json()["simulation_id"] == simulation_id
    for changed in (
        {"offset": 18},
        {"limit": 4},
        {"image_directory": "/workspace/new"},
    ):
        assert (await client.post(url, json={**body, **changed})).status_code == 409
    assert len(simulations.list(pid)) == 2
    prepared.assert_called_once()
    status.assert_called_once_with(endpoint="https://tracer.test", access_key="test")
