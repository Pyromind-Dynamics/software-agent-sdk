from __future__ import annotations

from unittest.mock import Mock
from uuid import uuid4

import httpx
import pytest
from pyromind_sdk.client.base import PyroMindAPIError
from pyromind_sdk.client.client import PyroMindAPIClient
from pyromind_sdk.client.models import EchoMindJobResponse, InferenceJobResponse

from pyromind_agent_server.echomind_deployments import (
    PlatformDeploymentProvider,
    ProjectDeploymentStore,
)
from pyromind_agent_server.echomind_graph_runs import GraphRunStore
from pyromind_agent_server.echomind_projects import (
    ProjectClusterMismatchError,
    ProjectGraph,
    ProjectInfo,
    ProjectRecord,
    ProjectStore,
)
from pyromind_agent_server.echomind_runtime_slots import (
    EchoMindTracerProvider,
    RuntimeHealth,
    RuntimeSlot,
    RuntimeSlotStore,
    ServingInferenceCreateConfig,
    ServingInferenceRuntime,
    TracerBufferConfig,
    TracerRuntime,
)
from pyromind_agent_server.echomind_storage import ProjectLease, ProjectLeaseTimeout
from pyromind_runtime.domain.context import RequestContext
from pyromind_runtime.domain.echomind_deployments import DeploymentTarget


@pytest.fixture
def platform_deployment(tmp_path):
    project_id = str(uuid4())
    ProjectStore(tmp_path).save(
        ProjectInfo(
            project=ProjectRecord(
                project_id=project_id,
                name="Manual deployment",
                description="test",
                skill="experiment-planning",
                conversation_id="conversation",
                owner_user_id="42",
                cluster="cn-east-1",
            ),
            graph=ProjectGraph(
                graph_id=str(uuid4()), project_id=project_id, nodes=[], edges=[]
            ),
        )
    )
    RuntimeSlotStore(tmp_path).save(
        RuntimeSlot(
            project_id=project_id,
            status="ready",
            serving_inference=ServingInferenceRuntime(
                resource_id="inf-old",
                model_uri="/workspace/old",
                endpoint="https://old.test",
                secret_ref="secret://old/key",
                health=RuntimeHealth(status="healthy"),
            ),
            serving_inference_create=ServingInferenceCreateConfig(
                model_path="/workspace/old",
                model_name="logical",
                gpu_card="NVIDIA-H20",
            ),
            tracer=TracerRuntime(
                resource_id="em-test",
                endpoint="https://tracer.test",
                admin_secret_ref="secret://admin/key",
                proxy_secret_ref="secret://proxy/key",
                target_inference_resource_id="inf-old",
                health=RuntimeHealth(status="healthy"),
                buffer_config=TracerBufferConfig(
                    batch_size=5,
                    trajectory_buffer_size=60,
                    time_per_round_seconds=1,
                    training_save_path="/workspace/tracer",
                ),
            ),
        )
    )
    client = Mock(spec=PyroMindAPIClient)
    client.echomind = Mock()
    client.inference = Mock()
    client.echomind.get_job.return_value = EchoMindJobResponse(
        job_id="em-test",
        name="my-tracer",
        status="Running",
        api_url="https://old.test",
        api_mode="openai",
        origin_model="logical",
        access_key="old-key",
        training_model="/workspace/training",
        training_batch_size="5",
        trajectory_buffer_size="60",
        time_per_round="1",
        training_round="1",
        training_save_path="/workspace/tracer",
        execution_mode="manual",
        custom_runtime_script_path="/workspace/trigger.py",
        secret_key="proxy-key",
    )
    client.inference.get.return_value = {
        "data": {
            "job_id": "inf-new",
            "name": "em-model-unique",
            "model_path": "/workspace/new",
            "model_name": "em-model-unique",
            "status": "Running",
            "endpoint_url": "https://new.test",
            "inf_sk": "new-key",
        }
    }
    client.inference.get_job.return_value = InferenceJobResponse.model_validate(
        client.inference.get.return_value["data"]
    )
    platform = EchoMindTracerProvider(client=client, cluster="cn-east-1")
    provider = PlatformDeploymentProvider(
        projects_root=tmp_path,
        project_id=project_id,
        context=RequestContext(user_id="42", x_cluster="cn-east-1"),
        platform=platform,
        resolve_candidate=Mock(),
    )
    return provider, client, tmp_path, project_id


def test_platform_switch_preserves_batch_configuration_and_uses_live_inference_key(
    platform_deployment,
):
    provider, client, _, _ = platform_deployment
    provider.switch(DeploymentTarget(resource_id="inf-new", model_uri="/workspace/new"))
    job_id, request = client.echomind.update.call_args.args
    assert job_id == "em-test"
    assert request.api_url == "https://new.test"
    assert request.access_key == "new-key"
    assert request.origin_model == "em-model-unique"
    assert request.training_batch_size == 5
    assert request.trajectory_buffer_size == 60
    assert request.custom_runtime_script_path == "/workspace/trigger.py"
    assert request.execution_mode == "manual"
    client.inference.delete.assert_not_called()


def test_route_verification_rejects_healthy_proxy_still_serving_old_model(
    platform_deployment,
    monkeypatch,
):
    provider, client, _, _ = platform_deployment
    client.echomind.get_job.return_value = (
        client.echomind.get_job.return_value.model_copy(
            update={"api_url": "https://new.test"}
        )
    )
    response = httpx.Response(
        200,
        json={"model": "logical"},
        request=httpx.Request("POST", "https://proxy.test"),
    )
    request = Mock(return_value=response)
    monkeypatch.setattr(httpx, "post", request)
    target = DeploymentTarget(
        resource_id="inf-new", model_uri="/workspace/new", endpoint="https://new.test"
    )
    assert provider.route_ready(target) is False
    assert request.call_args.kwargs["headers"]["X-Task-Trace"] == "false"
    request.return_value = httpx.Response(
        200, json={"model": "em-model-unique"}, request=response.request
    )
    assert provider.route_ready(target) is True


def test_commit_updates_serving_binding_without_deleting_old_inference(
    platform_deployment,
):
    provider, client, root, project_id = platform_deployment
    target = DeploymentTarget(
        resource_id="inf-new", model_uri="/workspace/new", endpoint="https://new.test"
    )
    provider.commit(target)
    slot = RuntimeSlotStore(root).load(project_id)
    assert slot.serving_inference is not None
    assert slot.serving_inference.resource_id == "inf-new"
    assert slot.tracer is not None
    assert slot.tracer.target_inference_resource_id == "inf-new"
    assert "new-key" not in slot.model_dump_json()
    client.inference.delete.assert_not_called()


def test_platform_deployment_enforces_project_owner_and_cluster(platform_deployment):
    provider, _, _, project_id = platform_deployment
    provider.authorize(project_id)
    with pytest.raises(PermissionError):
        provider.authorize(str(uuid4()))
    provider.context = RequestContext(user_id="other", x_cluster="cn-east-1")
    with pytest.raises(FileNotFoundError):
        provider.authorize(project_id)
    provider.context = RequestContext(user_id="42", x_cluster="us-west-2")
    with pytest.raises(ProjectClusterMismatchError):
        provider.authorize(project_id)


def test_deployment_and_graph_start_share_project_lease(tmp_path):
    project_id = str(uuid4())
    store = ProjectDeploymentStore(tmp_path)
    with store.lease(project_id):
        with pytest.raises(ProjectLeaseTimeout):
            with ProjectLease(tmp_path, project_id, timeout_seconds=0).acquire():
                pytest.fail("Graph startup acquired an active deployment lease")


def test_deployment_checks_actual_run_when_slot_pointer_is_stale(platform_deployment):
    provider, client, root, project_id = platform_deployment
    GraphRunStore(root).create(
        project_id=project_id,
        release_id=str(uuid4()),
        release_digest="sha256:" + "0" * 64,
        request_id="batch",
        trigger="manual",
        inputs={},
    )
    assert RuntimeSlotStore(root).load(project_id).active_graph_run_id is None
    with pytest.raises(ValueError, match="no active training round"):
        provider.current()
    client.inference.get.assert_not_called()


@pytest.mark.parametrize("protected_by", ["previous", "slot", "name", "model", "route"])
def test_candidate_cleanup_protects_serving_and_unowned_resources(
    platform_deployment, protected_by
):
    provider, client, root, project_id = platform_deployment
    target = DeploymentTarget(resource_id="inf-new", model_uri="/workspace/new")
    previous = DeploymentTarget(resource_id="inf-old", model_uri="/workspace/old")
    if protected_by == "previous":
        previous = target
    elif protected_by == "slot":
        slot = RuntimeSlotStore(root).load(project_id)
        assert slot.serving_inference is not None
        RuntimeSlotStore(root).save(
            slot.model_copy(
                update={
                    "serving_inference": slot.serving_inference.model_copy(
                        update={"resource_id": target.resource_id}
                    )
                }
            )
        )
    elif protected_by in {"name", "model"}:
        client.inference.get_job.return_value = (
            client.inference.get_job.return_value.model_copy(
                update={"name" if protected_by == "name" else "model_path": "other"}
            )
        )
    else:
        client.echomind.get_job.return_value = (
            client.echomind.get_job.return_value.model_copy(
                update={"api_url": "https://new.test/v1"}
            )
        )
    assert not provider.release_candidate(
        target, name="em-model-unique", previous=previous
    )
    client.inference.delete.assert_not_called()


def test_candidate_cleanup_requires_verified_absence_and_recovers_lost_response(
    platform_deployment,
):
    provider, client, _, _ = platform_deployment
    target = DeploymentTarget(resource_id="inf-new", model_uri="/workspace/new")
    previous = DeploymentTarget(resource_id="inf-old", model_uri="/workspace/old")
    assert not provider.release_candidate(
        target, name="em-model-unique", previous=previous
    )
    client.inference.delete.assert_called_once_with("inf-new")
    client.inference.delete.side_effect = TimeoutError("response lost")
    with pytest.raises(TimeoutError):
        provider.release_candidate(target, name="em-model-unique", previous=previous)
    client.inference.get_job.side_effect = PyroMindAPIError("Missing", status_code=404)
    assert provider.release_candidate(target, name="em-model-unique", previous=previous)
    assert client.inference.delete.call_count == 2
