import json
from datetime import UTC, datetime
from typing import Literal
from unittest.mock import Mock
from uuid import uuid4

import httpx
import pytest
from fastapi import FastAPI

from pyromind_agent_server import (
    echomind_deployments as deployment_module,
    project_flywheel as flywheel_module,
)
from pyromind_agent_server.api.project_flywheel import router
from pyromind_agent_server.config import Config
from pyromind_agent_server.echomind_projects import (
    ProjectGraph,
    ProjectInfo,
    ProjectRecord,
)
from pyromind_agent_server.project_flywheel import ProjectFlywheelServices
from pyromind_runtime.application.conversation_runtime import ConversationRuntime
from pyromind_runtime.application.model_reports import ProjectModelReports
from pyromind_runtime.domain.context import RequestContext
from pyromind_runtime.domain.echomind_deployments import (
    DeploymentConflictError,
    DeploymentRecord,
    DeploymentRequest,
    DeploymentStatus,
    DeploymentTarget,
)
from pyromind_runtime.domain.echomind_model_reports import ModelEvaluationRecord
from pyromind_runtime.domain.project_storage import (
    ProjectStorageBinding,
    StorageProxyConfig,
)
from pyromind_runtime.ports.storage import EvaluationStorageMissingError

from .fake_adapter import FakeAdapter


@pytest.fixture
def flywheel(tmp_path, monkeypatch):
    monkeypatch.setenv("APP_ENV", "dev")
    service = ProjectFlywheelServices(
        projects_root=tmp_path / "projects",
        conversations_dir=tmp_path / "conversations",
        credential_store=None,
        cipher=None,
    )
    project_id, candidate_id = str(uuid4()), str(uuid4())
    project = ProjectInfo(
        project=ProjectRecord(
            project_id=project_id,
            name="Example",
            description="",
            skill="test",
            owner_user_id="42",
            conversation_id=str(uuid4()),
            cluster="cn-east-1",
        ),
        graph=ProjectGraph(project_id=project_id, graph_id=str(uuid4())),
    )
    service.projects.save(project)
    candidate = ModelEvaluationRecord(
        candidate_id=candidate_id,
        project_id=project_id,
        run_id=candidate_id,
        release_id=str(uuid4()),
        release_digest="sha256:" + "0" * 64,
        round_no=1,
        execution_mode="published",
        model_uri="/workspace/models/adapter",
        deployment_model_uri="/workspace/models/merged",
        report_uri="/workspace/evaluation/report.json",
        report_artifact_id=str(uuid4()),
        evaluation_dataset_uri="/workspace/fixed-test.jsonl",
        created_at=datetime.now(UTC),
    )
    records = [candidate]
    report_store = Mock()
    report_store.list_candidates.side_effect = lambda identifier: [
        item for item in records if item.project_id == identifier
    ]
    report_store.save_report.side_effect = lambda item: records.__setitem__(0, item)
    service.report_service = ProjectModelReports(report_store)
    report = {
        "status": "succeeded",
        "evaluation_status": "validated",
        "report_status": "ready",
        "validation": {},
        "badcase_count": 1,
        "metrics": {
            "model_reference": candidate.model_uri,
            "dataset_path": candidate.evaluation_dataset_uri,
            "pass_rate": 0.5,
        },
        "artifacts": {"report.html": "/evaluation/report.html"},
    }
    files = {
        "/evaluation/report.json": json.dumps(report).encode(),
        "/evaluation/run_manifest.json": b"{}",
        "/evaluation/report.html": b"<html><body>Evaluation report</body></html>",
        "/evaluation/evaluation_cases_manifest.json": json.dumps(
            {"chunks": [{"name": "evaluation_cases/chunk-00000.json"}]}
        ).encode(),
        "/evaluation/evaluation_cases_index.json": json.dumps(
            {
                "cases": [
                    {
                        "id": "case-1",
                        "chunk": 0,
                        "offset": 0,
                        "media": [{"media_id": "a.png", "mime_type": "image/png"}],
                    }
                ]
            }
        ).encode(),
        "/evaluation/evaluation_cases/chunk-00000.json": json.dumps(
            {"items": [{"id": "case-1", "status": "failed", "score": 0.1}]}
        ).encode(),
        "/evaluation/evaluation_media/a.png": b"png-image",
    }

    def read(path):
        if path not in files:
            raise EvaluationStorageMissingError(path)
        return files[path]

    storage = Mock()
    storage.read.side_effect = read
    build_storage = Mock(return_value=storage)
    monkeypatch.setattr(flywheel_module, "build_evaluation_storage", build_storage)
    app = FastAPI()
    app.state.config = Config(
        enable_session_api_key_auth=False, enable_pyromind_jwt_auth=False
    )
    runtime = ConversationRuntime(
        tmp_path / "conversations",
        {"fake": FakeAdapter("fake")},
        default_harness_id="fake",
        project_flywheel=service,
    )
    app.state.product_runtime = runtime
    app.include_router(router)
    return app, service, candidate, files, build_storage


@pytest.fixture
async def client(flywheel):
    app, _, _, _, _ = flywheel
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
        headers={"x-pyromind-debug-user-id": "42", "x-cluster": "cn-east-1"},
    ) as client:
        yield client
    await app.state.product_runtime.close()


async def test_local_lists_require_owner_and_do_not_contact_platform(flywheel, client):
    _, service, candidate, _, storage = flywheel
    base = f"/api/echomind/projects/{candidate.project_id}"
    reports = await client.get(base + "/model-reports")
    assert reports.status_code == 200
    assert reports.json()["candidates"][0]["candidate_id"] == candidate.candidate_id
    schedule = await client.get(base + "/batch-schedule")
    assert schedule.json()["config"]["enabled"] is False
    assert (await client.get(base + "/deployments")).json() == []
    assert not list(service.root.glob("*/batch_schedule.json"))
    storage.assert_not_called()
    assert (
        await client.get(
            base + "/model-reports", headers={"x-pyromind-debug-user-id": "43"}
        )
    ).status_code == 404
    assert (
        await client.get(base + "/deployments", headers={"x-cluster": "cn-west-2"})
    ).status_code == 409
    assert (
        await client.get("/api/echomind/projects/missing/model-reports")
    ).status_code == 404


@pytest.mark.parametrize("evaluation_status", ["validated", "invalid"])
async def test_refresh_binds_html_cases_and_media_to_selected_report(
    flywheel, client, evaluation_status
):
    _, service, candidate, files, storage = flywheel
    report = json.loads(files["/evaluation/report.json"])
    report["evaluation_status"] = evaluation_status
    files["/evaluation/report.json"] = json.dumps(report).encode()
    base = (
        f"/api/echomind/projects/{candidate.project_id}"
        f"/model-reports/{candidate.candidate_id}"
    )
    assert (await client.post(base + "/refresh")).status_code == 503
    headers = {"authorization": "Bearer ephemeral-browser-session"}
    refresh = await client.post(base + "/refresh", headers=headers)
    assert refresh.status_code == 200
    assert refresh.json()["metrics"]["pass_rate"] == 0.5
    if evaluation_status == "invalid":
        assert refresh.json()["report_status"] == "invalid"
        with pytest.raises(ValueError, match="validated published"):
            service._candidate(candidate.project_id, candidate.candidate_id)
    storage.reset_mock()
    document = await client.get(base + "/document")
    assert document.status_code == 200
    assert "Evaluation report" in document.json()["html"]
    storage.assert_not_called()
    cases = await client.get(base + "/cases", headers=headers)
    assert cases.status_code == 200
    assert cases.json()["items"] == [{"id": "case-1", "status": "failed", "score": 0.1}]
    media = await client.get(base + "/media/a.png", headers=headers)
    assert media.content == b"png-image"
    assert media.headers["content-type"] == "image/png"
    assert (
        await client.get(base + "/media/secret.png", headers=headers)
    ).status_code == 404
    assert (
        await client.get(base + "/cases?limit=101", headers=headers)
    ).status_code == 422


async def test_unverified_candidate_and_arbitrary_model_cannot_deploy(flywheel, client):
    _, service, candidate, _, _ = flywheel
    with pytest.raises(ValueError, match="validated published"):
        service._candidate(candidate.project_id, candidate.candidate_id)
    with pytest.raises(LookupError):
        service._candidate(str(uuid4()), candidate.candidate_id)
    response = await client.post(
        f"/api/echomind/projects/{candidate.project_id}/deployments",
        json={
            "request_id": "one",
            "candidate_id": candidate.candidate_id,
            "model_uri": "/arbitrary",
        },
    )
    assert response.status_code == 422


async def test_conflicts_are_409_and_do_not_expose_provider_errors(
    flywheel, client, monkeypatch
):
    _, service, candidate, _, _ = flywheel
    monkeypatch.setattr(
        service, "deploy", Mock(side_effect=DeploymentConflictError("secret-key"))
    )
    response = await client.post(
        f"/api/echomind/projects/{candidate.project_id}/deployments",
        json={"request_id": "one", "candidate_id": candidate.candidate_id},
    )
    assert response.status_code == 409
    assert "secret-key" not in response.text


def test_recovery_only_advances_authorized_work_and_pending_cleanup(
    flywheel, monkeypatch
):
    _, service, candidate, _, _ = flywheel
    baseline = DeploymentTarget(resource_id="original", model_uri="/original")
    states: list[
        tuple[DeploymentStatus, Literal["not_required", "pending", "released"]]
    ] = [
        ("succeeded", "not_required"),
        ("failed", "pending"),
        ("waiting_inference", "not_required"),
    ]
    for status, cleanup in states:
        service.deployment_store.save(
            DeploymentRecord(
                deployment_id=str(uuid4()),
                project_id=candidate.project_id,
                request=DeploymentRequest(
                    request_id=status, candidate_id=candidate.candidate_id
                ),
                model_uri="/merged",
                previous=baseline,
                status=status,
                cleanup_status=cleanup,
            )
        )
    advance = Mock()
    schedules = Mock()
    monkeypatch.setattr(service, "advance_deployment", advance)
    monkeypatch.setattr(service, "_schedules", schedules)
    service.poll()
    assert advance.call_count == 2
    assert all(
        call.args[2] == RequestContext(user_id="42", x_cluster="cn-east-1")
        for call in advance.call_args_list
    )
    schedules.assert_not_called()


async def test_report_html_size_and_path_are_enforced(flywheel, client):
    _, _, candidate, files, _ = flywheel
    base = (
        f"/api/echomind/projects/{candidate.project_id}"
        f"/model-reports/{candidate.candidate_id}"
    )
    headers = {"authorization": "Bearer ephemeral-browser-session"}
    files["/evaluation/report.html"] = b"x" * (8 * 1024 * 1024 + 1)
    assert (await client.post(base + "/refresh", headers=headers)).status_code == 422
    assert (await client.get(base + "/document")).status_code == 503
    report = json.loads(files["/evaluation/report.json"])
    report["artifacts"]["report.html"] = "/other-project/report.html"
    files["/evaluation/report.json"] = json.dumps(report).encode()
    response = await client.post(base + "/refresh", headers=headers)
    assert response.status_code == 200
    assert response.json()["report_html_uri"] is None


async def test_bound_storage_reads_reports_without_browser_credentials(
    flywheel, client, monkeypatch
):
    _, _, candidate, _, browser_storage = flywheel
    bound_storage = Mock(return_value=browser_storage.return_value)
    monkeypatch.setattr(
        flywheel_module, "build_bound_evaluation_storage", bound_storage
    )
    response = await client.post(
        f"/api/echomind/projects/{candidate.project_id}/model-reports/"
        f"{candidate.candidate_id}/refresh"
    )
    assert response.status_code == 200
    assert response.json()["report_status"] == "ready"
    browser_storage.assert_not_called()
    context = bound_storage.call_args.kwargs["context"]
    assert context.user_id == "42"
    assert context.x_cluster == "cn-east-1"
    assert context.cookie is None and context.authorization is None


async def test_storage_binding_is_explicit_and_owner_scoped(
    flywheel, client, monkeypatch
):
    _, _, candidate, _, _ = flywheel
    base = f"/api/echomind/projects/{candidate.project_id}/storage-binding"
    assert (await client.get(base)).json() is None
    binding = ProjectStorageBinding(
        project_id=candidate.project_id,
        owner_user_id="42",
        cluster="cn-east-1",
        config=StorageProxyConfig(sandbox_id="sb-existing"),
        status="ready",
        last_checked_at=datetime.now(UTC),
    )
    bindings = Mock()
    bindings.bind.return_value = binding
    bindings.refresh.return_value = binding
    factory = Mock(return_value=bindings)
    monkeypatch.setattr(flywheel_module, "build_project_storage_bindings", factory)
    response = await client.put(base, json={"sandbox_id": "sb-existing"})
    assert response.status_code == 200
    assert response.json()["config"]["sandbox_id"] == "sb-existing"
    bindings.bind.assert_called_once_with(
        candidate.project_id, "42", StorageProxyConfig(sandbox_id="sb-existing")
    )
    assert (await client.post(base + "/refresh")).status_code == 200
    factory.reset_mock()
    response = await client.put(
        base,
        json={"sandbox_id": "sb-existing"},
        headers={"x-pyromind-debug-user-id": "43"},
    )
    assert response.status_code == 404
    factory.assert_not_called()


def test_batch_release_requires_refreshed_validated_report(flywheel):
    _, service, candidate, _, storage = flywheel
    assert not service._report_archived(candidate.project_id, candidate.run_id, None)
    assert service._report_archived(
        candidate.project_id, candidate.run_id, storage.return_value
    )
    record = service.report_service.get(candidate.project_id, candidate.candidate_id)
    assert record.report_status == "ready"
    service.report_service.store.save_report(
        record.model_copy(update={"report_status": "invalid"})
    )
    assert not service._report_archived(
        candidate.project_id, candidate.run_id, storage.return_value
    )


def test_deployment_factory_refreshes_owner_access_and_closes_client(
    flywheel, monkeypatch
):
    _, service, candidate, _, _ = flywheel
    credentials = Mock()
    credentials.load.return_value = "owner-platform-key"
    client = Mock()
    refresh = Mock()
    monkeypatch.setattr(deployment_module, "refresh_runtime_access", refresh)
    monkeypatch.setattr(
        deployment_module, "get_pyromind_api_client", Mock(return_value=client)
    )
    context = RequestContext(user_id="42", x_cluster="cn-east-1")
    with pytest.raises(RuntimeError, match="interrupted"):
        with deployment_module.build_manual_deployment_service(
            projects_root=service.root,
            project_id=candidate.project_id,
            context=context,
            credential_store=credentials,
            resolve_candidate=service._candidate,
        ):
            raise RuntimeError("interrupted")
    refresh.assert_called_once()
    credentials.load.assert_called_once_with("42", "cn-east-1")
    client.close.assert_called_once()
