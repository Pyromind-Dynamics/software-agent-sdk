from __future__ import annotations

from datetime import timedelta
from unittest.mock import Mock

import pytest

from pyromind_runtime.application.echomind_deployments import ManualDeploymentService
from pyromind_runtime.domain.base import utc_now
from pyromind_runtime.domain.echomind_deployments import (
    DeploymentCandidate,
    DeploymentConflictError,
    DeploymentRequest,
    DeploymentTarget,
)
from pyromind_runtime.infrastructure.file_deployment_store import FileDeploymentStore
from pyromind_runtime.ports.echomind_deployments import DeploymentProvider


PROJECT = "6742edad-2fd1-419e-9e6c-979847463fe2"
OLD = DeploymentTarget(
    resource_id="inf-old",
    model_uri="/workspace/models/old",
    endpoint="https://old.test",
)
NEW = DeploymentTarget(
    resource_id="inf-new",
    model_uri="/workspace/models/new",
    endpoint="https://new.test",
)


@pytest.fixture
def deployment_service(tmp_path):
    provider = Mock(spec=DeploymentProvider)
    provider.current.return_value = OLD
    provider.candidate.return_value = DeploymentCandidate(
        candidate_id="candidate", model_uri=NEW.model_uri
    )
    provider.find.return_value = None
    provider.create.return_value = NEW
    provider.ready.side_effect = lambda target: target
    provider.routes_to.side_effect = lambda target: target == OLD
    provider.route_ready.return_value = True
    provider.release_candidate.return_value = True
    store = FileDeploymentStore(tmp_path)
    return ManualDeploymentService(store, provider), provider, store


def test_manual_deployment_waits_for_verified_route_before_committing(
    deployment_service,
):
    service, provider, store = deployment_service
    request = DeploymentRequest(request_id="one", candidate_id="candidate")
    record = service.start(PROJECT, request)
    assert record.status == "waiting_inference"
    assert record.previous == OLD
    provider.switch.assert_not_called()
    provider.commit.assert_not_called()
    assert service.start(PROJECT, request) == record
    provider.create.assert_called_once()
    record = service.advance(PROJECT, record.deployment_id)
    assert record.status == "switching"
    record = service.advance(PROJECT, record.deployment_id)
    assert record.status == "verifying"
    provider.switch.assert_called_once_with(NEW)
    provider.route_ready.return_value = False
    assert service.advance(PROJECT, record.deployment_id).status == "verifying"
    provider.commit.assert_not_called()
    provider.route_ready.return_value = True
    result = service.advance(PROJECT, record.deployment_id)
    assert result.status == "succeeded"
    provider.commit.assert_called_once_with(NEW)
    assert store.list(PROJECT)[0].previous == OLD
    assert service.advance(PROJECT, record.deployment_id) == result
    provider.commit.assert_called_once()


def test_lost_submission_is_reconciled_without_second_create(deployment_service):
    service, provider, store = deployment_service
    provider.create.side_effect = TimeoutError("secret must not enter record")
    record = service.start(
        PROJECT, DeploymentRequest(request_id="one", candidate_id="candidate")
    )
    assert record.status == "submission_unknown"
    assert "secret" not in record.model_dump_json()
    restored = ManualDeploymentService(store, provider)
    assert (
        restored.advance(PROJECT, record.deployment_id).status == "submission_unknown"
    )
    provider.find.return_value = NEW
    assert restored.advance(PROJECT, record.deployment_id).status == "waiting_inference"
    provider.create.assert_called_once()


def test_partial_switch_failure_restores_and_verifies_previous_route(
    deployment_service,
):
    service, provider, _ = deployment_service
    record = service.start(
        PROJECT, DeploymentRequest(request_id="one", candidate_id="candidate")
    )
    service.advance(PROJECT, record.deployment_id)
    provider.switch.side_effect = TimeoutError("lost response")
    assert service.advance(PROJECT, record.deployment_id).status == "rolling_back"
    provider.switch.side_effect = None
    provider.routes_to.side_effect = lambda target: target == NEW
    assert service.advance(PROJECT, record.deployment_id).status == "rollback_verifying"
    provider.switch.assert_called_with(OLD)
    provider.route_ready.return_value = False
    assert service.advance(PROJECT, record.deployment_id).status == "rollback_verifying"
    provider.commit.assert_not_called()
    provider.route_ready.return_value = True
    result = service.advance(PROJECT, record.deployment_id)
    assert result.status == "failed"
    provider.commit.assert_called_once_with(OLD)


@pytest.mark.parametrize("endpoint", [NEW.endpoint, "https://moved.test"])
def test_user_rollback_reuses_previous_inference(deployment_service, endpoint):
    service, provider, _ = deployment_service
    record = service.start(
        PROJECT, DeploymentRequest(request_id="one", candidate_id="candidate")
    )
    for _ in range(3):
        record = service.advance(PROJECT, record.deployment_id)
    assert record.status == "succeeded"
    current = NEW.model_copy(update={"endpoint": endpoint})
    provider.current.return_value = current
    rollback = service.start(
        PROJECT,
        DeploymentRequest(
            request_id="rollback",
            rollback_deployment_id=record.deployment_id,
        ),
    )
    assert rollback.target == OLD
    assert rollback.previous == current
    assert rollback.candidate_id is None
    provider.create.assert_called_once()
    provider.routes_to.side_effect = lambda target: target == current
    for _ in range(3):
        rollback = service.advance(PROJECT, rollback.deployment_id)
    assert rollback.status == "succeeded"
    provider.commit.assert_called_with(OLD)
    provider.current.return_value = OLD
    restore_candidate = service.start(
        PROJECT,
        DeploymentRequest(
            request_id="restore-candidate",
            rollback_deployment_id=rollback.deployment_id,
        ),
    )
    assert restore_candidate.target == current
    assert restore_candidate.candidate_id == "candidate"


def test_deployment_rejects_conflicting_or_parallel_requests(deployment_service):
    service, provider, _ = deployment_service
    service.start(
        PROJECT, DeploymentRequest(request_id="one", candidate_id="candidate")
    )
    with pytest.raises(DeploymentConflictError, match="Request ID"):
        service.start(
            PROJECT, DeploymentRequest(request_id="one", candidate_id="other")
        )
    with pytest.raises(DeploymentConflictError, match="in progress"):
        service.start(
            PROJECT, DeploymentRequest(request_id="two", candidate_id="candidate")
        )
    provider.create.assert_called_once()
    provider.authorize.side_effect = PermissionError("owner mismatch")
    with pytest.raises(PermissionError):
        service.list(PROJECT)


def test_restart_after_switch_does_not_switch_twice(deployment_service):
    service, provider, store = deployment_service
    record = service.start(
        PROJECT, DeploymentRequest(request_id="one", candidate_id="candidate")
    )
    service.advance(PROJECT, record.deployment_id)
    provider.routes_to.side_effect = lambda target: target == NEW
    restored = ManualDeploymentService(store, provider)
    assert restored.advance(PROJECT, record.deployment_id).status == "verifying"
    provider.switch.assert_not_called()
    assert restored.advance(PROJECT, record.deployment_id).status == "succeeded"


def test_candidate_timeout_does_not_change_current_serving(deployment_service):
    service, provider, store = deployment_service
    record = service.start(
        PROJECT, DeploymentRequest(request_id="one", candidate_id="candidate")
    )
    store.save(
        record.model_copy(update={"phase_started_at": utc_now() - timedelta(hours=1)})
    )
    result = service.advance(PROJECT, record.deployment_id)
    assert result.status == "failed"
    assert result.cleanup_status == "pending"
    provider.switch.assert_not_called()
    provider.commit.assert_not_called()
    provider.release_candidate.side_effect = TimeoutError("private-key")
    result = service.advance(PROJECT, record.deployment_id)
    assert result.status == "failed"
    assert result.cleanup_status == "pending"
    assert "private-key" not in result.model_dump_json()
    provider.release_candidate.side_effect = None
    restored = ManualDeploymentService(store, provider)
    result = restored.advance(PROJECT, record.deployment_id)
    assert result.cleanup_status == "released"
    provider.release_candidate.assert_called_with(
        NEW, name=f"em-model-{record.deployment_id}", previous=OLD
    )
    provider.release_candidate.reset_mock()
    restored.advance(PROJECT, record.deployment_id)
    provider.release_candidate.assert_not_called()


def test_deployment_lock_is_nonblocking_and_project_scoped(tmp_path):
    store = FileDeploymentStore(tmp_path)
    with store.lease(PROJECT):
        with pytest.raises(RuntimeError, match="busy"):
            with store.lease(PROJECT):
                pytest.fail("another writer acquired the Project")
