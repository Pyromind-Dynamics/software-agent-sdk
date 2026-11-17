from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import Mock
from uuid import uuid4

import pytest
from pydantic import SecretStr
from pyromind_agent_server import echomind_runtime_access, echomind_train_backend
from pyromind_agent_server.echomind_runtime_access import (
    build_runtime_credential_store,
    refresh_runtime_access,
)
from pyromind_agent_server.echomind_train_backend import PlatformTrainWorkflowBackend
from pyromind_runtime.domain.context import RequestContext

from openhands.agent_server.config import Config
from openhands.agent_server.echomind_projects import (
    ProjectGraph,
    ProjectInfo,
    ProjectRecord,
)
from openhands.agent_server.echomind_runtime_contracts import (
    RuntimeCredentialUnavailableError,
)
from openhands.agent_server.echomind_train_runtime import TrainWorkflowSpec


def _project() -> ProjectInfo:
    project_id = str(uuid4())
    return ProjectInfo(
        project=ProjectRecord(
            project_id=project_id,
            name="Runtime access",
            description="Runtime access",
            skill="experiment-planning",
            owner_user_id="42",
            conversation_id="conversation",
            cluster="us-west-1#pre",
        ),
        graph=ProjectGraph(
            graph_id=str(uuid4()), project_id=project_id, nodes=[], edges=[]
        ),
    )


def test_background_uses_current_owner_key_after_refresh_and_restart(
    tmp_path, monkeypatch
):
    config = Config(secret_key=SecretStr("test-encryption-key"))
    projects_root = tmp_path / "projects"
    store = build_runtime_credential_store(projects_root, config)
    assert store is not None
    project = _project()
    context = RequestContext(
        user_id="42", cookie="auth_token=login-token", x_cluster="us-west-1#pre"
    )
    earlier = datetime.now(UTC) - timedelta(minutes=2)
    store.save(
        "42",
        project.project.cluster,
        "previous-key",
        platform_user_id="42",
        observed_at=earlier,
    )
    monkeypatch.setattr(
        echomind_runtime_access,
        "get_current_login_user_from_token",
        lambda token: SimpleNamespace(user_id=42),
    )
    exchange = Mock(return_value="current-key")
    monkeypatch.setattr(echomind_runtime_access, "get_api_key", exchange)

    refresh_runtime_access(store, project, context)
    refresh_runtime_access(store, project, context)
    exchange.assert_called_once()
    store.save(
        "42",
        project.project.cluster,
        "delayed-old-key",
        platform_user_id="42",
        observed_at=earlier,
    )

    client = Mock()
    client.studio.get_task.return_value = SimpleNamespace(
        status="Running", task_id="platform-task", error_message=None
    )
    build_client = Mock(return_value=client)
    monkeypatch.setattr(echomind_train_backend, "get_pyromind_api_client", build_client)
    monkeypatch.setenv("PYROMIND_RUNTIME_API_KEY", "unrelated-global-key")
    backend = PlatformTrainWorkflowBackend(
        credential_store=build_runtime_credential_store(projects_root, config)
    )
    spec = TrainWorkflowSpec(
        task_name="train",
        owner_user_id="42",
        conversation_id="conversation",
        cluster=project.project.cluster,
        workflow={},
    )
    assert (
        backend.query_task(
            task_id="platform-task", spec=spec, request_context=None
        ).status
        == "running"
    )
    build_client.assert_called_once_with(
        env="pre", cluster="us-west-1", api_key="current-key"
    )

    with pytest.raises(RuntimeCredentialUnavailableError, match="owner's Pyromind"):
        backend.query_task(
            task_id="platform-task",
            spec=spec.model_copy(update={"owner_user_id": "43"}),
            request_context=None,
        )


@pytest.mark.parametrize(
    ("login_user_id", "cluster"), [(43, "us-west-1#pre"), (42, "us-west-1")]
)
def test_refresh_requires_matching_authenticated_owner_and_cluster(
    tmp_path, monkeypatch, login_user_id, cluster
):
    config = Config(secret_key=SecretStr("test-encryption-key"))
    store = build_runtime_credential_store(tmp_path / "projects", config)
    assert store is not None
    project = _project()
    monkeypatch.setattr(
        echomind_runtime_access,
        "get_current_login_user_from_token",
        lambda token: SimpleNamespace(user_id=login_user_id),
    )
    exchange = Mock()
    monkeypatch.setattr(echomind_runtime_access, "get_api_key", exchange)

    refresh_runtime_access(
        store,
        project,
        RequestContext(
            user_id="42", cookie="auth_token=login-token", x_cluster=cluster
        ),
    )

    exchange.assert_not_called()
    assert store.load("42", project.project.cluster) is None
