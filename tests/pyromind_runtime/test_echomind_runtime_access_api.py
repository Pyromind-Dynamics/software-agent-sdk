from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
from uuid import uuid4

import httpx
import pytest
from fastapi import FastAPI
from pydantic import SecretStr

from pyromind_agent_server import (
    dependencies,
    echomind_runtime_access,
    echomind_train_backend,
)
from pyromind_agent_server.api.echomind_projects import router
from pyromind_agent_server.config import Config
from pyromind_agent_server.echomind_graph_runs import GraphRunStore
from pyromind_agent_server.echomind_projects import (
    ProjectGraph,
    ProjectInfo,
    ProjectRecord,
    ProjectStore,
)
from pyromind_agent_server.echomind_runtime_access import build_runtime_credential_store
from pyromind_agent_server.echomind_storage import canonical_sha256
from pyromind_agent_server.echomind_train_backend import PlatformTrainWorkflowBackend
from pyromind_agent_server.echomind_train_runtime import TrainWorkflowSpec
from pyromind_agent_server.service_context import ProductServiceContext
from pyromind_common.platform.auth import CurrentLoginUser


async def test_active_graph_run_detail_refreshes_owner_access_for_background_polling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("APP_ENV", "prod")
    owner_token = "test-owner-session"
    access_key = "test-platform-access"
    cluster = "us-west-1#pre"

    def verify_token(token: str | None) -> CurrentLoginUser | None:
        user_id = {owner_token: 42, "test-other-session": 43}.get(token or "")
        if user_id is None:
            return None
        return CurrentLoginUser(
            username="test-user", email="user@example.test", user_id=user_id
        )

    monkeypatch.setattr(dependencies, "get_current_login_user_from_token", verify_token)
    monkeypatch.setattr(
        echomind_runtime_access, "get_current_login_user_from_token", verify_token
    )
    exchange = Mock(return_value=access_key)
    monkeypatch.setattr(echomind_runtime_access, "get_api_key", exchange)

    conversations_dir = tmp_path / "conversations"
    config = Config(
        conversations_path=conversations_dir,
        workspace_path=tmp_path / "workspace",
        secret_key=SecretStr("test-runtime-encryption-key"),
        enable_session_api_key_auth=False,
        enable_pyromind_jwt_auth=True,
    )
    app = FastAPI()
    app.state.config = config
    app.state.conversation_service = ProductServiceContext(
        conversations_dir=conversations_dir
    )
    app.include_router(router)

    project_id, release_id = str(uuid4()), str(uuid4())
    projects = ProjectStore.for_conversations_dir(conversations_dir)
    project = projects.save(
        ProjectInfo(
            project=ProjectRecord(
                project_id=project_id,
                name="Runtime access",
                description="Runtime access",
                skill="experiment-planning",
                owner_user_id="42",
                conversation_id="conversation-1",
                cluster=cluster,
                lifecycle="frozen",
                active_release_id=release_id,
                frozen_at=datetime.now(UTC),
            ),
            graph=ProjectGraph(
                graph_id=str(uuid4()),
                project_id=project_id,
                lifecycle="frozen",
            ),
        )
    )
    runs = GraphRunStore(projects.root)
    run = runs.create(
        project_id=project_id,
        release_id=release_id,
        release_digest=canonical_sha256({"release_id": release_id}),
        request_id="test-request",
        trigger="manual",
        inputs={},
    )
    run = runs.save(
        run.model_copy(
            update={"status": "waiting_external", "started_at": run.created_at}
        )
    )

    detail_url = f"/api/echomind/projects/{project_id}/graph-runs/{run.run_id}"
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        other = await client.get(
            detail_url,
            headers={"cookie": "auth_token=test-other-session", "x-cluster": cluster},
        )
        assert other.status_code == 404
        cookie = f"auth_token={owner_token}"
        other_cluster = await client.get(
            detail_url, headers={"cookie": cookie, "x-cluster": "us-west-1"}
        )
        assert other_cluster.status_code == 200
        exchange.assert_not_called()

        detail = await client.get(
            detail_url, headers={"cookie": cookie, "x-cluster": cluster}
        )

    assert detail.status_code == 200
    assert detail.json()["run"]["status"] == "waiting_external"
    assert access_key not in detail.text
    exchange.assert_called_once()
    assert exchange.call_args.args[1:] == (
        owner_token,
        {"cookie": cookie, "x-cluster": cluster},
    )

    (credential_file,) = (tmp_path / ".echomind-runtime-credentials").glob("*.json")
    persisted = credential_file.read_text(encoding="utf-8")
    assert access_key not in persisted
    assert owner_token not in persisted

    restarted_config = Config(secret_key=config.secret_key)
    credential_store = build_runtime_credential_store(projects.root, restarted_config)
    assert credential_store is not None
    task_id = "existing-platform-task"
    platform = Mock()
    platform.studio.get_task.return_value = SimpleNamespace(
        task_id=task_id, status="running"
    )
    connect = Mock(return_value=platform)
    monkeypatch.setattr(echomind_train_backend, "get_pyromind_api_client", connect)

    status = PlatformTrainWorkflowBackend(credential_store=credential_store).query_task(
        task_id=task_id,
        spec=TrainWorkflowSpec(
            task_name="train-1",
            conversation_id=project.project.conversation_id,
            cluster=cluster,
            owner_user_id=project.project.owner_user_id,
            workflow={},
        ),
        request_context=None,
    )

    assert status.status == "running"
    assert connect.call_args.kwargs["api_key"] == access_key
    platform.studio.get_task.assert_called_once_with(task_id)
    exchange.assert_called_once()
