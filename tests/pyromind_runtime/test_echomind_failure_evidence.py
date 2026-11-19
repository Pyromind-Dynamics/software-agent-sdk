from types import SimpleNamespace
from unittest.mock import Mock
from uuid import uuid4

import pytest
from pyromind_sdk.client.base import PyroMindAPIError

from pyromind_agent_server import echomind_failure_evidence
from pyromind_agent_server.echomind_projects import (
    ProjectGraph,
    ProjectInfo,
    ProjectRecord,
)
from pyromind_runtime.domain.context import RequestContext


@pytest.fixture
def project():
    project_id = str(uuid4())
    return ProjectInfo(
        project=ProjectRecord(
            project_id=project_id,
            name="Diagnostics",
            description="Diagnostics",
            skill="experiment-planning",
            owner_user_id="42",
            conversation_id="conversation",
            cluster="us-west-1#pre",
        ),
        graph=ProjectGraph(
            graph_id=str(uuid4()), project_id=project_id, nodes=[], edges=[]
        ),
    )


def test_reads_bounded_node_logs_and_redacts_credentials(monkeypatch, project):
    client = Mock()
    client.studio.get_task.return_value = SimpleNamespace(
        status="Failed",
        error_message=None,
        nodes=[SimpleNamespace(node_id=43050, end_at="2026-11-12 10:19:51")],
    )
    client.studio.get.return_value = {
        "data": {
            "logs": [
                "Server started",
                "FileNotFoundError: No such file: evaluation_report.css",
                "cp: cannot create regular file 'evaluation/': Not a directory",
                "RuntimeError: request to https://storage.invalid/file?sig=signed-value",
                "ValueError: api_key='api-secret' password=secret-value",
                "ValueError: WANDB_API_KEY=env-secret",
                "Fatal: postgres://user:database-secret@storage.invalid/database",
                "Fatal: Bearer bearer-secret cookie-secret",
            ]
        }
    }
    create = Mock(return_value=client)
    monkeypatch.setattr(echomind_failure_evidence, "create_workflow_api_client", create)
    context = RequestContext(
        user_id="42",
        cookie="session=cookie-secret",
        authorization="Bearer bearer-secret",
    )

    evidence = echomind_failure_evidence.read_graph_failure_diagnostics(
        project, "10316", context
    )

    assert evidence is not None
    assert "Studio node 43050" in evidence
    assert "FileNotFoundError: No such file: evaluation_report.css" in evidence
    assert "cannot create regular file 'evaluation/': Not a directory" in evidence
    assert "Server started" not in evidence
    for secret in (
        "signed-value",
        "api-secret",
        "secret-value",
        "env-secret",
        "database-secret",
        "bearer-secret",
        "cookie-secret",
    ):
        assert secret not in evidence
    create.assert_called_once_with(
        env="pre",
        cluster="us-west-1",
        auth_token="bearer-secret",
        headers={"x-cluster": "us-west-1#pre"},
        timeout=4,
    )
    client.studio.get.assert_called_once_with(
        "/studio/tasks/10316/nodes/43050/logs/scroll",
        params={"expect_count": 200, "timestamp": "2026-11-12T10:18:51+00:00"},
    )
    client.close.assert_called_once()


def test_preserves_task_error_when_logs_are_unavailable(monkeypatch, project):
    client = Mock()
    client.studio.get_task.return_value = SimpleNamespace(
        status="Failed",
        error_message="Platform says credential=stored-key and " + ("x" * 5000),
        nodes=[SimpleNamespace(node_id=43050, end_at=None)],
    )
    client.studio.get.side_effect = PyroMindAPIError("Unavailable", status_code=404)
    build = Mock(return_value=client)
    monkeypatch.setattr(echomind_failure_evidence, "get_pyromind_api_client", build)
    credentials = Mock()
    credentials.load.return_value = "stored-key"

    evidence = echomind_failure_evidence.read_graph_failure_diagnostics(
        project, "10316", RequestContext(user_id="42"), credential_store=credentials
    )

    assert evidence is not None
    assert evidence.startswith("Platform says credential=[redacted]")
    assert "stored-key" not in evidence
    assert len(evidence) == 4096
    credentials.load.assert_called_once_with("42", "us-west-1#pre")
    client.close.assert_called_once()


@pytest.mark.parametrize(
    ("user_id", "route", "task_id"),
    [
        ("43", "us-west-1#pre", "10316"),
        ("42", "us-west-1", "10316"),
        ("42", "us-west-1#pre", "../10316"),
    ],
)
def test_rejects_out_of_scope_diagnostics(
    monkeypatch, project, user_id, route, task_id
):
    build = Mock()
    monkeypatch.setattr(echomind_failure_evidence, "create_workflow_api_client", build)
    context = RequestContext(
        user_id=user_id, x_cluster=route, authorization="Bearer bearer-secret"
    )

    assert (
        echomind_failure_evidence.read_graph_failure_diagnostics(
            project, task_id, context
        )
        is None
    )
    build.assert_not_called()
