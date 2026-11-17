from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from uuid import uuid4

from pyromind_agent_server import echomind_runtime_factory as runtime_factory
from pyromind_agent_server.echomind_runtime_factory import (
    GraphRuntimeCallbackService,
)

from openhands.agent_server.echomind_projects import (
    ProjectGraph,
    ProjectInfo,
    ProjectRecord,
    ProjectStore,
)


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
