from __future__ import annotations

import asyncio
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import harness_adapter.pi_adapter.adapter as pi_adapter_module
import httpx
from fastapi import FastAPI
from harness_adapter.pi_adapter import PiAdapter
from pydantic import SecretStr
from pyromind_agent_server.api import echomind_projects as echomind_project_api
from pyromind_agent_server.api.echomind_projects import router as echomind_router
from pyromind_agent_server.api.router import _resolve_cursor, _sse_stream
from pyromind_agent_server.app import create_app
from pyromind_agent_server.bootstrap import install_product_api
from pyromind_agent_server.echomind_resource_proposals import prepare_runtime_resources
from pyromind_agent_server.echomind_tracer_admin import load_tracer_admin_key
from pyromind_agent_server.profiles import (
    ConversationProfile,
)
from pyromind_runtime.application.conversation_runtime import ConversationRuntime
from pyromind_runtime.domain.context import RequestContext
from pyromind_runtime.domain.echomind_resources import RuntimeResourceProposalArguments
from pyromind_runtime.infrastructure.file_product_store import FileProductStore
from pyromind_runtime.ports.harness import SessionSpec
from pyromind_runtime.ports.storage import EvaluationStorageMissingError

from openhands.agent_server.config import Config
from openhands.agent_server.conversation_service import ConversationService
from openhands.agent_server.echomind_graph_runs import (
    GraphRunStore,
    RuntimeFailure,
    TaskRun,
)
from openhands.agent_server.echomind_projects import (
    ProjectGraph,
    ProjectInfo,
    ProjectNode,
    ProjectRecord,
    ProjectStore,
)
from openhands.agent_server.echomind_publication import (
    ProjectPublicationService,
    PromotionPolicy,
    PublicationStore,
    PublishedGraph,
    RuntimeRequirements,
)
from openhands.agent_server.echomind_runtime_slots import (
    RuntimeSlot,
    RuntimeSlotStore,
    TracerBufferConfig,
    TracerRuntime,
)
from openhands.agent_server.echomind_stage_runs import ProjectRunStore

from .fake_adapter import FakeAdapter


def _app(tmp_path, runtime: ConversationRuntime) -> FastAPI:
    app = FastAPI()
    app.state.config = Config(
        conversations_path=tmp_path / "conversations",
        workspace_path=tmp_path / "workspace",
        enable_session_api_key_auth=False,
        enable_pyromind_jwt_auth=False,
    )
    app.state.conversation_service = ConversationService(
        conversations_dir=tmp_path / "conversations"
    )
    app.state.product_runtime = runtime
    from pyromind_agent_server.api.router import create_product_router

    app.include_router(create_product_router())
    return app


async def test_composed_app_mounts_product_router(tmp_path) -> None:
    app = create_app(
        Config(
            conversations_path=tmp_path / "conversations",
            workspace_path=tmp_path / "workspace",
            enable_session_api_key_auth=False,
            enable_pyromind_jwt_auth=False,
        )
    )
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get(
            "/api/v2/pyromind/conversations",
            headers={"x-pyromind-debug-user-id": "42"},
        )

    # Lifespan is intentionally not entered, so the runtime is unavailable;
    # 503 proves the composed app matched the Product route instead of 404.
    assert response.status_code == 503


def test_composed_app_mounts_echomind_project_routes(tmp_path) -> None:
    app = create_app(
        Config(
            conversations_path=tmp_path / "conversations",
            workspace_path=tmp_path / "workspace",
            enable_session_api_key_auth=False,
            enable_pyromind_jwt_auth=False,
        )
    )
    paths = set(app.openapi()["paths"])

    assert "/api/v2/echomind/conversations" in paths
    assert "/api/echomind/projects/{project_id}" in paths
    assert "/api/echomind/projects" in paths
    assert "/api/echomind/projects/{project_id}/runtime-contract/draft" in paths
    assert "/api/echomind/projects/{project_id}/runtime-contract/apply" in paths
    assert "/api/echomind/projects/{project_id}/runtime-implementation/adopt" in paths
    assert "/api/echomind/projects/{project_id}/publish" in paths
    assert "/api/echomind/projects/{project_id}/reopen" in paths
    assert "/api/echomind/projects/{project_id}/publication/preflight" in paths
    assert "/api/echomind/projects/{project_id}/publication/compile" in paths
    assert "/api/echomind/projects/{project_id}/runtime-slot" in paths
    assert "/api/echomind/projects/{project_id}/runtime-slot/ensure" in paths
    assert "/api/echomind/projects/{project_id}/runtime-slot/health" in paths
    assert "/api/echomind/projects/{project_id}/runtime-slot/tracer-admin" in paths
    assert "/api/echomind/projects/{project_id}/runtime-slot/proposal" in paths
    assert "/api/echomind/projects/{project_id}/runtime-slot/proposal/apply" in paths
    assert "/api/echomind/projects/{project_id}/collection-simulations" in paths
    assert (
        "/api/echomind/projects/{project_id}/collection-simulations/"
        "{simulation_id}" in paths
    )
    assert "/api/echomind/projects/{project_id}/graph-runs" in paths
    assert "/api/echomind/projects/{project_id}/graph-runs/debug" in paths
    assert "/api/echomind/projects/{project_id}/graph-runs/{run_id}/advance" in paths
    assert "/api/echomind/projects/{project_id}/graph-runs/{run_id}/retry" in paths
    assert "/api/echomind/conversations" not in paths
    assert not any("project-request" in path for path in paths)


async def test_tracer_admin_is_verified_and_scoped_to_the_project(
    tmp_path, monkeypatch
):
    runtime = ConversationRuntime(tmp_path / "conversations", FakeAdapter())
    app = _app(tmp_path, runtime)
    app.state.config = app.state.config.model_copy(
        update={"secret_key": SecretStr("test")}
    )
    app.include_router(echomind_router)
    projects = ProjectStore.for_conversations_dir(tmp_path / "conversations")
    project_id = str(uuid4())
    projects.save(
        ProjectInfo(
            project=ProjectRecord(
                project_id=project_id,
                name="PCB AVI",
                description="Batch acceptance",
                skill="experiment-planning",
                owner_user_id="42",
                conversation_id=uuid4().hex,
                cluster="us-west-1#pre",
            ),
            graph=ProjectGraph(
                graph_id=str(uuid4()),
                project_id=project_id,
                nodes=[
                    ProjectNode(id="dataset", type="dataset", title="Dataset"),
                ],
            ),
        )
    )
    RuntimeSlotStore(projects.root).save(
        RuntimeSlot(
            project_id=project_id,
            tracer=TracerRuntime(
                resource_id="tracer-owned",
                buffer_config=TracerBufferConfig(
                    trajectory_buffer_size=30,
                    batch_size=10,
                    time_per_round_seconds=60,
                    training_save_path="/workspace/models",
                ),
            ),
        )
    )
    provider = Mock()
    provider.verify_batch_admin.side_effect = [False, True]
    monkeypatch.setattr(
        echomind_project_api,
        "_runtime_slot_service",
        AsyncMock(return_value=Mock(provider=provider)),
    )
    url = f"/api/echomind/projects/{project_id}/runtime-slot/tracer-admin"
    headers = {"x-pyromind-debug-user-id": "42", "x-cluster": "us-west-1#pre"}
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
        headers=headers,
    ) as client:
        invalid = await client.post(url, json={"admin_key": "bad-test-key"})
        forbidden = await client.post(
            url,
            headers={"x-pyromind-debug-user-id": "7"},
            json={"admin_key": "test-admin-key"},
        )
        saved = await client.post(url, json={"admin_key": "test-admin-key"})

    assert invalid.status_code == 409
    assert forbidden.status_code == 404
    assert saved.status_code == 200, saved.text
    assert saved.json()["batch_execution_available"] is True
    assert "test-admin-key" not in saved.text
    assert provider.verify_batch_admin.call_count == 2
    assert provider.verify_batch_admin.call_args.kwargs["resource_id"] == "tracer-owned"
    assert (
        "test-admin-key"
        not in (projects.root / project_id / "tracer_admin.json").read_text()
    )
    assert (
        load_tracer_admin_key(
            projects_root=projects.root,
            project_id=project_id,
            resource_id="tracer-owned",
            cipher=app.state.config.cipher,
        )
        == "test-admin-key"
    )
    assert (
        load_tracer_admin_key(
            projects_root=projects.root,
            project_id=project_id,
            resource_id="other-tracer",
            cipher=app.state.config.cipher,
        )
        is None
    )
    await runtime.close()


async def test_agent_resource_proposal_only_provisions_after_matching_confirmation(
    tmp_path, monkeypatch
):
    runtime = ConversationRuntime(tmp_path / "conversations", FakeAdapter())
    app = _app(tmp_path, runtime)
    app.include_router(echomind_router)
    project_id, graph_id, release_id = (str(uuid4()) for _ in range(3))
    projects = ProjectStore.for_conversations_dir(tmp_path / "conversations")
    node = ProjectNode(
        id="dataset", type="dataset", title="Dataset", executor="dataset_runtime"
    )
    project = projects.save(
        ProjectInfo(
            project=ProjectRecord(
                project_id=project_id,
                name="PCB AVI",
                description="Runtime resources",
                skill="experiment-planning",
                owner_user_id="42",
                conversation_id=uuid4().hex,
                cluster="us-west-1#pre",
                lifecycle="frozen",
                active_release_id=release_id,
            ),
            graph=ProjectGraph(
                graph_id=graph_id,
                project_id=project_id,
                nodes=[node],
                lifecycle="frozen",
            ),
        )
    )
    publications = PublicationStore(projects.root)
    release = publications.save(
        PublishedGraph.build(
            project_id=project_id,
            release_id=release_id,
            source_graph_id=graph_id,
            source_graph_digest="sha256:" + "a" * 64,
            runtime_requirements=RuntimeRequirements(),
            promotion_policy=PromotionPolicy(metric="accuracy"),
            publication_mode="production",
            nodes=[node],
            edges=[],
            published_by="42",
        )
    )
    publications.activate(release)
    proposal = prepare_runtime_resources(
        projects.root,
        project,
        release,
        RuntimeResourceProposalArguments(
            model_name="pcb-avi",
            model_path="/workspace/models/pcb-avi",
            gpu_card="L40S",
        ),
    )
    app.state.config = app.state.config.model_copy(
        update={"secret_key": SecretStr("test-cipher-key")}
    )
    slot = RuntimeSlot(
        project_id=project_id,
        active_release_id=release_id,
        tracer=TracerRuntime(
            resource_id="tracer-created",
            buffer_config=TracerBufferConfig(
                trajectory_buffer_size=60,
                batch_size=20,
                time_per_round_seconds=60,
                training_save_path="/workspace/models",
            ),
        ),
    )
    slot_store = RuntimeSlotStore(projects.root)
    provider = Mock()
    files: dict[str, bytes] = {}

    def read(path: str) -> bytes:
        if path not in files:
            raise EvaluationStorageMissingError(path)
        return files[path]

    async def save_provisioned_slot(*, project_id, release_id, request):
        assert project_id == project.project.project_id
        assert release_id == release.release_id
        script = request.tracer.custom_runtime_script_path
        assert script is not None and script.startswith(
            "/app/trigger/echomind-project-"
        )
        assert f"/EchoMind/trigger/{Path(script).name}" in files
        provider.tracer_script_path.return_value = script
        assert slot.tracer is not None
        return slot_store.save(
            slot.model_copy(
                update={
                    "tracer": slot.tracer.model_copy(
                        update={"custom_runtime_script_path": script}
                    ),
                }
            )
        )

    setup = AsyncMock(side_effect=save_provisioned_slot)
    service = Mock(store=slot_store, setup=setup, provider=provider)
    monkeypatch.setattr(
        echomind_project_api,
        "_runtime_slot_service",
        AsyncMock(return_value=service),
    )
    storage = Mock()
    storage.list.return_value = {"config.json", "model.safetensors"}
    storage.read.side_effect = read
    storage.write.side_effect = files.__setitem__
    build_storage = Mock(return_value=storage)
    monkeypatch.setattr(echomind_project_api, "build_evaluation_storage", build_storage)
    path = f"/api/echomind/projects/{project_id}/runtime-slot/proposal"
    headers = {"x-pyromind-debug-user-id": "42", "x-cluster": "us-west-1#pre"}
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test", headers=headers
    ) as client:
        preview = await client.get(path)
        assert preview.status_code == 200, preview.text
        assert preview.json()["digest"] == proposal.digest
        assert preview.json()["setup"]["serving_inference_create"]["gpu_count"] == 1
        wrong_owner = await client.post(
            f"{path}/apply",
            json={"expected_digest": proposal.digest},
            headers={"x-pyromind-debug-user-id": "7"},
        )
        stale = await client.post(
            f"{path}/apply", json={"expected_digest": "sha256:" + "b" * 64}
        )
        assert wrong_owner.status_code == 404
        assert stale.status_code == 409
        setup.assert_not_awaited()
        provider.batch_admin_key.assert_not_called()
        owner_cookie = "auth_token=test-owner-browser-token"
        confirmed = await client.post(
            f"{path}/apply",
            json={"expected_digest": proposal.digest},
            headers={"cookie": owner_cookie},
        )
        assert confirmed.status_code == 202, confirmed.text
        setup.assert_awaited_once()
        assert [invocation.args[0] for invocation in build_storage.call_args_list] == [
            RequestContext(
                user_id="42",
                profile_id="echomind",
                cookie=owner_cookie,
                x_cluster="us-west-1#pre",
            )
        ] * 2
        storage.list.assert_called_once_with("/models/pcb-avi")
        storage.write.assert_called_once()
        provider.tracer_script_path.assert_called_once_with("tracer-created")
        provider.batch_admin_key.assert_not_called()
        assert slot_store.load(project_id).batch_execution_available
    await runtime.close()


async def test_debug_graph_run_http_uses_inactive_snapshot(
    tmp_path, monkeypatch
) -> None:
    runtime = ConversationRuntime(tmp_path / "conversations", FakeAdapter())
    app = _app(tmp_path, runtime)
    app.include_router(echomind_router)
    project_store = ProjectStore.for_conversations_dir(tmp_path / "conversations")
    project_id = str(uuid4())
    graph_id = str(uuid4())
    release_id = str(uuid4())
    contract_digest = "sha256:" + "a" * 64
    node = ProjectNode(
        id="dataset", type="dataset", title="Dataset", executor="contract_probe"
    )
    project = ProjectInfo(
        project=ProjectRecord(
            project_id=project_id,
            name="Debug GraphRun",
            description="Original Project",
            skill="experiment-planning",
            owner_user_id="42",
            conversation_id=str(uuid4()),
            cluster="us-west-1#pre",
        ),
        graph=ProjectGraph(graph_id=graph_id, project_id=project_id, nodes=[node]),
    )
    project_store.save(project)
    snapshot = PublishedGraph.build(
        project_id=project_id,
        release_id=release_id,
        source_graph_id=graph_id,
        source_graph_digest=contract_digest,
        nodes=[node],
        edges=[],
        runtime_requirements=RuntimeRequirements(),
        promotion_policy=PromotionPolicy(metric="accuracy"),
        published_by="42",
    )
    expected_validation_project_id = str(uuid4())

    def prepare_debug(
        self,
        *,
        project: ProjectInfo,
        reviewed_by: str,
        expected_contract_sha256: str,
        validation_project_id: str,
    ) -> PublishedGraph:
        assert project.project.project_id == project_id
        assert reviewed_by == "42"
        assert expected_contract_sha256 == contract_digest
        assert validation_project_id == expected_validation_project_id
        return PublicationStore(project_store.root).save(snapshot)

    monkeypatch.setattr(ProjectPublicationService, "prepare_debug", prepare_debug)
    path = f"/api/echomind/projects/{project_id}/graph-runs"
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
        headers={"x-pyromind-debug-user-id": "42", "x-cluster": "us-west-1#pre"},
    ) as client:
        started = await client.post(
            f"{path}/debug",
            json={
                "request_id": "debug-http",
                "expected_contract_sha256": contract_digest,
                "validation_project_id": expected_validation_project_id,
            },
        )
        assert started.status_code == 202, started.text
        run = started.json()
        assert run["execution_mode"] == "debug"
        assert run["release_id"] == release_id
        run_path = f"{path}/{run['run_id']}"
        advanced = await client.post(f"{run_path}/advance")
        resumed = await client.post(f"{run_path}/resume")
        retried = await client.post(f"{run_path}/retry", json={})
        callback = await client.post(
            f"{run_path}/callback",
            json={"provider_run_id": "unknown-task", "status": "Succeeded"},
        )

    assert advanced.status_code == 200
    assert resumed.status_code == 200
    assert advanced.json() == resumed.json() == run
    assert retried.status_code == 409
    assert retried.json()["detail"] == "Only paused GraphRuns can be retried"
    assert callback.status_code == 404
    assert (
        callback.json()["detail"] == "Unknown provider task for callback: unknown-task"
    )
    assert project_store.load(project_id, owner_user_id="42") == project
    await runtime.close()


async def test_debug_repair_http_hands_off_failed_attempt_once(
    tmp_path, monkeypatch
) -> None:
    conversations = tmp_path / "conversations"
    conversation_id = uuid4().hex
    adapter = FakeAdapter()
    runtime = ConversationRuntime(conversations, adapter)
    context = RequestContext(user_id="42", profile_id="echomind")
    await runtime.create_conversation(
        SessionSpec(
            conversation_id=conversation_id,
            user_id=context.user_id,
            profile_id=context.profile_id,
            workspace_root=str(conversations),
        ),
        context,
    )
    (conversations / conversation_id / ".agent-profile").write_text("echomind")
    app = _app(tmp_path, runtime)
    app.include_router(echomind_router)
    project_store = ProjectStore.for_conversations_dir(conversations)
    project_id = str(uuid4())
    graph_id = str(uuid4())
    command = "python evaluate.py --report-assets-dir {output_uri}"
    node = ProjectNode(
        id="eval",
        type="evaluator",
        title="Eval",
        executor="evaluator_runtime",
        config={"evaluator_runtime": {"command_template": command}},
    )
    project = project_store.save(
        ProjectInfo(
            project=ProjectRecord(
                project_id=project_id,
                name="Fixture",
                description="",
                skill="experiment-planning",
                owner_user_id=context.user_id,
                conversation_id=conversation_id,
                cluster="us-west-1#pre",
            ),
            graph=ProjectGraph(graph_id=graph_id, project_id=project_id, nodes=[node]),
        )
    )
    digest = "sha256:" + "a" * 64
    snapshot = PublicationStore(project_store.root).save(
        PublishedGraph.build(
            project_id=project_id,
            release_id=str(uuid4()),
            source_graph_id=graph_id,
            source_graph_digest=digest,
            nodes=[node],
            edges=[],
            runtime_requirements=RuntimeRequirements(),
            promotion_policy=PromotionPolicy(metric="accuracy"),
            published_by=context.user_id,
        )
    )
    graph_runs = GraphRunStore(project_store.root)
    run = graph_runs.create(
        project_id=project_id,
        release_id=snapshot.release_id,
        release_digest=snapshot.content_sha256,
        request_id="debug-repair-http",
        trigger="manual",
        execution_mode="debug",
        inputs={},
    )
    stage_runs = ProjectRunStore(project_store.root)
    stage = stage_runs.start(
        project=project, node=node, graph_run_id=run.run_id, mode="published_run"
    )
    now = datetime.now(UTC)
    task = TaskRun(
        task_run_id=str(uuid4()),
        stage_run_id=stage.run_id,
        provider="studio",
        provider_run_id="10316",
        idempotency_key="fixture-task",
        request_digest=digest,
        status="failed",
        finished_at=now,
        error="CPU node failed",
    )
    stage_runs.save(
        stage.model_copy(
            update={
                "status": "failed",
                "error": task.error,
                "task_runs": [task],
                "finished_at": now,
            }
        )
    )
    graph_runs.save(
        run.model_copy(
            update={
                "status": "paused",
                "started_at": now,
                "current_stage_ids": [stage.run_id],
                "failure": RuntimeFailure(
                    category="external", message="CPU node failed"
                ),
            }
        )
    )
    observed_tasks: list[str] = []

    def diagnostics(project, task_id, context, *, credential_store):
        observed_tasks.append(task_id)
        return "FileNotFoundError: evaluation_report.css"

    monkeypatch.setattr(
        "pyromind_agent_server.api.echomind_projects.read_graph_failure_diagnostics",
        diagnostics,
    )
    path = f"/api/echomind/projects/{project_id}/graph-runs/{run.run_id}/repair"
    headers = {"x-pyromind-debug-user-id": "42", "x-cluster": "us-west-1#pre"}
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
            headers=headers,
        ) as client:
            denied = await client.post(path, headers={"x-pyromind-debug-user-id": "7"})
            wrong_cluster = await client.post(path, headers={"x-cluster": "us-west-2"})
            accepted = await client.post(path)
            repeated = await client.post(path)
        assert denied.status_code == 404
        assert wrong_cluster.status_code == 409
        assert accepted.status_code == repeated.status_code == 202, accepted.text
        assert accepted.json() == repeated.json()
        assert observed_tasks == ["10316"]
        handoff = graph_runs.get_repair_handoff(project_id, run.run_id)
        assert handoff is not None and handoff.delivered_at is not None
        [(recipient, notice, _)] = adapter.external_task_notifications
        assert recipient == conversation_id
        assert notice.trigger_turn
        assert "FileNotFoundError: evaluation_report.css" in notice.hidden_text
        assert command in notice.hidden_text
        proposal_path = f"public_data/graph-repairs/{handoff.handoff_id}.json"
        assert proposal_path in notice.hidden_text
        proposal_file = conversations / conversation_id / proposal_path
        proposal_file.parent.mkdir()
        proposal = {
            "release_digest": snapshot.content_sha256,
            "node_id": "eval",
            "reason": "Use the verified report assets.",
            "runtime_config": {"workflow_bindings": {"eval": {"input_paths": []}}},
        }
        proposal_file.write_text(json.dumps(proposal))
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
            headers=headers,
        ) as client:
            invalid = await client.get(f"{path}/proposal")
            assert invalid.status_code == 422
            field = "runtime_config.workflow_bindings.eval"
            assert field in invalid.json()["detail"]
            body = {"revise_invalid_proposal": True}
            correction = await client.post(path, json=body)
            duplicate = await client.post(path, json=body)
            assert correction.status_code == duplicate.status_code == 202
            assert correction.json() == duplicate.json()
            assert correction.json() != accepted.json()
            assert len(adapter.external_task_notifications) == 2
            _, correction_notice, _ = adapter.external_task_notifications[-1]
            assert field in correction_notice.hidden_text
            assert proposal_path in correction_notice.hidden_text
            proposal["runtime_config"] = {"command_template": "evaluate {output_dir}"}
            proposal_file.write_text(json.dumps(proposal))
            invalid_template = await client.get(f"{path}/proposal")
            assert invalid_template.status_code == 422
            assert "output_dir" in invalid_template.json()["detail"]
            proposal["runtime_config"] = {"command_template": command + "/assets"}
            proposal_file.write_text(json.dumps(proposal))
            ready = await client.get(f"{path}/proposal")
            assert ready.status_code == 200
            assert ready.json()["changed_fields"] == ["command_template"]
    finally:
        await runtime.close()


async def test_http_create_list_snapshot_and_command(tmp_path) -> None:
    conversations = tmp_path / "conversations"
    conversations.mkdir()
    adapter = FakeAdapter()
    runtime = ConversationRuntime(conversations, adapter)
    transport = httpx.ASGITransport(app=_app(tmp_path, runtime))
    headers = {"x-pyromind-debug-user-id": "42"}
    async with httpx.AsyncClient(
        transport=transport, base_url="http://test", headers=headers
    ) as client:
        created = await client.post(
            "/api/v2/pyromind/conversations",
            json={"llm": {"model": "test-model"}, "message": "hello"},
        )
        assert created.status_code == 201
        conversation_id = created.json()["conversation_id"]
        listed = await client.get("/api/v2/pyromind/conversations")
        snapshot = await client.get(
            f"/api/v2/pyromind/conversations/{conversation_id}/snapshot"
        )
        command = await client.post(
            f"/api/v2/pyromind/conversations/{conversation_id}/commands",
            json={
                "command_id": "command-1",
                "type": "user_message",
                "content": [{"type": "text", "text": "continue"}],
            },
        )
        retry = await client.post(
            f"/api/v2/pyromind/conversations/{conversation_id}/commands",
            json={
                "command_id": "command-1",
                "type": "user_message",
                "content": [{"type": "text", "text": "continue"}],
            },
        )

    assert [item["conversation_id"] for item in listed.json()] == [conversation_id]
    assert snapshot.json()["timeline"][0]["kind"] == "message"
    assert command.status_code == 202
    assert command.json() == retry.json()
    assert adapter.created_specs[0].workspace_root == str(
        conversations / conversation_id
    )
    await runtime.close()


async def test_product_profiles_isolate_conversations_with_the_same_profile_id(
    tmp_path,
) -> None:
    conversations = tmp_path / "conversations"
    conversations.mkdir()
    runtime = ConversationRuntime(conversations, FakeAdapter())
    first_profile = ConversationProfile(
        route_prefix="/api/first/conversations",
        profile_id="shared-profile",
        conversation_marker="first",
    )
    second_profile = ConversationProfile(
        route_prefix="/api/second/conversations",
        profile_id="shared-profile",
        conversation_marker="second",
    )
    app = _app(tmp_path, runtime)
    from pyromind_agent_server.api.router import create_product_router

    app.include_router(create_product_router(first_profile))
    app.include_router(create_product_router(second_profile))
    transport = httpx.ASGITransport(app=app)
    headers = {"x-pyromind-debug-user-id": "42"}
    async with httpx.AsyncClient(
        transport=transport, base_url="http://test", headers=headers
    ) as client:
        first = await client.post(
            "/api/first/conversations",
            json={"llm": {"model": "test-model"}},
        )
        second = await client.post(
            "/api/second/conversations",
            json={"llm": {"model": "test-model"}},
        )
        first_id = first.json()["conversation_id"]
        second_id = second.json()["conversation_id"]
        for conversation_id in (first_id, second_id):
            metadata_path = conversations / conversation_id / "product" / "meta.json"
            metadata = json.loads(metadata_path.read_text())
            metadata["profile_id"] = None
            metadata_path.write_text(json.dumps(metadata))

        assert [
            item["conversation_id"]
            for item in (await client.get("/api/first/conversations")).json()
        ] == [first_id]
        assert [
            item["conversation_id"]
            for item in (await client.get("/api/second/conversations")).json()
        ] == [second_id]
        assert (await client.get("/api/v2/pyromind/conversations")).json() == []
        assert (
            await client.get(f"/api/first/conversations/{second_id}/snapshot")
        ).status_code == 404
        assert (
            await client.get(f"/api/second/conversations/{first_id}/snapshot")
        ).status_code == 404

    await runtime.close()


async def test_http_workflow_execution_prepares_confirmation_without_submitting(
    tmp_path,
) -> None:
    class Submission:
        def __init__(self) -> None:
            self.submitted = False

        async def prepare(self, workflow_path, context):
            return {
                "workflow": {"name": "sft", "nodes": []},
                "fingerprint": "abcdef1234567890",
                "summary": {"name": "sft", "node_count": 0},
            }

        async def submit(self, workflow, conversation_id, context, *, correlation=None):
            self.submitted = True
            return "task-1"

    conversations = tmp_path / "conversations"
    conversations.mkdir()
    submission = Submission()
    runtime = ConversationRuntime(
        conversations, FakeAdapter(), workflow_submission=submission
    )
    context = RequestContext(user_id="42")
    await runtime.create_conversation(
        SessionSpec(
            conversation_id="workflow-api",
            user_id="42",
            workspace_root=str(conversations / "workflow-api"),
        ),
        context,
    )
    transport = httpx.ASGITransport(app=_app(tmp_path, runtime))
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/v2/pyromind/conversations/workflow-api/workflow-executions",
            headers={"x-pyromind-debug-user-id": "42"},
            json={
                "workflow_path": "/workspace/project/workflow.py",
                "request_id": "r",
            },
        )
    assert response.status_code == 200
    assert response.json()["status"] == "awaiting_confirmation"
    assert submission.submitted is False
    assert (
        len((await runtime.get_snapshot("workflow-api", context)).pending_permissions)
        == 1
    )
    await runtime.close()


async def test_create_conversation_reports_capacity_exceeded_as_429(tmp_path) -> None:
    conversations = tmp_path / "conversations"
    conversations.mkdir()
    adapter = FakeAdapter()
    runtime = ConversationRuntime(conversations, adapter, max_active_conversations=1)
    transport = httpx.ASGITransport(app=_app(tmp_path, runtime))
    headers = {"x-pyromind-debug-user-id": "42"}
    async with httpx.AsyncClient(
        transport=transport, base_url="http://test", headers=headers
    ) as client:
        created = await client.post(
            "/api/v2/pyromind/conversations", json={"llm": {"model": "test-model"}}
        )
        assert created.status_code == 201
        runtime.register_external_task(
            created.json()["conversation_id"],
            {
                "task_id": "task-1",
                "kind": "data_cleaning",
                "run_id": "run-1",
                "status": "running",
                "output_dir": "/outputs/run-1",
                "submitted_at": "2026-09-01T00:00:00+00:00",
                "updated_at": "2026-09-01T00:00:00+00:00",
                "resume_pending": False,
            },
        )

        rejected = await client.post(
            "/api/v2/pyromind/conversations", json={"llm": {"model": "test-model"}}
        )

    assert rejected.status_code == 429
    assert rejected.json()["detail"]["code"] == "capacity_exceeded"
    await runtime.close()


async def test_sse_uses_persisted_sequence_as_event_id(tmp_path) -> None:
    conversations = tmp_path / "conversations"
    conversations.mkdir()
    runtime = ConversationRuntime(conversations, FakeAdapter())
    context = RequestContext(user_id="42")
    from pyromind_runtime.ports.harness import SessionSpec

    snapshot = await runtime.create_conversation(
        SessionSpec(
            conversation_id="conversation-sse",
            user_id="42",
            workspace_root=str(conversations),
        ),
        context,
    )
    stream = _sse_stream(runtime, snapshot.conversation_id, 0, context)
    first = await asyncio.wait_for(anext(stream), timeout=1)
    assert first == ": connected\n\n"
    event_frame = await asyncio.wait_for(anext(stream), timeout=1)
    assert "id: 1\n" in event_frame
    assert "event: conversation.created\n" in event_frame
    assert '"seq":1' in event_frame
    await stream.aclose()
    await runtime.close()


def test_sse_cursor_prefers_after_seq_and_validates_last_event_id() -> None:
    assert _resolve_cursor(4, "2") == 4
    assert _resolve_cursor(None, "2") == 2


def test_install_product_api_does_not_modify_openhands_routes() -> None:
    app = FastAPI()
    original = app.router.lifespan_context
    installed = install_product_api(app)
    assert installed is app
    assert app.router.lifespan_context is not original


async def test_product_api_creates_pi_metadata_and_reports_missing_checkpoint(
    tmp_path, monkeypatch
) -> None:
    class FakeRunner:
        def __init__(self, **_kwargs) -> None:
            self.running = True

        async def start(self, _config) -> None:
            return None

        async def request(self, method, params):
            if method == "context.append":
                return {"checkpoint_entry_id": "workflow-checkpoint"}
            if method == "fork":
                target = Path(params["target_session_dir"]) / "branched.jsonl"
                target.write_text('{"type":"session"}\n', encoding="utf-8")
                return {"session_path": str(target)}
            raise AssertionError(f"unexpected runner request: {method}")

        async def close(self) -> None:
            self.running = False

    monkeypatch.setattr(pi_adapter_module, "PiRunnerProcess", FakeRunner)
    conversations = tmp_path / "conversations"
    conversations.mkdir()
    runtime = ConversationRuntime(
        conversations,
        {
            "openhands": FakeAdapter(),
            "pi": PiAdapter(conversations, terminal_backend="os-sandbox"),
        },
        default_harness_id="pi",
    )
    transport = httpx.ASGITransport(app=_app(tmp_path, runtime))
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
        headers={"x-pyromind-debug-user-id": "42"},
    ) as client:
        created = await client.post(
            "/api/v2/pyromind/conversations",
            json={
                "llm": {"model": "gpt-4o", "api_key": "request-secret"},
                "workflow_xyflow": {
                    "name": "Workflow",
                    "nodes": [
                        {"id": "n1", "data": {"nodeType": "CloneAndCacheDataset"}}
                    ],
                    "edges": [],
                },
            },
        )
        assert created.status_code == 201
        conversation_id = created.json()["conversation_id"]
        forked = await client.post(
            f"/api/v2/pyromind/conversations/{conversation_id}/forks",
            json={"eventId": "event-1"},
        )
        assert created.json()["current_workflow"] is None
        adapter = runtime.adapters["pi"]
        assert isinstance(adapter, PiAdapter)
        session = adapter._session(conversation_id)
        frame = {
            "protocolVersion": 2,
            "type": "pi.event",
            "sessionId": conversation_id,
            "runId": "edited-run",
            "eventId": "edit-final",
            "kind": "tool.completed",
            "payload": {
                "tool_call_id": "edit",
                "tool_name": "edit",
                "arguments": {"path": "public_data/workflow_canvas/workflow.py"},
                "content": [],
            },
        }
        await adapter._runner_event(session, frame)
        await adapter._runner_event(
            session,
            {
                **frame,
                "eventId": "finish-final",
                "kind": "run.finished",
                "payload": {
                    "outcome": {"status": "completed"},
                    "checkpoint_entry_id": "workflow-checkpoint",
                },
            },
        )
        async with asyncio.timeout(2):
            while (
                not FileProductStore(conversations / conversation_id)
                .load_snapshot()
                .current_workflow
            ):
                await asyncio.sleep(0.01)
        workflow_event = next(
            event
            for event in FileProductStore(conversations / conversation_id).replay()
            if event.type == "workflow.updated"
        )
        valid_fork = await client.post(
            f"/api/v2/pyromind/conversations/{conversation_id}/forks",
            json={"eventId": workflow_event.event_id, "title": "Pi fork"},
        )
    metadata = (conversations / conversation_id / "product" / "meta.json").read_text()
    assert '"harness_id":"pi"' in metadata
    assert "request-secret" not in metadata
    workflow = (
        conversations
        / conversation_id
        / "public_data"
        / "workflow_canvas"
        / "workflow.py"
    )
    workflow_text = workflow.read_text()
    assert workflow_text.startswith("# workflow: Workflow")
    assert 'CloneAndCacheDataset(id="n1")' in workflow_text
    canvas = workflow_event.payload["canvas"]
    assert isinstance(canvas, dict) and canvas["nodes"]
    assert forked.status_code == 404
    assert forked.json()["detail"]["code"] == "checkpoint_not_found"
    assert valid_fork.status_code == 201
    target_id = valid_fork.json()["conversation_id"]
    assert target_id != conversation_id
    assert (
        valid_fork.json()["current_workflow"]["version"]
        == workflow_event.payload["version"]
    )
    assert (conversations / target_id / "pi" / "session.jsonl").is_file()
    await runtime.close()


async def test_evidence_http_checks_owner_and_digest(tmp_path) -> None:
    conversations = tmp_path / "conversations"
    runtime = ConversationRuntime(conversations, FakeAdapter())
    context = RequestContext(user_id="42")
    await runtime.create_conversation(
        SessionSpec(
            conversation_id="evidence",
            user_id="42",
            workspace_root=str(conversations / "evidence"),
        ),
        context,
    )
    content = b'{"content":"immutable evidence"}'
    digest = hashlib.sha256(content).hexdigest()
    directory = conversations / "evidence" / ".observability"
    directory.mkdir()
    path = directory / f"{digest}.json"
    path.write_bytes(content)
    transport = httpx.ASGITransport(app=_app(tmp_path, runtime))
    url = f"/api/v2/pyromind/conversations/evidence/execution-evidence/{digest}"
    try:
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            owner = await client.get(url, headers={"x-pyromind-debug-user-id": "42"})
            stranger = await client.get(url, headers={"x-pyromind-debug-user-id": "99"})
            assert owner.status_code == 200
            assert owner.content == content
            assert stranger.status_code == 404
            path.write_bytes(b"changed")
            corrupt = await client.get(url, headers={"x-pyromind-debug-user-id": "42"})
            assert corrupt.status_code == 404
    finally:
        await runtime.close()
