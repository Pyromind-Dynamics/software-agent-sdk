from __future__ import annotations

import hashlib
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import httpx
import pytest
from fastapi import FastAPI

from pyromind_agent_server.api import echomind_projects as project_api
from pyromind_agent_server.api.echomind_projects import router
from pyromind_agent_server.config import Config
from pyromind_agent_server.echomind_graph_runs import (
    ActiveGraphRunExistsError,
    GraphRunStore,
    RuntimeFailure,
    TaskRun,
)
from pyromind_agent_server.echomind_platform_tracer import (
    _HOOK,
    is_platform_tracer_script,
)
from pyromind_agent_server.echomind_projects import (
    ProjectGraph,
    ProjectInfo,
    ProjectNode,
    ProjectNodeInput,
    ProjectNodeInputBinding,
    ProjectNodeOutput,
    ProjectStore,
)
from pyromind_agent_server.echomind_publication import (
    RUNTIME_VALIDATION_GRAPH_TEMPLATE,
    ProjectLifecycleError,
    ProjectPublicationService,
    PublicationImmutableError,
    PublicationStore,
    PublicationValidationError,
    RuntimeGraphRepairProposal,
    RuntimeImplementationAdoptionError,
    adopt_runtime_implementation,
    apply_runtime_contract_draft,
    build_publication_preflight,
    build_runtime_contract_draft,
    compile_working_graph,
)
from pyromind_agent_server.echomind_resource_proposals import prepare_runtime_resources
from pyromind_agent_server.echomind_runtime_contracts import (
    ContractProbeRuntimeExecutor,
    ExternalTaskIntent,
    RuntimeArtifact,
    RuntimeStageContext,
    StageCancellationResult,
    StagePollResult,
    StagePreparationResult,
    StageSubmissionResult,
)
from pyromind_agent_server.echomind_runtime_scheduler import RuntimeGraphScheduler
from pyromind_agent_server.echomind_runtime_slots import (
    RuntimeHealth,
    RuntimeSlot,
    RuntimeSlotStore,
    ServingInferenceRuntime,
    TracerBufferConfig,
    TracerRuntime,
)
from pyromind_agent_server.echomind_stage_executors import StageArtifactPayload
from pyromind_agent_server.echomind_storage import project_directory
from pyromind_agent_server.service_context import ProductServiceContext
from pyromind_runtime.domain.echomind_resources import RuntimeResourceProposalArguments
from pyromind_runtime.domain.experiment_plans import PlanRevision, WorkflowPlan


def test_6742_shape_can_upgrade_publish_and_execute_without_mutating_source(
    tmp_path,
) -> None:
    store, source = _golden_shape_project(tmp_path)
    before = _file_snapshot(store.root)

    draft = build_runtime_contract_draft(source)

    assert draft.ready
    assert set(draft.nodes[0].outputs) == {"train_messages", "val_messages"}
    assert {item.code for item in draft.assumptions} >= {
        "executor_launch_derived",
        "promotion_policy_defaulted",
        "candidate_inference_policy_defaulted",
    }
    assert _file_snapshot(store.root) == before

    application = apply_runtime_contract_draft(
        store=store,
        project=source,
        draft=draft,
    )
    derived = application.project
    assert derived.project.project_id != source.project.project_id
    assert derived.project.lifecycle == "development"
    assert build_publication_preflight(derived).ready
    assert compile_working_graph(derived).content_sha256
    assert _file_snapshot(store.root) != before

    published = ProjectPublicationService(store.root).publish(
        project=derived,
        published_by="42",
    )
    frozen = store.load(
        derived.project.project_id,
        owner_user_id=derived.project.owner_user_id,
    )
    assert frozen.project.lifecycle == "frozen"
    assert frozen.project.active_release_id == published.release_id
    active = PublicationStore(store.root).load_active(frozen.project.project_id)
    assert active == published

    run = RuntimeGraphScheduler(projects_root=store.root).start(
        project=frozen,
        published=published,
        request_id="contract-upgrade-run-1",
        inputs={
            "data": "/workspace/datasets/hql_test/pcb_avi_recheck_0823_0828_test5",
            "base_model": "/workspace/models/Qwen/Qwen3-VL-2B-Instruct",
        },
    )

    assert run.status == "succeeded"
    assert run.outcome == "candidate_recorded"
    scheduler = RuntimeGraphScheduler(projects_root=store.root)
    stages = scheduler.stage_run_store.list_for_graph_run(
        project_id=frozen.project.project_id,
        graph_run_id=run.run_id,
    )
    assert [stage.node_id for stage in stages] == ["dataset", "train", "eval"]
    assert all(stage.status == "succeeded" for stage in stages)
    assert {ref.metadata["port"] for ref in stages[0].output_refs} == {
        "train_messages",
        "val_messages",
    }

    original = store.load(
        source.project.project_id,
        owner_user_id=source.project.owner_user_id,
    )
    assert original == source
    assert original.project.lifecycle == "development"


@pytest.mark.parametrize(
    ("held_out_path", "valid"),
    [
        ("/workspace/datasets/held-out/val.jsonl", True),
        ("/workspace/datasets/../val.jsonl", False),
    ],
)
def test_runtime_draft_preserves_fixed_eval_dataset(tmp_path, held_out_path, valid):
    _, source = _golden_shape_project(tmp_path)
    dataset, train, evaluator = source.graph.nodes
    validation = ProjectNodeInput(
        artifact_type="dataset.messages.validation",
        schema_ref="echomind://schemas/messages-dataset/v1",
        binding=ProjectNodeInputBinding(kind="literal", value=held_out_path),
    )
    evaluator = evaluator.model_copy(
        update={"inputs": {"validation_dataset": validation}}
    )
    source = store_project_graph(source, nodes=[dataset, train, evaluator])
    draft = build_runtime_contract_draft(source)
    assert draft.nodes[-1].inputs["validation_dataset"] == validation
    assert draft.nodes[1].inputs["validation_dataset"].binding.kind == "node_output"
    prepared = store_project_graph(source, nodes=draft.nodes)
    preflight = build_publication_preflight(prepared)
    assert preflight.ready == valid
    if not valid:
        assert {issue.code for issue in preflight.issues} == {
            "input_binding_source_invalid"
        }


async def test_runtime_validation_recipe_applies_to_copy_without_mutating_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("APP_ENV", "dev")
    store, source = _golden_shape_project(tmp_path)
    validation = store.save(
        _runtime_validation_project(source, train_binding_port="train_messages")
    )
    incompatible = store.save(
        _runtime_validation_project(source, train_binding_port="old_train_messages")
    )
    legacy_recipe = _runtime_validation_project(source)
    legacy_dataset, *legacy_nodes = legacy_recipe.graph.nodes
    manifest_only = store.save(
        store_project_graph(
            legacy_recipe,
            nodes=[
                legacy_dataset.model_copy(update={"config": {"resources": []}}),
                *legacy_nodes,
            ],
        )
    )
    original = _file_snapshot(store.root)
    source_id = source.project.project_id
    validation_id = validation.project.project_id

    legacy_preview = adopt_runtime_implementation(
        store=store, project=source, validation_project_id=validation_id
    )
    assert [node.executor for node in legacy_preview.project.graph.nodes] == [
        "dataset_runtime",
        "train_runtime",
        "evaluator_runtime",
    ]
    assert _file_snapshot(store.root) == original

    app = FastAPI()
    app.state.config = Config(
        conversations_path=tmp_path / "conversations",
        workspace_path=tmp_path / "workspace",
        enable_session_api_key_auth=False,
        enable_pyromind_jwt_auth=False,
    )
    app.state.conversation_service = ProductServiceContext(
        conversations_dir=tmp_path / "conversations"
    )
    app.include_router(router)
    base = f"/api/echomind/projects/{source_id}"
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
        headers={"x-pyromind-debug-user-id": "42", "x-cluster": source.project.cluster},
    ) as client:
        response = await client.get(f"{base}/runtime-contract/draft")
        unbound = response.raise_for_status().json()
        assert unbound["validation_project_id"] is None
        assert unbound["validation_candidates"] == [
            {"project_id": validation_id, "name": validation.project.name}
        ]
        incompatible_preview = await client.get(
            f"{base}/runtime-contract/draft",
            params={"validation_project_id": incompatible.project.project_id},
        )
        assert incompatible_preview.status_code == 409
        assert "bindings do not match" in incompatible_preview.json()["detail"]
        old_preview = await client.get(
            f"{base}/runtime-contract/draft",
            params={"validation_project_id": manifest_only.project.project_id},
        )
        assert old_preview.status_code == 200
        assert (
            old_preview.json()["validation_candidates"]
            == unbound["validation_candidates"]
        )
        assert (
            "full platform processing workflow"
            in old_preview.json()["issues"][0]["message"]
        )
        preview = await client.get(
            f"{base}/runtime-contract/draft",
            params={"validation_project_id": validation_id},
        )
        assert preview.status_code == 200
        draft = preview.json()
        assert draft["validation_project_id"] == validation_id
        assert not draft["issues"]
        assert (
            draft["nodes"][0]["config"]["tasks"]
            == source.graph.nodes[0].config["tasks"]
        )
        assert draft["nodes"][0]["config"]["runtime_input_defaults"] == {
            "data": "/workspace/raw"
        }
        assert _file_snapshot(store.root) == original

        rejected = await client.post(
            f"{base}/runtime-contract/apply",
            json={
                "expected_draft_sha256": unbound["content_sha256"],
                "validation_project_id": validation_id,
            },
        )
        assert rejected.status_code == 409
        body = {
            "expected_draft_sha256": draft["content_sha256"],
            "validation_project_id": validation_id,
        }
        recipe_dataset, *other_nodes = validation.graph.nodes
        changed_node = recipe_dataset.model_copy(
            update={
                "config": {
                    **recipe_dataset.config,
                    "runtime_input_defaults": {"data": "/workspace/changed"},
                }
            }
        )
        store.save(
            ProjectInfo(
                project=validation.project,
                graph=validation.graph.model_copy(
                    update={"nodes": [changed_node, *other_nodes]}
                ),
            )
        )
        stale = await client.post(f"{base}/runtime-contract/apply", json=body)
        assert stale.status_code == 409
        store.save(validation)

        applied = await client.post(f"{base}/runtime-contract/apply", json=body)
        assert applied.status_code == 200
        assert (
            await client.post(f"{base}/runtime-contract/apply", json=body)
        ).json() == (applied.json())
        derived = ProjectInfo.model_validate(applied.json()["project"])
        summaries = (await client.get("/api/echomind/projects")).json()
        summary = next(
            item
            for item in summaries
            if item["project_id"] == derived.project.project_id
        )
        assert summary["source_project_id"] == source_id
        assert summary["validation_project_id"] == validation_id
        assert summary["runtime_contract_sha256"] == draft["content_sha256"]

    assert derived.project.project_id != source_id
    assert store.load(source_id, owner_user_id=source.project.owner_user_id) == source
    assert all(
        _file_snapshot(store.root)[path] == value for path, value in original.items()
    )
    assert build_publication_preflight(derived, publication_mode="production").ready


def test_runtime_implementation_adoption_requires_validation_project(
    tmp_path,
) -> None:
    store, source = _golden_shape_project(tmp_path)

    with pytest.raises(
        RuntimeImplementationAdoptionError, match="No runtime validation recipe"
    ):
        adopt_runtime_implementation(store=store, project=source)


def test_reviewed_contract_publishes_original_project_without_debug(
    tmp_path: Path,
) -> None:
    store, source = _golden_shape_project(tmp_path)
    validation = store.save(_runtime_validation_project(source))
    draft = build_runtime_contract_draft(source, validation_project=validation)
    service = ProjectPublicationService(store.root)
    project_id = source.project.project_id
    project_ids = {entry.project_id for entry in store.list(owner_user_id="42")}
    assert not service.publication_store.list(project_id)

    unbound_draft = build_runtime_contract_draft(source)
    with pytest.raises(PublicationValidationError, match="review the updated"):
        service.publish(
            project=source,
            published_by="42",
            publication_mode="production",
            expected_contract_sha256=unbound_draft.content_sha256,
            validation_project_id=validation.project.project_id,
        )
    assert not service.publication_store.list(project_id)

    published = service.publish(
        project=source,
        published_by="42",
        publication_mode="production",
        expected_contract_sha256=draft.content_sha256,
        validation_project_id=validation.project.project_id,
    )
    assert published.project_id == project_id
    assert published.publication_mode == "production"
    assert {node.executor for node in published.nodes} == {
        "dataset_runtime",
        "train_runtime",
        "evaluator_runtime",
    }
    assert {entry.project_id for entry in store.list(owner_user_id="42")} == project_ids

    frozen = store.load(project_id, owner_user_id="42")
    assert frozen.project.active_release_id == published.release_id
    assert frozen.graph.nodes == source.graph.nodes
    before_rebind = _file_snapshot(store.root)
    rebound = store.bind_plan(
        PlanRevision(
            project_id=project_id,
            revision=2,
            plan=WorkflowPlan.model_validate(
                {
                    "skill": "experiment-planning",
                    "name": "A new working plan",
                    "goal": "A plan submitted after publication.",
                    "stages": [{"id": "dataset", "name": "Dataset"}],
                }
            ),
            confirmed_by="42",
        ),
        conversation_id=source.project.conversation_id,
        cluster=source.project.cluster,
        owner_user_id="42",
    )
    assert rebound == frozen
    assert _file_snapshot(store.root) == before_rebind

    republished = service.publish(
        project=frozen,
        published_by="42",
        publication_mode="production",
        expected_contract_sha256=draft.content_sha256,
        validation_project_id=validation.project.project_id,
    )
    assert republished == published


def test_graph_repair_changes_only_the_selected_stage(tmp_path: Path) -> None:
    store, source = _golden_shape_project(tmp_path)
    recipe = store.save(_runtime_validation_project(source))
    service = ProjectPublicationService(store.root)
    draft = build_runtime_contract_draft(source, validation_project=recipe)
    previous = service.prepare_debug(
        project=source,
        reviewed_by="42",
        expected_contract_sha256=draft.content_sha256,
        validation_project_id=recipe.project.project_id,
    )
    proposal = RuntimeGraphRepairProposal.model_validate(
        {
            "release_digest": previous.content_sha256,
            "node_id": "eval",
            "reason": "Read evaluator report assets from their verified directory.",
            "runtime_config": {
                "workflow_parameter_updates": {"1": {"source": "/report-assets"}}
            },
        }
    )
    with pytest.raises(ProjectLifecycleError, match="different snapshot"):
        service.prepare_repair_debug(
            project=source,
            release_id=previous.release_id,
            proposal=proposal.model_copy(
                update={"release_digest": "sha256:" + "0" * 64}
            ),
            reviewed_by="42",
        )
    repaired = service.prepare_repair_debug(
        project=source,
        release_id=previous.release_id,
        proposal=proposal,
        reviewed_by="42",
    )
    assert repaired.release_id != previous.release_id
    assert repaired.edges == previous.edges
    assert [node for node in repaired.nodes if node.id != "eval"] == [
        node for node in previous.nodes if node.id != "eval"
    ]
    eval_node = next(node for node in repaired.nodes if node.id == "eval")
    workflow = eval_node.config["evaluator_runtime"]["workflow"]
    assert workflow["nodes"][0]["data"]["config"]["source"] == "/report-assets"
    assert (
        service.publication_store.load(
            project_id=source.project.project_id, release_id=previous.release_id
        )
        == previous
    )
    assert store.load(source.project.project_id, owner_user_id="42") == source


def test_debug_draft_snapshots_do_not_activate_or_mutate_project(
    tmp_path: Path,
) -> None:
    store, source = _golden_shape_project(tmp_path)
    validation = store.save(_runtime_validation_project(source))
    service = ProjectPublicationService(store.root)
    project_id = source.project.project_id
    unbound_draft = build_runtime_contract_draft(source)
    with pytest.raises(PublicationValidationError, match="debug-only executor"):
        service.prepare_debug(
            project=source,
            reviewed_by="42",
            expected_contract_sha256=unbound_draft.content_sha256,
        )

    draft = build_runtime_contract_draft(source, validation_project=validation)
    selection = {
        "expected_contract_sha256": draft.content_sha256,
        "validation_project_id": validation.project.project_id,
    }
    debug = service.prepare_debug(project=source, reviewed_by="42", **selection)
    assert debug.project_id == project_id
    assert debug.publication_mode == "debug"
    legacy_draft = compile_working_graph(store_project_graph(source, nodes=draft.nodes))
    assert debug.release_id != legacy_draft.candidate_release_id
    assert store.load(project_id, owner_user_id="42") == source
    with pytest.raises(FileNotFoundError):
        service.publication_store.load_active(project_id)

    release = service.publish(
        project=source,
        published_by="42",
        publication_mode="production",
        expected_contract_sha256=draft.content_sha256,
        validation_project_id=validation.project.project_id,
    )
    assert release.release_id != debug.release_id
    assert release.content_sha256 != debug.content_sha256
    frozen = store.load(project_id, owner_user_id="42")
    assert service.prepare_debug(project=frozen, reviewed_by="42", **selection) == debug

    dataset, *other_nodes = validation.graph.nodes
    changed_dataset = dataset.model_copy(
        update={
            "config": {
                **dataset.config,
                "runtime_input_defaults": {"data": "/workspace/updated-data"},
            }
        }
    )
    updated_recipe = store.save(
        store_project_graph(validation, nodes=[changed_dataset, *other_nodes])
    )
    updated_draft = build_runtime_contract_draft(
        frozen, validation_project=updated_recipe
    )
    next_debug = service.prepare_debug(
        project=frozen,
        reviewed_by="42",
        expected_contract_sha256=updated_draft.content_sha256,
        validation_project_id=updated_recipe.project.project_id,
    )
    assert next_debug.release_id != debug.release_id
    historical_debug = service.publication_store.load(
        project_id=project_id, release_id=debug.release_id
    )
    assert historical_debug == debug
    assert service.publication_store.load_active(project_id) == release
    assert store.load(project_id, owner_user_id="42") == frozen


async def test_successful_debug_adoption_publishes_exact_snapshot_on_original_project(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("APP_ENV", "dev")
    store, target = _golden_shape_project(tmp_path)
    validation = store.save(_runtime_validation_project(target))
    service = ProjectPublicationService(store.root)
    validation_id = validation.project.project_id
    target_id = target.project.project_id
    target_recipe = build_runtime_contract_draft(target, validation_project=validation)
    runtime_project = store.save(
        store_project_graph(validation, nodes=target_recipe.nodes)
    )
    pre_repair = service.prepare_debug(
        project=runtime_project,
        reviewed_by="42",
        expected_contract_sha256=service.runtime_contract_draft(
            runtime_project
        ).content_sha256,
    )
    repaired = service.prepare_repair_debug(
        project=runtime_project,
        release_id=pre_repair.release_id,
        proposal=RuntimeGraphRepairProposal.model_validate(
            {
                "release_digest": pre_repair.content_sha256,
                "node_id": "eval",
                "reason": "Read verified evaluator assets.",
                "runtime_config": {
                    "workflow_parameter_updates": {
                        "1": {"source": "/workspace/report-assets"}
                    }
                },
            }
        ),
        reviewed_by="42",
    )
    run = service.graph_run_store.create(
        project_id=validation_id,
        release_id=repaired.release_id,
        release_digest=repaired.content_sha256,
        request_id="test-adoption",
        trigger="manual",
        execution_mode="debug",
        inputs={"data": "/workspace/debug-only-run-input"},
    )
    assert not service.list_debug_execution_candidates(target)
    timestamp = datetime.now(UTC)
    for node in repaired.nodes:
        stage = service.stage_run_store.start(
            project=runtime_project,
            node=node,
            graph_run_id=run.run_id,
            mode="published_run",
        )
        service.stage_run_store.save(
            stage.model_copy(
                update={
                    "status": "succeeded",
                    "finished_at": timestamp,
                    "reused_from_stage_run_id": (
                        str(uuid4()) if node.id in {"dataset", "train"} else None
                    ),
                }
            )
        )
    service.graph_run_store.save(
        run.model_copy(
            update={
                "status": "succeeded",
                "outcome": "baseline_retained",
                "finished_at": timestamp,
            }
        )
    )

    app = FastAPI()
    app.state.config = Config(
        conversations_path=tmp_path / "conversations",
        workspace_path=tmp_path / "workspace",
        enable_session_api_key_auth=False,
        enable_pyromind_jwt_auth=False,
    )
    app.state.conversation_service = ProductServiceContext(
        conversations_dir=tmp_path / "conversations"
    )
    app.include_router(router)
    base = f"/api/echomind/projects/{target_id}"
    selection = {"source_project_id": validation_id, "run_id": run.run_id}
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
        headers={"x-pyromind-debug-user-id": "42", "x-cluster": target.project.cluster},
    ) as client:
        candidates = (
            (await client.get(f"{base}/runtime-contract/debug-candidates"))
            .raise_for_status()
            .json()
        )
        assert [candidate["run_id"] for candidate in candidates] == [run.run_id]
        assert candidates[0]["reused_node_ids"] == ["dataset", "train"]
        preview = (
            (
                await client.get(
                    f"{base}/runtime-contract/debug-adoption", params=selection
                )
            )
            .raise_for_status()
            .json()
        )
        assert preview["source"] == candidates[0]
        assert any(change["node_id"] == "eval" for change in preview["changes"])
        assert store.load(target_id, owner_user_id="42") == target
        rejected = await client.post(
            f"{base}/runtime-contract/debug-adoption",
            json={**selection, "expected_preview_sha256": "sha256:" + "0" * 64},
        )
        assert rejected.status_code == 409
        adoption_body = {
            **selection,
            "expected_preview_sha256": preview["content_sha256"],
        }
        adopted = (
            (
                await client.post(
                    f"{base}/runtime-contract/debug-adoption", json=adoption_body
                )
            )
            .raise_for_status()
            .json()
        )
        assert adopted["adopted_debug_run"] == candidates[0]
        assert (
            await client.post(
                f"{base}/runtime-contract/debug-adoption", json=adoption_body
            )
        ).raise_for_status().json() == adopted
        assert store.load(target_id, owner_user_id="42") == target
        with pytest.raises(FileNotFoundError):
            service.publication_store.load_active(target_id)

        with pytest.raises(ProjectLifecycleError, match="Review the adopted"):
            service.publish(
                project=target, published_by="42", publication_mode="production"
            )
        changed = target.graph.nodes[0].model_copy(update={"title": "Changed stage"})
        changed_target = store.save(
            store_project_graph(target, nodes=[changed, *target.graph.nodes[1:]])
        )
        stale = service.runtime_contract_draft(changed_target)
        assert any(
            issue.code == "adopted_debug_execution_stale" for issue in stale.issues
        )
        assert (
            await client.post(
                f"{base}/runtime-contract/debug-adoption", json=adoption_body
            )
        ).status_code == 409
        store.save(target)

        shutil.rmtree(project_directory(store.root, validation_id))
        draft = (
            (await client.get(f"{base}/runtime-contract/draft"))
            .raise_for_status()
            .json()
        )
        assert draft["content_sha256"] == adopted["content_sha256"]
        resource_release_id = draft["resource_release_id"]
        assert resource_release_id
        assert resource_release_id != repaired.release_id
        publish_request = {
            "publication_mode": "production",
            "expected_contract_sha256": draft["content_sha256"],
        }
        response = await client.post(f"{base}/publish", json=publish_request)
        assert response.status_code == 409
        assert "Inference" in response.json()["detail"]
        assert (await client.get(f"{base}/publication")).status_code == 404

        candidate = service.resource_setup_release(target)
        assert candidate.release_id == resource_release_id
        assert candidate.nodes == repaired.nodes
        slot_store = RuntimeSlotStore(store.root)
        staged_slot = _ready_test_slot(target_id, resource_release_id)

        async def provision(*, project_id, release_id, request):
            assert project_id == target_id
            assert release_id == resource_release_id
            assert is_platform_tracer_script(
                project_id, request.tracer.custom_runtime_script_path
            )
            assert request.serving_inference_id == proposal.setup.serving_inference_id
            return slot_store.save(staged_slot)

        setup = AsyncMock(side_effect=provision)
        monkeypatch.setattr(
            project_api,
            "_runtime_slot_service",
            AsyncMock(return_value=Mock(store=slot_store, setup=setup)),
        )
        storage = Mock()
        storage.list.return_value = {"adapter_config.json", "adapter_model.safetensors"}
        storage.read.return_value = _HOOK
        build_storage = Mock(return_value=storage)
        monkeypatch.setattr(project_api, "build_evaluation_storage", build_storage)
        adapter_proposal = prepare_runtime_resources(
            store.root,
            target,
            candidate,
            RuntimeResourceProposalArguments(
                model_name="test-pcb-avi",
                model_path="/workspace/models/adapter",
                gpu_card="L40S",
            ),
        )
        response = await client.post(
            f"{base}/runtime-slot/proposal/apply",
            json={"expected_digest": adapter_proposal.digest},
        )
        assert response.status_code == 409
        assert "LoRA adapter" in response.json()["detail"]
        storage.list.assert_called_once_with("/models/adapter")

        storage.list.reset_mock()
        ensure = await client.post(
            f"{base}/runtime-slot/ensure",
            json=adapter_proposal.setup.model_dump(mode="json"),
        )
        assert ensure.status_code == 409
        assert "LoRA adapter" in ensure.json()["detail"]
        storage.list.assert_called_once_with("/models/adapter")
        setup.assert_not_awaited()
        assert slot_store.load_optional(target_id) is None

        proposal = prepare_runtime_resources(
            store.root,
            target,
            candidate,
            RuntimeResourceProposalArguments(
                model_name="test-pcb-avi",
                inference_id="test-serving-inference",
            ),
        )

        preview = await client.get(f"{base}/runtime-slot/proposal")
        assert preview.raise_for_status().json()["digest"] == proposal.digest
        assert slot_store.load_optional(target_id) is None
        response = await client.post(
            f"{base}/runtime-slot/proposal/apply",
            json={"expected_digest": "sha256:" + "0" * 64},
        )
        assert response.status_code == 409
        setup.assert_not_awaited()

        slot_store.save(_ready_test_slot(target_id, str(uuid4())))
        response = await client.post(f"{base}/publish", json=publish_request)
        assert response.status_code == 409
        assert store.load(target_id, owner_user_id="42") == target

        slot_store.save(staged_slot.model_copy(update={"status": "initializing"}))
        response = await client.post(f"{base}/publish", json=publish_request)
        assert response.status_code == 409
        assert (await client.get(f"{base}/publication")).status_code == 404

        response = await client.post(
            f"{base}/runtime-slot/proposal/apply",
            json={"expected_digest": proposal.digest},
        )
        assert response.status_code == 202
        assert build_storage.call_count == 3
        setup.assert_awaited_once()
        assert slot_store.load(target_id) == staged_slot
        assert store.load(target_id, owner_user_id="42") == target
        assert (await client.get(f"{base}/publication")).status_code == 404

        release = (
            (await client.post(f"{base}/publish", json=publish_request))
            .raise_for_status()
            .json()
        )
        assert release["adopted_debug_run"] == candidates[0]
        assert release["project_id"] == target_id
        assert release["release_id"] == resource_release_id
        assert release["nodes"] == repaired.model_dump(mode="json")["nodes"]
        assert release["edges"] == repaired.model_dump(mode="json")["edges"]
        assert "/workspace/debug-only-run-input" not in json.dumps(release)
        assert (
            service.publication_store.load_active(target_id).release_id
            == release["release_id"]
        )


def _ready_test_slot(project_id: str, release_id: str) -> RuntimeSlot:
    health = RuntimeHealth(status="healthy")
    inference_id = "test-serving-inference"
    return RuntimeSlot(
        project_id=project_id,
        status="ready",
        active_release_id=release_id,
        serving_inference=ServingInferenceRuntime(
            resource_id=inference_id,
            endpoint="https://example.test/serving",
            model_uri="/workspace/models/test-pcb-avi",
            secret_ref="secret://test/inference",
            health=health,
        ),
        tracer=TracerRuntime(
            resource_id="test-tracer",
            endpoint="https://example.test/tracer",
            admin_secret_ref="secret://test/admin",
            proxy_secret_ref="secret://test/proxy",
            target_inference_resource_id=inference_id,
            buffer_config=TracerBufferConfig(
                trajectory_buffer_size=60,
                batch_size=20,
                time_per_round_seconds=60,
                training_save_path="/workspace/models",
            ),
            health=health,
        ),
    )


def test_release_is_immutable_and_reopen_preserves_history(tmp_path) -> None:
    store, source = _golden_shape_project(tmp_path)
    draft = build_runtime_contract_draft(source)
    derived = apply_runtime_contract_draft(
        store=store,
        project=source,
        draft=draft,
    ).project
    service = ProjectPublicationService(store.root)
    published = service.publish(project=derived, published_by="42")
    publication_store = PublicationStore(store.root)

    with pytest.raises(PublicationImmutableError):
        publication_store.save(
            published.model_copy(update={"published_by": "someone-else"})
        )

    active_run = GraphRunStore(store.root).create(
        project_id=derived.project.project_id,
        release_id=published.release_id,
        release_digest=published.content_sha256,
        request_id="active-before-reopen",
        trigger="manual",
        inputs={},
    )
    frozen = store.load(
        derived.project.project_id,
        owner_user_id=derived.project.owner_user_id,
    )
    with pytest.raises(ActiveGraphRunExistsError):
        service.reopen(project=frozen)

    GraphRunStore(store.root).save(
        active_run.model_copy(
            update={
                "status": "failed",
                "outcome": "failed",
                "started_at": active_run.created_at,
                "finished_at": active_run.created_at,
                "failure": RuntimeFailure(
                    category="infrastructure",
                    message="test cleanup",
                ),
            }
        )
    )
    reopened = service.reopen(project=frozen)

    assert reopened.project.lifecycle == "development"
    assert reopened.project.active_release_id is None
    assert publication_store.load_active(derived.project.project_id) == published
    with pytest.raises(ProjectLifecycleError):
        service.reopen(project=reopened)

    with pytest.raises(PublicationValidationError, match="stale"):
        service.publish(
            project=reopened, published_by="42", expected_draft_sha256="stale"
        )

    manifest = (
        store.root
        / derived.project.project_id
        / "published_graphs"
        / published.release_id
        / "manifest.json"
    )
    manifest.unlink()
    draft = compile_working_graph(reopened)
    republished = service.publish(
        project=reopened,
        published_by="other",
        expected_draft_sha256=draft.content_sha256,
    )
    assert republished == published
    assert manifest.is_file()
    assert publication_store.list(derived.project.project_id) == [published]
    assert service.publish(project=reopened, published_by="42") == published


def test_golden_artifacts_replay_through_published_ports(tmp_path) -> None:
    store, source = _golden_shape_project(tmp_path)
    replay_source = _with_replay_artifacts(source)
    draft = build_runtime_contract_draft(replay_source)
    derived = apply_runtime_contract_draft(
        store=store,
        project=replay_source,
        draft=draft,
    ).project
    published = ProjectPublicationService(store.root).publish(
        project=derived,
        published_by="42",
    )
    frozen = store.load(
        derived.project.project_id,
        owner_user_id=derived.project.owner_user_id,
    )
    scheduler = RuntimeGraphScheduler(projects_root=store.root)

    run = scheduler.start(
        project=frozen,
        published=published,
        request_id="golden-artifact-replay",
        inputs={
            "data": "/workspace/datasets/hql_test/test5",
            "base_model": "/workspace/models/Qwen/Qwen3-VL-2B-Instruct",
        },
    )

    stages = scheduler.stage_run_store.list_for_graph_run(
        project_id=frozen.project.project_id,
        graph_run_id=run.run_id,
    )
    dataset, train, evaluator = stages
    assert run.status == "succeeded"
    assert {ref.metadata["source_uri"] for ref in dataset.output_refs} == {
        "/workspace/datasets/hql_test/test5/cleaned_train_normalized.jsonl",
        "/workspace/datasets/hql_test/test5/val_normalized.jsonl",
    }
    assert train.input_refs == [
        "/workspace/models/Qwen/Qwen3-VL-2B-Instruct",
        "/workspace/datasets/hql_test/test5/cleaned_train_normalized.jsonl",
        "/workspace/datasets/hql_test/test5/val_normalized.jsonl",
    ]
    assert train.output_refs[0].metadata["source_uri"] == (
        "/workspace/models/pcb_avi_sft_test5"
    )
    assert evaluator.input_refs == [
        "/workspace/models/pcb_avi_sft_test5",
        "/workspace/datasets/hql_test/test5/val_normalized.jsonl",
    ]
    assert evaluator.output_refs[0].metadata["source_uri"] == (
        "platform://tasks/10246/results"
    )


def test_workspace_6742_replays_in_temporary_directory(tmp_path) -> None:
    source_dir = (
        Path(__file__).resolve().parents[2]
        / "workspace/projects/6742edad-2fd1-419e-9e6c-979847463fe2"
    )
    project_path = source_dir / "project.json"
    graph_path = source_dir / "graph.json"
    if not project_path.is_file() or not graph_path.is_file():
        pytest.skip("workspace 6742 fixture is not available")
    before = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (project_path, graph_path)
    }
    source = ProjectInfo(
        project=json.loads(project_path.read_text(encoding="utf-8")),
        graph=json.loads(graph_path.read_text(encoding="utf-8")),
    )
    replay_source = _with_replay_artifacts(
        source,
        data_root=("/workspace/datasets/hql_test/pcb_avi_recheck_0823_0828_test5"),
        model_uri="/workspace/models/pcb_avi_sft_test5",
        eval_uri="platform://tasks/10246/results",
    )
    store = ProjectStore.for_conversations_dir(tmp_path / "conversations")
    copied = store.save(replay_source)
    draft = build_runtime_contract_draft(copied)
    derived = apply_runtime_contract_draft(
        store=store,
        project=copied,
        draft=draft,
    ).project
    published = ProjectPublicationService(store.root).publish(
        project=derived,
        published_by="42",
    )
    frozen = store.load(
        derived.project.project_id,
        owner_user_id=derived.project.owner_user_id,
    )
    scheduler = RuntimeGraphScheduler(projects_root=store.root)

    run = scheduler.start(
        project=frozen,
        published=published,
        request_id="workspace-6742-temp-replay",
        inputs={
            "data": ("/workspace/datasets/hql_test/pcb_avi_recheck_0823_0828_test5"),
            "base_model": "/workspace/models/Qwen/Qwen3-VL-2B-Instruct",
        },
    )

    stages = scheduler.stage_run_store.list_for_graph_run(
        project_id=frozen.project.project_id,
        graph_run_id=run.run_id,
    )
    assert run.status == "succeeded"
    assert [stage.node_id for stage in stages] == ["dataset", "train", "eval"]
    assert [stage.status for stage in stages] == [
        "succeeded",
        "succeeded",
        "succeeded",
    ]
    after = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (project_path, graph_path)
    }
    assert after == before


def test_waiting_external_run_recovers_without_resubmission(tmp_path) -> None:
    store, source = _golden_shape_project(tmp_path)
    draft = build_runtime_contract_draft(source)
    derived = apply_runtime_contract_draft(
        store=store,
        project=source,
        draft=draft,
    ).project
    published = ProjectPublicationService(store.root).publish(
        project=derived,
        published_by="42",
    )
    frozen = store.load(
        derived.project.project_id,
        owner_user_id=derived.project.owner_user_id,
    )
    slow_executor = _SlowTrainExecutor()
    scheduler = RuntimeGraphScheduler(
        projects_root=store.root,
        executors={"contract_probe": slow_executor},
    )

    waiting = scheduler.start(
        project=frozen,
        published=published,
        request_id="waiting-external",
        inputs={
            "data": "/workspace/datasets/hql_test/test5",
            "base_model": "/workspace/models/Qwen/Qwen3-VL-2B-Instruct",
        },
    )

    assert waiting.status == "waiting_external"
    assert len(slow_executor.submissions) == 1
    recovered_scheduler = RuntimeGraphScheduler(
        projects_root=store.root,
        executors={"contract_probe": slow_executor},
    )
    recovered = recovered_scheduler.advance(
        project=frozen,
        published=published,
        run_id=waiting.run_id,
    )
    assert recovered.status == "succeeded"
    assert len(slow_executor.submissions) == 1
    tasks = recovered_scheduler.graph_run_store.list_task_runs(
        project_id=frozen.project.project_id,
        graph_run_id=recovered.run_id,
    )
    assert [(task.provider, task.status) for task in tasks] == [
        ("fake-platform", "succeeded")
    ]


class _SlowTrainExecutor(ContractProbeRuntimeExecutor):
    def __init__(self) -> None:
        self.submissions: list[str] = []

    def prepare(self, context: RuntimeStageContext) -> StagePreparationResult:
        if context.node.type != "train":
            return super().prepare(context)
        return StagePreparationResult(
            status="waiting_external",
            message="Train submitted.",
            external_task=ExternalTaskIntent(provider="fake-platform"),
        )

    def submit(
        self,
        context: RuntimeStageContext,
        task_run: TaskRun,
    ) -> StageSubmissionResult:
        del context
        self.submissions.append(task_run.idempotency_key)
        return StageSubmissionResult(
            status="submitted",
            message="Train submitted.",
            provider_run_id="fake-train-task-1",
        )

    def poll(
        self,
        context: RuntimeStageContext,
        task_run: TaskRun,
    ) -> StagePollResult:
        del context, task_run
        return StagePollResult(
            status="completed",
            message="Train completed.",
            artifacts=[
                RuntimeArtifact(
                    port="trained_model",
                    payload=StageArtifactPayload(
                        kind="model.checkpoint",
                        content={"uri": "/workspace/models/candidate"},
                    ),
                )
            ],
        )

    def cancel(
        self,
        context: RuntimeStageContext,
        task_run: TaskRun,
    ) -> StageCancellationResult:
        del task_run
        return StageCancellationResult(
            status="canceled",
            message=f"{context.node.title} canceled.",
        )


def _with_replay_artifacts(
    project: ProjectInfo,
    *,
    data_root: str = "/workspace/datasets/hql_test/test5",
    model_uri: str = "/workspace/models/pcb_avi_sft_test5",
    eval_uri: str = "platform://tasks/10246/results",
) -> ProjectInfo:
    replay = {
        "dataset": {
            "train_messages": {"uri": f"{data_root}/cleaned_train_normalized.jsonl"},
            "val_messages": {"uri": f"{data_root}/val_normalized.jsonl"},
        },
        "train": {"trained_model": {"uri": model_uri}},
        "eval": {"eval_report": {"uri": eval_uri}},
    }
    nodes = [
        node.model_copy(
            update={
                "executor": "artifact_replay",
                "config": {
                    **node.config,
                    "replay_artifacts": replay[node.id],
                },
            }
        )
        for node in project.graph.nodes
    ]
    return store_project_graph(project, nodes=nodes)


def store_project_graph(
    project: ProjectInfo,
    *,
    nodes: list[ProjectNode],
) -> ProjectInfo:
    return ProjectInfo(
        project=project.project,
        graph=ProjectGraph(
            graph_id=project.graph.graph_id,
            project_id=project.project.project_id,
            template=project.graph.template,
            lifecycle=project.graph.lifecycle,
            nodes=nodes,
            edges=project.graph.edges,
        ),
    )


def _runtime_validation_project(
    source: ProjectInfo, *, train_binding_port: str | None = None
) -> ProjectInfo:
    executors = {
        "dataset": "dataset_runtime",
        "train": "train_runtime",
        "eval": "evaluator_runtime",
    }
    workflow = {
        "nodes": [
            {"id": "1", "data": {"config": {"source": "/input", "output_path": "/out"}}}
        ]
    }
    configs = {
        "dataset": {
            "workflow": workflow,
            "workflow_bindings": {"nodes[0].data.config.source": "source_batch"},
            "output_uri": "/workspace/dataset/{graph_run_id}",
            "output_node_id": "1",
        },
        "train": {"workflow": workflow},
        "eval": {"workflow": workflow},
    }
    nodes = [
        node.model_copy(
            update={
                "launch_mode": "executor",
                "executor": executors[node.id],
                "config": {
                    "tasks": [],
                    executors[node.id]: configs[node.id],
                    "runtime_input_defaults": {"data": "/workspace/raw"},
                },
            }
        )
        for node in source.graph.nodes
    ]
    if train_binding_port is not None:
        dataset, train, *rest = nodes
        output = ProjectNodeOutput(
            artifact_type="dataset.messages.train",
            schema_ref="echomind://schemas/messages-dataset/v1",
        )
        dataset = dataset.model_copy(update={"outputs": {train_binding_port: output}})
        training_data = ProjectNodeInput(
            artifact_type=output.artifact_type,
            schema_ref=output.schema_ref,
            binding=ProjectNodeInputBinding(
                kind="node_output", node_id=dataset.id, port=train_binding_port
            ),
        )
        train = train.model_copy(
            update={"inputs": {**train.inputs, "recipe_training_data": training_data}}
        )
        nodes = [dataset, train, *rest]
    project_id = str(uuid4())
    return ProjectInfo(
        project=source.project.model_copy(
            update={
                "project_id": project_id,
                "name": f"{source.project.name} [Runtime Validation]",
                "lifecycle": "development",
                "active_release_id": None,
                "frozen_at": None,
            }
        ),
        graph=ProjectGraph(
            graph_id=str(uuid4()),
            project_id=project_id,
            template=RUNTIME_VALIDATION_GRAPH_TEMPLATE,
            lifecycle="development",
            nodes=nodes,
            edges=source.graph.edges,
        ),
    )


def _golden_shape_project(tmp_path) -> tuple[ProjectStore, ProjectInfo]:
    plan = WorkflowPlan.model_validate(
        {
            "skill": "experiment-planning",
            "name": "PCB AVI runtime contract",
            "goal": "Run Dataset -> Train -> Evaluator.",
            "stages": [
                {"id": "dataset", "name": "Dataset"},
                {"id": "train", "name": "Train"},
                {"id": "eval", "name": "Evaluator"},
            ],
            "edges": [
                {"source": "dataset", "target": "train"},
                {"source": "train", "target": "eval"},
            ],
        }
    )
    store = ProjectStore.for_conversations_dir(tmp_path / "conversations")
    project = store.bind_plan(
        PlanRevision(
            project_id=uuid4().hex,
            revision=1,
            plan=plan,
            confirmed_by="42",
        ),
        conversation_id="conversation-1",
        cluster="us-west-1",
        owner_user_id="42",
    )
    resources = [
        {
            "id": "data",
            "name": "Dataset directory",
            "uri": "/workspace/datasets/hql_test/test5",
            "description": "",
        },
        {
            "id": "base_model",
            "name": "Base model",
            "uri": "/workspace/models/Qwen/Qwen3-VL-2B-Instruct",
            "description": "",
        },
    ]
    nodes = [
        ProjectNode(
            id="dataset",
            type="dataset",
            title="Dataset",
            launch_mode="agent",
            executor="contract_probe",
            config={
                "resources": resources,
                "tasks": [
                    {
                        "kind": "clean_label",
                        "id": "clean",
                        "stage_id": "dataset",
                        "inputs": [
                            {
                                "name": "data",
                                "resource_id": "data",
                                "task_id": None,
                                "output_name": None,
                            }
                        ],
                        "outputs": [
                            {"name": "train_messages"},
                            {"name": "val_messages"},
                        ],
                    }
                ],
                "capability": "data-processing",
            },
        ),
        ProjectNode(
            id="train",
            type="train",
            title="Train",
            executor="contract_probe",
            config={
                "resources": resources,
                "tasks": [
                    {
                        "kind": "train_run",
                        "id": "run_train",
                        "stage_id": "train",
                        "inputs": [
                            {
                                "name": "base_model",
                                "resource_id": "base_model",
                                "task_id": None,
                                "output_name": None,
                            },
                            {
                                "name": "train_messages",
                                "resource_id": None,
                                "task_id": "clean",
                                "output_name": "train_messages",
                            },
                            {
                                "name": "val_messages",
                                "resource_id": None,
                                "task_id": "clean",
                                "output_name": "val_messages",
                            },
                        ],
                        "outputs": [{"name": "trained_model"}],
                    }
                ],
            },
        ),
        ProjectNode(
            id="eval",
            type="evaluator",
            title="Evaluator",
            executor="contract_probe",
            config={
                "resources": resources,
                "tasks": [
                    {
                        "kind": "checkpoint_eval",
                        "id": "cand_eval",
                        "stage_id": "eval",
                        "inputs": [
                            {
                                "name": "trained_model",
                                "resource_id": None,
                                "task_id": "run_train",
                                "output_name": "trained_model",
                            },
                            {
                                "name": "val_messages",
                                "resource_id": None,
                                "task_id": "clean",
                                "output_name": "val_messages",
                            },
                        ],
                        "outputs": [{"name": "eval_report"}],
                    }
                ],
            },
        ),
    ]
    graph = ProjectGraph(
        graph_id=project.graph.graph_id,
        project_id=project.project.project_id,
        template=project.graph.template,
        lifecycle="development",
        nodes=nodes,
        edges=project.graph.edges,
    )
    return store, store.save(ProjectInfo(project=project.project, graph=graph))


def _file_snapshot(root) -> dict[str, bytes]:
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }
