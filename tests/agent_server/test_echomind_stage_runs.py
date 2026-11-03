from uuid import UUID, uuid4

import pytest
from pyromind_runtime.domain.experiment_plans import PlanRevision, WorkflowPlan

from openhands.agent_server.echomind_projects import ProjectStore
from openhands.agent_server.echomind_stage_runs import (
    ProjectRunStore,
    StageRun,
    project_node_config_digest,
    run_project_node,
)


def _plan() -> WorkflowPlan:
    return WorkflowPlan.model_validate(
        {
            "skill": "experiment-planning",
            "name": "AVI experiment",
            "goal": "Validate a Dataset -> Inference -> Evaluator -> Train graph.",
            "resources": [
                {
                    "id": "images",
                    "name": "AVI images",
                    "uri": "/datasets/pcb_avi_eval_3/",
                }
            ],
            "stages": [
                {"id": "dataset", "name": "Dataset"},
                {"id": "inference", "name": "Inference"},
                {"id": "evaluator", "name": "Evaluator"},
                {"id": "train", "name": "Train"},
            ],
            "edges": [
                {"source": "dataset", "target": "inference"},
                {"source": "inference", "target": "evaluator"},
                {"source": "evaluator", "target": "train"},
            ],
        }
    )


def test_stage_runs_and_artifacts_persist_under_canonical_project_directory(
    tmp_path,
):
    project_id = uuid4().hex
    canonical_project_id = str(UUID(project_id))
    project_store = ProjectStore.for_conversations_dir(tmp_path / "conversations")
    project = project_store.bind_plan(
        PlanRevision(
            project_id=project_id,
            revision=1,
            plan=_plan(),
            confirmed_by="42",
        ),
        echomind_conversation_id="conversation-1",
        owner_user_id="42",
    )

    run = run_project_node(
        project=project,
        node=project.graph.nodes[1],
        run_store=ProjectRunStore(project_store.root),
    )

    assert run.status == "succeeded"
    assert run.project_id == project_id
    assert len(run.artifacts) == 1
    assert not (project_store.root / project_id).exists()
    assert (
        project_store.root / canonical_project_id / "runs" / f"{run.run_id}.json"
    ).is_file()

    reloaded_store = ProjectRunStore(project_store.root)
    assert [item.run_id for item in reloaded_store.list(project_id)] == [run.run_id]
    artifact = reloaded_store.load_artifact(
        project_id=project_id,
        artifact_id=run.artifacts[0].artifact_id,
    )
    assert artifact.artifact.project_id == project_id
    assert artifact.content["node_id"] == "inference"
    assert artifact.content["validation_mode"] == "contract_probe"


def test_all_four_nodes_create_persistent_validation_runs(tmp_path):
    project_store = ProjectStore.for_conversations_dir(tmp_path / "conversations")
    project = project_store.bind_plan(
        PlanRevision(
            project_id=uuid4().hex,
            revision=1,
            plan=_plan(),
            confirmed_by="42",
        ),
        echomind_conversation_id="conversation-1",
        owner_user_id="42",
    )
    run_store = ProjectRunStore(project_store.root)

    runs = [
        run_project_node(
            project=project,
            node=node,
            run_store=run_store,
            request_id=f"validation-{node.id}",
        )
        for node in project.graph.nodes
    ]

    assert {run.node_type for run in runs} == {
        "dataset",
        "inference",
        "evaluator",
        "train",
    }
    assert all(run.mode == "node_validation" for run in runs)
    assert all(run.status == "succeeded" for run in runs)
    assert all(run.input_refs == ["/datasets/pcb_avi_eval_3/"] for run in runs)
    assert [artifact.kind for run in runs for artifact in run.artifacts] == [
        "node_validation_report"
    ] * 4
    assert len(run_store.list(project.project.project_id)) == 4


def test_node_validation_request_id_is_idempotent(tmp_path):
    project_store = ProjectStore.for_conversations_dir(tmp_path / "conversations")
    project = project_store.bind_plan(
        PlanRevision(
            project_id=uuid4().hex,
            revision=1,
            plan=_plan(),
            confirmed_by="42",
        ),
        echomind_conversation_id="conversation-1",
        owner_user_id="42",
    )
    run_store = ProjectRunStore(project_store.root)
    node = project.graph.nodes[0]

    first = run_project_node(
        project=project,
        node=node,
        run_store=run_store,
        request_id="validation-dataset",
    )
    second = run_project_node(
        project=project,
        node=node,
        run_store=run_store,
        request_id="validation-dataset",
    )

    assert second.run_id == first.run_id
    assert len(run_store.list(project.project.project_id)) == 1
    assert first.node_config_digest == project_node_config_digest(node)


def test_request_id_cannot_be_reused_for_another_node(tmp_path):
    project_store = ProjectStore.for_conversations_dir(tmp_path / "conversations")
    project = project_store.bind_plan(
        PlanRevision(
            project_id=uuid4().hex,
            revision=1,
            plan=_plan(),
            confirmed_by="42",
        ),
        echomind_conversation_id="conversation-1",
        owner_user_id="42",
    )
    run_store = ProjectRunStore(project_store.root)
    run_project_node(
        project=project,
        node=project.graph.nodes[0],
        run_store=run_store,
        request_id="shared-request",
    )

    with pytest.raises(ValueError, match="request_id"):
        run_project_node(
            project=project,
            node=project.graph.nodes[1],
            run_store=run_store,
            request_id="shared-request",
        )


def test_executor_failure_is_persisted(monkeypatch, tmp_path):
    project_store = ProjectStore.for_conversations_dir(tmp_path / "conversations")
    project = project_store.bind_plan(
        PlanRevision(
            project_id=uuid4().hex,
            revision=1,
            plan=_plan(),
            confirmed_by="42",
        ),
        echomind_conversation_id="conversation-1",
        owner_user_id="42",
    )
    run_store = ProjectRunStore(project_store.root)

    def fail_executor(_: str):
        raise ValueError("executor unavailable")

    monkeypatch.setattr(
        "openhands.agent_server.echomind_stage_runs.get_project_node_executor",
        fail_executor,
    )
    run = run_project_node(
        project=project,
        node=project.graph.nodes[1],
        run_store=run_store,
        request_id="failing-validation",
    )

    assert run.status == "failed"
    assert "executor unavailable" in (run.error or "")
    assert run_store.list(project.project.project_id)[0].status == "failed"


def test_legacy_stage_run_defaults_validation_metadata():
    run = StageRun.model_validate(
        {
            "run_id": "legacy-run",
            "project_id": uuid4().hex,
            "node_id": "dataset",
            "node_type": "dataset",
            "executor": "contract_probe",
            "status": "succeeded",
            "message": "Legacy run",
            "created_at": "2026-11-02T08:00:00Z",
        }
    )

    assert run.mode == "node_validation"
    assert run.request_id is None
    assert run.node_config_digest == ""
