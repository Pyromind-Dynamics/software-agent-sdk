from uuid import UUID, uuid4

from pyromind_runtime.domain.experiment_plans import PlanRevision, WorkflowPlan

from openhands.agent_server.echomind_projects import ProjectStore
from openhands.agent_server.echomind_stage_runs import ProjectRunStore, run_project_node


def _plan() -> WorkflowPlan:
    return WorkflowPlan.model_validate(
        {
            "skill": "avi-single-turn",
            "name": "AVI experiment",
            "goal": "Validate a Dataset -> Inference -> Evaluator -> Train graph.",
            "resources": [],
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
