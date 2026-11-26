from datetime import UTC, datetime
from uuid import uuid4

import pytest

from pyromind_agent_server.echomind_graph_runs import ArtifactRef, GraphRunStore
from pyromind_agent_server.echomind_model_reports import GraphModelReportStore
from pyromind_agent_server.echomind_projects import (
    ProjectEdge,
    ProjectNode,
    ProjectNodeInput,
    ProjectNodeInputBinding,
    ProjectNodeOutput,
)
from pyromind_agent_server.echomind_publication import (
    PromotionPolicy,
    PublicationStore,
    PublishedGraph,
    RuntimeRequirements,
)
from pyromind_agent_server.echomind_stage_executors import StageArtifactPayload
from pyromind_agent_server.echomind_stage_runs import ProjectRunStore, StageRun


@pytest.fixture
def archived_candidate(tmp_path):
    root = tmp_path / "projects"
    project_id, release_id, run_id, stage_id = (str(uuid4()) for _ in range(4))
    node = ProjectNode(
        id="eval",
        type="evaluator",
        title="Eval",
        executor="evaluator_runtime",
        inputs={
            name: ProjectNodeInput(
                artifact_type="model.checkpoint",
                schema_ref="test://input",
                binding=ProjectNodeInputBinding(kind="run_input", name=name),
            )
            for name in ("candidate_model", "validation_dataset", "base_model")
        },
        outputs={
            "eval_report": ProjectNodeOutput(
                artifact_type="evaluation.report", schema_ref="test://report"
            )
        },
        config={
            "evaluator_runtime": {
                "workflow": {"nodes": [{"data": {"nodeType": "ModelMergeLoraNode"}}]},
                "workflow_bindings": {
                    "nodes[0].data.config.lora_path": "candidate_model"
                },
                "workflow_templates": {
                    "nodes[0].data.config.output_path": (
                        "/workspace/runs/{graph_run_id}/{stage_run_id}/merged"
                    )
                },
                "command_template": (
                    "python score.py --evaluation-config "
                    "/workspace/versions/v1/rules.json"
                ),
            }
        },
    )
    published = PublishedGraph.build(
        release_id=release_id,
        project_id=project_id,
        source_graph_id=str(uuid4()),
        source_graph_digest="sha256:" + "a" * 64,
        runtime_requirements=RuntimeRequirements(),
        promotion_policy=PromotionPolicy(metric="pass_rate"),
        nodes=[node],
        edges=[],
        published_by="owner",
    )
    PublicationStore(root).save(published)
    graph_runs = GraphRunStore(root)
    run = graph_runs.create(
        project_id=project_id,
        release_id=release_id,
        release_digest=published.content_sha256,
        request_id="first-round",
        run_id=run_id,
        trigger="manual",
        inputs={
            "candidate_model": "/workspace/model/adapter",
            "validation_dataset": "/workspace/dataset/val.jsonl",
            "base_model": "/workspace/model/base",
            "tracer_batch": {"batch_id": "batch-0"},
        },
    )
    now = datetime.now(UTC)
    graph_runs.save(
        run.model_copy(
            update={
                "status": "succeeded",
                "outcome": "baseline_retained",
                "started_at": now,
                "finished_at": now,
            }
        )
    )
    stages = ProjectRunStore(root)
    stage = StageRun(
        run_id=stage_id,
        graph_run_id=run_id,
        project_id=project_id,
        node_id="eval",
        node_type="evaluator",
        mode="published_run",
        executor="evaluator_runtime",
        status="succeeded",
        message="Evaluated",
        created_at=now,
        started_at=now,
        finished_at=now,
    )
    artifact, _ = stages.write_artifact(
        stage,
        StageArtifactPayload(
            kind="evaluation.report", content={"uri": "/workspace/eval/report.json"}
        ),
    )
    stages.save(
        stage.model_copy(
            update={
                "artifacts": [artifact],
                "output_refs": [
                    ArtifactRef(
                        artifact_type="evaluation.report",
                        uri=artifact.uri,
                        metadata={"artifact_id": artifact.artifact_id},
                    )
                ],
            }
        )
    )
    return root, project_id, run_id, stage_id


def test_catalog_projects_existing_run_and_does_not_promote(archived_candidate):
    root, project_id, run_id, stage_id = archived_candidate
    store = GraphModelReportStore(root)
    (record,) = store.list_candidates(project_id)
    assert record.candidate_id == run_id
    assert record.source_batch_id == "batch-0"
    assert record.model_uri == "/workspace/model/adapter"
    assert record.deployment_model_uri == f"/workspace/runs/{run_id}/{stage_id}/merged"
    assert record.evaluation_config_uri == "/workspace/versions/v1/rules.json"
    assert record.report_status == "pending"
    assert record.metrics == {}
    assert store.list_candidates(str(uuid4())) == []
    assert GraphRunStore(root).load(project_id=project_id, run_id=run_id).outcome == (
        "baseline_retained"
    )
    refreshed = record.model_copy(
        update={"report_status": "ready", "metrics": {"x": 1}}
    )
    store.save_report(refreshed)
    assert store.list_candidates(project_id) == [refreshed]


@pytest.mark.parametrize("selection", ["latest_lora_checkpoint", "merged_model"])
def test_catalog_resolves_the_evaluated_train_output(archived_candidate, selection):
    root, project_id, run_id, _ = archived_candidate
    store = GraphModelReportStore(root)
    run = store.runs.load(project_id=project_id, run_id=run_id)
    published = store.publications.load(
        project_id=project_id, release_id=run.release_id
    )
    eval_node = published.nodes[0]
    training = ProjectNode(
        id="sft",
        type="train",
        title="SFT",
        executor="train_runtime",
        config={"train_runtime": {"artifact_selection": selection}},
        inputs={
            "base_model": ProjectNodeInput(
                artifact_type="model.checkpoint",
                schema_ref="test://input",
                binding=ProjectNodeInputBinding(
                    kind="literal", value="/workspace/models/training-start"
                ),
            )
        },
        outputs={
            "trained_model": ProjectNodeOutput(
                artifact_type="model.checkpoint", schema_ref="test://input"
            )
        },
    )
    input_model = eval_node.inputs["candidate_model"].model_copy(
        update={
            "binding": ProjectNodeInputBinding(
                kind="node_output", node_id=training.id, port="trained_model"
            )
        }
    )
    node = eval_node.model_copy(
        update={
            "inputs": {
                "candidate_model": input_model,
                "validation_dataset": eval_node.inputs["validation_dataset"],
            },
            "config": {"evaluator_runtime": {}},
        }
    )
    replacement = PublishedGraph.build(
        release_id=str(uuid4()),
        project_id=project_id,
        source_graph_id=published.source_graph_id,
        source_graph_digest=published.source_graph_digest,
        runtime_requirements=published.runtime_requirements,
        promotion_policy=published.promotion_policy,
        nodes=[training, node],
        edges=[ProjectEdge(source=training.id, target=node.id)],
        published_by="owner",
    )
    store.publications.save(replacement)
    store.runs.save(
        run.model_copy(
            update={
                "release_id": replacement.release_id,
                "release_digest": replacement.content_sha256,
            }
        )
    )
    (evaluation,) = store.stages.list_for_graph_run(
        project_id=project_id, graph_run_id=run_id
    )
    trained_model = "/workspace/runs/selected-output"
    store.stages.save(
        evaluation.model_copy(
            update={
                "run_id": str(uuid4()),
                "node_id": training.id,
                "node_type": "train",
                "executor": training.executor,
                "artifacts": [],
                "output_refs": [
                    ArtifactRef(
                        artifact_type="model.checkpoint",
                        uri=trained_model,
                        metadata={"port": "trained_model", "source_uri": trained_model},
                    )
                ],
            }
        )
    )
    (record,) = store.list_candidates(project_id)
    store.save_report(
        record.model_copy(
            update={"deployment_model_uri": None, "report_status": "ready"}
        )
    )
    (cached,) = store.list_candidates(project_id)
    assert cached.report_status == "ready"
    assert cached.model_uri == trained_model
    assert cached.training_base_uri == "/workspace/models/training-start"
    assert cached.deployment_model_uri == (
        trained_model if selection == "merged_model" else None
    )
